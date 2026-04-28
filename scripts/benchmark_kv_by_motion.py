import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_experiment(frames_dir: Path, max_frames: int, model: str, r_match: float, tmp_csv: Path) -> None:
    """
    Runs the KV-cache experiment script for one video and one r_match value.
    """
    cmd = [
        sys.executable,
        "scripts/run_kv_cache_reuse.py",
        "--frames", str(frames_dir),
        "--max-frames", str(max_frames),
        "--model", model,
        "--r-match", str(r_match),
        "--out-csv", str(tmp_csv),
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def summarize_run(csv_path: Path, r_match: float, motion_bucket: str, video_id: str) -> dict:
    """
    Summarizes one per-frame CSV into one row.
    """
    df = pd.read_csv(csv_path)

    reuse_df = df[df["mode"].isin(["reuse", "match"])].copy()
    if reuse_df.empty:
        raise ValueError(f"No reuse rows found in {csv_path}")

    avg_baseline_ms = reuse_df["baseline_ms"].dropna().mean()
    reuse_col = "reuse_ms" if "reuse_ms" in reuse_df.columns else "latency_ms"
    avg_reuse_ms = reuse_df[reuse_col].dropna().mean()
    avg_speedup = avg_baseline_ms / avg_reuse_ms if avg_reuse_ms > 0 else 0.0
    match_rate = reuse_df["match"].dropna().mean()

    return {
        "video_id": video_id,
        "motion_bucket": motion_bucket,
        "r_match": r_match,
        "stable_ratio": r_match,
        "frames_used": int(len(df)),
        "reuse_frames": int(len(reuse_df)),
        "avg_baseline_ms": round(float(avg_baseline_ms), 3),
        "avg_reuse_ms": round(float(avg_reuse_ms), 3),
        "baseline_fps": round(1000.0 / float(avg_baseline_ms), 3),
        "reuse_fps": round(1000.0 / float(avg_reuse_ms), 3),
        "avg_speedup": round(float(avg_speedup), 3),
        "top1_match_rate": round(float(match_rate), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/experiments/video_manifest.csv",
        help="CSV listing video_id,path,motion_bucket",
    )
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--r-values",
        type=str,
        default="0.00,0.25,0.50,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1.00",
        help="Comma-separated r_match/stable-ratio values across the full 0–1 range",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/kv_motion_comparison.csv",
        help="Output summary CSV",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)

    required_cols = {"video_id", "path", "motion_bucket"}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    r_values = [float(x.strip()) for x in args.r_values.split(",") if x.strip()]
    out_csv = Path(args.out_csv)
    tmp_dir = out_csv.parent / "tmp_kv_motion"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n=== KV-CACHE MOTION-BUCKET BENCHMARK ===")
    print(f"manifest: {manifest_path}")
    print(f"r_match values: {r_values}")

    for _, row in manifest_df.iterrows():
        video_id = str(row["video_id"])
        frames_dir = Path(str(row["path"]))
        motion_bucket = str(row["motion_bucket"])

        for r_match in r_values:
            tmp_csv = tmp_dir / f"{video_id}_r_{str(r_match).replace('.', '_')}.csv"

            run_experiment(
                frames_dir=frames_dir,
                max_frames=args.max_frames,
                model=args.model,
                r_match=r_match,
                tmp_csv=tmp_csv,
            )

            summary = summarize_run(
                csv_path=tmp_csv,
                r_match=r_match,
                motion_bucket=motion_bucket,
                video_id=video_id,
            )
            rows.append(summary)

            print(
                f"[{motion_bucket}] {video_id} | r_match={r_match:.2f} | "
                f"speedup={summary['avg_speedup']:.3f}x | "
                f"match={summary['top1_match_rate']:.3f}"
            )

    summary_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print(f"\nSaved motion comparison CSV: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()