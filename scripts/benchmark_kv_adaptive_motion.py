import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ADAPTIVE_R_MATCH = {
    "low": 0.90,
    "medium": 0.75,
    "high": 0.50,
}


def run_experiment(frames_dir: Path, max_frames: int, model: str, r_match: float, tmp_csv: Path) -> None:
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
    df = pd.read_csv(csv_path)

    reuse_df = df[df["mode"].isin(["reuse", "match"])].copy()
    if reuse_df.empty:
        raise ValueError(f"No reuse rows found in {csv_path}")

    avg_baseline_ms = reuse_df["baseline_ms"].dropna().mean()
    avg_reuse_ms = reuse_df["latency_ms"].dropna().mean()
    avg_speedup = reuse_df["speedup"].dropna().mean()
    match_rate = reuse_df["match"].dropna().mean()

    return {
        "video_id": video_id,
        "motion_bucket": motion_bucket,
        "policy": "adaptive_bucket",
        "r_match": r_match,
        "frames_used": int(len(df)),
        "reuse_frames": int(len(reuse_df)),
        "avg_baseline_ms": round(float(avg_baseline_ms), 3),
        "avg_reuse_ms": round(float(avg_reuse_ms), 3),
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
        "--out-csv",
        type=str,
        default="results/kv_motion_adaptive.csv",
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

    out_csv = Path(args.out_csv)
    tmp_dir = out_csv.parent / "tmp_kv_motion_adaptive"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n=== KV-CACHE ADAPTIVE MOTION BENCHMARK ===")
    print(f"manifest: {manifest_path}")
    print(f"adaptive policy: {ADAPTIVE_R_MATCH}")

    for _, row in manifest_df.iterrows():
        video_id = str(row["video_id"])
        frames_dir = Path(str(row["path"]))
        motion_bucket = str(row["motion_bucket"]).strip().lower()

        if motion_bucket not in ADAPTIVE_R_MATCH:
            raise ValueError(
                f"Unknown motion bucket '{motion_bucket}' for video '{video_id}'. "
                f"Expected one of: {list(ADAPTIVE_R_MATCH.keys())}"
            )

        r_match = ADAPTIVE_R_MATCH[motion_bucket]
        tmp_csv = tmp_dir / f"{video_id}_adaptive.csv"

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
            f"[{motion_bucket}] {video_id} | adaptive r_match={r_match:.2f} | "
            f"speedup={summary['avg_speedup']:.3f}x | "
            f"match={summary['top1_match_rate']:.3f}"
        )

    summary_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print(f"\nSaved adaptive motion CSV: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()