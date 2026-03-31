import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_experiment(frames_dir: Path, max_frames: int, model: str, r_match: float, tmp_csv: Path) -> None:
    """
    Runs the KV-cache experiment script for a single r_match value.
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


def summarize_run(csv_path: Path, r_match: float) -> dict:
    """
    Reads one per-frame CSV and converts it into one summary row.
    """
    df = pd.read_csv(csv_path)

    # Frame 1+ rows only for reuse-phase metrics
    reuse_df = df[df["mode"] == "reuse"].copy()

    if reuse_df.empty:
        raise ValueError(f"No reuse rows found in {csv_path}")

    video_id = str(df["video_id"].dropna().iloc[0]) if "video_id" in df.columns else csv_path.stem

    avg_baseline_ms = reuse_df["baseline_ms"].dropna().mean()
    avg_reuse_ms = reuse_df["latency_ms"].dropna().mean()
    avg_speedup = reuse_df["speedup"].dropna().mean()
    match_rate = reuse_df["match"].dropna().mean()
    frames_used = len(df)
    reuse_frames = len(reuse_df)

    return {
        "video_id": video_id,
        "r_match": r_match,
        "frames_used": int(frames_used),
        "reuse_frames": int(reuse_frames),
        "avg_baseline_ms": round(float(avg_baseline_ms), 3),
        "avg_reuse_ms": round(float(avg_reuse_ms), 3),
        "avg_speedup": round(float(avg_speedup), 3),
        "top1_match_rate": round(float(match_rate), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Folder of sequential frame images")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to process")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument(
        "--r-values",
        type=str,
        default="0.25,0.50,0.75,0.90",
        help="Comma-separated r_match values to sweep",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/kv_tradeoff_summary.csv",
        help="Summary CSV path",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    out_csv = Path(args.out_csv)
    tmp_dir = out_csv.parent / "tmp_kv_tradeoff"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    r_values = [float(x.strip()) for x in args.r_values.split(",") if x.strip()]
    rows = []

    print("\n=== KV-CACHE TRADEOFF BENCHMARK ===")
    print(f"frames: {frames_dir}")
    print(f"r_match values: {r_values}")

    for r_match in r_values:
        tmp_csv = tmp_dir / f"kv_reuse_r_{str(r_match).replace('.', '_')}.csv"

        run_experiment(
            frames_dir=frames_dir,
            max_frames=args.max_frames,
            model=args.model,
            r_match=r_match,
            tmp_csv=tmp_csv,
        )

        summary = summarize_run(tmp_csv, r_match)
        rows.append(summary)

        print(
            f"r_match={r_match:.2f} | "
            f"baseline={summary['avg_baseline_ms']:.2f} ms | "
            f"reuse={summary['avg_reuse_ms']:.2f} ms | "
            f"speedup={summary['avg_speedup']:.3f}x | "
            f"match={summary['top1_match_rate']:.3f}"
        )

    summary_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print(f"\nSaved summary CSV: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()