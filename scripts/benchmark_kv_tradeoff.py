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
    reuse_col = "reuse_ms" if "reuse_ms" in reuse_df.columns else "latency_ms"
    avg_reuse_ms = reuse_df[reuse_col].dropna().mean()
    avg_speedup = avg_baseline_ms / avg_reuse_ms if avg_reuse_ms > 0 else 0.0
    match_rate = reuse_df["match"].dropna().mean()

    top3_match_rate = reuse_df["top3_match"].dropna().mean() if "top3_match" in reuse_df.columns else None
    top5_match_rate = reuse_df["top5_match"].dropna().mean() if "top5_match" in reuse_df.columns else None
    avg_kl = reuse_df["kl_divergence"].dropna().mean() if "kl_divergence" in reuse_df.columns else None
    avg_conf_delta = reuse_df["conf_delta"].dropna().mean() if "conf_delta" in reuse_df.columns else None
    frames_used = len(df)
    reuse_frames = len(reuse_df)

    return {
        "video_id": video_id,
        "r_match": r_match,
        "stable_ratio": r_match,
        "frames_used": int(frames_used),
        "reuse_frames": int(reuse_frames),
        "avg_baseline_ms": round(float(avg_baseline_ms), 3),
        "avg_reuse_ms": round(float(avg_reuse_ms), 3),
        "baseline_fps": round(1000.0 / float(avg_baseline_ms), 3),
        "reuse_fps": round(1000.0 / float(avg_reuse_ms), 3),
        "avg_speedup": round(float(avg_speedup), 3),
        "top1_match_rate": round(float(match_rate), 3),
        "top3_match_rate": round(float(top3_match_rate), 3) if top3_match_rate is not None else "",
        "top5_match_rate": round(float(top5_match_rate), 3) if top5_match_rate is not None else "",
        "avg_kl_divergence": round(float(avg_kl), 6) if avg_kl is not None else "",
        "avg_conf_delta": round(float(avg_conf_delta), 6) if avg_conf_delta is not None else "",
    }

def select_best_result(rows: list[dict]) -> dict:
    """
    Best result is not just maximum speedup.
    We choose the highest balanced score while preserving prediction quality.

    Score:
        speedup * top1_match
        - 0.25 * KL
        - 0.50 * confidence_delta

    If KL/confidence are unavailable, they are treated as zero.
    """
    scored = []
    for row in rows:
        speedup = float(row["avg_speedup"])
        top1 = float(row["top1_match_rate"])
        kl = float(row["avg_kl_divergence"]) if row["avg_kl_divergence"] != "" else 0.0
        conf = float(row["avg_conf_delta"]) if row["avg_conf_delta"] != "" else 0.0

        score = (speedup * top1) - (0.25 * kl) - (0.50 * conf)
        row["selection_score"] = round(score, 6)
        scored.append(row)

    # Prefer safe operating points first
    safe = [r for r in scored if float(r["top1_match_rate"]) >= 0.90]
    candidates = safe if safe else scored
    return max(candidates, key=lambda r: float(r["selection_score"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Folder of sequential frame images")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to process")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument(
        "--r-values",
        type=str,
        default="0.00,0.25,0.50,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1.00",
        help="Comma-separated r_match/stable-ratio values to sweep across the full 0–1 range",
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

    best = select_best_result(rows)

    summary_df = pd.DataFrame(rows)
    summary_df["is_best"] = summary_df["r_match"].astype(float).eq(float(best["r_match"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print(f"\nSaved summary CSV: {out_csv}")
    print("\nBEST OPERATING POINT")
    print(
        f"r_match={best['r_match']:.2f} | "
        f"speedup={best['avg_speedup']:.3f}x | "
        f"top1={best['top1_match_rate']:.3f} | "
        f"KL={best['avg_kl_divergence']} | "
        f"conf_delta={best['avg_conf_delta']} | "
        f"score={best['selection_score']}"
    )
    print("Done.")


if __name__ == "__main__":
    main()