from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    in_csv = Path("results/kv_motion_comparison.csv")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df = pd.read_csv(in_csv)

    required_cols = {
        "motion_bucket",
        "r_match",
        "avg_speedup",
        "top1_match_rate",
        "avg_baseline_ms",
        "avg_reuse_ms",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Aggregate by motion bucket
    bucket_df = (
        df.groupby("motion_bucket", as_index=False)
        .agg({
            "avg_speedup": "mean",
            "top1_match_rate": "mean",
            "avg_baseline_ms": "mean",
            "avg_reuse_ms": "mean",
        })
        .sort_values("motion_bucket")
    )

    # ----------------------------
    # Plot 1: Speedup by motion bucket
    # ----------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(bucket_df["motion_bucket"], bucket_df["avg_speedup"])
    plt.xlabel("Motion Bucket")
    plt.ylabel("Average Speedup")
    plt.title("KV-Cache Speedup by Motion Type")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_motion_speedup.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot 2: Match rate by motion bucket
    # ----------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(bucket_df["motion_bucket"], bucket_df["top1_match_rate"])
    plt.xlabel("Motion Bucket")
    plt.ylabel("Top-1 Match Rate")
    plt.title("Prediction Consistency by Motion Type")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_motion_match.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot 3: Baseline vs Reuse latency by motion bucket
    # ----------------------------
    x = range(len(bucket_df))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar([i - width/2 for i in x], bucket_df["avg_baseline_ms"], width=width, label="Baseline")
    plt.bar([i + width/2 for i in x], bucket_df["avg_reuse_ms"], width=width, label="KV-Cache")
    plt.xticks(list(x), bucket_df["motion_bucket"])
    plt.xlabel("Motion Bucket")
    plt.ylabel("Latency (ms)")
    plt.title("Baseline vs KV-Cache Latency by Motion Type")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_motion_latency.png", dpi=200)
    plt.close()

    print("\nSaved plots:")
    print(f"- {out_dir / 'kv_motion_speedup.png'}")
    print(f"- {out_dir / 'kv_motion_match.png'}")
    print(f"- {out_dir / 'kv_motion_latency.png'}")
    print("Done.")


if __name__ == "__main__":
    main()