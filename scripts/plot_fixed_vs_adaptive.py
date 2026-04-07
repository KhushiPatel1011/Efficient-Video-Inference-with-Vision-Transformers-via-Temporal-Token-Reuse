import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metric(df, metric, ylabel, filename):
    plt.figure()

    motion_buckets = sorted(df["motion_bucket"].unique())
    x = range(len(motion_buckets))

    fixed_vals = []
    adaptive_vals = []

    for bucket in motion_buckets:
        bucket_df = df[df["motion_bucket"] == bucket]

        fixed = bucket_df[bucket_df["policy"] == "fixed"][metric].mean()
        adaptive = bucket_df[bucket_df["policy"] == "adaptive"][metric].mean()

        fixed_vals.append(fixed)
        adaptive_vals.append(adaptive)

    width = 0.35

    plt.bar([i - width/2 for i in x], fixed_vals, width, label="Fixed")
    plt.bar([i + width/2 for i in x], adaptive_vals, width, label="Adaptive")

    plt.xticks(x, motion_buckets)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Fixed vs Adaptive)")
    plt.legend()

    out_path = Path("results") / filename
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def main():
    csv_path = Path("results/kv_motion_fixed_vs_adaptive.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    plot_metric(df, "avg_speedup", "Average Speedup", "fixed_vs_adaptive_speedup.png")
    plot_metric(df, "avg_reuse_ms", "Latency (ms)", "fixed_vs_adaptive_latency.png")
    plot_metric(df, "top1_match_rate", "Top-1 Match Rate", "fixed_vs_adaptive_match.png")


if __name__ == "__main__":
    main()