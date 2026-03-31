from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    in_csv = Path("results/kv_tradeoff_summary.csv")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df = pd.read_csv(in_csv)

    required_cols = {
        "r_match",
        "avg_speedup",
        "top1_match_rate",
        "avg_baseline_ms",
        "avg_reuse_ms",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Sort for clean plotting
    df = df.sort_values("r_match").reset_index(drop=True)

    # ----------------------------
    # Plot 1: Speedup vs r_match
    # ----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(df["r_match"], df["avg_speedup"], marker="o")
    plt.xlabel("r_match")
    plt.ylabel("Average Speedup (baseline / reuse)")
    plt.title("KV-Cache Speedup vs Matching Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_tradeoff_speedup.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot 2: Match rate vs r_match
    # ----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(df["r_match"], df["top1_match_rate"], marker="o")
    plt.xlabel("r_match")
    plt.ylabel("Top-1 Match Rate")
    plt.title("Prediction Consistency vs Matching Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_tradeoff_match.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot 3: Speedup vs Match Rate
    # ----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(df["avg_speedup"], df["top1_match_rate"], marker="o")
    for _, row in df.iterrows():
        plt.annotate(
            f"r={row['r_match']}",
            (row["avg_speedup"], row["top1_match_rate"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    plt.xlabel("Average Speedup")
    plt.ylabel("Top-1 Match Rate")
    plt.title("KV-Cache Tradeoff: Speedup vs Consistency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_tradeoff_pareto.png", dpi=200)
    plt.close()

    print("\nSaved plots:")
    print(f"- {out_dir / 'kv_tradeoff_speedup.png'}")
    print(f"- {out_dir / 'kv_tradeoff_match.png'}")
    print(f"- {out_dir / 'kv_tradeoff_pareto.png'}")
    print("Done.")


if __name__ == "__main__":
    main()