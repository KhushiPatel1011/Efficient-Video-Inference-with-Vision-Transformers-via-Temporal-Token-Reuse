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

    best_row = None
    if "is_best" in df.columns and df["is_best"].astype(str).str.lower().isin(["true", "1"]).any():
        best_row = df[df["is_best"].astype(str).str.lower().isin(["true", "1"])].iloc[0]
    else:
        safe_df = df[df["top1_match_rate"] >= 0.90]
        best_row = safe_df.sort_values("avg_speedup", ascending=False).iloc[0] if not safe_df.empty else df.sort_values("avg_speedup", ascending=False).iloc[0]

    # ----------------------------
    # Plot 1: Speedup vs r_match
    # ----------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["r_match"], df["avg_speedup"], marker="o", linewidth=2)
    plt.scatter([best_row["r_match"]], [best_row["avg_speedup"]], marker="*", s=180, zorder=5)
    plt.annotate(
        f"Best r={best_row['r_match']:.2f}",
        xy=(best_row["r_match"], best_row["avg_speedup"]),
        xytext=(best_row["r_match"] + 0.04, best_row["avg_speedup"]),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    plt.xlim(0.0, 1.0)
    plt.xlabel("Stable Ratio / r_match (0 = no reuse, 1 = aggressive reuse)")
    plt.ylabel("Average Speedup (baseline / reuse)")
    plt.title("KV-Cache Speedup Across Full Reuse Range")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_tradeoff_speedup.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot 2: Match rate vs r_match
    # ----------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["r_match"], df["top1_match_rate"], marker="o", linewidth=2)
    plt.axhline(0.90, linestyle="--", linewidth=1, label="Acceptable floor = 0.90")
    plt.scatter([best_row["r_match"]], [best_row["top1_match_rate"]], marker="*", s=180, zorder=5)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Stable Ratio / r_match (0 = no reuse, 1 = aggressive reuse)")
    plt.ylabel("Top-1 Match Rate")
    plt.title("Prediction Consistency Across Full Reuse Range")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_tradeoff_match.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot 3: Speedup vs Match Rate
    # ----------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["avg_speedup"], df["top1_match_rate"], marker="o", linewidth=2)
    for _, row in df.iterrows():
        label = f"r={row['r_match']:.2f}"
        plt.annotate(
            label,
            (row["avg_speedup"], row["top1_match_rate"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    plt.scatter([best_row["avg_speedup"]], [best_row["top1_match_rate"]], marker="*", s=180, zorder=5)
    plt.axhline(0.90, linestyle="--", linewidth=1, label="Acceptable floor = 0.90")
    plt.xlabel("Average Speedup")
    plt.ylabel("Top-1 Match Rate")
    plt.title("Best Operating Point: Speedup vs Prediction Consistency")
    plt.legend()
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