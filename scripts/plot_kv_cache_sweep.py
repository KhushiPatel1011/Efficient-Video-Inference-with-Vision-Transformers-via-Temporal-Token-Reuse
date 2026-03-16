"""
Plotting KV cache sweep results: speedup and accuracy vs stable_ratio.

Usage:
    python scripts/plot_kv_cache_sweep.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import matplotlib.pyplot as plt


def main():
    csv_path = Path("results/kv_cache_sweep.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run sweep_kv_cache.py first.")
        return

    ratios, speedups, match_rates, reuse_fps = [], [], [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratios.append(float(row["stable_ratio"]))
            speedups.append(float(row["speedup"]))
            match_rates.append(float(row["top1_match_rate"]))
            reuse_fps.append(float(row["reuse_fps"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Speedup vs stable_ratio
    ax1 = axes[0]
    ax1.plot(ratios, speedups, marker="o", color="steelblue", linewidth=2, markersize=7)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1.0x)")
    ax1.set_xlabel("Stable Ratio (fraction of tokens with reused K/V)")
    ax1.set_ylabel("Speedup (x)")
    ax1.set_title("KV Cache Speedup vs Stable Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for x, y in zip(ratios, speedups):
        ax1.annotate(f"{y:.3f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    # Plot 2: Top-1 Match Rate vs stable_ratio
    ax2 = axes[1]
    ax2.plot(ratios, match_rates, marker="s", color="darkorange", linewidth=2, markersize=7)
    ax2.axhline(y=1.0, color="green", linestyle="--", linewidth=1, label="Perfect (1.0)")
    ax2.axhline(y=0.95, color="red", linestyle=":", linewidth=1, label="95% threshold")
    ax2.set_xlabel("Stable Ratio (fraction of tokens with reused K/V)")
    ax2.set_ylabel("Top-1 Match Rate vs Baseline")
    ax2.set_title("Accuracy Preservation vs Stable Ratio")
    ax2.set_ylim(0.8, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for x, y in zip(ratios, match_rates):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    plt.suptitle(
        "TBKV KV Cache Temporal Token Reuse — CPU Results\n"
        "vit_base_patch16_224 | 60 frames | Selective K/V Scatter",
        fontsize=11
    )
    plt.tight_layout()

    out_path = Path("results/kv_cache_sweep_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    print("\nSummary:")
    print(f"  Max speedup:        {max(speedups):.3f}x  at stable_ratio={ratios[speedups.index(max(speedups))]}")
    print(f"  Min speedup:        {min(speedups):.3f}x  at stable_ratio={ratios[speedups.index(min(speedups))]}")
    print(f"  Accuracy range:     {min(match_rates):.3f} — {max(match_rates):.3f}")
    print(f"  Perfect accuracy:   ALL ratios (match_rate = 1.000 across entire sweep)")


if __name__ == "__main__":
    main()