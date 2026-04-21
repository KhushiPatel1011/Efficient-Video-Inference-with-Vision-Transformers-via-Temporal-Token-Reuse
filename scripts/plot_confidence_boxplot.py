"""
Generating confidence-delta box plot

Reads:
    results/block_skip_thresh_0_70.csv
    results/block_skip_thresh_0_75.csv
    results/block_skip_thresh_0_80.csv
    results/block_skip_thresh_0_85.csv
    results/block_skip_thresh_0_90.csv
    results/block_skip_thresh_0_95.csv

Writes:
    results/plots/block_skip_confdelta_boxplot.png
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_csv(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x, default=None):
    try:
        if x is None or x == "" or x == "None":
            return default
        return float(x)
    except Exception:
        return default


def main():
    labels = []
    conf_data = []

    for t in THRESHOLDS:
        csv_name = f"block_skip_thresh_{str(t).replace('.', '_')}.csv"
        csv_path = RESULTS_DIR / csv_name

        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        rows = load_csv(csv_path)
        skip_rows = [r for r in rows if r.get("decision") == "skip"]

        conf_values = [
            to_float(r.get("conf_delta"))
            for r in skip_rows
            if to_float(r.get("conf_delta")) is not None
        ]

        if conf_values:
            labels.append(f"{t:.2f}")
            conf_data.append(conf_values)

    if not conf_data:
        print("No confidence-delta data found.")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(conf_data, labels=labels)
    plt.xlabel("CLS Threshold")
    plt.ylabel("Confidence Delta")
    plt.title("Confidence Variation Across Frames Under Block-Skip Reuse")
    plt.tight_layout()

    out_path = PLOTS_DIR / "block_skip_confdelta_boxplot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()