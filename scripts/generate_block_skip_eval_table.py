"""
Here, we are generating evaluation table for block-skip reuse.

Needs: 
    results/block_skip_threshold_sweep.csv

output:
    results/block_skip_eval_table.csv

This will reformat the sweep output into a comparison table
"""

import csv
from pathlib import Path


INPUT_CSV = Path("results/block_skip_threshold_sweep.csv")
OUTPUT_CSV = Path("results/block_skip_eval_table.csv")


def load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def main():
    rows = load_rows(INPUT_CSV)

    table_rows = []
    for row in rows:
        threshold = to_float(row, "cls_threshold")
        skip_rate = to_float(row, "skip_rate")
        speedup = to_float(row, "speedup")
        top1 = to_float(row, "top1_match_rate")
        top3 = to_float(row, "top3_match_rate")
        top5 = to_float(row, "top5_match_rate")
        avg_kl = to_float(row, "avg_kl_divergence")
        avg_cd = to_float(row, "avg_conf_delta")

        table_rows.append({
            "Method": "Block Skip Reuse",
            "Dataset": "medical_frames",
            "Motion": "low",
            "Cache_After_Block": 7,
            "CLS_Threshold": round(threshold, 2),
            "Skip_Rate": round(skip_rate, 3),
            "Speedup_x": round(speedup, 3),
            "Top1_Match": round(top1, 3),
            "Top3_Match": round(top3, 3),
            "Top5_Match": round(top5, 3),
            "Avg_KL_Div": round(avg_kl, 6),
            "Avg_Conf_Delta": round(avg_cd, 6),
        })

    # Save formatted table
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)

    print("\n=== BLOCK-SKIP EVALUATION TABLE ===")
    print(
        f"{'thr':>6} | {'skip':>6} | {'spd':>6} | {'top1':>6} | {'top3':>6} | {'top5':>6} | {'KL':>10} | {'conf_d':>8}"
    )
    print("-" * 78)

    for r in table_rows:
        print(
            f"{r['CLS_Threshold']:>6.2f} | "
            f"{r['Skip_Rate']:>6.3f} | "
            f"{r['Speedup_x']:>6.3f} | "
            f"{r['Top1_Match']:>6.3f} | "
            f"{r['Top3_Match']:>6.3f} | "
            f"{r['Top5_Match']:>6.3f} | "
            f"{r['Avg_KL_Div']:>10.6f} | "
            f"{r['Avg_Conf_Delta']:>8.6f}"
        )

    # Best by speedup
    best_speedup = max(table_rows, key=lambda r: r["Speedup_x"])

    # Best balanced point:
    # score = top1 + 0.5*top5 + 0.5*speedup - 0.25*KL
    best_balanced = max(
        table_rows,
        key=lambda r: (
            r["Top1_Match"]
            + 0.5 * r["Top5_Match"]
            + 0.5 * r["Speedup_x"]
            - 0.25 * r["Avg_KL_Div"]
        )
    )

    print("\n=== BEST OPERATING POINTS ===")
    print(
        f"Best speedup: threshold={best_speedup['CLS_Threshold']:.2f}, "
        f"speedup={best_speedup['Speedup_x']:.3f}x, "
        f"top1={best_speedup['Top1_Match']:.3f}, "
        f"top5={best_speedup['Top5_Match']:.3f}"
    )
    print(
        f"Best balanced point: threshold={best_balanced['CLS_Threshold']:.2f}, "
        f"speedup={best_balanced['Speedup_x']:.3f}x, "
        f"top1={best_balanced['Top1_Match']:.3f}, "
        f"top5={best_balanced['Top5_Match']:.3f}, "
        f"KL={best_balanced['Avg_KL_Div']:.6f}"
    )

    print(f"\nSaved: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()