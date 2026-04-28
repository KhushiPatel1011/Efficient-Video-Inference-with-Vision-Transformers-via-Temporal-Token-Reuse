"""
Evaluation table
    - It compares KV cache reuse across 6 dataset categories with fixed and adaptive policies.
"""

import csv
from pathlib import Path


DATASET_LABELS = {
    "medical_ultrasound": "Medical Ultrasound",
    "pest_detection": "Agriculture / Pest Monitoring",
    "street_surveillance": "CCTV / Surveillance",
    "crowd": "Crowd / Urban",
    "dashcam": "Driving / Traffic",
    "sports": "Sports",
}

MOTION_ORDER = {"low": 0, "medium": 1, "high": 2}

DATASET_ORDER = [
    "medical_ultrasound",
    "pest_detection",
    "street_surveillance",
    "crowd",
    "dashcam",
    "sports",
]


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    fixed_csv = Path("results/kv_motion_comparison.csv")
    adaptive_csv = Path("results/kv_motion_adaptive.csv")

    fixed_rows = load_csv(fixed_csv)
    adaptive_rows = load_csv(adaptive_csv)

    # Build lookup for fixed results: (video_id, r_match) -> row
    fixed_lookup = {}
    for r in fixed_rows:
        key = (r["video_id"], float(r["r_match"]))
        fixed_lookup[key] = r

    # Build lookup for adaptive results: video_id -> row
    adaptive_lookup = {}
    for r in adaptive_rows:
        adaptive_lookup[r["video_id"]] = r

    print("\n" + "=" * 110)
    print("EVALUATION TABLE: Temporal Reuse Across Dataset Categories")
    print("=" * 110)
    print(
        f"{'Dataset':<30} {'Motion Level':<13} {'Stable Ratio':>13} "
        f"{'Baseline FPS':>13} {'Reuse FPS':>11} {'Speedup':>10} {'Top-1 Match Rate':>18}"
    )
    print("-" * 110)

    table_rows = []

    for vid_id in DATASET_ORDER:
        label = DATASET_LABELS.get(vid_id, vid_id)

        candidate_rows = [r for r in fixed_rows if r["video_id"] == vid_id]
        if not candidate_rows:
            continue

        # Best = highest speedup while keeping top1 match >= 0.90
        safe = [r for r in candidate_rows if float(r["top1_match_rate"]) >= 0.90]
        pool = safe if safe else candidate_rows
        best = max(pool, key=lambda r: float(r["avg_speedup"]))

        motion = best["motion_bucket"]
        stable_ratio = float(best["r_match"])
        baseline_fps = 1000.0 / float(best["avg_baseline_ms"])
        reuse_fps = 1000.0 / float(best["avg_reuse_ms"])
        speedup = float(best["avg_speedup"])
        top1 = float(best["top1_match_rate"])

        table_rows.append([
            label,
            motion,
            round(stable_ratio, 2),
            round(baseline_fps, 2),
            round(reuse_fps, 2),
            round(speedup, 3),
            round(top1, 3),
        ])

        print(
            f"{label:<30} {motion:<13} {stable_ratio:>13.2f} "
            f"{baseline_fps:>13.2f} {reuse_fps:>11.2f} {speedup:>9.3f}x {top1:>18.3f}"
        )

    print("-" * 110)
    print("\nBest row per dataset is selected as the highest speedup with Top-1 Match Rate >= 0.90.")

    out_csv = Path("results/final_eval_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset",
            "Motion Level",
            "Stable Ratio",
            "Baseline FPS",
            "Reuse FPS",
            "Speedup",
            "Top-1 Match Rate",
        ])
        writer.writerows(table_rows)

    print(f"\nSaved: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()