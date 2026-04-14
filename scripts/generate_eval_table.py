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

    print("\n" + "=" * 100)
    print("EVALUATION TABLE: KV Cache Temporal Token Reuse Across Dataset Categories")
    print("=" * 100)
    print(
        f"{'Dataset':<30} {'Motion':<8} "
        f"{'Base FPS':>9} "
        f"{'r=0.50':>8} {'Acc':>6} "
        f"{'r=0.75':>8} {'Acc':>6} "
        f"{'r=0.90':>8} {'Acc':>6} "
        f"{'Adaptive':>9} {'Acc':>6}"
    )
    print("-" * 100)

    for vid_id in DATASET_ORDER:
        label = DATASET_LABELS.get(vid_id, vid_id)

        # Get motion bucket from any fixed row
        motion = ""
        for r in fixed_rows:
            if r["video_id"] == vid_id:
                motion = r["motion_bucket"]
                break

        # Baseline FPS (from r=0.75 fixed row baseline)
        base_row = fixed_lookup.get((vid_id, 0.75))
        base_fps = 1000.0 / float(base_row["avg_baseline_ms"]) if base_row else 0.0

        # Fixed r=0.50
        r50 = fixed_lookup.get((vid_id, 0.5))
        r50_speedup = float(r50["avg_speedup"]) if r50 else 0.0
        r50_acc = float(r50["top1_match_rate"]) if r50 else 0.0
        r50_fps = 1000.0 / float(r50["avg_reuse_ms"]) if r50 else 0.0

        # Fixed r=0.75
        r75 = fixed_lookup.get((vid_id, 0.75))
        r75_speedup = float(r75["avg_speedup"]) if r75 else 0.0
        r75_acc = float(r75["top1_match_rate"]) if r75 else 0.0
        r75_fps = 1000.0 / float(r75["avg_reuse_ms"]) if r75 else 0.0

        # Fixed r=0.90
        r90 = fixed_lookup.get((vid_id, 0.9))
        r90_speedup = float(r90["avg_speedup"]) if r90 else 0.0
        r90_acc = float(r90["top1_match_rate"]) if r90 else 0.0
        r90_fps = 1000.0 / float(r90["avg_reuse_ms"]) if r90 else 0.0

        # Adaptive
        adp = adaptive_lookup.get(vid_id)
        adp_speedup = float(adp["avg_speedup"]) if adp else 0.0
        adp_acc = float(adp["top1_match_rate"]) if adp else 0.0
        adp_fps = 1000.0 / float(adp["avg_reuse_ms"]) if adp else 0.0

        print(
            f"{label:<30} {motion:<8} "
            f"{base_fps:>9.2f} "
            f"{r50_speedup:>7.3f}x {r50_acc:>6.3f} "
            f"{r75_speedup:>7.3f}x {r75_acc:>6.3f} "
            f"{r90_speedup:>7.3f}x {r90_acc:>6.3f} "
            f"{adp_speedup:>8.3f}x {adp_acc:>6.3f}"
        )

    print("-" * 100)
    print("\nColumns: Base FPS = baseline throughput")
    print("         r=0.50/0.75/0.90 = fixed stable ratio speedup and top-1 match rate")
    print("         Adaptive = motion-aware stable ratio (low=0.90, medium=0.75, high=0.50)")
    print("         Acc = top-1 match rate vs baseline (1.000 = perfect)")

    # Also save as CSV for paper
    out_csv = Path("results/final_eval_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Motion", "Baseline_FPS",
            "Speedup_r050", "Acc_r050",
            "Speedup_r075", "Acc_r075",
            "Speedup_r090", "Acc_r090",
            "Speedup_Adaptive", "Acc_Adaptive"
        ])
        for vid_id in DATASET_ORDER:
            label = DATASET_LABELS.get(vid_id, vid_id)
            motion = ""
            for r in fixed_rows:
                if r["video_id"] == vid_id:
                    motion = r["motion_bucket"]
                    break
            base_row = fixed_lookup.get((vid_id, 0.75))
            base_fps = round(1000.0 / float(base_row["avg_baseline_ms"]), 2) if base_row else 0.0
            r50 = fixed_lookup.get((vid_id, 0.5))
            r75 = fixed_lookup.get((vid_id, 0.75))
            r90 = fixed_lookup.get((vid_id, 0.9))
            adp = adaptive_lookup.get(vid_id)
            writer.writerow([
                label, motion, base_fps,
                round(float(r50["avg_speedup"]), 3) if r50 else "",
                round(float(r50["top1_match_rate"]), 3) if r50 else "",
                round(float(r75["avg_speedup"]), 3) if r75 else "",
                round(float(r75["top1_match_rate"]), 3) if r75 else "",
                round(float(r90["avg_speedup"]), 3) if r90 else "",
                round(float(r90["top1_match_rate"]), 3) if r90 else "",
                round(float(adp["avg_speedup"]), 3) if adp else "",
                round(float(adp["top1_match_rate"]), 3) if adp else "",
            ])

    print(f"\nSaved: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()