"""
Generate one aggregated evaluation table for block-skip reuse
across multiple video datasets and thresholds.

For each dataset:
    - add one Baseline row
    - run block-skip reuse at 3 thresholds
    - collect:
        skip_rate, speedup, top1/top3/top5, KL divergence, confidence delta

Outputs:
    results/block_skip_all_datasets_eval_table.csv
"""

import csv
import subprocess
import sys
from pathlib import Path

DATASETS = [
    {"name": "medical_ultrasound", "frames_dir": "data/raw/medical_frames"},
    {"name": "crowd", "frames_dir": "data/experiments/crowd/frames"},
    {"name": "dashcam", "frames_dir": "data/experiments/dashcam/frames"},
    {"name": "pest_detection", "frames_dir": "data/experiments/pest_detection/frames"},
    {"name": "sports", "frames_dir": "data/experiments/sports/frames"},
    {"name": "street_surveillance", "frames_dir": "data/experiments/street_surveillance/frames"},
]

# Low / medium / aggressive reuse
THRESHOLDS = [0.70, 0.85, 0.95]

MAX_FRAMES = 60
CACHE_AFTER_BLOCK = 7

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_TABLE_CSV = RESULTS_DIR / "block_skip_all_datasets_eval_table.csv"


def read_csv_rows(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value, default=0.0):
    try:
        if value is None or value == "" or value == "None":
            return default
        return float(value)
    except Exception:
        return default


def run_block_skip_for_dataset(dataset_name: str, frames_dir: str, threshold: float) -> Path:
    out_csv = RESULTS_DIR / f"{dataset_name}_block_skip_thresh_{str(threshold).replace('.', '_')}.csv"

    cmd = [
        sys.executable,
        "scripts/run_block_skip_reuse.py",
        "--frames", frames_dir,
        "--max-frames", str(MAX_FRAMES),
        "--cache-after-block", str(CACHE_AFTER_BLOCK),
        "--cls-threshold", str(threshold),
        "--out-csv", str(out_csv),
    ]

    print(f"Running: dataset={dataset_name}, threshold={threshold}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\nERROR while running dataset={dataset_name}, threshold={threshold}")
        print(result.stderr)
        raise RuntimeError(f"Run failed for {dataset_name} @ {threshold}")

    return out_csv


def summarize_block_skip_csv(dataset_name: str, threshold: float, csv_path: Path):
    rows = read_csv_rows(csv_path)
    skip_rows = [r for r in rows if r.get("decision") == "skip"]

    n_total = len(rows)
    n_skip = len(skip_rows)

    skip_rate = n_skip / (n_total - 1) if n_total > 1 else 0.0

    if n_skip == 0:
        return {
            "dataset": dataset_name,
            "method": "Block-Skip",
            "threshold": threshold,
            "skip_rate": round(skip_rate, 4),
            "speedup": 0.0,
            "top1_match_rate": 0.0,
            "top3_match_rate": 0.0,
            "top5_match_rate": 0.0,
            "avg_kl_divergence": 0.0,
            "avg_conf_delta": 0.0,
        }

    avg_base = sum(to_float(r.get("baseline_ms")) for r in skip_rows) / n_skip
    avg_skip = sum(to_float(r.get("skip_ms")) for r in skip_rows) / n_skip
    speedup = avg_base / avg_skip if avg_skip > 0 else 0.0

    top1 = sum(int(to_float(r.get("top1_match"))) for r in skip_rows) / n_skip
    top3 = sum(int(to_float(r.get("top3_match"))) for r in skip_rows) / n_skip
    top5 = sum(int(to_float(r.get("top5_match"))) for r in skip_rows) / n_skip
    avg_kl = sum(to_float(r.get("kl_divergence")) for r in skip_rows) / n_skip
    avg_cd = sum(to_float(r.get("conf_delta")) for r in skip_rows) / n_skip

    return {
        "dataset": dataset_name,
        "method": "Block-Skip",
        "threshold": round(threshold, 2),
        "skip_rate": round(skip_rate, 4),
        "speedup": round(speedup, 4),
        "top1_match_rate": round(top1, 4),
        "top3_match_rate": round(top3, 4),
        "top5_match_rate": round(top5, 4),
        "avg_kl_divergence": round(avg_kl, 6),
        "avg_conf_delta": round(avg_cd, 6),
    }


def make_baseline_row(dataset_name: str):
    return {
        "dataset": dataset_name,
        "method": "Baseline",
        "threshold": "-",
        "skip_rate": 0.0,
        "speedup": 1.0,
        "top1_match_rate": 1.0,
        "top3_match_rate": 1.0,
        "top5_match_rate": 1.0,
        "avg_kl_divergence": 0.0,
        "avg_conf_delta": 0.0,
    }


def print_table(rows):
    print("\n=== AGGREGATED BLOCK-SKIP EVALUATION TABLE ===")
    print(
        f"{'dataset':<18} | {'method':<10} | {'thr':>5} | {'skip':>6} | {'spd':>7} | "
        f"{'top1':>6} | {'top3':>6} | {'top5':>6} | {'KL':>10} | {'conf_d':>8}"
    )
    print("-" * 108)

    for r in rows:
        print(
            f"{r['dataset']:<18} | "
            f"{r['method']:<10} | "
            f"{str(r['threshold']):>5} | "
            f"{float(r['skip_rate']):>6.3f} | "
            f"{float(r['speedup']):>7.3f} | "
            f"{float(r['top1_match_rate']):>6.3f} | "
            f"{float(r['top3_match_rate']):>6.3f} | "
            f"{float(r['top5_match_rate']):>6.3f} | "
            f"{float(r['avg_kl_divergence']):>10.6f} | "
            f"{float(r['avg_conf_delta']):>8.6f}"
        )


def main():
    all_rows = []

    for ds in DATASETS:
        dataset_name = ds["name"]
        frames_dir = ds["frames_dir"]

        if not Path(frames_dir).exists():
            print(f"WARNING: Skipping missing dataset folder: {frames_dir}")
            continue

        # Add baseline row first
        all_rows.append(make_baseline_row(dataset_name))

        # Run selected thresholds
        for threshold in THRESHOLDS:
            out_csv = run_block_skip_for_dataset(dataset_name, frames_dir, threshold)
            summary_row = summarize_block_skip_csv(dataset_name, threshold, out_csv)
            all_rows.append(summary_row)

    if not all_rows:
        print("No rows generated. Check dataset paths.")
        return

    with open(FINAL_TABLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "threshold",
                "skip_rate",
                "speedup",
                "top1_match_rate",
                "top3_match_rate",
                "top5_match_rate",
                "avg_kl_divergence",
                "avg_conf_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print_table(all_rows)
    print(f"\nSaved: {FINAL_TABLE_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()