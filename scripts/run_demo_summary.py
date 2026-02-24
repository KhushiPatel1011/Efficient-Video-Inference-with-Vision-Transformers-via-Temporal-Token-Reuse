import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results/temporal_simulation.csv",
        help="Path to temporal simulation CSV",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="results/temporal_masks",
        help="Directory containing mask visualizations",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    masks_dir = Path(args.masks_dir)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No rows found in CSV.")
        return

    baseline_ms = [float(r["baseline_ms"]) for r in rows]
    simulated_ms = [float(r["simulated_ms"]) for r in rows]
    stable_ratio = [float(r["stable_ratio"]) for r in rows]

    avg_base = sum(baseline_ms) / len(baseline_ms)
    avg_sim = sum(simulated_ms) / len(simulated_ms)
    avg_stable = sum(stable_ratio) / len(stable_ratio)

    base_fps = 1000.0 / avg_base if avg_base > 0 else 0.0
    sim_fps = 1000.0 / avg_sim if avg_sim > 0 else 0.0
    speedup = avg_base / avg_sim if avg_sim > 0 else 0.0

    mask_count = len(list(masks_dir.glob("*.jpg"))) if masks_dir.exists() else 0

    print("TEMPORAL TOKEN REUSE DEMO")

    print("Baseline Performance:")
    print(f"Avg Latency: {avg_base:.2f} ms")
    print(f"Avg FPS:     {base_fps:.2f}\n")

    print("Temporal Stability:")
    print(f"Avg Stable Patch Ratio: {avg_stable:.3f}")
    print(f"(≈ {avg_stable * 100:.1f}% patches reusable)\n")

    print("Simulated Reuse Performance:")
    print(f"Avg Simulated Latency: {avg_sim:.2f} ms")
    print(f"Avg Simulated FPS:     {sim_fps:.2f}")
    print(f"Estimated Speedup:     {speedup:.2f}x\n")

    print("Artifacts:")
    print(f"CSV File: {csv_path}")
    print(f"Mask Images: {mask_count} saved in {masks_dir}")
    
if __name__ == "__main__":
    main()