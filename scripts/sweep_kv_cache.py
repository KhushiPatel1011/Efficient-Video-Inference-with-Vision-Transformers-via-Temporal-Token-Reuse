"""
Here, we are sweeping stable_ratio values for KV cache reuse.
What it is doing?: Finding the best tradeoff between speedup and accuracy.

Usage:
    python scripts/sweep_kv_cache.py --frames data/raw/sample_video_frames --max-frames 60
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.timm_vit import load_timm_vit
from src.models.patching import apply_patch, set_caching_mode, reset_cache
from src.data.frames_dataset import load_frames_from_folder


def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    return int(idx.item()), float(conf.item())


def run_one_ratio(model, transform, frames, stable_ratio: float, device) -> Dict:
    """Running full evaluation for one stable_ratio value."""

    # Re-patching with new ratio
    reset_cache(model)
    for module in model.modules():
        from src.models.patching import TBKVBlock
        if isinstance(module, TBKVBlock):
            module._tbkv_stable_ratio = stable_ratio

    baseline_times = []
    reuse_times = []
    match_count = 0
    n_reuse = 0

    for frame_idx, img in enumerate(frames):
        x = transform(img).unsqueeze(0).to(device)

        if frame_idx == 0:
            set_caching_mode(model, caching=True)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            t1 = time.perf_counter()
            baseline_times.append(ms(t0, t1))
        else:
            # Baseline
            set_caching_mode(model, caching=True)
            t0b = time.perf_counter()
            with torch.no_grad():
                base_logits = model(x)
            t1b = time.perf_counter()
            base_ms = ms(t0b, t1b)
            baseline_times.append(base_ms)
            base_pred, _ = top1(base_logits)

            # Reuse
            set_caching_mode(model, caching=False)
            t0 = time.perf_counter()
            with torch.no_grad():
                reuse_logits = model(x)
            t1 = time.perf_counter()
            reuse_ms_val = ms(t0, t1)
            reuse_times.append(reuse_ms_val)
            reuse_pred, _ = top1(reuse_logits)

            match_count += int(base_pred == reuse_pred)
            n_reuse += 1

    avg_base = sum(baseline_times) / len(baseline_times)
    avg_reuse = sum(reuse_times) / len(reuse_times) if reuse_times else 0.0
    speedup = avg_base / avg_reuse if avg_reuse > 0 else 0.0
    match_rate = match_count / n_reuse if n_reuse > 0 else 0.0

    return {
        "stable_ratio": stable_ratio,
        "avg_baseline_ms": round(avg_base, 2),
        "avg_reuse_ms": round(avg_reuse, 2),
        "baseline_fps": round(1000.0 / avg_base, 3),
        "reuse_fps": round(1000.0 / avg_reuse, 3) if avg_reuse > 0 else 0.0,
        "speedup": round(speedup, 4),
        "top1_match_rate": round(match_rate, 4),
        "matched_frames": match_count,
        "total_reuse_frames": n_reuse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--ratios", type=str, default="0.3,0.5,0.6,0.7,0.75,0.8,0.85,0.9",
        help="Comma-separated stable_ratio values to sweep"
    )
    parser.add_argument("--out-csv", type=str, default="results/kv_cache_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_dir = Path(args.frames)
    ratios = [float(r.strip()) for r in args.ratios.split(",")]

    print(f"\nLoading model: {args.model}")
    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    # Patching once with a neutral ratio (this will be overridden per sweep step)
    apply_patch(model, stable_ratio=0.75)

    frames = load_frames_from_folder(frames_dir, max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames\n")

    print(f"{'stable_ratio':>12} | {'speedup':>8} | {'match_rate':>10} | "
          f"{'reuse_ms':>9} | {'base_ms':>8} | {'reuse_fps':>9}")
    print("-" * 70)

    rows = []
    for ratio in ratios:
        result = run_one_ratio(model, transform, frames, ratio, device)
        rows.append(result)
        print(
            f"{result['stable_ratio']:>12.2f} | "
            f"{result['speedup']:>8.3f} | "
            f"{result['top1_match_rate']:>10.3f} | "
            f"{result['avg_reuse_ms']:>9.2f} | "
            f"{result['avg_baseline_ms']:>8.2f} | "
            f"{result['reuse_fps']:>9.3f}"
        )

    # Saving the results to CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Printing best results (highest speedup with match_rate >= 0.95)
    print("\n=== BEST OPERATING POINTS ===")
    high_acc = [r for r in rows if r["top1_match_rate"] >= 0.95]
    if high_acc:
        best = max(high_acc, key=lambda r: r["speedup"])
        print(f"Best speedup with match_rate >= 0.95:")
        print(f"  stable_ratio={best['stable_ratio']}  speedup={best['speedup']}x  "
              f"match_rate={best['top1_match_rate']}  reuse_fps={best['reuse_fps']}")

    perfect = [r for r in rows if r["top1_match_rate"] == 1.0]
    if perfect:
        best_perfect = max(perfect, key=lambda r: r["speedup"])
        print(f"Best speedup with perfect accuracy (1.000):")
        print(f"  stable_ratio={best_perfect['stable_ratio']}  "
              f"speedup={best_perfect['speedup']}x  "
              f"reuse_fps={best_perfect['reuse_fps']}")

    print(f"\nsaved_csv: {out_csv}")


if __name__ == "__main__":
    main()
