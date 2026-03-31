"""
KV Cache Temporal Token Reuse

Two phases:
Phase 1 (frame 0):  CACHING  — full attention, stores K/V/tokens per block.
Phase 2 (frame 1+): MATCHING — reuses cached K/V for stable bg tokens,
                    recomputes K/V only for changed/fg tokens.

Usage:
    python scripts/run_kv_cache_reuse.py --frames data/raw/sample_video_frames --max-frames 60
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Make repo root importable when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.timm_vit import load_timm_vit
from src.models.patching import apply_patch, set_caching_mode, reset_cache
from src.data.frames_dataset import load_frames_from_folder


# ----------------------------
# Helper functions
# ----------------------------
def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    return int(idx.item()), float(conf.item())


def save_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        type=str,
        required=True,
        help="Folder of sequential frame images"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Max frames to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name"
    )
    parser.add_argument(
        "--r-match",
        type=float,
        default=0.75,
        help="Fraction of bg tokens matched to cache (0.0–1.0)"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/kv_cache_reuse.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    # Safer device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_dir = Path(args.frames)

    print(f"\nLoading model: {args.model}")
    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    print(f"Applying TBKV patch (r_match={args.r_match})...")
    apply_patch(model, stable_ratio=args.r_match)
    reset_cache(model)

    print("\n    KV-CACHE TEMPORAL TOKEN REUSE")
    print("Frame 0:      CACHING  mode (full attention, builds K/V cache)")
    print("Frames 1+:    MATCHING mode (reuses cached K/V for stable tokens)")
    print(f"r_match:      {args.r_match}  ({int(args.r_match * 100)}% bg tokens matched to cache)")
    print(f"device:       {device}")
    print()
    print("t | mode     | latency_ms | base_ms | speedup | top1 | base_top1 | match")
    print("-" * 82)

    rows: List[Dict] = []
    baseline_times: List[float] = []
    reuse_times: List[float] = []
    match_count = 0

    all_frames = load_frames_from_folder(frames_dir, max_frames=args.max_frames)

    if len(all_frames) < 2:
        print("ERROR: Need at least 2 frames.")
        return

    video_id = frames_dir.name

    for frame_idx, img in enumerate(all_frames):
        x = transform(img).unsqueeze(0).to(device)

        if frame_idx == 0:
            # ----------------------------
            # Frame 0: CACHING mode
            # ----------------------------
            set_caching_mode(model, caching=True)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(x)
            t1 = time.perf_counter()

            latency = ms(t0, t1)
            pred, conf = top1(logits)

            print(
                f"{frame_idx:2d} | CACHING  | {latency:10.2f} | {'N/A':>7} | {'N/A':>7} | "
                f"{pred:4d} | {'N/A':>9} | {'N/A':>5}"
            )

            rows.append({
                "video_id": video_id,
                "frame": frame_idx,
                "mode": "cache",
                "latency_ms": round(latency, 3),
                "baseline_ms": None,
                "speedup": None,
                "top1": pred,
                "baseline_top1": None,
                "match": None,
                "r_match": args.r_match,
            })

            # Frame 0 is a full forward, so it also counts as baseline-like timing
            baseline_times.append(latency)

        else:
            # ----------------------------
            # Frame 1+: MATCHING mode
            # ----------------------------

            # Baseline comparison: patched model in caching=True behaves like full attention
            set_caching_mode(model, caching=True)
            t0b = time.perf_counter()
            with torch.no_grad():
                base_logits = model(x)
            t1b = time.perf_counter()

            base_ms_val = ms(t0b, t1b)
            base_pred, _ = top1(base_logits)
            baseline_times.append(base_ms_val)

            # Reuse / matching mode
            set_caching_mode(model, caching=False)
            t0 = time.perf_counter()
            with torch.no_grad():
                reuse_logits = model(x)
            t1 = time.perf_counter()

            reuse_ms_val = ms(t0, t1)
            reuse_pred, _ = top1(reuse_logits)
            reuse_times.append(reuse_ms_val)

            matched = int(base_pred == reuse_pred)
            match_count += matched

            speedup_val = base_ms_val / reuse_ms_val if reuse_ms_val > 0 else None

            print(
                f"{frame_idx:2d} | MATCHING | {reuse_ms_val:10.2f} | {base_ms_val:7.2f} | "
                f"{speedup_val:7.3f} | {reuse_pred:4d} | {base_pred:9d} | "
                f"{'YES' if matched else 'NO':>5}"
            )

            rows.append({
                "video_id": video_id,
                "frame": frame_idx,
                "mode": "reuse",
                "latency_ms": round(reuse_ms_val, 3),
                "baseline_ms": round(base_ms_val, 3),
                "speedup": round(speedup_val, 3) if speedup_val is not None else None,
                "top1": reuse_pred,
                "baseline_top1": base_pred,
                "match": matched,
                "r_match": args.r_match,
            })

    # ----------------------------
    # Summary
    # ----------------------------
    n_reuse = len(reuse_times)
    n_total = len(rows)

    if reuse_times and baseline_times:
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_reuse = sum(reuse_times) / len(reuse_times)
        speedup = avg_baseline / avg_reuse if avg_reuse > 0 else 0.0
        match_rate = match_count / n_reuse if n_reuse > 0 else 0.0

        print("\n    SUMMARY")
        print(f"video_id:           {video_id}")
        print(f"total_frames:       {n_total}")
        print(f"matching_frames:    {n_reuse}")
        print(f"avg_baseline_ms:    {avg_baseline:.2f}  | fps: {1000 / avg_baseline:.2f}")
        print(f"avg_reuse_ms:       {avg_reuse:.2f}  | fps: {1000 / avg_reuse:.2f}")
        print(f"measured_speedup:   {speedup:.3f}x")
        print(f"top1_match_rate:    {match_rate:.3f}  ({match_count}/{n_reuse} frames)")

    # ----------------------------
    # Save CSV
    # ----------------------------
    out_csv = Path(args.out_csv)
    save_csv(rows, out_csv)
    print(f"\nsaved_csv: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()