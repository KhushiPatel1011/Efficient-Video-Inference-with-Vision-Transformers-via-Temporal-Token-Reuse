"""
KV Cache Temporal Token Reuse — Corrected Two-Model Architecture

Two separate models:
  baseline_model: unpatched ViT, runs full attention every frame
  reuse_model:    patched ViT, caches K/V on frame 0, reuses on frames 1+

This ensures baseline and reuse are genuinely different computations.

Metrics per frame:
    top1_match, top3_match, top5_match
    kl_divergence
    conf_delta, baseline_conf, reuse_conf
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


# ----------------------------
# Helper functions
# ----------------------------
def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    flat = logits.squeeze()
    probs = torch.softmax(flat, dim=0)
    conf, idx = probs.max(dim=0)
    return int(idx.item()), float(conf.item())


def topk_match(baseline_logits: torch.Tensor, reuse_logits: torch.Tensor, k: int) -> int:
    """
    Returns 1 if baseline top-1 class appears in reuse top-K predictions.

    Formula:
        topk_match = 1[ y_baseline_top1 in TopK(logits_reuse) ]
    """
    base_flat = baseline_logits.squeeze()
    reuse_flat = reuse_logits.squeeze()
    baseline_top1_idx = int(base_flat.argmax().item())
    reuse_topk_indices = torch.topk(reuse_flat, k=k).indices.tolist()
    return int(baseline_top1_idx in reuse_topk_indices)


def kl_divergence(baseline_logits: torch.Tensor, reuse_logits: torch.Tensor) -> float:
    """
    KL divergence from baseline to reuse probability distribution.

    Formula:
        KL(P_baseline || P_reuse) = sum_c [ P(c) * log( P(c) / Q(c) ) ]
    """
    p = torch.softmax(baseline_logits.squeeze(), dim=0)
    q = torch.softmax(reuse_logits.squeeze(), dim=0)
    kl = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum()
    return float(kl.item())


def confidence_delta(baseline_logits: torch.Tensor, reuse_logits: torch.Tensor) -> float:
    """
    Absolute change in confidence for the baseline top-1 class.

    Formula:
        conf_delta = | P_baseline(y_top1) - P_reuse(y_top1) |
    """
    base_flat = baseline_logits.squeeze()
    reuse_flat = reuse_logits.squeeze()
    baseline_top1_idx = int(base_flat.argmax().item())
    p = torch.softmax(base_flat, dim=0)
    q = torch.softmax(reuse_flat, dim=0)
    delta = float(p[baseline_top1_idx].item()) - float(q[baseline_top1_idx].item())
    return abs(delta)


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
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--r-match", type=float, default=0.75)
    parser.add_argument("--out-csv", type=str, default="results/kv_cache_reuse.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    frames_dir = Path(args.frames)

    # ── Load TWO separate models ──────────────────────────────────────────────
    print(f"\nLoading baseline model (unpatched)...")
    baseline_model, transform, _ = load_timm_vit(
        model_name=args.model, pretrained=True
    )
    baseline_model.to(device)
    baseline_model.eval()

    print(f"Loading reuse model (patched, r_match={args.r_match})...")
    reuse_model, _, _ = load_timm_vit(
        model_name=args.model, pretrained=True
    )
    reuse_model.to(device)
    reuse_model.eval()
    apply_patch(reuse_model, stable_ratio=args.r_match)
    reset_cache(reuse_model)

    print("\n    KV-CACHE TEMPORAL TOKEN REUSE (Two-Model Architecture)")
    print("baseline_model: unpatched ViT — full attention every frame")
    print("reuse_model:    patched ViT  — cache frame 0, reuse frames 1+")
    print(f"r_match:        {args.r_match}")
    print()
    print("t | mode     | base_ms | reuse_ms | speedup | base_top1 | reuse_top1 | t1 | t3 | t5 | kl_div   | conf_d")
    print("-" * 115)

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

        # ── Baseline: always full attention, unpatched ────────────────────────
        t0b = time.perf_counter()
        with torch.no_grad():
            base_logits = baseline_model(x)
        t1b = time.perf_counter()
        base_ms_val = ms(t0b, t1b)
        base_pred, base_conf_val = top1(base_logits)

        # ── Reuse model ───────────────────────────────────────────────────────
        if frame_idx == 0:
            # Frame 0: build cache
            set_caching_mode(reuse_model, caching=True)
            t0r = time.perf_counter()
            with torch.no_grad():
                reuse_logits = reuse_model(x)
            t1r = time.perf_counter()
            reuse_ms_val = ms(t0r, t1r)
            reuse_pred, reuse_conf_val = top1(reuse_logits)

            baseline_times.append(base_ms_val)

            print(
                f"{frame_idx:2d} | CACHING  | {base_ms_val:7.2f} | {reuse_ms_val:8.2f} | "
                f"{'N/A':>7} | {base_pred:9d} | {reuse_pred:10d} | "
                f"{'N/A':>2} | {'N/A':>2} | {'N/A':>2} | {'N/A':>8} | {'N/A':>6}"
            )

            rows.append({
                "video_id": video_id,
                "frame": frame_idx,
                "mode": "cache",
                "baseline_ms": round(base_ms_val, 3),
                "reuse_ms": round(reuse_ms_val, 3),
                "speedup": None,
                "baseline_top1": base_pred,
                "reuse_top1": reuse_pred,
                "match": None,
                "top3_match": None,
                "top5_match": None,
                "kl_divergence": None,
                "conf_delta": None,
                "baseline_conf": round(base_conf_val, 6),
                "reuse_conf": round(reuse_conf_val, 6),
                "r_match": args.r_match,
            })

        else:
            # Frames 1+: reuse cached K/V
            set_caching_mode(reuse_model, caching=False)
            t0r = time.perf_counter()
            with torch.no_grad():
                reuse_logits = reuse_model(x)
            t1r = time.perf_counter()
            reuse_ms_val = ms(t0r, t1r)
            reuse_pred, reuse_conf_val = top1(reuse_logits)

            baseline_times.append(base_ms_val)
            reuse_times.append(reuse_ms_val)

            # Compute all metrics
            matched_top1 = topk_match(base_logits, reuse_logits, k=1)
            matched_top3 = topk_match(base_logits, reuse_logits, k=3)
            matched_top5 = topk_match(base_logits, reuse_logits, k=5)
            kl = kl_divergence(base_logits, reuse_logits)
            conf_d = confidence_delta(base_logits, reuse_logits)
            speedup_val = base_ms_val / reuse_ms_val if reuse_ms_val > 0 else None

            match_count += matched_top1

            print(
                f"{frame_idx:2d} | MATCHING | {base_ms_val:7.2f} | {reuse_ms_val:8.2f} | "
                f"{speedup_val:7.3f} | {base_pred:9d} | {reuse_pred:10d} | "
                f"{'Y' if matched_top1 else 'N':>2} | "
                f"{'Y' if matched_top3 else 'N':>2} | "
                f"{'Y' if matched_top5 else 'N':>2} | "
                f"{kl:8.6f} | {conf_d:6.4f}"
            )

            rows.append({
                "video_id": video_id,
                "frame": frame_idx,
                "mode": "reuse",
                "baseline_ms": round(base_ms_val, 3),
                "reuse_ms": round(reuse_ms_val, 3),
                "speedup": round(speedup_val, 3) if speedup_val else None,
                "baseline_top1": base_pred,
                "reuse_top1": reuse_pred,
                "match": matched_top1,
                "top3_match": matched_top3,
                "top5_match": matched_top5,
                "kl_divergence": round(kl, 6),
                "conf_delta": round(conf_d, 6),
                "baseline_conf": round(base_conf_val, 6),
                "reuse_conf": round(reuse_conf_val, 6),
                "r_match": args.r_match,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_reuse = len(reuse_times)

    if reuse_times and baseline_times:
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_reuse = sum(reuse_times) / len(reuse_times)
        speedup = avg_baseline / avg_reuse if avg_reuse > 0 else 0.0
        match_rate = match_count / n_reuse if n_reuse > 0 else 0.0

        reuse_rows = [r for r in rows if r["mode"] == "reuse"]
        top3_rate = sum(r["top3_match"] for r in reuse_rows) / n_reuse
        top5_rate = sum(r["top5_match"] for r in reuse_rows) / n_reuse
        avg_kl = sum(r["kl_divergence"] for r in reuse_rows) / n_reuse
        avg_conf_d = sum(r["conf_delta"] for r in reuse_rows) / n_reuse

        print("\n    SUMMARY")
        print(f"video_id:           {video_id}")
        print(f"total_frames:       {len(rows)}")
        print(f"reuse_frames:       {n_reuse}")
        print(f"avg_baseline_ms:    {avg_baseline:.2f}  | fps: {1000/avg_baseline:.2f}")
        print(f"avg_reuse_ms:       {avg_reuse:.2f}  | fps: {1000/avg_reuse:.2f}")
        print(f"measured_speedup:   {speedup:.3f}x")
        print(f"top1_match_rate:    {match_rate:.3f}  ({match_count}/{n_reuse})")
        print(f"top3_match_rate:    {top3_rate:.3f}")
        print(f"top5_match_rate:    {top5_rate:.3f}")
        print(f"avg_kl_divergence:  {avg_kl:.6f}")
        print(f"avg_conf_delta:     {avg_conf_d:.6f}")

    save_csv(rows, Path(args.out_csv))
    print(f"\nsaved_csv: {args.out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()