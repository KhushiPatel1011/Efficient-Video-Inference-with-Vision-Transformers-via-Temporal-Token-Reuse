"""
Sweeping stable_ratio values for KV cache reuse.
Useing two-model architecture for honest evaluation.
Collects: speedup, top1/top3/top5 match rate, KL divergence, conf delta.
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
    flat = logits.squeeze()
    probs = torch.softmax(flat, dim=0)
    conf, idx = probs.max(dim=0)
    return int(idx.item()), float(conf.item())


def topk_match(base_logits: torch.Tensor, reuse_logits: torch.Tensor, k: int) -> int:
    base_flat = base_logits.squeeze()
    reuse_flat = reuse_logits.squeeze()
    baseline_top1_idx = int(base_flat.argmax().item())
    reuse_topk = torch.topk(reuse_flat, k=k).indices.tolist()
    return int(baseline_top1_idx in reuse_topk)


def kl_divergence(base_logits: torch.Tensor, reuse_logits: torch.Tensor) -> float:
    p = torch.softmax(base_logits.squeeze(), dim=0)
    q = torch.softmax(reuse_logits.squeeze(), dim=0)
    kl = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum()
    return float(kl.item())


def confidence_delta(base_logits: torch.Tensor, reuse_logits: torch.Tensor) -> float:
    base_flat = base_logits.squeeze()
    reuse_flat = reuse_logits.squeeze()
    top1_idx = int(base_flat.argmax().item())
    p = torch.softmax(base_flat, dim=0)
    q = torch.softmax(reuse_flat, dim=0)
    return abs(float(p[top1_idx].item()) - float(q[top1_idx].item()))


def run_one_ratio(
    baseline_model,
    reuse_model,
    transform,
    frames,
    stable_ratio: float,
    device,
) -> Tuple[Dict, List[Dict]]:
    """
    Run one stable_ratio evaluation.
    Returns summary dict and per-frame rows list.
    """
    reset_cache(reuse_model)
    for module in reuse_model.modules():
        from src.models.patching import TBKVBlock
        if isinstance(module, TBKVBlock):
            module._tbkv_stable_ratio = stable_ratio
            module._prev_attn = None

    baseline_times = []
    reuse_times = []
    top1_matches = []
    top3_matches = []
    top5_matches = []
    kl_values = []
    conf_deltas = []
    per_frame_rows = []

    for frame_idx, img in enumerate(frames):
        x = transform(img).unsqueeze(0).to(device)

        # Baseline: always unpatched full attention
        t0b = time.perf_counter()
        with torch.no_grad():
            base_logits = baseline_model(x)
        t1b = time.perf_counter()
        base_ms_val = ms(t0b, t1b)
        baseline_times.append(base_ms_val)

        # Reuse model
        if frame_idx == 0:
            set_caching_mode(reuse_model, caching=True)
            with torch.no_grad():
                reuse_logits = reuse_model(x)
        else:
            set_caching_mode(reuse_model, caching=False)
            t0r = time.perf_counter()
            with torch.no_grad():
                reuse_logits = reuse_model(x)
            t1r = time.perf_counter()
            reuse_ms_val = ms(t0r, t1r)
            reuse_times.append(reuse_ms_val)

            m1 = topk_match(base_logits, reuse_logits, k=1)
            m3 = topk_match(base_logits, reuse_logits, k=3)
            m5 = topk_match(base_logits, reuse_logits, k=5)
            kl = kl_divergence(base_logits, reuse_logits)
            cd = confidence_delta(base_logits, reuse_logits)

            top1_matches.append(m1)
            top3_matches.append(m3)
            top5_matches.append(m5)
            kl_values.append(kl)
            conf_deltas.append(cd)

            speedup_val = base_ms_val / reuse_ms_val if reuse_ms_val > 0 else 0.0

            per_frame_rows.append({
                "stable_ratio": stable_ratio,
                "frame": frame_idx,
                "baseline_ms": round(base_ms_val, 3),
                "reuse_ms": round(reuse_ms_val, 3),
                "speedup": round(speedup_val, 4),
                "top1_match": m1,
                "top3_match": m3,
                "top5_match": m5,
                "kl_divergence": round(kl, 6),
                "conf_delta": round(cd, 6),
            })

    n = len(reuse_times)
    avg_base = sum(baseline_times) / len(baseline_times)
    avg_reuse = sum(reuse_times) / n if n > 0 else 0.0
    speedup = avg_base / avg_reuse if avg_reuse > 0 else 0.0

    summary = {
        "stable_ratio": stable_ratio,
        "avg_baseline_ms": round(avg_base, 3),
        "avg_reuse_ms": round(avg_reuse, 3),
        "baseline_fps": round(1000.0 / avg_base, 3),
        "reuse_fps": round(1000.0 / avg_reuse, 3) if avg_reuse > 0 else 0.0,
        "speedup": round(speedup, 4),
        "top1_match_rate": round(sum(top1_matches) / n, 4) if n > 0 else 0.0,
        "top3_match_rate": round(sum(top3_matches) / n, 4) if n > 0 else 0.0,
        "top5_match_rate": round(sum(top5_matches) / n, 4) if n > 0 else 0.0,
        "avg_kl_divergence": round(sum(kl_values) / n, 6) if n > 0 else 0.0,
        "avg_conf_delta": round(sum(conf_deltas) / n, 6) if n > 0 else 0.0,
        "reuse_frames": n,
    }

    return summary, per_frame_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--ratios", type=str, default="0.3,0.5,0.6,0.7,0.75,0.8,0.85,0.9",
    )
    parser.add_argument("--out-csv", type=str, default="results/kv_cache_sweep.csv")
    parser.add_argument(
        "--out-perframe-csv", type=str,
        default="results/kv_cache_sweep_perframe.csv"
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    frames_dir = Path(args.frames)
    ratios = [float(r.strip()) for r in args.ratios.split(",")]

    print(f"\nLoading baseline model (unpatched)...")
    baseline_model, transform, _ = load_timm_vit(
        model_name=args.model, pretrained=True
    )
    baseline_model.to(device)
    baseline_model.eval()

    print(f"Loading reuse model (patched)...")
    reuse_model, _, _ = load_timm_vit(
        model_name=args.model, pretrained=True
    )
    reuse_model.to(device)
    reuse_model.eval()
    apply_patch(reuse_model, stable_ratio=0.75)

    frames = load_frames_from_folder(frames_dir, max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames\n")

    print(
        f"{'stable_ratio':>12} | {'speedup':>8} | "
        f"{'top1':>6} | {'top3':>6} | {'top5':>6} | "
        f"{'avg_kl':>10} | {'avg_conf_d':>10}"
    )
    print("-" * 80)

    all_summaries = []
    all_perframe = []

    for ratio in ratios:
        summary, perframe = run_one_ratio(
            baseline_model, reuse_model, transform,
            frames, ratio, device
        )
        all_summaries.append(summary)
        all_perframe.extend(perframe)

        print(
            f"{summary['stable_ratio']:>12.2f} | "
            f"{summary['speedup']:>8.4f} | "
            f"{summary['top1_match_rate']:>6.3f} | "
            f"{summary['top3_match_rate']:>6.3f} | "
            f"{summary['top5_match_rate']:>6.3f} | "
            f"{summary['avg_kl_divergence']:>10.6f} | "
            f"{summary['avg_conf_delta']:>10.6f}"
        )

    # Save summary CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    # Save per-frame CSV for box plots
    out_pf = Path(args.out_perframe_csv)
    with open(out_pf, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_perframe[0].keys())
        writer.writeheader()
        writer.writerows(all_perframe)

    print(f"\nsaved_summary_csv:  {out_csv}")
    print(f"saved_perframe_csv: {out_pf}")

    # Best operating points
    print("\n=== BEST OPERATING POINTS ===")
    best_t1 = max(all_summaries, key=lambda r: r["top1_match_rate"])
    best_spd = max(all_summaries, key=lambda r: r["speedup"])
    print(
        f"Best top1 accuracy: stable_ratio={best_t1['stable_ratio']} "
        f"top1={best_t1['top1_match_rate']} speedup={best_t1['speedup']}x"
    )
    print(
        f"Best speedup:       stable_ratio={best_spd['stable_ratio']} "
        f"speedup={best_spd['speedup']}x top1={best_spd['top1_match_rate']}"
    )
    print("Done.")


if __name__ == "__main__":
    main()