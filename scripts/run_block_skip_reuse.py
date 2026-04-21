"""
Block-Skip Temporal Token Reuse.

For stable frames, skips the last N transformer blocks entirely
and reuses cached token embeddings from the previous frame.
This gives real wall-clock speedup on CPU.

Two-model architecture:
  baseline_model: unpatched ViT, full attention every frame
  skip_model:     same ViT, but blocks after cache_point are skipped
                  on stable frames using cached token embeddings

Usage:
    python scripts/run_block_skip_reuse.py --frames data/raw/medical_frames --max-frames 60
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
from src.cache.block_skip_cache import BlockSkipCache
from src.data.frames_dataset import load_frames_from_folder


# ----------------------------
# Helpers
# ----------------------------
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
    top1_idx = int(base_flat.argmax().item())
    topk_idx = torch.topk(reuse_flat, k=k).indices.tolist()
    return int(top1_idx in topk_idx)


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
# Block-skip forward pass
# ----------------------------
def forward_with_block_skip(
    model,
    x: torch.Tensor,
    cache: BlockSkipCache,
    cls_threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, str, float]:
    """
    Run ViT forward pass with optional block skipping.

    Phase 1 (cache empty or unstable): run all blocks, cache tokens
    Phase 2 (stable): run blocks 0..cache_after_block, check stability,
                      if stable reuse cached tokens and skip remaining blocks

    Returns:
        logits:   [1, num_classes]
        decision: "full" or "skip"
        cls_dist: cosine distance used for decision
    """
    with torch.no_grad():
        # Patch embedding + positional embedding
        x_tokens = model.patch_embed(x)

        if hasattr(model, "_pos_embed"):
            x_tokens = model._pos_embed(x_tokens)
        else:
            cls_tok = model.cls_token.expand(x.shape[0], -1, -1)
            x_tokens = torch.cat((cls_tok, x_tokens), dim=1)
            x_tokens = x_tokens + model.pos_embed
            x_tokens = model.pos_drop(x_tokens)

        cache_point = cache.cache_after_block
        total_blocks = len(model.blocks)

        # Run blocks 0 to cache_point
        for i in range(cache_point + 1):
            x_tokens = model.blocks[i](x_tokens)

        # Check stability
        cls_dist = cache.cls_distance(x_tokens)

        if cache.is_empty() or cls_dist >= cls_threshold:
            # Run remaining blocks fully
            for i in range(cache_point + 1, total_blocks):
                x_tokens = model.blocks[i](x_tokens)
            decision = "full"
            cache.store(x_tokens)
        else:
            # Skip remaining blocks — reuse cached tokens
            x_tokens = cache.get_cached_tokens()
            decision = "skip"

        # Norm + head
        if hasattr(model, "norm") and model.norm is not None:
            x_tokens = model.norm(x_tokens)

        cls_final = x_tokens[:, 0]

        if hasattr(model, "head") and model.head is not None:
            logits = model.head(cls_final)
        elif hasattr(model, "fc"):
            logits = model.fc(cls_final)
        else:
            raise RuntimeError("Cannot find model head.")

    return logits, decision, cls_dist


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--cache-after-block", type=int, default=7,
        help="Cache tokens after this block. Skip all subsequent blocks on stable frames."
    )
    parser.add_argument(
        "--cls-threshold", type=float, default=0.02,
        help="CLS cosine distance threshold. Below = stable = skip blocks."
    )
    parser.add_argument("--out-csv", type=str, default="results/block_skip_reuse.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    frames_dir = Path(args.frames)

    n_blocks_total = 12
    n_blocks_skipped = n_blocks_total - (args.cache_after_block + 1)

    print(f"\nLoading baseline model (unpatched)...")
    baseline_model, transform, _ = load_timm_vit(
        model_name=args.model, pretrained=True
    )
    baseline_model.to(device)
    baseline_model.eval()

    print(f"Loading skip model...")
    skip_model, _, _ = load_timm_vit(
        model_name=args.model, pretrained=True
    )
    skip_model.to(device)
    skip_model.eval()

    cache = BlockSkipCache(cache_after_block=args.cache_after_block)

    print(f"\n=== BLOCK-SKIP TEMPORAL TOKEN REUSE ===")
    print(f"Total blocks:     {n_blocks_total}")
    print(f"Cache after:      block {args.cache_after_block}")
    print(f"Blocks skipped:   {n_blocks_skipped} (blocks {args.cache_after_block+1} to {n_blocks_total-1})")
    print(f"CLS threshold:    {args.cls_threshold}")
    print(f"Expected speedup: ~{n_blocks_total / (args.cache_after_block + 1):.2f}x (theoretical max)")
    print()
    print("t | decision | base_ms | skip_ms | speedup | base_t1 | skip_t1 | t1 | t3 | t5 | kl_div   | conf_d  | cls_dist")
    print("-" * 120)

    rows: List[Dict] = []
    baseline_times: List[float] = []
    skip_times: List[float] = []
    match_counts = [0, 0, 0]
    skip_count = 0
    video_id = frames_dir.name

    all_frames = load_frames_from_folder(frames_dir, max_frames=args.max_frames)

    for frame_idx, img in enumerate(all_frames):
        x = transform(img).unsqueeze(0).to(device)

        # Baseline: always full unpatched forward
        t0b = time.perf_counter()
        with torch.no_grad():
            base_logits = baseline_model(x)
        t1b = time.perf_counter()
        base_ms_val = ms(t0b, t1b)
        base_pred, base_conf_val = top1(base_logits)
        baseline_times.append(base_ms_val)

        # Block-skip forward
        t0s = time.perf_counter()
        skip_logits, decision, cls_dist = forward_with_block_skip(
            skip_model, x, cache, args.cls_threshold, device
        )
        t1s = time.perf_counter()
        skip_ms_val = ms(t0s, t1s)
        skip_pred, skip_conf_val = top1(skip_logits)

        if decision == "skip":
            skip_count += 1
            skip_times.append(skip_ms_val)

            m1 = topk_match(base_logits, skip_logits, k=1)
            m3 = topk_match(base_logits, skip_logits, k=3)
            m5 = topk_match(base_logits, skip_logits, k=5)
            kl = kl_divergence(base_logits, skip_logits)
            cd = confidence_delta(base_logits, skip_logits)
            speedup_val = base_ms_val / skip_ms_val if skip_ms_val > 0 else 0.0

            match_counts[0] += m1
            match_counts[1] += m3
            match_counts[2] += m5

            print(
                f"{frame_idx:2d} | {'SKIP':>8} | {base_ms_val:7.2f} | {skip_ms_val:7.2f} | "
                f"{speedup_val:7.3f} | {base_pred:7d} | {skip_pred:7d} | "
                f"{'Y' if m1 else 'N':>2} | {'Y' if m3 else 'N':>2} | {'Y' if m5 else 'N':>2} | "
                f"{kl:8.6f} | {cd:7.4f} | {cls_dist:.6f}"
            )

            rows.append({
                "video_id": video_id,
                "frame": frame_idx,
                "decision": "skip",
                "baseline_ms": round(base_ms_val, 3),
                "skip_ms": round(skip_ms_val, 3),
                "speedup": round(speedup_val, 4),
                "baseline_top1": base_pred,
                "skip_top1": skip_pred,
                "top1_match": m1,
                "top3_match": m3,
                "top5_match": m5,
                "kl_divergence": round(kl, 6),
                "conf_delta": round(cd, 6),
                "cls_dist": round(cls_dist, 6),
                "cache_after_block": args.cache_after_block,
                "cls_threshold": args.cls_threshold,
            })
        else:
            print(
                f"{frame_idx:2d} | {'FULL':>8} | {base_ms_val:7.2f} | {skip_ms_val:7.2f} | "
                f"{'N/A':>7} | {base_pred:7d} | {skip_pred:7d} | "
                f"{'N/A':>2} | {'N/A':>2} | {'N/A':>2} | "
                f"{'N/A':>8} | {'N/A':>7} | {cls_dist:.6f}"
            )

            rows.append({
                "video_id": video_id,
                "frame": frame_idx,
                "decision": "full",
                "baseline_ms": round(base_ms_val, 3),
                "skip_ms": round(skip_ms_val, 3),
                "speedup": None,
                "baseline_top1": base_pred,
                "skip_top1": skip_pred,
                "top1_match": None,
                "top3_match": None,
                "top5_match": None,
                "kl_divergence": None,
                "conf_delta": None,
                "cls_dist": round(cls_dist, 6),
                "cache_after_block": args.cache_after_block,
                "cls_threshold": args.cls_threshold,
            })

    # Summary
    n_skip = len(skip_times)
    n_total = len(all_frames)
    skip_rate = skip_count / (n_total - 1) if n_total > 1 else 0.0

    if skip_times and baseline_times:
        avg_base = sum(baseline_times) / len(baseline_times)
        avg_skip = sum(skip_times) / len(skip_times)
        speedup = avg_base / avg_skip if avg_skip > 0 else 0.0

        t1_rate = match_counts[0] / n_skip if n_skip > 0 else 0.0
        t3_rate = match_counts[1] / n_skip if n_skip > 0 else 0.0
        t5_rate = match_counts[2] / n_skip if n_skip > 0 else 0.0

        skip_rows = [r for r in rows if r["decision"] == "skip"]
        avg_kl = sum(r["kl_divergence"] for r in skip_rows) / n_skip if n_skip > 0 else 0.0
        avg_cd = sum(r["conf_delta"] for r in skip_rows) / n_skip if n_skip > 0 else 0.0

        print(f"\n=== SUMMARY ===")
        print(f"video_id:            {video_id}")
        print(f"total_frames:        {n_total}")
        print(f"skip_frames:         {n_skip}  (rate: {skip_rate:.3f})")
        print(f"avg_baseline_ms:     {avg_base:.2f}  | fps: {1000/avg_base:.2f}")
        print(f"avg_skip_ms:         {avg_skip:.2f}  | fps: {1000/avg_skip:.2f}")
        print(f"measured_speedup:    {speedup:.3f}x  (on skipped frames)")
        print(f"top1_match_rate:     {t1_rate:.3f}")
        print(f"top3_match_rate:     {t3_rate:.3f}")
        print(f"top5_match_rate:     {t5_rate:.3f}")
        print(f"avg_kl_divergence:   {avg_kl:.6f}")
        print(f"avg_conf_delta:      {avg_cd:.6f}")
        print(f"\nNote: speedup measured on skipped frames only.")
        print(f"On skipped frames, {n_blocks_skipped} of {n_blocks_total} blocks were bypassed.")

    save_csv(rows, Path(args.out_csv))
    print(f"\nsaved_csv: {args.out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()