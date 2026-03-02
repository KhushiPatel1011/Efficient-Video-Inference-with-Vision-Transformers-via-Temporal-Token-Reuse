import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit
from src.utils.token_extract import extract_patch_tokens_pre_blocks
from src.methods.embedding_change import compute_embedding_change
from src.models.vit_forward import build_tokens_pre_blocks, forward_from_tokens


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _top1(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    v, idx = torch.max(probs, dim=-1)
    return int(idx.item()), float(v.item())


def _save_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Top fraction changed tokens")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out-csv", type=str, default="results/temporal_reuse_embed.csv")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    frames_dir = Path(args.frames)
    rows: List[Dict] = []

    # Warmup (baseline only)
    if args.warmup > 0:
        it = iter_frame_pairs(frames_dir, max_frames=min(args.max_frames, args.warmup + 1))
        for _, _, curr_img in it:
            x = transform(curr_img).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(x)

    print("\n    TEMPORAL TOKEN REUSE (Embedding Mask, Patch-Token Reuse)")
    print("We reused pre-block patch tokens for stable patches and recomputed only changed patches")
    print("This is reuse of real token embeddings.\n")
    print("t, changed, reused, stable_ratio, baseline_ms, reuse_ms, base_top1, reuse_top1, match")

    baseline_times: List[float] = []
    reuse_times: List[float] = []
    match_count = 0

    # Cache: previous frame patch tokens (N, C)
    cached_patch_tokens = None

    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        prev_x = transform(prev_img).unsqueeze(0).to(device)
        curr_x = transform(curr_img).unsqueeze(0).to(device)

        # Baseline timing with normal forward
        t0 = time.perf_counter()
        with torch.no_grad():
            base_logits = model(curr_x)
        t1 = time.perf_counter()
        baseline_ms = _ms(t0, t1)
        base_top1, base_conf = _top1(base_logits)

        # Extract patch tokens
        with torch.no_grad():
            prev_tokens, _ = extract_patch_tokens_pre_blocks(model, prev_x, return_batch=False)  # (196,C)
            curr_tokens, _ = extract_patch_tokens_pre_blocks(model, curr_x, return_batch=False)  # (196,C)

        # Initializing cache on first pair
        if cached_patch_tokens is None:
            cached_patch_tokens = prev_tokens.clone().detach()

        # Computing embedding-based change mask 
        change = compute_embedding_change(cached_patch_tokens, curr_tokens, keep_ratio=args.keep_ratio)
        changed_mask = change.changed_mask  # (196,) bool 
        changed_idx = torch.where(changed_mask)[0]
        stable_idx = torch.where(~changed_mask)[0]

        total = change.total
        changed = int(changed_mask.sum().item())
        reused = total - changed
        stable_ratio = reused / float(total)

        # Building reused patch tokens
        # Starting from cached tokens and replacing changed positions with current
        reused_patch_tokens = cached_patch_tokens.clone()
        reused_patch_tokens[changed_idx] = curr_tokens[changed_idx]

        # Forward from tokens, reuse path
        # Building full token sequence (CLS + pos embed + drop), then forward blocks+head
        patch_tokens_b = reused_patch_tokens.unsqueeze(0).to(device)  # (1,196,C)
        tokens_with_special, _ = build_tokens_pre_blocks(model, patch_tokens_b)

        t2 = time.perf_counter()
        with torch.no_grad():
            reuse_logits = forward_from_tokens(model, tokens_with_special)
        t3 = time.perf_counter()
        reuse_ms = _ms(t2, t3)
        reuse_top1, reuse_conf = _top1(reuse_logits)

        # Updating cache for future development pipeline, we are chaching what we are going to use in future development phases
        cached_patch_tokens = reused_patch_tokens.detach().cpu()

        match = int(base_top1 == reuse_top1)
        match_count += match

        baseline_times.append(baseline_ms)
        reuse_times.append(reuse_ms)

        row = {
            "t": t,
            "changed": changed,
            "reused": reused,
            "stable_ratio": round(stable_ratio, 6),
            "baseline_ms": round(baseline_ms, 3),
            "reuse_ms": round(reuse_ms, 3),
            "baseline_top1": base_top1,
            "reuse_top1": reuse_top1,
            "top1_match": match,
            "baseline_conf": round(base_conf, 6),
            "reuse_conf": round(reuse_conf, 6),
            "threshold": round(float(change.threshold), 6),
            "keep_ratio": args.keep_ratio,
            "model": args.model,
        }
        rows.append(row)

        print(
            f"{t}, {changed}, {reused}, {stable_ratio:.3f}, "
            f"{baseline_ms:.2f}, {reuse_ms:.2f}, {base_top1}, {reuse_top1}, {match}"
        )

    if rows:
        avg_base = sum(baseline_times) / len(baseline_times)
        avg_reuse = sum(reuse_times) / len(reuse_times)
        base_fps = 1000.0 / avg_base if avg_base > 0 else 0.0
        reuse_fps = 1000.0 / avg_reuse if avg_reuse > 0 else 0.0
        speedup = (avg_base / avg_reuse) if avg_reuse > 0 else 0.0
        top1_acc = match_count / len(rows)

        print("\n    REUSE SUMMARY")
        print(f"pairs: {len(rows)}")
        print(f"avg_baseline_ms: {avg_base:.2f}  | fps: {base_fps:.2f}")
        print(f"avg_reuse_ms:    {avg_reuse:.2f}  | fps: {reuse_fps:.2f}")
        print(f"measured_speedup: {speedup:.2f}x")
        print(f"top1_match_rate_vs_baseline: {top1_acc:.3f}")
        print("note: speedup is smaller than simulation because transformer blocks still run;")
        print("      but this is token embedding reuse across frames.")

    out_csv = Path(args.out_csv)
    _save_csv(rows, out_csv)
    print(f"\nsaved_csv: {out_csv.as_posix()}\nDone.")


if __name__ == "__main__":
    main()