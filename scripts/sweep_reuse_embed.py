import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
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


@torch.no_grad()
def run_one_keep_ratio(
    model,
    transform,
    frames_dir: Path,
    max_frames: int,
    keep_ratio: float,
    warmup: int = 1,
) -> Dict:
    device = torch.device("cpu")
    model.eval()

    # warmup baseline forward
    if warmup > 0:
        it = iter_frame_pairs(frames_dir, max_frames=min(max_frames, warmup + 1))
        for _, _, curr_img in it:
            x = transform(curr_img).unsqueeze(0).to(device)
            _ = model(x)

    baseline_times: List[float] = []
    reuse_times: List[float] = []
    stable_ratios: List[float] = []
    match_count = 0
    n_pairs = 0

    cached_patch_tokens = None

    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=max_frames):
        prev_x = transform(prev_img).unsqueeze(0).to(device)
        curr_x = transform(curr_img).unsqueeze(0).to(device)

        # Baseline
        t0 = time.perf_counter()
        base_logits = model(curr_x)
        t1 = time.perf_counter()
        baseline_ms = _ms(t0, t1)
        base_top1, _ = _top1(base_logits)

        # Patch tokens pre-block
        prev_tokens, _ = extract_patch_tokens_pre_blocks(model, prev_x, return_batch=False)
        curr_tokens, _ = extract_patch_tokens_pre_blocks(model, curr_x, return_batch=False)

        if cached_patch_tokens is None:
            cached_patch_tokens = prev_tokens.clone().detach()

        # Embedding change mask
        change = compute_embedding_change(cached_patch_tokens, curr_tokens, keep_ratio=keep_ratio)
        changed_mask = change.changed_mask  # (196,) bool
        changed_idx = torch.where(changed_mask)[0]

        total = int(change.total)
        changed = int(changed_mask.sum().item())
        reused = total - changed
        stable_ratio = reused / float(total)

        # Building reused tokens
        reused_patch_tokens = cached_patch_tokens.clone()
        reused_patch_tokens[changed_idx] = curr_tokens[changed_idx]

        # Forward-from-tokens (reuse path)
        patch_tokens_b = reused_patch_tokens.unsqueeze(0).to(device)
        tokens_with_special, _ = build_tokens_pre_blocks(model, patch_tokens_b)

        t2 = time.perf_counter()
        reuse_logits = forward_from_tokens(model, tokens_with_special)
        t3 = time.perf_counter()
        reuse_ms = _ms(t2, t3)
        reuse_top1, _ = _top1(reuse_logits)

        # updating cache (what we actually used)
        cached_patch_tokens = reused_patch_tokens.detach().cpu()

        match = int(base_top1 == reuse_top1)
        match_count += match

        baseline_times.append(baseline_ms)
        reuse_times.append(reuse_ms)
        stable_ratios.append(stable_ratio)
        n_pairs += 1

    avg_base = sum(baseline_times) / n_pairs
    avg_reuse = sum(reuse_times) / n_pairs
    base_fps = 1000.0 / avg_base if avg_base > 0 else 0.0
    reuse_fps = 1000.0 / avg_reuse if avg_reuse > 0 else 0.0
    speedup = (avg_base / avg_reuse) if avg_reuse > 0 else 0.0
    top1_match = match_count / n_pairs
    avg_stable = sum(stable_ratios) / n_pairs

    return {
        "keep_ratio": keep_ratio,
        "avg_stable_ratio": avg_stable,
        "avg_baseline_ms": avg_base,
        "avg_reuse_ms": avg_reuse,
        "baseline_fps": base_fps,
        "reuse_fps": reuse_fps,
        "measured_speedup": speedup,
        "top1_match_rate": top1_match,
        "pairs": n_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--keep-ratios",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated keep ratios (fraction changed tokens).",
    )
    parser.add_argument("--out-csv", type=str, default="results/reuse_sweep.csv")
    parser.add_argument("--out-plot", type=str, default="results/reuse_sweep_plot.png")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    frames_dir = Path(args.frames)
    keep_ratios = [float(x.strip()) for x in args.keep_ratios.split(",") if x.strip()]

    print("\n   SWEEP: Real Temporal Token Reuse (Embedding Mask)")
    print("We are varying keep_ratio (changed-token fractions) to measure stability vs correctness trade-off.\n")

    rows: List[Dict] = []
    for kr in keep_ratios:
        res = run_one_keep_ratio(
            model=model,
            transform=transform,
            frames_dir=frames_dir,
            max_frames=args.max_frames,
            keep_ratio=kr,
            warmup=1,
        )
        rows.append(res)
        print(
            f"keep_ratio={kr:.2f} | stable={res['avg_stable_ratio']:.3f} | "
            f"speedup={res['measured_speedup']:.2f}x | top1_match={res['top1_match_rate']:.3f}"
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Plot of trade-off curve
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    # top1 match vs stable ratio
    plt.figure()
    plt.plot(df["avg_stable_ratio"], df["top1_match_rate"], marker="o")
    plt.xlabel("Avg Stable Ratio (reused tokens fraction)")
    plt.ylabel("Top-1 Match Rate vs Baseline")
    plt.title("Correctness vs Reuse (Real Patch-Token Reuse)")
    plt.savefig(out_plot.as_posix(), dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(f"- CSV:  {out_csv.as_posix()}")
    print(f"- Plot: {out_plot.as_posix()}")
    print("Done.")


if __name__ == "__main__":
    main()