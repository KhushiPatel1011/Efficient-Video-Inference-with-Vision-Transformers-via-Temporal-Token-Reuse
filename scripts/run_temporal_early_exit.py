import argparse
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# impotable repo roots
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit


# Utilities
def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return int(idx.item()), float(conf.item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    # returns 1 - cosine_similarity
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    sim = F.cosine_similarity(a, b, dim=-1)
    return float((1.0 - sim).item())


def unpack_pair(item):
    """
    Robustly unpack whatever iter_frame_pairs returns.
    Supports:
      1) (t, (prev_img, cur_img))
      2) (t, prev_img, cur_img)
      3) (t, prev_img, cur_img, *extra)
    Returns: (t, prev_img, cur_img)
    """
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise ValueError(f"Unexpected iter_frame_pairs item: {type(item)} {item}")

    t = item[0]
    rest = item[1:]

    # Case (t, (prev, cur))
    if len(rest) == 1 and isinstance(rest[0], (tuple, list)) and len(rest[0]) >= 2:
        prev_img, cur_img = rest[0][0], rest[0][1]
        return t, prev_img, cur_img

    # Case (t, prev, cur, ...)
    if len(rest) >= 2:
        prev_img, cur_img = rest[0], rest[1]
        return t, prev_img, cur_img

    raise ValueError(f"Could not unpack iter_frame_pairs item: {item}")


# Probe forward (partial ViT)
def vit_probe_cls(model, x: torch.Tensor, probe_block: int) -> torch.Tensor:
    with torch.no_grad():
        x = model.patch_embed(x)

        if hasattr(model, "_pos_embed"):
            x = model._pos_embed(x)
        else:
            cls_tok = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tok, x), dim=1)
            x = x + model.pos_embed
            x = model.pos_drop(x)

        probe_block = max(0, min(probe_block, len(model.blocks) - 1))
        for i in range(probe_block + 1):
            x = model.blocks[i](x)

        cls_emb = x[:, 0, :]
        return cls_emb.squeeze(0).detach().cpu()


def vit_full_logits(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).detach().cpu()


# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--probe-block", type=int, default=5)
    parser.add_argument("--cls-threshold", type=float, default=8e-05)
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument("--csv-out", type=str, default="results/early_exit_logits_reuse.csv")
    args = parser.parse_args()

    device = torch.device("cpu")

    model, transform, _ = load_timm_vit(model_name="vit_base_patch16_224", pretrained=True)
    model.to(device)
    model.eval()

    print("\n    TEMPORAL TOKEN REUSE (PROBE CLS + LOGITS REUSE)")
    print("Idea: Probe CLS → if stable, reuse previous FULL logits (skip full forward).")
    print(f"probe_block: {args.probe_block}")
    print(f"cls_threshold: {args.cls_threshold}\n")
    print("t, cls_dist, decision, probe_ms, full_ms(if_run), total_decision_ms, base_top1, out_top1, match")

    prev_probe_cls: Optional[torch.Tensor] = None
    prev_full_logits: Optional[torch.Tensor] = None

    decision_times = []
    baseline_times = []
    reuse_count = 0
    match_count = 0
    match_total = 0

    step = 0

    for item in iter_frame_pairs(args.frames, max_frames=args.max_frames):
        t, _img_prev, img_cur = unpack_pair(item)
        step += 1

        x = transform(img_cur).unsqueeze(0).to(device)

        #Probe
        t0p = time.perf_counter()
        probe_cls = vit_probe_cls(model, x, args.probe_block)
        t1p = time.perf_counter()
        probe_ms_val = ms(t0p, t1p)

        # Decision Pipeline
        if prev_probe_cls is None or prev_full_logits is None:
            cls_dist = float("nan")
            decision = "full(first)"
        else:
            cls_dist = cosine_distance(probe_cls, prev_probe_cls)
            decision = "reuse_logits" if cls_dist < args.cls_threshold else "full"

        # Optional Baseline
        base_top1 = None
        if args.eval_baseline:
            t0b = time.perf_counter()
            base_logits = vit_full_logits(model, x)
            t1b = time.perf_counter()
            baseline_ms_val = ms(t0b, t1b)
            baseline_times.append(baseline_ms_val)
            base_top1, _ = top1(base_logits)

        # Deployment path
        full_ms_val = 0.0
        if decision.startswith("full"):
            t0f = time.perf_counter()
            out_logits = vit_full_logits(model, x)
            t1f = time.perf_counter()
            full_ms_val = ms(t0f, t1f)
            prev_full_logits = out_logits
        else:
            out_logits = prev_full_logits
            reuse_count += 1

        prev_probe_cls = probe_cls

        out_top1, _ = top1(out_logits)

        if args.eval_baseline:
            match = int(base_top1 == out_top1)
            match_count += match
            match_total += 1
        else:
            match = -1

        decision_ms_val = probe_ms_val + full_ms_val
        decision_times.append(decision_ms_val)

        base_str = str(base_top1) if base_top1 is not None else "-"
        print(
            f"{step}, {cls_dist}, {decision}, "
            f"{probe_ms_val:.2f}, {full_ms_val:.2f}, {decision_ms_val:.2f}, {base_str}, {out_top1}, {match}"
        )

        if step >= args.max_frames - 1:
            break

    # Summary
    pairs = len(decision_times)
    avg_decision_ms = sum(decision_times) / max(1, pairs)
    avg_decision_fps = 1000.0 / avg_decision_ms
    reuse_rate = reuse_count / max(1, pairs)

    print("\n   SUMMARY")
    print(f"pairs: {pairs}")
    if args.eval_baseline and baseline_times:
        avg_base = sum(baseline_times) / len(baseline_times)
        print(f"avg_baseline_ms:   {avg_base:.2f}  | fps: {1000.0/avg_base:.2f}")
        print(f"top1_match_rate:   {match_count / max(1, match_total):.3f}")
    else:
        print("avg_baseline_ms:   (skipped; run with --eval-baseline)")
        print("top1_match_rate:   (skipped)")

    print(f"avg_decision_ms:   {avg_decision_ms:.2f}  | fps: {avg_decision_fps:.2f}")
    print(f"reuse_rate:        {reuse_rate:.3f}")

    print("\nsaved_csv:", args.csv_out)
    print("Done.")


if __name__ == "__main__":
    main()
