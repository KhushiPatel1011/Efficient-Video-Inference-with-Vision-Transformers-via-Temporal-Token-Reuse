"""
TEMPORAL TOKEN REUSE (PROBE CLS + LOGITS REUSE)

Brainstorming + Efficient ViT + MaskVD + ToMe + TBKV:
- Running a cheap "probe" forward pass up to a chosen transformer block
- Computing cosine distance between current CLS probe embedding and previous CLS probe embedding.
- If stable (distance < threshold): reusing previous FULL logits and skip full forward.
- Else: run full forward and refresh cached logits.

Note:
- decision_ms is the realistic end-to-end time of the reuse system (probe + optional full).
- Baseline evaluation (full model every frame) is OPTIONAL via --eval-baseline and is NOT part of decision_ms.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# solving previous error: importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit

# Helper functions
def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    """Return (top1_index, top1_confidence)."""
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return int(idx.item()), float(conf.item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a, b: [D] or [1, D]
    returns scalar distance = 1 - cosine_similarity
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    sim = F.cosine_similarity(a, b, dim=-1)  # [1]
    dist = 1.0 - sim
    return float(dist.item())


def vit_probe_cls(model, x: torch.Tensor, probe_block: int) -> torch.Tensor:
    """
    Run timm ViT forward up to probe_block (inclusive) and return CLS token embedding.
    Works for timm VisionTransformer-style models.

    Returns:
        cls_emb: [D]
    """
    # Most timm ViT models expose these components.
    # We are not running all the blocks to make it less expensive
    with torch.no_grad():
        # Patch embedding
        x = model.patch_embed(x)  # [B, N, C]
        # Positional embedding / cls token
        if hasattr(model, "_pos_embed"):
            x = model._pos_embed(x)
        else:
            # fallback for older timm versions
            cls_tok = getattr(model, "cls_token", None)
            pos_embed = getattr(model, "pos_embed", None)
            if cls_tok is not None:
                cls_tokens = cls_tok.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            if pos_embed is not None:
                x = x + pos_embed
            x = getattr(model, "pos_drop", torch.nn.Identity())(x)

        # Run blocks up to probe_block (inclusive)
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Model does not have .blocks; expected timm VisionTransformer.")

        probe_block = int(probe_block)
        probe_block = max(0, min(probe_block, len(blocks) - 1))
        for i in range(probe_block + 1):
            x = blocks[i](x)

        # CLS token is token 0
        cls_emb = x[:, 0, :]  # [B, C]
        return cls_emb.squeeze(0).detach().cpu()


def vit_full_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Run full forward and return logits."""
    with torch.no_grad():
        logits = model(x)
        return logits.detach().cpu()


@dataclass
class Row:
    t: int
    cls_dist: float
    decision: str
    probe_ms: float
    full_ms: float
    decision_ms: float
    base_top1: Optional[int]
    out_top1: int
    match: int


def write_csv(rows, out_path: str):
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "cls_dist",
                "decision",
                "probe_ms",
                "full_ms(if_run)",
                "total_decision_ms",
                "base_top1",
                "out_top1",
                "match",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.t,
                    r.cls_dist if not (r.cls_dist != r.cls_dist) else "nan",  # keep nan readable
                    r.decision,
                    f"{r.probe_ms:.4f}",
                    f"{r.full_ms:.4f}",
                    f"{r.decision_ms:.4f}",
                    r.base_top1 if r.base_top1 is not None else "-",
                    r.out_top1,
                    r.match,
                ]
            )

#Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Path to frames folder (sorted images).")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to use.")
    parser.add_argument("--probe-block", type=int, default=5, help="Probe block index (0-based, inclusive).")
    parser.add_argument("--cls-threshold", type=float, default=8e-05, help="Cosine distance threshold for stability.")
    parser.add_argument(
        "--eval-baseline",
        action="store_true",
        help="If set, also run full baseline for correctness (adds extra compute).",
    )
    parser.add_argument("--csv-out", type=str, default="results/early_exit_logits_reuse.csv", help="CSV output path.")
    args = parser.parse_args()

    device = "cpu"
    model, preprocess = load_timm_vit(device=device)
    model.eval()

    print("\n    TEMPORAL TOKEN REUSE (PROBE CLS + LOGITS REUSE)")
    print("Idea: Running a cheap probe to test stability first; if stable, then reusing previous FULL logits (skipping full forward).")
    print(f"probe_block: {args.probe_block}")
    print(f"cls_threshold: {args.cls_threshold}\n")
    print("t, cls_dist, decision, probe_ms, full_ms(if_run), total_decision_ms, base_top1, out_top1, match")

    prev_probe_cls: Optional[torch.Tensor] = None
    prev_full_logits: Optional[torch.Tensor] = None

    rows = []
    decision_times = []
    baseline_times = []
    reuse_count = 0
    match_count = 0
    match_total = 0

    t_idx = 0

    # iter_frame_pairs should yield (t, img_t, img_t1) or similar. we only need the current frame for decisions.
    # We will treat each "pair index" as one decision step using the 2nd frame (current).
    for t, (_img_prev, img_cur) in iter_frame_pairs(args.frames, max_frames=args.max_frames):
        t_idx += 1

        # Preprocess current frame -> tensor [1, 3, 224, 224]
        x = preprocess(img_cur).unsqueeze(0).to(device)

        # Probe timing
        t0p = time.perf_counter()
        probe_cls = vit_probe_cls(model, x, args.probe_block) 
        t1p = time.perf_counter()
        probe_ms = ms(t0p, t1p)

        # Decision if else
        if prev_probe_cls is None or prev_full_logits is None:
            cls_dist = float("nan")
            decision = "full(first)"
        else:
            cls_dist = cosine_distance(probe_cls, prev_probe_cls)
            decision = "reuse_logits" if cls_dist < args.cls_threshold else "full"

        # baseline correctness
        base_top1 = None
        base_ms = 0.0
        if args.eval_baseline:
            # If decision is full, we can reuse that computation for baseline later
            # BUT we don't know decision full_ms yet, so we will compute baseline only when decision is reuse.
            if decision == "reuse_logits":
                t0b = time.perf_counter()
                base_logits = vit_full_logits(model, x)
                t1b = time.perf_counter()
                base_ms = ms(t0b, t1b)
                base_top1, _ = top1(base_logits)
                baseline_times.append(base_ms)

        # applying decision pipeline
        full_ms = 0.0
        if decision.startswith("full"):
            t0f = time.perf_counter()
            out_logits = vit_full_logits(model, x)
            t1f = time.perf_counter()
            full_ms = ms(t0f, t1f)

            # refresh caches
            prev_full_logits = out_logits
        else:
            # reuse the cached logits
            out_logits = prev_full_logits
            reuse_count += 1

        # Always update probe cache 
        prev_probe_cls = probe_cls

        out_top1, _ = top1(out_logits)

        # If baseline eval mode and decision was full, baseline = out (same logits).
        if args.eval_baseline and decision.startswith("full"):
            base_top1 = out_top1
            base_ms = full_ms
            baseline_times.append(base_ms)

        # Match tracking
        if args.eval_baseline:
            match = int(base_top1 == out_top1)  # base_top1 must exist
            match_count += match
            match_total += 1
        else:
            match = -1  # not evaluated

        decision_ms = probe_ms + full_ms
        decision_times.append(decision_ms)

        base_str = str(base_top1) if base_top1 is not None else "-"
        print(
            f"{t_idx}, {cls_dist if prev_full_logits is not None else 'nan'}, {decision}, "
            f"{probe_ms:.2f}, {full_ms:.2f}, {decision_ms:.2f}, {base_str}, {out_top1}, {match}"
        )

        rows.append(
            Row(
                t=t_idx,
                cls_dist=cls_dist,
                decision=decision,
                probe_ms=probe_ms,
                full_ms=full_ms,
                decision_ms=decision_ms,
                base_top1=base_top1,
                out_top1=out_top1,
                match=match,
            )
        )

        if t_idx >= args.max_frames - 1:
            break

    # Summary
    pairs = len(rows)
    avg_decision_ms = sum(decision_times) / max(1, len(decision_times))
    avg_decision_fps = 1000.0 / avg_decision_ms if avg_decision_ms > 0 else 0.0
    reuse_rate = reuse_count / max(1, pairs)

    # Baseline summary 
    avg_baseline_ms = (sum(baseline_times) / len(baseline_times)) if baseline_times else 0.0
    avg_baseline_fps = (1000.0 / avg_baseline_ms) if avg_baseline_ms > 0 else 0.0

    print("\n   SUMMARY")
    print(f"pairs: {pairs}")
    if args.eval_baseline and avg_baseline_ms > 0:
        print(f"avg_baseline_ms:   {avg_baseline_ms:.2f}  | fps: {avg_baseline_fps:.2f}")
    else:
        print("avg_baseline_ms:   (skipped; run with --eval-baseline)")

    print(f"avg_decision_ms:   {avg_decision_ms:.2f}  | fps: {avg_decision_fps:.2f}")
    print(f"reuse_rate:        {reuse_rate:.3f}")

    if args.eval_baseline:
        match_rate = match_count / max(1, match_total)
        print(f"top1_match_rate:   {match_rate:.3f}")
    else:
        print("top1_match_rate:   (skipped; run with --eval-baseline)")

    print("\nMeaning:")
    print("- decision_ms is the realistic end-to-end time of the reuse system (probe + optional full).")
    print("- If reuse_rate is high AND match_rate stays high, we get real speedups without breaking correctness.\n")

    write_csv(rows, args.csv_out)
    print(f"saved_csv: {args.csv_out}")
    print("Done.")


if __name__ == "__main__":
    main()
