import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit


def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    v, idx = torch.max(probs, dim=-1)
    return int(idx.item()), float(v.item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float()
    b = b.float()
    na = torch.norm(a) + eps
    nb = torch.norm(b) + eps
    cos = torch.dot(a, b) / (na * nb)
    return float(1.0 - cos.item())


@torch.no_grad()
def vit_probe_cls(model, x: torch.Tensor, stop_block: int) -> torch.Tensor:
    """
    running ViT only up to stop_block and return CLS embedding (C,).
    """
    if not hasattr(model, "patch_embed") or not hasattr(model, "blocks"):
        raise RuntimeError("Model does not look like a timm ViT")

    B = x.shape[0]
    x = model.patch_embed(x)

    cls_tok = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tok, x), dim=1)

    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        x = x + model.pos_embed
    if hasattr(model, "pos_drop"):
        x = model.pos_drop(x)

    stop_block = int(stop_block)
    stop_block = max(0, min(stop_block, len(model.blocks) - 1))

    for i in range(stop_block + 1):
        x = model.blocks[i](x)

    cls_vec = x[:, 0, :].squeeze(0).detach().cpu()
    return cls_vec


@torch.no_grad()
def vit_full_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")

    # Probe controls
    parser.add_argument("--probe-block", type=int, default=5)
    parser.add_argument(
        "--cls-threshold",
        type=float,
        default=0.00008,
        help="Cosine distance threshold on probed CLS. Smaller => fewer reuses.",
    )

    # Logging
    parser.add_argument("--out-csv", type=str, default="results/early_exit_logits_reuse.csv")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    frames_dir = Path(args.frames)
    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    prev_probe_cls: Optional[torch.Tensor] = None
    prev_full_logits: Optional[torch.Tensor] = None
    prev_full_top1: Optional[int] = None

    rows: List[Dict] = []

    print("\n    TEMPORAL TOKEN REUSE (PROBE CLS + LOGITS REUSE)")
    print("Idea: Running a cheap probe to test stability first; if stable, then reusing previous FULL logits (skipping full forward).")
    print(f"probe_block: {args.probe_block}")
    print(f"cls_threshold: {args.cls_threshold}\n")
    print("t, cls_dist, decision, probe_ms, full_ms(if_run), total_decision_ms, base_top1, out_top1, match")

    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        x = transform(curr_img).unsqueeze(0).to(device)

        # Baseline for correctness reporting ONLY and is not part of decision pipeline
        t0b = time.perf_counter()
        base_logits = vit_full_logits(model, x)
        t1b = time.perf_counter()
        base_ms = ms(t0b, t1b)
        base_top1, _ = top1(base_logits)

        # Decision pipeline 
        t0 = time.perf_counter()

        # Probe CLS
        tp0 = time.perf_counter()
        curr_probe_cls = vit_probe_cls(model, x, stop_block=args.probe_block)
        tp1 = time.perf_counter()
        probe_ms = ms(tp0, tp1)

        if prev_probe_cls is None or prev_full_logits is None or prev_full_top1 is None:
            # First pair: must run full to populate cache
            cls_dist = float("nan")
            decision = "full(first)"
            tf0 = time.perf_counter()
            out_logits = vit_full_logits(model, x)
            tf1 = time.perf_counter()
            full_ms = ms(tf0, tf1)
            out_top1, _ = top1(out_logits)

            prev_full_logits = out_logits.detach().cpu()
            prev_full_top1 = out_top1
        else:
            cls_dist = cosine_distance(prev_probe_cls, curr_probe_cls)
            if cls_dist < args.cls_threshold:
                decision = "reuse_logits"
                out_logits = prev_full_logits  # reusing last full logits
                out_top1 = prev_full_top1
                full_ms = 0.0
            else:
                decision = "full"
                tf0 = time.perf_counter()
                out_logits = vit_full_logits(model, x)
                tf1 = time.perf_counter()
                full_ms = ms(tf0, tf1)
                out_top1, _ = top1(out_logits)

                prev_full_logits = out_logits.detach().cpu()
                prev_full_top1 = out_top1

        t1 = time.perf_counter()
        decision_ms = ms(t0, t1)

        match = int(base_top1 == out_top1)

        rows.append(
            {
                "t": t,
                "cls_dist": cls_dist,
                "decision": decision,
                "probe_block": args.probe_block,
                "cls_threshold": args.cls_threshold,
                "baseline_ms": base_ms,
                "probe_ms": probe_ms,
                "full_ms": full_ms,
                "decision_ms": decision_ms,
                "base_top1": base_top1,
                "out_top1": out_top1,
                "match": match,
            }
        )

        print(
            f"{t}, {cls_dist if prev_probe_cls is not None else 'nan'}, {decision}, "
            f"{probe_ms:.2f}, {full_ms:.2f}, {decision_ms:.2f}, {base_top1}, {out_top1}, {match}"
        )

        prev_probe_cls = curr_probe_cls

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    avg_base = df["baseline_ms"].mean()
    avg_decision = df["decision_ms"].mean()
    reuse_rate = (df["decision"] == "reuse_logits").mean()
    match_rate = df["match"].mean()

    print("\n   SUMMARY")
    print(f"pairs: {len(df)}")
    print(f"avg_baseline_ms:   {avg_base:.2f}  | fps: {1000.0/avg_base:.2f}")
    print(f"avg_decision_ms:   {avg_decision:.2f}  | fps: {1000.0/avg_decision:.2f}")
    print(f"reuse_rate:        {reuse_rate:.3f}")
    print(f"top1_match_rate:   {match_rate:.3f}")
    print("\nMeaning:")
    print("- decision_ms is the realistic end-to-end time of the reuse system (probe + optional full).")
    print("- If reuse_rate is high AND match_rate stays high, we get real speedups without breaking correctness.")
    print(f"\nsaved_csv: {out_csv.as_posix()}")
    print("Done.")


if __name__ == "__main__":
    main()