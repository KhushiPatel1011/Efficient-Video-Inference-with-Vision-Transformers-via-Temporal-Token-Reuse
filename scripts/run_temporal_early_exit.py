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
    # a, b: (C,)
    a = a.float()
    b = b.float()
    na = torch.norm(a) + eps
    nb = torch.norm(b) + eps
    cos = torch.dot(a, b) / (na * nb)
    return float(1.0 - cos.item())


@torch.no_grad()
def vit_forward_partial(
    model,
    x: torch.Tensor,
    stop_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forwarding ViT to stop_block (inclusive), and then producing logits using model.norm + model.head.
    Returns:
      logits: (1, num_classes)
      cls_vec: (C,) CLS token embedding after stop_block
    """
    # timm ViT layout (common):
    # x -> patch_embed -> _pos_embed -> blocks -> norm -> head
    # We implemented a minimal version compatible with standard timm ViT.
    if not hasattr(model, "patch_embed") or not hasattr(model, "blocks"):
        raise RuntimeError("Model does not look like a timm ViT (missing patch_embed/blocks).")

    B = x.shape[0]

    x = model.patch_embed(x)  # (B, N, C) without CLS

    # Adding CLS token
    cls_tok = model.cls_token.expand(B, -1, -1)  # (B, 1, C)
    x = torch.cat((cls_tok, x), dim=1)  # (B, 1+N, C)

    # Positional embedding
    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        x = x + model.pos_embed
    if hasattr(model, "pos_drop"):
        x = model.pos_drop(x)

    # Running blocks up to stop_block
    stop_block = int(stop_block)
    if stop_block < 0:
        stop_block = 0
    if stop_block >= len(model.blocks):
        stop_block = len(model.blocks) - 1

    for i in range(stop_block + 1):
        x = model.blocks[i](x)

    # CLS vector after stop_block
    cls_vec = x[:, 0, :].squeeze(0).detach().cpu()  # (C,)

    # Norm + head
    if hasattr(model, "norm") and model.norm is not None:
        x = model.norm(x)

    cls_final = x[:, 0]  # (B, C)
    if hasattr(model, "head") and model.head is not None:
        logits = model.head(cls_final)
    else:
        # some timm models use fc or classifier
        if hasattr(model, "fc"):
            logits = model.fc(cls_final)
        elif hasattr(model, "classifier"):
            logits = model.classifier(cls_final)
        else:
            raise RuntimeError("Cannot find head/fc/classifier on model.")

    return logits, cls_vec


@torch.no_grad()
def vit_forward_full(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")

    # Controls for early exit decision
    parser.add_argument(
        "--exit-block",
        type=int,
        default=5,
        help="Exit after this transformer block index (0-based). Example: 5 means run blocks 0..5 then head.",
    )
    parser.add_argument(
        "--cls-threshold",
        type=float,
        default=0.01,
        help="Cosine distance threshold on CLS between consecutive frames. Smaller => more conservative reuse.",
    )

    # Logging
    parser.add_argument("--out-csv", type=str, default="results/early_exit.csv")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    frames_dir = Path(args.frames)
    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    prev_cls: Optional[torch.Tensor] = None

    rows: List[Dict] = []

    print("\n    TEMPORAL TOKEN REUSE (EARLY EXIT via CLS Stability)")
    print(f"exit_block: {args.exit_block}")
    print(f"cls_threshold (cosine distance): {args.cls_threshold}\n")
    print("t, cls_dist, decision, baseline_ms, earlyexit_ms, base_top1, ee_top1, match")

    # We are evaluating pairs (prev, curr); baseline is always full model on curr for comparison.
    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        curr_x = transform(curr_img).unsqueeze(0).to(device)

        # Baseline forward
        t0 = time.perf_counter()
        base_logits = vit_forward_full(model, curr_x)
        t1 = time.perf_counter()
        baseline_ms = ms(t0, t1)
        base_top1, _ = top1(base_logits)

        # Computing CLS at exit depth
        # We need CLS from current frame at exit depth; for decision we are comparing to prev frame's CLS at same depth.
        t2 = time.perf_counter()
        ee_logits, curr_cls = vit_forward_partial(model, curr_x, stop_block=args.exit_block)
        t3 = time.perf_counter()
        earlyexit_ms = ms(t2, t3)
        ee_top1, _ = top1(ee_logits)

        if prev_cls is None:
            cls_dist = float("nan")
            decision = "early_exit(first)"
        else:
            cls_dist = cosine_distance(prev_cls, curr_cls)
            decision = "early_exit" if cls_dist < args.cls_threshold else "no_exit"

            # If not stable, we will use full model output for "no_exit" mode (so we don’t harm accuracy)
            if decision == "no_exit":
                ee_logits = base_logits
                ee_top1 = base_top1
                # In this case, the "earlyexit_ms" is not a fair timing for the chosen decision
                # (because we had already computed that). For demo, we still report measured partial time,
                # but the real deployment would skip this partial compute and just run full.
                

        match = int(base_top1 == ee_top1)

        rows.append(
            {
                "t": t,
                "cls_dist": cls_dist,
                "decision": decision,
                "exit_block": args.exit_block,
                "cls_threshold": args.cls_threshold,
                "baseline_ms": baseline_ms,
                "earlyexit_ms": earlyexit_ms,
                "base_top1": base_top1,
                "ee_top1": ee_top1,
                "match": match,
            }
        )

        print(
            f"{t}, {cls_dist if prev_cls is not None else 'nan'}, {decision}, "
            f"{baseline_ms:.2f}, {earlyexit_ms:.2f}, {base_top1}, {ee_top1}, {match}"
        )

        prev_cls = curr_cls

    # Summary + save CSV
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    avg_base = df["baseline_ms"].mean()
    avg_partial = df["earlyexit_ms"].mean()
    match_rate = df["match"].mean()
    # We are counting how often the model *would* early-exit (excluding first row with NaN)
    ee_rate = (df["decision"] == "early_exit").mean()

    print("\n    SUMMARY")
    print(f"pairs: {len(df)}")
    print(f"avg_baseline_ms: {avg_base:.2f}  | fps: {1000.0/avg_base:.2f}")
    print(f"avg_partial_ms:  {avg_partial:.2f}  | fps: {1000.0/avg_partial:.2f}")
    print(f"early_exit_rate: {ee_rate:.3f}")
    print(f"top1_match_rate: {match_rate:.3f}")
    print("\nNote: This is the FIRST compute-skipping strategy. Next we’ll make timing faithful to decisions")
    print("      (i.e., run partial only when stable; else run full), so we can report true end-to-end speedup.")
    print(f"\nsaved_csv: {out_csv.as_posix()}")
    print("Done.")


if __name__ == "__main__":
    main()