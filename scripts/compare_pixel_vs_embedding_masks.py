import argparse
from pathlib import Path
from typing import Dict, List

import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit

# Patch change (Phase 2 implementation)
from src.methods.temporal_change import patch_change_scores, change_mask_from_scores

# Embedding change
from src.methods.embedding_change import compute_embedding_change
from src.utils.token_extract import extract_patch_tokens_pre_blocks


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU for two boolean vectors."""
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    return float(inter / union) if union > 0 else 1.0


def agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Percent agreement (same boolean value)."""
    return float((a == b).float().mean().item())


def save_csv(rows: List[Dict], out_path: Path) -> None:
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
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--out-csv", type=str, default="results/mask_comparison.csv")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    frames_dir = Path(args.frames)
    rows: List[Dict] = []

    print("\n    MASK COMPARISON: Pixel vs Embedding ")
    print("Pixel mask: patch-level change scores on input tensors")
    print("Embedding mask: cosine similarity on pre-block patch tokens\n")
    print("t, pixel_changed, embed_changed, intersection, union, iou, agreement")

    ious: List[float] = []
    agrees: List[float] = []

    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        prev_x = transform(prev_img).to(device)
        curr_x = transform(curr_img).to(device)

        # Patch Mask
        pix_scores = patch_change_scores(prev_x, curr_x, patch_size=args.patch_size)
        pix_mask, pix_thr = change_mask_from_scores(pix_scores, keep_ratio=args.keep_ratio)

        # Embedding Mask
        with torch.no_grad():
            prev_tokens, _ = extract_patch_tokens_pre_blocks(model, prev_x.unsqueeze(0), return_batch=False)
            curr_tokens, _ = extract_patch_tokens_pre_blocks(model, curr_x.unsqueeze(0), return_batch=False)

        emb_change = compute_embedding_change(prev_tokens, curr_tokens, keep_ratio=args.keep_ratio)
        emb_mask = emb_change.changed_mask

        # Comparison
        inter = int(torch.logical_and(pix_mask, emb_mask).sum().item())
        union = int(torch.logical_or(pix_mask, emb_mask).sum().item())
        m_iou = iou(pix_mask, emb_mask)
        m_agree = agreement(pix_mask, emb_mask)

        ious.append(m_iou)
        agrees.append(m_agree)

        row = {
            "t": t,
            "pixel_changed": int(pix_mask.sum().item()),
            "embed_changed": int(emb_mask.sum().item()),
            "intersection": inter,
            "union": union,
            "iou": round(m_iou, 6),
            "agreement": round(m_agree, 6),
            "pixel_threshold": round(float(pix_thr), 6),
            "embed_threshold": round(float(emb_change.threshold), 6),
            "keep_ratio": args.keep_ratio,
        }
        rows.append(row)

        print(
            f"{t}, {row['pixel_changed']}, {row['embed_changed']}, "
            f"{inter}, {union}, {m_iou:.3f}, {m_agree:.3f}"
        )

    if rows:
        avg_iou = sum(ious) / len(ious)
        avg_agree = sum(agrees) / len(agrees)
        print("\n=== SUMMARY ===")
        print(f"pairs: {len(rows)}")
        print(f"avg_iou: {avg_iou:.3f}")
        print(f"avg_agreement: {avg_agree:.3f}")
        print("Interpretation: higher IoU means both masks identify similar changed regions;")
        print("lower IoU indicates embedding-space change differs from pixel-space change.")

    out_csv = Path(args.out_csv)
    save_csv(rows, out_csv)
    print(f"\nsaved_csv: {out_csv.as_posix()}\nDone.")


if __name__ == "__main__":
    main()
