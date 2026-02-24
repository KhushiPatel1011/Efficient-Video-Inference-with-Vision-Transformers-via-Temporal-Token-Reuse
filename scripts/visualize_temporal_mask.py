import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit
from src.methods.temporal_change import patch_change_scores, change_mask_from_scores


def draw_patch_overlay(
    img_rgb: Image.Image,
    changed_mask: torch.Tensor,
    patch_size: int = 16,
    alpha: int = 90,
) -> Image.Image:
    """
    Overlay changed patches on an RGB image.

    changed_mask: [num_patches] bool tensor (True = changed)
    patch_size: patch size in pixels (assumes image size divisible by patch_size)
    """
    img = img_rgb.copy().convert("RGBA")
    w, h = img.size
    if w % patch_size != 0 or h % patch_size != 0:
        raise ValueError(f"Image size {(w,h)} not divisible by patch_size={patch_size}")

    grid_w = w // patch_size
    grid_h = h // patch_size
    num_patches = grid_w * grid_h
    if changed_mask.numel() != num_patches:
        raise ValueError(
            f"Mask patches={changed_mask.numel()} but grid patches={num_patches} "
            f"({grid_h}x{grid_w})"
        )

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color: red with transparency
    fill = (255, 0, 0, alpha)

    idx = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            if bool(changed_mask[idx].item()):
                x0 = gx * patch_size
                y0 = gy * patch_size
                x1 = x0 + patch_size
                y1 = y0 + patch_size
                draw.rectangle([x0, y0, x1, y1], fill=fill)
            idx += 1

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out


def draw_grid_lines(img_rgb: Image.Image, patch_size: int = 16) -> Image.Image:
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # light grid lines
    for x in range(0, w + 1, patch_size):
        draw.line([(x, 0), (x, h)], width=1)
    for y in range(0, h + 1, patch_size):
        draw.line([(0, y), (w, y)], width=1)

    return img


def annotate(img_rgb: Image.Image, text: str) -> Image.Image:
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # black shadow and white text
    x, y = 8, 8
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Folder of sequential frames")
    parser.add_argument("--outdir", type=str, default="results/temporal_masks", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to visualize (including first)")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name (for preprocess)")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size in pixels (ViT-B/16 uses 16)")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Top fraction patches marked as changed")
    parser.add_argument("--alpha", type=int, default=90, help="Overlay transparency 0-255")
    parser.add_argument("--draw-grid", type=int, default=1, help="1=draw patch grid lines")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Model not used here
    _, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)

    saved = 0
    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        # Normalised Preprocessed tensors used for scoring and not for visualization
        prev = transform(prev_img)
        curr = transform(curr_img)

        scores = patch_change_scores(prev, curr, patch_size=args.patch_size)
        mask, thresh = change_mask_from_scores(scores, keep_ratio=args.keep_ratio)

        total = int(scores.numel())
        changed = int(mask.sum().item())
        stable_ratio = (total - changed) / float(total)

        # For visualization, *resized* image that matches patches is used.
        # We reconstruct it by applying the same resize as transform: easiest is to
        # apply transform and then unnormalize-ish is messy; instead, we rely on PIL resize.
        # timm transform resizes/crops to 224. We mimic that by applying transform then
        # reading size from tensor.
        # We'll just resize curr_img to 224x224 directly for consistency.
        vis_img = curr_img.convert("RGB").resize((224, 224), resample=Image.BICUBIC)

        over = draw_patch_overlay(vis_img, mask, patch_size=args.patch_size, alpha=args.alpha)
        if args.draw_grid == 1:
            over = draw_grid_lines(over, patch_size=args.patch_size)

        text = f"t={t} | changed={changed}/{total} | stable={stable_ratio:.3f} | thr={float(thresh):.6f}"
        over = annotate(over, text)

        out_path = outdir / f"mask_{t:04d}.jpg"
        over.save(out_path, quality=95)
        saved += 1

    print(f"Saved {saved} visualizations to: {outdir.as_posix()}")


if __name__ == "__main__":
    main()