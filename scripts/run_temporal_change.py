import argparse
from pathlib import Path

import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit
from src.methods.temporal_change import patch_change_scores, change_mask_from_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Folder of sequential frames")
    parser.add_argument("--max-frames", type=int, default=60, help="Limit frames used (including first)")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name (for preprocess cfg)")
    parser.add_argument("--patch-size", type=int, default=16, help="ViT patch size (16 for ViT-B/16)")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Top fraction of patches marked as changed")
    args = parser.parse_args()

    device = torch.device("cpu")

    # We reuse the model's preprocessing transform for consistent resize/normalize
    _, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)

    frames_dir = Path(args.frames)

    num_patches = None
    print("t, changed_patches, total_patches, threshold")

    # max_frames applies to total frames; iter_frame_pairs needs that too
    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        prev = transform(prev_img).to(device)  # [3,H,W]
        curr = transform(curr_img).to(device)

        scores = patch_change_scores(prev, curr, patch_size=args.patch_size)
        changed_mask, thresh = change_mask_from_scores(scores, keep_ratio=args.keep_ratio)

        if num_patches is None:
            num_patches = scores.numel()

        changed = int(changed_mask.sum().item())
        print(f"{t}, {changed}, {num_patches}, {float(thresh):.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
