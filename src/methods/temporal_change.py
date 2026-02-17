from typing import Tuple

import torch


def patch_change_scores(
    prev_tensor: torch.Tensor,
    curr_tensor: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Compute per-patch change score between two preprocessed image tensors.

    Args:
        prev_tensor: [3, H, W] float tensor (already resized/normalized)
        curr_tensor: [3, H, W] float tensor (already resized/normalized)
        patch_size: patch size (ViT-B/16 uses 16)

    Returns:
        scores: [num_patches] tensor, mean absolute difference per patch
    """
    if prev_tensor.shape != curr_tensor.shape:
        raise ValueError(f"Shape mismatch: {prev_tensor.shape} vs {curr_tensor.shape}")

    c, h, w = prev_tensor.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch_size}. Got {(h, w)}")

    diff = (curr_tensor - prev_tensor).abs()  # [3, H, W]

    # Reshape into patches: [3, H//p, p, W//p, p] -> [H//p, W//p, 3, p, p]
    p = patch_size
    diff = diff.view(c, h // p, p, w // p, p).permute(1, 3, 0, 2, 4)

    # Mean over channels and patch pixels => [H//p, W//p]
    patch_scores = diff.mean(dim=(2, 3, 4))

    # Flatten to [num_patches]
    return patch_scores.reshape(-1)


def change_mask_from_scores(
    scores: torch.Tensor,
    keep_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert patch scores to a binary "changed" mask by selecting top keep_ratio patches.

    Args:
        scores: [num_patches]
        keep_ratio: fraction of patches to mark as changed (e.g., 0.2 = top 20%)

    Returns:
        changed_mask: [num_patches] bool tensor (True = changed)
        threshold: scalar tensor threshold used
    """
    if scores.dim() != 1:
        raise ValueError("scores must be a 1D tensor")

    n = scores.numel()
    k = max(1, int(n * keep_ratio))
    topk_vals, _ = torch.topk(scores, k=k, largest=True)
    thresh = topk_vals.min()
    mask = scores >= thresh
    return mask, thresh
