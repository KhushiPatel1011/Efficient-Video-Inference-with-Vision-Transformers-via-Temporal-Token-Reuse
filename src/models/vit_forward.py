from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def forward_from_tokens(
    model,
    tokens_with_special: torch.Tensor,
) -> torch.Tensor:
    """
    Forwarding timm ViT starting from already-built token sequence

    tokens_with_special:
        Shape (B, T, C) where T includes CLS (+ dist token if present) + patch tokens after pos embed/drop

    Returns:
        logits: (B, num_classes)
    """
    x = tokens_with_special

    # Forwarding through transformer blocks
    if not hasattr(model, "blocks"):
        raise AttributeError("Model has no blocks attribute")

    for blk in model.blocks:
        x = blk(x)

    # Final norm
    if hasattr(model, "norm") and model.norm is not None:
        x = model.norm(x)

    # Classification head expects pooled token (timm ViT uses CLS token at index 0)
    if hasattr(model, "fc_norm") and model.fc_norm is not None:
        # Some timm variants use global pool with fc_norm
        # If global_pool is set, then using mean pooling over tokens (excluding special)
        if getattr(model, "global_pool", None) in ("avg", "token"):
            # avg: average over patch tokens (exclude CLS)
            # token: uses CLS token
            if model.global_pool == "avg":
                x_pooled = x[:, 1:, :].mean(dim=1)
            else:
                x_pooled = x[:, 0, :]
            x_pooled = model.fc_norm(x_pooled)
        else:
            x_pooled = x[:, 0, :]
    else:
        x_pooled = x[:, 0, :]

    # Head
    if not hasattr(model, "head"):
        raise AttributeError("Model has no head attribute.")
    logits = model.head(x_pooled)
    return logits


@torch.no_grad()
def build_tokens_pre_blocks(
    model,
    patch_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Building a ViT token sequence from patch tokens by adding special tokens + pos embed + pos_drop
    This matches the 'pre-block' stage of timm ViT

    patch_tokens:
        Shape (B, N, C) without CLS/dist.

    Returns:
        tokens_with_special: (B, T, C) including CLS/dist + patch tokens after pos embed/drop.
        start_idx: index where patch tokens begin in the full sequence.
                  (1 if CLS only, 2 if CLS+dist, 0 if neither)
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens shape (B,N,C), got {tuple(patch_tokens.shape)}")

    B = patch_tokens.shape[0]
    x_tokens = patch_tokens
    start_idx = 0

    # CLS token
    if hasattr(model, "cls_token") and model.cls_token is not None:
        cls_tok = model.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tok, x_tokens), dim=1)
        start_idx = 1

    # Dist token (if any)
    if hasattr(model, "dist_token") and model.dist_token is not None:
        dist_tok = model.dist_token.expand(B, -1, -1)
        x_tokens = torch.cat((x_tokens[:, :start_idx], dist_tok, x_tokens[:, start_idx:]), dim=1)
        start_idx = start_idx + 1

    # Positional embedding
    if hasattr(model, "_pos_embed") and callable(getattr(model, "_pos_embed")):
        x_tokens = model._pos_embed(x_tokens)
    else:
        if not hasattr(model, "pos_embed") or model.pos_embed is None:
            raise AttributeError("Model has no pos_embed and no _pos_embed().")
        x_tokens = x_tokens + model.pos_embed

    # Positional dropout
    if hasattr(model, "pos_drop") and model.pos_drop is not None:
        x_tokens = model.pos_drop(x_tokens)

    return x_tokens, start_idx