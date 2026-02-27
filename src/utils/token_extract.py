from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def extract_patch_tokens_pre_blocks(
    model,
    x: torch.Tensor,
    *,
    return_batch: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract ViT patch token embeddings BEFORE transformer blocks (ViT-aware, non-invasive).

    This function mirrors the early part of timm VisionTransformer.forward_features:
      1) patch_embed
      2) prepend cls token (and dist token if present)
      3) add positional embedding
      4) pos_drop

    Then returns ONLY the spatial patch tokens (excluding CLS / dist tokens).

    Parameters
    ----------
    model:
        timm VisionTransformer-like model (e.g., vit_base_patch16_224).
    x:
        Input image tensor of shape (B, 3, H, W), typically (B, 3, 224, 224).
    return_batch:
        If False, returns patch tokens of shape (N, C) for B=1.
        If True, returns patch tokens of shape (B, N, C).

    Returns
    -------
    patch_tokens:
        Patch token embeddings BEFORE blocks.
        Shape: (N, C) if B=1 and return_batch=False, else (B, N, C).
        For ViT-B/16 at 224x224, N=196 and C=768.
    cls_tokens:
        CLS token embedding after pos embed/drop (optional), shape (B, 1, C) if present.
        Returned for future use (e.g., comparison), else None.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (B,3,H,W), got {tuple(x.shape)}")

    B = x.shape[0]

    # 1) Patch embedding: (B, N, C)
    if not hasattr(model, "patch_embed"):
        raise AttributeError("Model has no patch_embed attribute (not a ViT-like timm model).")
    x_tokens = model.patch_embed(x)

    if x_tokens.ndim != 3:
        raise ValueError(f"patch_embed output must be (B,N,C). Got {tuple(x_tokens.shape)}")

    # 2) Prepend CLS token (and dist token if present)
    cls_out = None
    start_idx = 0

    if hasattr(model, "cls_token") and model.cls_token is not None:
        cls_tok = model.cls_token.expand(B, -1, -1)
        cls_out = cls_tok  # will be updated after pos embed/drop
        x_tokens = torch.cat((cls_tok, x_tokens), dim=1)
        start_idx = 1

    # Some timm ViT variants have distillation token
    if hasattr(model, "dist_token") and model.dist_token is not None:
        dist_tok = model.dist_token.expand(B, -1, -1)
        x_tokens = torch.cat((x_tokens[:, :start_idx], dist_tok, x_tokens[:, start_idx:]), dim=1)
        # If CLS existed: sequence is [CLS, DIST, patches...], else [DIST, patches...]
        start_idx = start_idx + 1

    # 3) Positional embedding
    # Older timm: x = x + pos_embed
    # Newer timm sometimes uses model._pos_embed(x)
    if hasattr(model, "_pos_embed") and callable(getattr(model, "_pos_embed")):
        x_tokens = model._pos_embed(x_tokens)
    else:
        if not hasattr(model, "pos_embed") or model.pos_embed is None:
            raise AttributeError("Model has no pos_embed and no _pos_embed().")
        x_tokens = x_tokens + model.pos_embed

    # 4) Positional dropout
    if hasattr(model, "pos_drop") and model.pos_drop is not None:
        x_tokens = model.pos_drop(x_tokens)

    # Update cls token output (post pos)
    if cls_out is not None:
        cls_out = x_tokens[:, :1, :]

    # Return ONLY patch tokens (exclude CLS and/or DIST tokens)
    patch_tokens = x_tokens[:, start_idx:, :]

    if not return_batch:
        if B != 1:
            raise ValueError("return_batch=False requires batch size B=1.")
        patch_tokens = patch_tokens.squeeze(0)  # (N, C)
        # cls_out remains (1,1,C) to avoid ambiguity; caller can squeeze if needed

    return patch_tokens, cls_out


def cosine_similarity_per_token(
    tokens_prev: torch.Tensor,
    tokens_curr: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    This is to Compute cosine similarity per token between two token tensors.

    It Accepts:
      - (N, C) and (N, C)  -> returns (N,)
      - (B, N, C) and (B, N, C) -> returns (B, N)

    and Returns cosine similarity in [-1, 1].
    """
    if tokens_prev.shape != tokens_curr.shape:
        raise ValueError(f"Shape mismatch: {tuple(tokens_prev.shape)} vs {tuple(tokens_curr.shape)}")

    # Normalize last dimension (C)
    a = tokens_prev / (tokens_prev.norm(dim=-1, keepdim=True) + eps)
    b = tokens_curr / (tokens_curr.norm(dim=-1, keepdim=True) + eps)
    sim = (a * b).sum(dim=-1)
    return sim