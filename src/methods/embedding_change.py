from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from src.utils.token_extract import cosine_similarity_per_token


@dataclass
class EmbeddingChangeResult:
    """
    Output container for embedding-level temporal change.
    """
    scores: torch.Tensor        # (N,) change scores in [0, 2] approximately (1 - cosine)
    changed_mask: torch.Tensor  # (N,) bool mask
    threshold: float            # chosen cutoff value (score)
    changed: int                # number of changed tokens
    total: int                  # total tokens (N)
    stable_ratio: float         # fraction of tokens considered stable


def embedding_change_scores(
    tokens_prev: torch.Tensor,
    tokens_curr: torch.Tensor,
) -> torch.Tensor:
    """
    This is to Compute embedding-space change score per token.

    Change score is defined as:
        score_i = 1 - cosine_similarity(token_prev_i, token_curr_i)

    Inputs:
        tokens_prev, tokens_curr: (N, C) or (B, N, C)

    Returns:
        scores: (N,) or (B, N)
    """
    sim = cosine_similarity_per_token(tokens_prev, tokens_curr)
    scores = 1.0 - sim
    return scores


def change_mask_from_embedding_scores(
    scores: torch.Tensor,
    keep_ratio: float = 0.2,
) -> Tuple[torch.Tensor, float]:
    """
    This is to Convert change scores into a boolean changed mask using top-k selection.

    Parameters
    ----------
    scores:
        (N,) tensor of change scores (higher = more change)
    keep_ratio:
        Fraction of tokens marked as "changed". Example: 0.2 -> top 20% changed.

    Returns
    -------
    changed_mask:
        (N,) bool tensor
    threshold:
        The score threshold used (kth largest score)
    """
    if scores.ndim != 1:
        raise ValueError(f"Expected scores shape (N,), got {tuple(scores.shape)}")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")

    N = scores.numel()
    k = max(1, int(round(N * keep_ratio)))

    # kth largest threshold: top-k changed => threshold is min of selected top-k
    topk_vals, topk_idx = torch.topk(scores, k=k, largest=True, sorted=True)
    thresh = float(topk_vals[-1].item())

    changed_mask = torch.zeros_like(scores, dtype=torch.bool)
    changed_mask[topk_idx] = True
    return changed_mask, thresh


def compute_embedding_change(
    tokens_prev: torch.Tensor,
    tokens_curr: torch.Tensor,
    keep_ratio: float = 0.2,
) -> EmbeddingChangeResult:
    """
    This is a Convenience wrapper that returns scores + mask + summary stats.

    The Inputs should be patch tokens only (no CLS), typically (196, C).
    """
    if tokens_prev.shape != tokens_curr.shape:
        raise ValueError(
            f"Token shape mismatch: {tuple(tokens_prev.shape)} vs {tuple(tokens_curr.shape)}"
        )
    if tokens_prev.ndim != 2:
        raise ValueError("Expected tokens shape (N, C). Provide tokens for a single frame.")

    scores = embedding_change_scores(tokens_prev, tokens_curr)
    changed_mask, thresh = change_mask_from_embedding_scores(scores, keep_ratio=keep_ratio)

    total = int(scores.numel())
    changed = int(changed_mask.sum().item())
    stable_ratio = (total - changed) / float(total)

    return EmbeddingChangeResult(
        scores=scores.detach().cpu(),
        changed_mask=changed_mask.detach().cpu(),
        threshold=thresh,
        changed=changed,
        total=total,
        stable_ratio=stable_ratio,
    )