"""
Block Skip Cache for Temporal Token Reuse.

Instead of reusing K/V inside attention, it will cache the entire
output token tensor after a chosen block. For stable frames,
we are skipping all subsequent blocks and jump directly to the head.

"""

import torch
from typing import Optional


class BlockSkipCache:
    """
    Caches token embeddings after a specific transformer block.
    On stable frames, skips all blocks after the cache point.
    """

    def __init__(self, cache_after_block: int = 7):
        """
        Args:
            cache_after_block: block index after which to cache tokens.
                               For ViT-Base (12 blocks), 7 means skip blocks 8-11.
        """
        self.cache_after_block = cache_after_block
        self.cached_tokens: Optional[torch.Tensor] = None  # [B, N, C]
        self.cached_cls: Optional[torch.Tensor] = None     # [B, C]
        self.frame_count = 0

    def is_empty(self) -> bool:
        return self.cached_tokens is None

    def store(self, tokens: torch.Tensor) -> None:
        """
        Store token tensor after cache_after_block.
        tokens: [B, N, C]
        """
        self.cached_tokens = tokens.detach().clone()
        self.cached_cls = tokens[:, 0, :].detach().clone()
        self.frame_count += 1

    def get_cached_tokens(self) -> Optional[torch.Tensor]:
        return self.cached_tokens

    def cls_distance(self, current_tokens: torch.Tensor, eps: float = 1e-8) -> float:
        """
        Cosine distance between cached CLS and current CLS token.
        Used to decide whether to skip blocks.

        Formula:
            d = 1 - cos(cached_cls, current_cls)

        Returns:
            float distance in [0, 2]. 0 = identical, 2 = opposite.
        """
        if self.cached_cls is None:
            return 1.0

        curr_cls = current_tokens[:, 0, :]  # [B, C]

        a = self.cached_cls / (self.cached_cls.norm(dim=-1, keepdim=True) + eps)
        b = curr_cls / (curr_cls.norm(dim=-1, keepdim=True) + eps)
        cos_sim = (a * b).sum(dim=-1).mean()
        return float(1.0 - cos_sim.item())

    def clear(self) -> None:
        self.cached_tokens = None
        self.cached_cls = None
        self.frame_count = 0