"""
This file is for KV Caching for Temporal Token Reuse.

Logic:
Here, it stores Key and Value tensors from one frame's attention computation,
then matches and reuses them for subsequent similar frames.
"""

import torch
from typing import Optional, Tuple


class KVCache:
    """
    Per-layer KV cache for temporal token reuse.

    Stores:
        K: [1, num_heads, N_cache, head_dim] — cached key tensors
        V: [1, num_heads, N_cache, head_dim] — cached value tensors
        tokens: [1, N_cache, C] — cached token embeddings (for matching)
    """

    def __init__(self):
        self.K: Optional[torch.Tensor] = None
        self.V: Optional[torch.Tensor] = None
        self.tokens: Optional[torch.Tensor] = None
        self.prev_attn: Optional[torch.Tensor] = None  # attention map from the previous frame

    def is_empty(self) -> bool:
        return self.K is None

    def store(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        tokens: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Now, here we are caching the K, V, and token embeddings
        that we are actually going to reuse in future steps.
         We are not caching the original K, V, and tokens from 
         the current frame; instead, we are caching the K, V, and tokens 
         that we are going to use in future development phases (after matching and reuse decisions).
        
        Storing K, V, and token embeddings into cache.

        Args:
            K:      [B, num_heads, N, head_dim]
            V:      [B, num_heads, N, head_dim]
            tokens: [B, N, C]
            attn:   [B, num_heads, N, N] attention map (optional)
        """
        self.K = K.detach().clone()
        self.V = V.detach().clone()
        self.tokens = tokens.detach().clone()
        if attn is not None:
            self.prev_attn = attn.detach().clone()

    def match_tokens(
        self,
        x: torch.Tensor,
        r_match: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Now, here it will match current tokens against cached tokens using cosine similarity.
        It will returns cached K/V for matched tokens and indices for unmatched tokens.

        Args:
            x:       [B, N, C] current token embeddings (normalized or raw)
            r_match: fraction of tokens to match against cache (e.g., 0.75 = top 75%)

        Returns:
            k_matched:    [B, num_heads, N_matched_unique, head_dim]
            v_matched:    [B, num_heads, N_matched_unique, head_dim]
            mc:           [B, N_matched_unique, C] matched cache token embeddings
            mt_idx:       [B, r] indices into x of matched tokens
            unm_idx:      [B, N-r] indices into x of unmatched tokens
        """
        assert not self.is_empty(), "Cache is empty — call store() first."

        B, N, C = x.shape

        cache_tokens = self.tokens  # [B, N_cache, C]

        # Normalizing for cosine similarity
        a = x / (x.norm(dim=-1, keepdim=True) + 1e-8)           # [B, N, C]
        b = cache_tokens / (cache_tokens.norm(dim=-1, keepdim=True) + 1e-8)  # [B, N_cache, C]

        scores = a @ b.transpose(-1, -2)  # [B, N, N_cache]

        # Number of tokens to match
        r = min(N, max(1, int(N * r_match)))

        # For each current token, it will find the best-matching cache token
        node_max, node_idx = scores.max(dim=-1)  # [B, N]

        # It will rank tokens by their best-match score (descending)
        edge_idx = node_max.argsort(dim=-1, descending=True)  # [B, N]

        # Top-r = matched, rest = unmatched
        mt_idx  = edge_idx[:, :r]   # [B, r]   — matched
        unm_idx = edge_idx[:, r:]   # [B, N-r] — unmatched

        # Cache indices for matched tokens
        mc_idx = torch.gather(node_idx, dim=-1, index=mt_idx)  # [B, r]

        # Deduplicating cache indices (multiple current tokens can map to same cache token)
        mc_idx_unique = mc_idx[0].unique(sorted=False).unsqueeze(0)  # [1, N_unique]

        # Clamp for safety
        max_cache = cache_tokens.shape[1] - 1
        mc_idx_unique = mc_idx_unique.clamp(0, max_cache)

        # Gather cached K, V, tokens for matched unique cache indices
        idx_k = mc_idx_unique.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.K.shape[1], -1, self.K.shape[-1]
        )  # [1, num_heads, N_unique, head_dim]

        k_matched = torch.gather(self.K, dim=2, index=idx_k)
        v_matched = torch.gather(self.V, dim=2, index=idx_k)

        idx_tok = mc_idx_unique.unsqueeze(-1).expand(-1, -1, C)  # [1, N_unique, C]
        mc = torch.gather(cache_tokens, dim=1, index=idx_tok)

        return k_matched, v_matched, mc, mt_idx, unm_idx

    def clear(self) -> None:
        """Reset cache (call between video clips)."""
        self.K = None
        self.V = None
        self.tokens = None
        self.prev_attn = None