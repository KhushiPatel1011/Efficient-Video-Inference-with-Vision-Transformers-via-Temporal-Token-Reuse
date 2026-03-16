"""
KV caching for timm VisionTransformer.

Key design decision for correctness:
  - Token ORDER and COUNT are preserved across all blocks (no token dropping).
  - For stable (background) tokens: reuse cached K and V directly.
  - For changed (foreground) tokens: recompute K and V fresh.
  - Q is always computed for all tokens.
  - This preserves residual connections and CLS token position.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from timm.models.vision_transformer import Attention, Block, VisionTransformer

from src.cache.kv_cache import KVCache

# Split fused qkv -> separate q, k, v

def _split_qkv_linear(attn_module: Attention) -> None:
    if hasattr(attn_module, "q") and hasattr(attn_module, "k") and hasattr(attn_module, "v"):
        return
    qkv: nn.Linear = attn_module.qkv
    dim = qkv.in_features
    has_bias = qkv.bias is not None
    device = qkv.weight.device
    dtype = qkv.weight.dtype
    w_q, w_k, w_v = qkv.weight.data.chunk(3, dim=0)
    if has_bias:
        b_q, b_k, b_v = qkv.bias.data.chunk(3, dim=0)

    def _make_linear(w, b=None):
        lin = nn.Linear(dim, dim, bias=has_bias, device=device, dtype=dtype)
        lin.weight.data.copy_(w)
        if has_bias and b is not None:
            lin.bias.data.copy_(b)
        return lin

    attn_module.q = _make_linear(w_q, b_q if has_bias else None)
    attn_module.k = _make_linear(w_k, b_k if has_bias else None)
    attn_module.v = _make_linear(w_v, b_v if has_bias else None)

# Computing stable/changed mask (no token reordering)

def _compute_stable_mask(
    attn: torch.Tensor,
    N: int,
    stable_ratio: float = 0.75,
) -> torch.Tensor:
    """
    Computing a boolean mask identifying stable (background) tokens.
    Does NOT reorder tokens — mask indices correspond to original token positions.

    attn:        [B, num_heads, N_attn, N_attn]
    N:           number of tokens in current x (may differ from N_attn)
    stable_ratio: fraction of tokens to treat as stable (e.g. 0.75 = 75% stable)

    Returns: stable_mask [B, N] — True = stable (reuse K/V), False = changed (recompute)
    """
    B = attn.shape[0]

    if attn.dim() == 4:
        attn_avg = attn.mean(dim=1)   # [B, N_attn, N_attn]
    else:
        attn_avg = attn               # [B, N_attn, N_attn]

    # Entropy-based saliency per token
    scores = (attn_avg * torch.log(attn_avg + 1e-6)).sum(dim=-1)  # [B, N_attn]

    N_attn = scores.shape[1]

    # Aligning scores size with N (the current token count)
    if N_attn > N:
        scores = scores[:, :N]
    elif N_attn < N:
        pad = scores.mean(dim=1, keepdim=True).expand(-1, N - N_attn)
        scores = torch.cat([scores, pad], dim=1)

    # Lower score = more stable (background)
    # Mark bottom `stable_ratio` fraction as stable
    k_stable = max(1, int(N * stable_ratio))
    k_stable = min(k_stable, N - 1)  # always leave at least 1 changed

    # topk with largest=False gives the k_stable LOWEST (most stable) indices
    stable_idx = torch.topk(scores, k=k_stable, dim=-1, largest=False).indices  # [B, k_stable]

    stable_mask = torch.zeros(B, N, dtype=torch.bool, device=attn.device)
    stable_mask.scatter_(1, stable_idx, True)

    return stable_mask  # [B, N]

# Patched Attention: selective K/V computation

class TBKVAttention(Attention):
    """
    Replacing timm Attention.

    Caching mode:
        - Compute Q, K, V for all tokens normally.
        - Return K, V, attn_map for caching.

    Matching mode:
        - Compute Q for ALL tokens (order preserved).
        - For STABLE tokens: reuse cached K, V.
        - For CHANGED tokens: recompute K, V fresh.
        - Merge into full K, V tensors and run attention normally.
        - Token count and order are NEVER changed.
    """

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        mode: str = "cache",
        prev_attn: Optional[torch.Tensor] = None,
        stable_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (attn_out, K, V, attn_map)
            attn_out: [B, N, C]
            K:        [B, num_heads, N, head_dim]
            V:        [B, num_heads, N, head_dim]
            attn_map: [B, num_heads, N, N]
        """
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # Always compute Q for all tokens
        q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        # [B, heads, N, head_dim]

        if (mode == "match"
                and cache is not None
                and not cache.is_empty()
                and prev_attn is not None
                and cache.K is not None
                and cache.K.shape[2] == N):  # cache must match current token count

            # Computing stable mask (which tokens can reuse K/V from cache)
            stable_mask = _compute_stable_mask(prev_attn, N, stable_ratio)
            # [B, N] — True = stable

            changed_mask = ~stable_mask  # [B, N] — True = changed

            # Recomputing K, V ONLY for changed tokens 
            # Starting from cached K, V
            K = cache.K.clone()  # [B, heads, N, head_dim]
            V = cache.V.clone()

            # Getting indices of changed tokens
            # changed_mask: [B, N]
            # For simplicity with B=1, using batch dim 0
            K_fresh = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            V_fresh = self.v(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            changed_expanded = changed_mask.unsqueeze(1).unsqueeze(-1).expand_as(K)
            K = torch.where(changed_expanded, K_fresh, K)
            V = torch.where(changed_expanded, V_fresh, V)
        else:
            # Caching mode or first frame: compute everything fresh
            K = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            V = self.v(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # Standard attention
        attn = (q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_map = attn.clone()
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, K, V, attn_map

# Patched Block

class TBKVBlock(Block):

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)

        mode         = getattr(self, "_tbkv_mode",         "cache")
        prev_attn    = getattr(self, "_prev_attn",         None)
        stable_ratio = getattr(self, "_tbkv_stable_ratio", 0.75)

        attn_out, K, V, attn_map = self.attn(
            norm_x,
            cache=self.cache,
            mode=mode,
            prev_attn=prev_attn,
            stable_ratio=stable_ratio,
        )

        # Standard residual — token count never changes
        x = x + self._drop_path1(attn_out)

        # Store K/V/tokens in cache after each caching-mode block
        if mode == "cache":
            self.cache.store(K, V, norm_x, attn=attn_map)

        # Pass this block's attn_map to the next block
        self._prev_attn = attn_map.detach() if attn_map is not None else None

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

# Public API

def apply_patch(model: VisionTransformer, stable_ratio: float = 0.75) -> VisionTransformer:
    """
    Patching a timm ViT in-place for TBKV temporal token reuse.

    stable_ratio: fraction of tokens treated as stable per block (0.0–1.0).
                  Higher = more reuse, potentially lower accuracy.
                  Lower  = less reuse, closer to baseline accuracy.
    """
    for module in model.modules():
        if isinstance(module, Block) and not isinstance(module, TBKVBlock):
            module.__class__ = TBKVBlock
            module.cache = KVCache()
            module._tbkv_mode = "cache"
            module._tbkv_stable_ratio = stable_ratio
            module._prev_attn = None

        elif isinstance(module, Attention) and not isinstance(module, TBKVAttention):
            module.__class__ = TBKVAttention
            _split_qkv_linear(module)

    return model


def set_caching_mode(model: VisionTransformer, caching: bool) -> None:
    """True = build cache (first frame), False = reuse cache (subsequent frames)."""
    mode = "cache" if caching else "match"
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            module._tbkv_mode = mode


def reset_cache(model: VisionTransformer) -> None:
    """Clear all caches. Call between unrelated video clips."""
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            module.cache.clear()
            module._prev_attn = None
