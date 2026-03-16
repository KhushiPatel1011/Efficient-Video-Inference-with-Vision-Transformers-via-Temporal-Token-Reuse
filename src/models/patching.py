"""
This file does patching for timm VisionTransformer.

It will patch the ViT so each transformer block:
  - Caching mode (first frame):  runs normal attention, stores K/V + tokens into cache
  - Matching mode (next frames): reuses cached K/V for stable background tokens,
                                 recomputes K/V only for changed/foreground tokens
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from src.cache.kv_cache import KVCache

# Utility: split fused qkv → separate q, k, v linear layers

def _split_qkv_linear(attn_module: Attention) -> None:
    """
    timm Attention uses a single fused Linear (qkv) of shape [3*dim, dim].
    In this file, we will split it into three separate Linear layers so we can compute
    K and V selectively only for tokens that need recomputation and it will skip this step if already split.
    """
    
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

# Separating forgroud and background tokens based on attention; matching tokens to cache

def _get_saliency_scores(attn: torch.Tensor) -> torch.Tensor:
    """
    Computing per-token saliency using attention entropy.
    Low entropy = background (predictable), High entropy = foreground (important).

    attn: [B, num_heads, N, N]
    returns: scores [B, N, 1]  (0 = background, >0 = foreground)
    """
    if attn.dim() == 4:
        attn_avg = attn.mean(dim=1)  # [B, N, N]
    else:
        attn_avg = attn

    # Entropy-based saliency
    scores = (attn_avg * torch.log(attn_avg + 1e-6)).sum(dim=-1, keepdim=True)  # [B, N, 1]

    # Normalizing per sample
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / (scores.amax(dim=1, keepdim=True) + 1e-6)

    # Threshold: tokens below mean are background
    score_mask = scores >= scores.mean(dim=1, keepdim=True)
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores / (scores.amax(dim=1, keepdim=True) + 1e-6)
    scores[score_mask] = 0.0

    return scores  # [B, N, 1]


def _extract_bg_fg(
    x: torch.Tensor,
    attn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split tokens into background (low saliency) and foreground (high saliency).

    x:    [B, N, C]
    attn: [B, num_heads, N, N]

    Returns:
        x_bg:   [B, N_bg, C]
        x_fg:   [B, N_fg, C]
        idx_bg: [B, N_bg]
        idx_fg: [B, N_fg]
    """
    B, N, C = x.shape
    scores = _get_saliency_scores(attn).squeeze(-1)  # [B, N]

    # Count bg tokens (score == 0) per sample, take the max across batch
    n_bg = int((scores == 0).float().sum(dim=-1).max().item())

    # Strictly enforce n_bg + n_fg == N
    n_bg = max(1, min(n_bg, N - 1))
    n_fg = N - n_bg  # always >= 1, always sums to N

    idx_bg = torch.topk(scores, k=n_bg, dim=-1, largest=False).indices  # [B, n_bg]
    idx_fg = torch.topk(scores, k=n_fg, dim=-1, largest=True).indices   # [B, n_fg]

    x_bg = torch.gather(x, dim=1, index=idx_bg.unsqueeze(-1).expand(-1, -1, C))
    x_fg = torch.gather(x, dim=1, index=idx_fg.unsqueeze(-1).expand(-1, -1, C))

    return x_bg, x_fg, idx_bg, idx_fg

# Patched Attention

class TBKVAttention(Attention):
    """
    Drop-in replacement for timm Attention.

    In caching mode:  runs standard attention, returns K, V, and attn_map.
    In matching mode: reuses cached K/V for bg tokens matched to cache;
                      recomputes K/V only for unmatched bg + fg tokens.
    """

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        x_unnorm: Optional[torch.Tensor] = None,
        mode: str = "cache",
        prev_attn: Optional[torch.Tensor] = None,
        r_match: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns: (attn_out, K, V, new_tokens, attn_map)
            attn_out:   [B, N_q, C]
            K:          [B, num_heads, N_kv, head_dim]
            V:          [B, num_heads, N_kv, head_dim]
            new_tokens: [B, N_reduced, C] or None
            attn_map:   [B, num_heads, N_q, N_kv]
        """
        B, N, C = x.shape
        head_dim = C // self.num_heads
        new_tokens = None

        # Matching: reuse cached K/V for stable background token 
        if mode == "match" and cache is not None and not cache.is_empty() and prev_attn is not None:
            x_bg, x_fg, idx_bg, idx_fg = _extract_bg_fg(x, prev_attn)

            # Match background tokens against cache
            k_matched, v_matched, mc, mt_idx, unm_idx = cache.match_tokens(
                x_bg, r_match=r_match
            )

            # Unmatched background tokens that needs fresh KV
            unm_bg = torch.gather(
                x_bg, dim=1,
                index=unm_idx.unsqueeze(-1).expand(-1, -1, C)
            )

            # Tokens for Q computation: matched_cache + unmatched background + foreground
            tokens_fwd = torch.cat([mc, unm_bg, x_fg], dim=1)

            # For residual: use unnormalized if available
            if x_unnorm is not None:
                x_bg_un, x_fg_un, _, _ = _extract_bg_fg(x_unnorm, prev_attn)
                unm_bg_un = torch.gather(
                    x_bg_un, dim=1,
                    index=unm_idx.unsqueeze(-1).expand(-1, -1, C)
                )
                new_tokens = torch.cat([mc, unm_bg_un, x_fg_un], dim=1)
            else:
                new_tokens = tokens_fwd

            # Q for ALL tokens (cache + unmatched background + foreground)
            q = self.q(tokens_fwd).reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

            # K/V only for unmatched + fg (skipping matched cache tokens)
            tokens_kv = torch.cat([unm_bg, x_fg], dim=1)
            k_new = self.k(tokens_kv).reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_new = self.v(tokens_kv).reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

            # Merge cached + fresh K/V
            K = torch.cat([k_matched, k_new], dim=2)
            V = torch.cat([v_matched, v_new], dim=2)

        # Caching mode: standard attention 
        else:
            q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            K = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            V = self.v(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # Shared Attention computation 
        attn = (q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_map = attn.clone()
        attn = self.attn_drop(attn)

        N_q = q.shape[2]
        out = (attn @ V).transpose(1, 2).reshape(B, N_q, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, K, V, new_tokens, attn_map


# Patched Block

class TBKVBlock(Block):
    """
    Drop-in replacement for timm Block.

    Adds:
      - self.cache: KVCache for this layer
      - self._tbkv_mode: "cache" | "match"
      - self._tbkv_r_match: matching ratio
      - self._prev_attn: previous frame's attention map (set by model)
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)

        mode      = getattr(self, "_tbkv_mode",    "cache")
        prev_attn = getattr(self, "_prev_attn",    None)
        r_match   = getattr(self, "_tbkv_r_match", 0.75)

        # Patched Attention
        attn_out, K, V, new_tokens, attn_map = self.attn(
            norm_x,
            cache=self.cache,
            x_unnorm=x,
            mode=mode,
            prev_attn=prev_attn,
            r_match=r_match,
        )

        # Residual Connection
        if new_tokens is not None and new_tokens.shape[1] != x.shape[1]:
            x = new_tokens + self._drop_path1(attn_out)
        else:
            x = x + self._drop_path1(attn_out)

        # Update cache in caching mode
        if mode == "cache":
            self.cache.store(K, V, x, attn=attn_map)

        # Store attention map for next layer's bg/fg split
        self._prev_attn = attn_map.detach() if attn_map is not None else None

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

# apply_patch: entry point to patch a timm ViT

def apply_patch(model: VisionTransformer, r_match: float = 0.75) -> VisionTransformer:
    """
    Patching a timm ViT in-place for KV temporal token reuse.

    After patching:
      - Call set_caching_mode(True) before processing first frame
      - Call set_caching_mode(False) before processing subsequent frames

    Args:
        model:    timm ViT (e.g., vit_base_patch16_224)
        r_match:  fraction of background tokens matched to cache (default 0.75)

    Returns:
        model (patched in-place)
    """
    for module in model.modules():
        if isinstance(module, Block) and not isinstance(module, TBKVBlock):
            module.__class__ = TBKVBlock
            module.cache = KVCache()
            module._tbkv_mode = "cache"
            module._tbkv_r_match = r_match
            module._prev_attn = None

        elif isinstance(module, Attention) and not isinstance(module, TBKVAttention):
            module.__class__ = TBKVAttention
            _split_qkv_linear(module)

    return model


def set_caching_mode(model: VisionTransformer, caching: bool) -> None:
    """
    Switching all patched blocks between caching and matching 

    Args:
        model:   patched timm ViT
        caching: True = build cache, False = reuse cache
    """
    mode = "cache" if caching else "match"
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            module._tbkv_mode = mode


def reset_cache(model: VisionTransformer) -> None:
    """
    Clearing all per-block caches. Call between video clips.
    """
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            module.cache.clear()
            module._prev_attn = None