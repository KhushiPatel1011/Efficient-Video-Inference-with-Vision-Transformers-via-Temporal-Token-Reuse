"""
Patching for timm ViT.
Fixed: handling variable token counts across blocks during matching mode.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from timm.models.vision_transformer import Attention, Block, VisionTransformer

from src.cache.kv_cache import KVCache


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


def _get_saliency_scores(attn: torch.Tensor, N_tokens: int) -> torch.Tensor:
    """
    Computing per-token saliency. Handling size mismatch between attn and N_tokens.
    attn:     [B, num_heads, N_attn, N_attn]
    N_tokens: actual size of x we will split
    Returns:  [B, N_tokens]  — 0=background, >0=foreground
    """
    if attn.dim() == 4:
        attn_avg = attn.mean(dim=1)       # [B, N_attn, N_attn]
    else:
        attn_avg = attn

    scores = (attn_avg * torch.log(attn_avg + 1e-6)).sum(dim=-1)  # [B, N_attn]

    N_attn = scores.shape[1]
    if N_attn != N_tokens:
        if N_attn > N_tokens:
            scores = scores[:, :N_tokens]
        else:
            pad = scores.mean(dim=1, keepdim=True).expand(-1, N_tokens - N_attn)
            scores = torch.cat([scores, pad], dim=1)

    # Normalizing to [0, 1]
    s_min = scores.amin(dim=-1, keepdim=True)
    s_max = scores.amax(dim=-1, keepdim=True)
    scores = (scores - s_min) / (s_max - s_min + 1e-6)

    # Tokens below mean = background (set to 0)
    mean_s = scores.mean(dim=-1, keepdim=True)
    bg_mask = scores < mean_s
    scores = scores.clone()
    scores[bg_mask] = 0.0

    return scores  # [B, N_tokens]


def _extract_bg_fg(
    x: torch.Tensor,
    attn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spliting x into background and foreground tokens
    Should be n_bg + n_fg == N exactly
    x:    [B, N, C]
    attn: [B, num_heads, N_attn, N_attn]  (N_attn may differ from N)
    """
    B, N, C = x.shape
    scores = _get_saliency_scores(attn, N_tokens=N)  # [B, N]

    n_bg = int((scores == 0).float().sum(dim=-1).max().item())
    n_bg = max(1, min(n_bg, N - 1))
    n_fg = N - n_bg                        # always >= 1, sums to N

    idx_bg = torch.topk(scores, k=n_bg, dim=-1, largest=False).indices
    idx_fg = torch.topk(scores, k=n_fg, dim=-1, largest=True).indices

    x_bg = torch.gather(x, dim=1, index=idx_bg.unsqueeze(-1).expand(-1, -1, C))
    x_fg = torch.gather(x, dim=1, index=idx_fg.unsqueeze(-1).expand(-1, -1, C))

    return x_bg, x_fg, idx_bg, idx_fg


class TBKVAttention(Attention):

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

        B, N, C = x.shape
        head_dim = C // self.num_heads
        new_tokens = None

        if mode == "match" and cache is not None and not cache.is_empty() and prev_attn is not None:

            x_bg, x_fg, idx_bg, idx_fg = _extract_bg_fg(x, prev_attn)

            k_matched, v_matched, mc, mt_idx, unm_idx = cache.match_tokens(
                x_bg, r_match=r_match
            )

            unm_bg = torch.gather(
                x_bg, dim=1,
                index=unm_idx.unsqueeze(-1).expand(-1, -1, C)
            )

            tokens_fwd = torch.cat([mc, unm_bg, x_fg], dim=1)

            if x_unnorm is not None:
                x_bg_un, x_fg_un, _, _ = _extract_bg_fg(x_unnorm, prev_attn)
                unm_bg_un = torch.gather(
                    x_bg_un, dim=1,
                    index=unm_idx.unsqueeze(-1).expand(-1, -1, C)
                )
                new_tokens = torch.cat([mc, unm_bg_un, x_fg_un], dim=1)
            else:
                new_tokens = tokens_fwd

            q = self.q(tokens_fwd).reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

            tokens_kv = torch.cat([unm_bg, x_fg], dim=1)
            k_new = self.k(tokens_kv).reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_new = self.v(tokens_kv).reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

            K = torch.cat([k_matched, k_new], dim=2)
            V = torch.cat([v_matched, v_new], dim=2)

        else:
            q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            K = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            V = self.v(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_map = attn.clone()
        attn = self.attn_drop(attn)

        N_q = q.shape[2]
        out = (attn @ V).transpose(1, 2).reshape(B, N_q, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, K, V, new_tokens, attn_map


class TBKVBlock(Block):

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)

        mode      = getattr(self, "_tbkv_mode",    "cache")
        prev_attn = getattr(self, "_prev_attn",    None)
        r_match   = getattr(self, "_tbkv_r_match", 0.75)

        attn_out, K, V, new_tokens, attn_map = self.attn(
            norm_x,
            cache=self.cache,
            x_unnorm=x,
            mode=mode,
            prev_attn=prev_attn,
            r_match=r_match,
        )

        if new_tokens is not None and new_tokens.shape[1] != x.shape[1]:
            x = new_tokens + self._drop_path1(attn_out)
        else:
            x = x + self._drop_path1(attn_out)

        if mode == "cache":
            self.cache.store(K, V, x, attn=attn_map)

        self._prev_attn = attn_map.detach() if attn_map is not None else None

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


def apply_patch(model: VisionTransformer, r_match: float = 0.75) -> VisionTransformer:
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
    mode = "cache" if caching else "match"
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            module._tbkv_mode = mode


def reset_cache(model: VisionTransformer) -> None:
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            module.cache.clear()
            module._prev_attn = None
