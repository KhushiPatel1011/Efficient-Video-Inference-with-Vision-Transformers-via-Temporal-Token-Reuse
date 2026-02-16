from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class HookState:
    token_shapes: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)

    def pretty_print(self) -> None:
        for name, shape in self.token_shapes:
            print(f"{name}: {shape}")


def _make_hook(name: str, state: HookState):
    def hook(module, inputs, output):
        # We are expecting ViT blocks to output tokens shaped [B, N, C]
        if isinstance(output, torch.Tensor):
            state.token_shapes.append((name, tuple(output.shape)))
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            state.token_shapes.append((name, tuple(output[0].shape)))
    return hook


def register_vit_block_hooks(model) -> HookState:
    """
    Registering forward hooks on timm ViT blocks to record token shapes.
    It works with timm ViT models that expose model.blocks.
    """
    state = HookState()

    if not hasattr(model, "blocks"):
        raise AttributeError("Model has no attribute 'blocks'. Is this a timm ViT?")

    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(_make_hook(f"block_{i}", state))

    return state
