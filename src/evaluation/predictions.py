from typing import Any, Dict, List, Optional

import torch

def topk_from_logits(
    logits: torch.Tensor,
    k: int = 5,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Converting logits into a top-k list of {index, label, prob} dicts.

    Args:
        logits: Tensor of shape [B, num_classes] or [num_classes]
        k: number of top classes
        class_names: optional list of class labels (length num_classes)

    It will return:
        List of dicts sorted by probability descending.
    """
    if logits.dim() == 2:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)

    k = min(k, probs.numel())
    top_probs, top_idx = torch.topk(probs, k=k)

    results: List[Dict[str, Any]] = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        label = class_names[idx] if class_names is not None and idx < len(class_names) else str(idx)
        results.append({"index": idx, "label": label, "prob": float(p)})

    return results
