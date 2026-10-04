from collections.abc import Sequence

import torch
import torch.nn as nn

Critics = nn.Module | Sequence[nn.Module]


def as_list(critics: Critics) -> list[nn.Module]:
    """Accept a single critic or an ensemble (list / nn.ModuleList)."""
    return [critics] if isinstance(critics, nn.Module) and not isinstance(critics, nn.ModuleList) else list(critics)


def ensemble_q(critics: Critics, x: torch.Tensor, a: torch.Tensor, reduction: str = "min") -> torch.Tensor:
    """Evaluate every critic on (x, a) and reduce over the ensemble to shape (B,).

    ``min`` is the clipped double-Q estimate used by SAC; ``mean`` averages.
    """
    q = torch.stack([critic(x, a) for critic in as_list(critics)])
    if reduction == "min":
        return q.min(dim=0).values
    if reduction == "mean":
        return q.mean(dim=0)
    raise ValueError(f"unknown ensemble reduction {reduction!r}, expected 'min' or 'mean'")
