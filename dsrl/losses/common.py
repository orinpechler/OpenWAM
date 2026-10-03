from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

# Frozen diffusion policy steered through its initial noise: pi_dp(obs, noise) -> action chunk.
# ``obs`` is the batch's observation tensor and ``noise`` has shape (B, noise_dim); the
# returned chunk may be unflattened and is reshaped to (B, action_dim) before use.
ActionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

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


def decode_action(action_fn: ActionFn, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Run the frozen diffusion policy and flatten its chunk to (B, action_dim)."""
    return action_fn(obs, noise).reshape(obs.shape[0], -1).to(obs.dtype)
