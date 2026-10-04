from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Type-only: importing OpenWAM pulls in its video/VLM backbones.
    from openwam.model.architectures import BaseWAMArchitecture


def steering_noise_layout(
    wam: "BaseWAMArchitecture", horizon: int, seed: int = 42
) -> tuple[list[int] | None, torch.Tensor | None]:
    """Which action-noise dims the latent actor steers, and the fixed noise of the others.

    Unified-action checkpoints (``dataloader.unify_action``) have an action width
    of which only the embodiment's dims are real; ``generate`` keeps the other,
    inactive dims on the analytic noise path (their initial noise, rescaled each
    step) and the normalizer drops them from the returned actions. Steering them
    does next to nothing, so the actor only steers the active dims and the
    inactive ones get the same fixed draw OpenWAM uses by default
    (``seed``, as ``WAMInputsBuilder``), which keeps them in-distribution.

    Both come from the same ``BaseWAMArchitecture`` methods ``generate`` uses.
    Returns (active_dims, base_noise (horizon, action_dim) float32), or
    (None, None) when every dim is active.
    """
    inactive = wam._resolve_inactive_action_dims(None, wam.device)
    if inactive is None:
        return None, None
    active_dims = (~inactive).nonzero().flatten().tolist()
    base_noise = wam._initial_action_latents(None, horizon + 1, seed, wam.device, wam.dtype)
    return active_dims, base_noise.squeeze(0).float()
