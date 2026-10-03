import torch


def temperature_loss(
    log_alpha: torch.Tensor,
    log_prob: torch.Tensor,
    target_entropy: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SAC automatic entropy tuning for the latent actor.

        L = -mean (log_alpha * (log pi^W(w | s) + target_entropy))

    ``log_prob`` comes from the actor's samples (e.g. ``info["actor/log_prob"]``
    from ``actor_loss``) and is treated as a constant. A common choice is
    ``target_entropy = -noise_dim``. The entropy coefficient used by the other
    losses is ``log_alpha.exp().detach()``.
    """
    loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    info = {
        "temperature/loss": loss.detach(),
        "temperature/alpha": log_alpha.exp().detach(),
    }
    return loss, info
