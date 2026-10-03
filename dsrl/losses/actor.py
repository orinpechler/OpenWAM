import torch

from dsrl.losses.common import Critics, ensemble_q
from dsrl.models.latent_actor import LatentActor


def actor_loss(
    batch: dict[str, torch.Tensor],
    actor: LatentActor,
    q_noise: Critics,
    alpha: float | torch.Tensor,
    reduction: str = "min",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DSRL-NA latent-actor update: SAC policy loss against the noise critic Q^W.

        w ~ pi^W(. | s)  (reparameterized)
        L = mean (alpha * log pi^W(w | s) - Q^W(s, w))

    The actor never touches the diffusion policy; Q^W already scores noise
    directly. Gradients also flow into ``q_noise``'s parameters, so step only the
    actor's optimizer (and zero the critic grads before their own update).

    ``info["actor/log_prob"]`` is detached and can be fed to ``temperature_loss``.
    """
    noise, log_prob = actor.sample(batch["obs"])
    q = ensemble_q(q_noise, batch["obs"], noise, reduction)
    loss = (alpha * log_prob - q).mean()

    info = {
        "actor/loss": loss.detach(),
        "actor/q": q.mean().detach(),
        "actor/log_prob": log_prob.detach(),
        "actor/entropy": -log_prob.mean().detach(),
    }
    return loss, info
