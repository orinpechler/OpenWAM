import torch
import torch.nn.functional as F

from dsrl.losses.common import ActionFn, Critics, as_list, decode_action, ensemble_q


def noise_critic_loss(
    batch: dict[str, torch.Tensor],
    q_noise: Critics,
    q_action: Critics,
    action_fn: ActionFn,
    noise_dim: int,
    reduction: str = "min",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DSRL-NA noise-critic update: distill Q^A into the latent-noise critic Q^W.

    For each state in the batch a fresh noise vector is drawn from the diffusion
    policy's prior and decoded through the frozen policy, so Q^W learns the value
    of every noise that aliases to a given action, not only the ones the actor
    picked:
        w ~ N(0, I),  a = pi_dp(s, w)
        L = sum_j mean (Q^W_j(s, w) - Q^A(s, a))^2
    The regression target carries no entropy term. Q^A is reduced over its
    ensemble with ``reduction``; pass the target critics as ``q_action`` to
    distill from those instead. Only ``q_noise`` receives gradients.
    """
    obs = batch["obs"]
    noise = torch.randn(obs.shape[0], noise_dim, device=obs.device, dtype=obs.dtype)
    with torch.no_grad():
        action = decode_action(action_fn, obs, noise)
        target = ensemble_q(q_action, obs, action, reduction)

    qs = [critic(obs, noise) for critic in as_list(q_noise)]
    loss = sum(F.mse_loss(q, target) for q in qs)

    info = {
        "noise_critic/loss": loss.detach(),
        "noise_critic/q": torch.stack(qs).mean().detach(),
        "noise_critic/target": target.mean(),
    }
    return loss, info
