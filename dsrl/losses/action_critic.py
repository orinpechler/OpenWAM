import torch
import torch.nn.functional as F

from dsrl.losses.common import ActionFn, Critics, as_list, decode_action, ensemble_q
from dsrl.models.latent_actor import LatentActor


def action_critic_loss(
    batch: dict[str, torch.Tensor],
    q_action: Critics,
    q_action_target: Critics,
    actor: LatentActor,
    action_fn: ActionFn,
    gamma: float,
    target_reduction: str = "min",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DSRL-NA action-critic update: TD learning of Q^A(s, a) on executed chunks.

    The bootstrap action comes from the current steered policy, i.e. the latent
    actor's noise decoded by the frozen diffusion policy:
        w' ~ pi^W(. | s'),  a' = pi_dp(s', w')
        y  = r + gamma * (1 - done) * Q^A_target(s', a')
        L  = sum_i mean (Q^A_i(s, a) - y)^2
    The target is a plain TD backup with no entropy term; entropy enters only the
    actor loss. ``gamma`` is the per-transition discount; since one transition is
    a whole action chunk this is typically gamma_step ** chunk_length.

    Only the online critics ``q_action`` receive gradients.
    """
    obs, action = batch["obs"], batch["action"]
    with torch.no_grad():
        next_obs = batch["next_obs"]
        next_noise, _ = actor.sample(next_obs)
        next_action = decode_action(action_fn, next_obs, next_noise)
        next_q = ensemble_q(q_action_target, next_obs, next_action, target_reduction)
        target = batch["reward"] + gamma * (1.0 - batch["done"]) * next_q

    qs = [critic(obs, action) for critic in as_list(q_action)]
    loss = sum(F.mse_loss(q, target) for q in qs)

    info = {
        "action_critic/loss": loss.detach(),
        "action_critic/q": torch.stack(qs).mean().detach(),
        "action_critic/target": target.mean(),
    }
    return loss, info
