import torch
import torch.nn.functional as F

from dsrl.losses.common import Critics, as_list, ensemble_q
from dsrl.models.latent_actor import LatentActor


def critic_loss(
    batch: dict[str, torch.Tensor],
    critic: Critics,
    critic_target: Critics,
    actor: LatentActor,
    gamma: float,
    target_reduction: str = "min",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DSRL-SAC critic update: TD learning of Q(s, w) in the latent-noise space.

    The frozen diffusion policy is treated as part of the environment, so the
    critic scores the stored noise ``w`` directly and is bootstrapped with the
    latent actor's next noise; nothing is decoded:
        w' ~ pi^W(. | s')
        y  = r + gamma * (1 - done) * Q_target(s', w')
        L  = sum_i mean (Q_i(s, w) - y)^2
    The target is a plain TD backup of the rewards with no entropy term; entropy
    enters only the actor loss. (An entropy bonus in the target would also make
    the critic prefer bootstrapped, truncated episodes over ones that terminate
    on success.) ``gamma`` is the per-transition discount; since one transition
    is a whole action chunk this is typically gamma_step ** chunk_length.

    Only the online critics ``critic`` receive gradients.
    """
    obs, noise = batch["obs"], batch["noise"]
    with torch.no_grad():
        next_obs = batch["next_obs"]
        next_noise, _ = actor.sample(next_obs)
        next_q = ensemble_q(critic_target, next_obs, next_noise, target_reduction)
        target = batch["reward"] + gamma * (1.0 - batch["done"]) * next_q

    qs = [q(obs, noise) for q in as_list(critic)]
    loss = sum(F.mse_loss(q, target) for q in qs)

    info = {
        "critic/loss": loss.detach(),
        "critic/q": torch.stack(qs).mean().detach(),
        "critic/target": target.mean(),
    }
    return loss, info
