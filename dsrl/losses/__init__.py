"""DSRL-SAC losses (Wagenmaker et al., "Steering Your Diffusion Policy with Latent
Space Reinforcement Learning"), one per update so a SAC agent can step each
network with its own optimizer. The frozen diffusion policy is part of the
environment, so every loss works on (state, latent noise) only:

    critic_loss         Q(s, w): TD on the rewards, on the stored latent noise (no entropy)
    actor_loss          pi^W(w | s): maximize Q - alpha * log pi^W
    temperature_loss    alpha: SAC entropy tuning
"""

from dsrl.losses.actor import actor_loss
from dsrl.losses.critic import critic_loss
from dsrl.losses.temperature import temperature_loss

__all__ = ["critic_loss", "actor_loss", "temperature_loss"]
