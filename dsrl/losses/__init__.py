"""DSRL-NA losses (Wagenmaker et al., "Steering Your Diffusion Policy with Latent
Space Reinforcement Learning"), one per update so a SAC agent can step each
network with its own optimizer:

    action_critic_loss  Q^A(s, a): plain TD on executed action chunks (no entropy)
    noise_critic_loss   Q^W(s, w): distilled from Q^A(s, pi_dp(s, w)), w ~ N(0, I) (no entropy)
    actor_loss          pi^W(w | s): maximize Q^W - alpha * log pi^W
    temperature_loss    alpha: SAC entropy tuning
"""

from dsrl.losses.action_critic import action_critic_loss
from dsrl.losses.actor import actor_loss
from dsrl.losses.noise_critic import noise_critic_loss
from dsrl.losses.temperature import temperature_loss

__all__ = ["action_critic_loss", "noise_critic_loss", "actor_loss", "temperature_loss"]
