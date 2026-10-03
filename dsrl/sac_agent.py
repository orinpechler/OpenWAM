import copy
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from dsrl.losses import action_critic_loss, actor_loss, noise_critic_loss, temperature_loss
from dsrl.models.latent_actor import LatentActor
from dsrl.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    # Type-only: importing OpenWAM pulls in its video/VLM backbones.
    from openwam.model.architectures import BaseWAMArchitecture

# Loss-info prefixes; each becomes a wandb section plotted against the agent's update step.
_METRIC_SECTIONS = ("action_critic", "noise_critic", "actor", "temperature")
_STEP_METRIC = "agent/update_step"


@contextmanager
def _frozen(module: nn.Module) -> Iterator[None]:
    """Stop gradients into ``module``'s parameters (inputs still get gradients)."""
    module.requires_grad_(False)
    try:
        yield
    finally:
        module.requires_grad_(True)


class SACAgent:
    """DSRL-NA agent: SAC in the latent-noise space of a frozen diffusion policy.

    The latent actor pi^W outputs one noise vector w of width ``noise_dim`` (the
    diffusion policy's per-step action width). It is replicated over the
    ``horizon`` time steps to form the policy's initial noise (B, horizon,
    noise_dim), so the actor steers a single per-dimension noise rather than the
    full chunk-sized noise space.

    ``update`` is the training for one environment timestep: first the critics,
    then the policy, each for ``gradient_steps`` gradient steps on fresh batches,
    with one optimizer per network:

        critic phase (x gradient_steps)
            1. action critic Q^A    TD on executed chunks   (action_critic_loss)
            2. noise critic Q^W     distilled from Q^A      (noise_critic_loss)
            3. Polyak update of the Q^A targets every ``target_update_interval`` critic steps
        policy phase (x gradient_steps)
            4. latent actor pi^W    maximize Q^W + entropy  (actor_loss)
            5. temperature alpha                            (temperature_loss)

    ``gamma`` is the per-transition discount; one transition is one executed
    chunk, so this is typically gamma_step ** executed_chunk_length.

    wandb: pass an initialized run as ``wandb_run`` to log the loss metrics after
    every ``update`` (each averaged over its ``gradient_steps`` steps, against
    ``agent/update_step``) and to upload checkpoints as model artifacts.
    """

    def __init__(
        self,
        actor: LatentActor,
        q_action: nn.Module | Sequence[nn.Module],
        q_noise: nn.Module | Sequence[nn.Module],
        replay_buffer: ReplayBuffer,
        wam: "BaseWAMArchitecture",
        horizon: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        gradient_steps: int = 20,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        noise_critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 1.0,
        target_entropy: float | None = None,
        target_reduction: str = "min",
        target_update_interval: int = 1,
        max_grad_norm: float | None = None,
        device: str | torch.device = "cuda",
        wandb_run=None,
    ):
        self.device = torch.device(device)
        self.horizon = horizon
        self.noise_dim = actor.output_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        if gradient_steps < 1:
            raise ValueError("gradient_steps must be at least 1")
        self.gradient_steps = gradient_steps
        self.target_entropy = -float(self.noise_dim) if target_entropy is None else target_entropy
        self.target_reduction = target_reduction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm

        self.actor = actor.to(self.device)
        self.q_action = self._as_module_list(q_action).to(self.device)
        self.q_action_target = copy.deepcopy(self.q_action).requires_grad_(False)
        self.q_noise = self._as_module_list(q_noise).to(self.device)
        self.log_alpha = torch.tensor(float(init_alpha), device=self.device).log().requires_grad_(True)
        self.replay_buffer = replay_buffer
        self.wam = wam

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.q_action.parameters(), lr=critic_lr)
        self.noise_critic_optimizer = torch.optim.Adam(self.q_noise.parameters(), lr=noise_critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.update_step = 0  # update() calls, i.e. trained environment timesteps
        self.critic_step = 0  # critic gradient steps, which drive the target updates
        self.wandb_run = wandb_run
        if wandb_run is not None:
            wandb_run.define_metric(_STEP_METRIC)
            for section in _METRIC_SECTIONS:
                wandb_run.define_metric(f"{section}/*", step_metric=_STEP_METRIC)

    @staticmethod
    def _as_module_list(critics: nn.Module | Sequence[nn.Module]) -> nn.ModuleList:
        if isinstance(critics, nn.ModuleList):
            return critics
        return nn.ModuleList([critics] if isinstance(critics, nn.Module) else critics)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # ------------------------------------------------------------------ sampling

    def expand_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Replicate per-dimension noise (B, noise_dim) over time -> (B, horizon, noise_dim)."""
        return noise.unsqueeze(1).expand(-1, self.horizon, -1)

    def _action_fn(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Steered diffusion policy as the losses see it: pi_dp(s, w) with w replicated over time."""
        # TODO: BaseWAMArchitecture's __call__ is its training forward, and generate() draws
        # its own action noise; replace this with WAM generation from the injected noise.
        return self.wam(obs, self.expand_noise(noise))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample latent noise for a batch of observations (B, obs_dim) and decode it.

        Returns (action_chunk, noise): the diffusion policy's output and the
        per-dimension noise w of shape (B, noise_dim) that produced it.
        ``deterministic`` uses the actor's (squashed) mean instead of a sample.
        """
        obs = obs.to(self.device)
        noise = self.actor.deterministic(obs) if deterministic else self.actor.sample(obs)[0]
        return self._action_fn(obs, noise), noise

    def add_transition(self, obs, action, reward, next_obs, done, noise=None) -> None:
        """Store one or a batch of chunk-level transitions (see ``ReplayBuffer.add``)."""
        self.replay_buffer.add(obs, action, reward, next_obs, done, noise=noise)

    # ------------------------------------------------------------------ updates

    def _step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """Polyak-average the action critics into their targets: target <- tau * online + (1 - tau) * target."""
        for online, target in zip(self.q_action.parameters(), self.q_action_target.parameters()):
            target.lerp_(online, self.tau)

    def _sample_batch(self) -> dict[str, torch.Tensor]:
        return self.replay_buffer.sample(self.batch_size, device=self.device)

    def _critic_step(self) -> dict[str, torch.Tensor]:
        """One gradient step of Q^A then Q^W on a fresh batch, followed by the target update."""
        batch = self._sample_batch()

        loss, info = action_critic_loss(
            batch, self.q_action, self.q_action_target, self.actor, self._action_fn, self.gamma, self.target_reduction
        )
        self._step(self.critic_optimizer, loss)

        loss, noise_info = noise_critic_loss(
            batch, self.q_noise, self.q_action, self._action_fn, self.noise_dim, self.target_reduction
        )
        self._step(self.noise_critic_optimizer, loss)

        self.critic_step += 1
        if self.critic_step % self.target_update_interval == 0:
            self.soft_update_targets()
        return {**info, **noise_info}

    def _policy_step(self) -> dict[str, torch.Tensor]:
        """One gradient step of the latent actor then the temperature on a fresh batch."""
        batch = self._sample_batch()

        with _frozen(self.q_noise):
            loss, info = actor_loss(batch, self.actor, self.q_noise, self.alpha, self.target_reduction)
        self._step(self.actor_optimizer, loss)
        log_prob = info.pop("actor/log_prob")

        loss, temperature_info = temperature_loss(self.log_alpha, log_prob, self.target_entropy)
        self._step(self.alpha_optimizer, loss)
        return {**info, **temperature_info}

    def update(self) -> dict[str, float]:
        """Train for one environment timestep.

        Runs ``gradient_steps`` critic steps (Q^A, Q^W, target update), then
        ``gradient_steps`` policy steps (actor, temperature), each on its own
        batch. Once both phases are done, every metric is averaged over its
        phase's ``gradient_steps`` steps, logged to wandb (if a run is set) and
        returned. The replay buffer must hold at least one transition; warm-up
        gating is left to the training loop.
        """
        sums: dict[str, float] = defaultdict(float)
        for step in (self._critic_step, self._policy_step):
            for _ in range(self.gradient_steps):
                for k, v in step().items():
                    sums[k] += v.mean().item()

        self.update_step += 1
        metrics = {k: v / self.gradient_steps for k, v in sums.items()}
        if self.wandb_run is not None:
            self.wandb_run.log({**metrics, _STEP_METRIC: self.update_step})
        return metrics

    # ------------------------------------------------------------------ checkpointing

    def state_dict(self, include_replay_buffer: bool = False) -> dict:
        state = {
            "actor": self.actor.state_dict(),
            "q_action": self.q_action.state_dict(),
            "q_action_target": self.q_action_target.state_dict(),
            "q_noise": self.q_noise.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "noise_critic_optimizer": self.noise_critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_step": self.update_step,
            "critic_step": self.critic_step,
        }
        if include_replay_buffer:
            state["replay_buffer"] = self.replay_buffer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q_action.load_state_dict(state["q_action"])
        self.q_action_target.load_state_dict(state["q_action_target"])
        self.q_noise.load_state_dict(state["q_noise"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.noise_critic_optimizer.load_state_dict(state["noise_critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.update_step = state["update_step"]
        self.critic_step = state["critic_step"]
        if "replay_buffer" in state:
            self.replay_buffer.load_state_dict(state["replay_buffer"])

    def save_checkpoint(
        self,
        path: str | Path,
        include_replay_buffer: bool = False,
        upload: bool = True,
        aliases: Sequence[str] = (),
    ) -> Path:
        """Save the agent to ``path`` and, with a wandb run and ``upload``, log it as an artifact.

        The artifact is named ``dsrl-agent-<run id>`` (type ``model``) and tagged
        ``latest``, ``step_<update_step>`` and any extra ``aliases``, so a run's
        checkpoints form one versioned artifact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(include_replay_buffer), path)

        if upload and self.wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=f"dsrl-agent-{self.wandb_run.id}",
                type="model",
                metadata={"update_step": self.update_step, "includes_replay_buffer": include_replay_buffer},
            )
            artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact, aliases=["latest", f"step_{self.update_step}", *aliases])
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))

    def load_wandb_checkpoint(self, artifact_ref: str) -> None:
        """Download and load a checkpoint artifact, e.g. ``"dsrl-agent-<run id>:latest"``.

        Requires ``wandb_run``; using the artifact records it as an input of this run.
        """
        if self.wandb_run is None:
            raise RuntimeError("load_wandb_checkpoint needs a wandb run")
        artifact_dir = Path(self.wandb_run.use_artifact(artifact_ref, type="model").download())
        (checkpoint,) = artifact_dir.glob("*.pt")
        self.load_checkpoint(checkpoint)
