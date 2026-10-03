import copy
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
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

# Maps one observation (obs_dim,) to the keyword arguments of ``BaseWAMArchitecture.generate``
# for that state (schedule, prompt, first_frame_image, proprio, num_frames, ...), excluding
# ``action_noise`` and ``decode_video``, which the agent sets itself.
WAMInputsFn = Callable[[torch.Tensor], dict]

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

    ``update`` is the training for one environment timestep: three phases run in
    order, each gradient step on a fresh batch, with one optimizer per network:

        action-critic phase (x gradient_steps)
            1. action critic Q^A    TD on executed chunks   (action_critic_loss)
            2. Polyak update of the Q^A targets every ``target_update_interval`` critic steps
        noise-critic phase (x noise_critic_steps)
            3. noise critic Q^W     distilled from Q^A      (noise_critic_loss)
        policy phase (x gradient_steps)
            4. latent actor pi^W    maximize Q^W + entropy  (actor_loss)
            5. temperature alpha                            (temperature_loss)

    ``gamma`` is the per-transition discount; one transition is one executed
    chunk of ``executed_steps`` actions (default: the full ``horizon``), so this
    is typically gamma_step ** executed_steps.

    wandb: pass an initialized run as ``wandb_run`` to log the loss metrics after
    every ``update`` (each averaged over the steps of its phase, against
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
        wam_inputs_fn: WAMInputsFn | None = None,
        executed_steps: int | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        gradient_steps: int = 20,
        noise_critic_steps: int = 10,
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
        self.executed_steps = horizon if executed_steps is None else executed_steps
        if not 1 <= self.executed_steps <= horizon:
            raise ValueError(f"executed_steps must be in [1, horizon={horizon}], got {executed_steps}")
        self.noise_dim = actor.output_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        if gradient_steps < 1 or noise_critic_steps < 1:
            raise ValueError("gradient_steps and noise_critic_steps must be at least 1")
        self.gradient_steps = gradient_steps
        self.noise_critic_steps = noise_critic_steps
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
        self.wam_inputs_fn = wam_inputs_fn

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.q_action.parameters(), lr=critic_lr)
        self.noise_critic_optimizer = torch.optim.Adam(self.q_noise.parameters(), lr=noise_critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.update_step = 0  # update() calls, i.e. trained environment timesteps
        self.critic_step = 0  # action-critic gradient steps, which drive the target updates
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

    @torch.no_grad()
    def decode(self, wam_inputs: Sequence[dict], noise: torch.Tensor) -> torch.Tensor:
        """Steered WAM pi_dp(s, w) -> executed chunks (B, executed_steps, raw_action_dim), float32.

        ``wam_inputs[i]`` are the ``generate`` kwargs of sample i. ``generate`` is
        batch-1, so this runs one call per sample, with w replicated over time as
        the initial action noise; the video noise is still OpenWAM's own seeded
        draw and the video is never decoded. Only the first ``executed_steps``
        actions of each chunk are kept, matching what the environment executes.
        """
        chunks = []
        for inputs, n in zip(wam_inputs, self.expand_noise(noise)):
            # The DiT velocity cache holds the previous generation's predictions.
            if inputs.get("dit_cache") is not None:
                inputs["dit_cache"].reset()
            out = self.wam.generate(**inputs, action_noise=n, decode_video=False)
            chunks.append(torch.as_tensor(out["actions"])[: self.executed_steps])
        return torch.stack(chunks).to(device=self.device, dtype=torch.float32)

    def _action_fn(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """pi_dp(s, w) as the losses see it, with each state's WAM inputs from ``wam_inputs_fn``."""
        if self.wam_inputs_fn is None:
            raise RuntimeError("updates need wam_inputs_fn to rebuild the WAM inputs of replayed states")
        return self.decode([self.wam_inputs_fn(o) for o in obs], noise)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, wam_inputs: Sequence[dict] | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample latent noise for a batch of observations (B, obs_dim) and decode it.

        ``wam_inputs`` are the samples' ``generate`` kwargs; without them they come
        from ``wam_inputs_fn``. Returns (action_chunk, noise): the executed chunk
        (B, executed_steps, raw_action_dim) and the per-dimension noise w
        (B, noise_dim) that produced it. ``deterministic`` uses the actor's
        (squashed) mean instead of a sample.
        """
        obs = obs.to(self.device)
        noise = self.actor.deterministic(obs) if deterministic else self.actor.sample(obs)[0]
        if wam_inputs is None:
            return self._action_fn(obs, noise), noise
        return self.decode(wam_inputs, noise), noise

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

    def _action_critic_step(self) -> dict[str, torch.Tensor]:
        """One gradient step of Q^A on a fresh batch, followed by the target update."""
        loss, info = action_critic_loss(
            self._sample_batch(),
            self.q_action,
            self.q_action_target,
            self.actor,
            self._action_fn,
            self.gamma,
            self.target_reduction,
        )
        self._step(self.critic_optimizer, loss)

        self.critic_step += 1
        if self.critic_step % self.target_update_interval == 0:
            self.soft_update_targets()
        return info

    def _noise_critic_step(self) -> dict[str, torch.Tensor]:
        """One gradient step of Q^W on a fresh batch."""
        loss, info = noise_critic_loss(
            self._sample_batch(), self.q_noise, self.q_action, self._action_fn, self.noise_dim, self.target_reduction
        )
        self._step(self.noise_critic_optimizer, loss)
        return info

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

        Runs ``gradient_steps`` action-critic steps (Q^A, target update), then
        ``noise_critic_steps`` noise-critic steps (Q^W), then ``gradient_steps``
        policy steps (actor, temperature), each on its own batch. Once all phases
        are done, every metric is averaged over the steps of its phase, logged to
        wandb (if a run is set) and returned. The replay buffer must hold at
        least one transition; warm-up gating is left to the training loop.
        """
        metrics: dict[str, float] = {}
        phases = (
            (self._action_critic_step, self.gradient_steps),
            (self._noise_critic_step, self.noise_critic_steps),
            (self._policy_step, self.gradient_steps),
        )
        for step, num_steps in phases:
            sums: dict[str, float] = defaultdict(float)
            for _ in range(num_steps):
                for k, v in step().items():
                    sums[k] += v.mean().item()
            metrics.update({k: v / num_steps for k, v in sums.items()})

        self.update_step += 1
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
