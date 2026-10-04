"""Train the DSRL-SAC agent online in RoboTwin.

This process is the policy server: it loads the frozen WAM, builds the encoder,
networks, replay buffer and agent, and serves ``dsrl/data/robotwin/client.py`` (run
separately with ``dsrl/data/robotwin/rollout.sh``; ``jobs/train_dsrl.job`` runs both).
Each executed chunk arrives as one (s, w, r, s') transition from
``RoboTwinRLServer``; once the buffer holds ``learning_starts`` transitions,
each stored one is followed by one ``agent.update()`` (``agent.gradient_steps``
gradient rounds), run while the simulator waits for the reply. Updates never
run the WAM, so they are cheap next to acting.

Every ``eval_every_episodes``-th episode acts with the actor's mean and is not
stored. Checkpoints go to ``checkpoint.dir`` on shared scratch, next to the
run's resolved ``config.yaml``: ``agent_ep<N>.pt`` every ``every_episodes``
training episodes (local only) and ``final.pt`` at the end, which is also
uploaded as the run's ``dsrl-agent-<run id>`` model artifact (alias ``final``).
SIGINT / SIGTERM stop training after the current message and save final.pt.

wandb sections:
    critic/*, actor/*, temperature/*   loss metrics, per agent update
                                       (logged by SACAgent, against agent/update_step)
    rollout/*, eval/*                  per training / evaluation episode
    train/*, buffer/*, time/*          per episode
All episode metrics are plotted against ``env/episode``.

Usage:
    python dsrl/train_dsrl.py wam.ckpt_dir=<checkpoint dir> server.port=<port> [overrides]
"""

import asyncio
import json
import logging
import signal
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from dsrl.data.robotwin.server import RoboTwinRLServer, Transition  # noqa: E402
from dsrl.models.wam_noise import steering_noise_layout  # noqa: E402
from dsrl.sac_agent import SACAgent  # noqa: E402
from openwam.deploy.server import ERR_INTERNAL, ERROR, MAX_MESSAGE_BYTES  # noqa: E402

logger = logging.getLogger("train_dsrl")

_EPISODE_METRIC = "env/episode"
_EPISODE_SECTIONS = ("env", "rollout", "eval", "train", "buffer", "time")


class TrainingServer(RoboTwinRLServer):
    """``RoboTwinRLServer`` that also times each chunk's encode + act + decode and
    calls ``on_reset()`` whenever an episode starts or ends (or is dropped)."""

    def __init__(self, *args, on_reset=None, **kwargs):
        self.on_reset = on_reset
        self.chunk_seconds: list[float] = []
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.chunk_seconds = []
        if self.on_reset is not None:
            self.on_reset()

    def _start_chunk(self, obs: dict) -> None:
        t0 = time.monotonic()
        super()._start_chunk(obs)
        self.chunk_seconds.append(time.monotonic() - t0)


def build_engine(cfg: DictConfig):
    """The WAM's ``JointInferenceEngine``, built exactly as the OpenWAM policy server builds it."""
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    deploy_cfg = _load_deploy_yaml(cfg.wam.deploy_config)
    if cfg.wam.overrides:
        deploy_cfg = OmegaConf.merge(deploy_cfg, OmegaConf.from_dotlist(list(cfg.wam.overrides)))
    # PolicyServer is lazy, so this only loads the checkpoint and builds the engine.
    server = build_server_from_config(deploy_cfg, cfg.wam.ckpt_dir, device=cfg.wam.device, ckpt_name=cfg.wam.ckpt_name)
    server.engine.architecture.requires_grad_(False)
    return server.engine


def raw_action_dim(architecture, horizon: int) -> int:
    """Width of the actions ``generate`` returns, after unnormalization (may differ from the model's)."""
    normalizer = getattr(architecture, "normalizer", None)
    if normalizer is None:
        return architecture.action_dim
    return np.asarray(normalizer.unnormalize(np.zeros((horizon, architecture.action_dim), np.float32))).shape[-1]


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def run_dir() -> Path:
    """Hydra's output dir for this run (logs, .hydra, wandb files); the cwd outside Hydra."""
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        return Path.cwd()


def init_wandb(cfg: DictConfig):
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        id=cfg.wandb.id,
        resume="allow" if cfg.wandb.id else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir()),
    )
    run.define_metric(_EPISODE_METRIC)
    for section in _EPISODE_SECTIONS:
        run.define_metric(f"{section}/*", step_metric=_EPISODE_METRIC)
    logger.info("wandb run: %s (%s)", run.name, run.url)
    return run


class Trainer:
    """Wires the WAM, encoder, agent and server together and runs the training loop."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

        self.engine = build_engine(cfg)
        wam = self.engine.architecture
        self.encoder = instantiate(cfg.encoder)(wam=wam)

        # generate's action noise is (action_num_frames - 1, action_dim), as in WAMInputsBuilder.
        horizon = int(getattr(self.engine.cfg.inference, "num_frames", 49)) - 1
        executed_steps = cfg.agent.executed_steps or horizon
        obs_dim = self.encoder.feature_dim
        # Unified-action checkpoints: steer only the embodiment's dims (see steering_noise_layout).
        active_dims, base_noise = steering_noise_layout(wam, horizon)
        noise_dim = wam.action_dim if active_dims is None else len(active_dims)
        action_dim = executed_steps * raw_action_dim(wam, horizon)
        self.dims = {
            "obs_dim": obs_dim,
            "noise_dim": noise_dim,
            "wam_action_dim": wam.action_dim,
            "active_dims": active_dims,
            "horizon": horizon,
            "executed_steps": executed_steps,
            "action_dim": action_dim,
        }
        logger.info("DSRL dims: %s", self.dims)

        actor = instantiate(cfg.actor, input_dim=obs_dim, output_dim=noise_dim)
        critic = [instantiate(cfg.critic, input_dim=obs_dim, action_dim=noise_dim) for _ in range(cfg.num_critics)]
        self.replay_buffer = instantiate(cfg.replay_buffer, obs_dim=obs_dim, action_dim=action_dim, noise_dim=noise_dim)

        self.run = init_wandb(cfg)
        self.agent: SACAgent = instantiate(cfg.agent, horizon=horizon)(
            active_dims=active_dims,
            base_noise=base_noise,
            actor=actor,
            critic=critic,
            replay_buffer=self.replay_buffer,
            wam=wam,
            wandb_run=self.run,
        )
        if cfg.checkpoint.resume:
            self._resume(str(cfg.checkpoint.resume))

        # The checkpoint dir carries the run's config, so a checkpoint can be evaluated wherever it is.
        self.checkpoint_dir = Path(cfg.checkpoint.dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.checkpoint_dir / "config.yaml", resolve=True)
        logger.info("Checkpoints and config in %s", self.checkpoint_dir)

        self.server = TrainingServer(
            self.engine,
            self.agent,
            self.encoder,
            on_transition=self.on_transition,
            on_episode_end=self.on_episode_end,
            on_reset=self.on_reset,
        )
        if self.run is not None:
            self.run.config.update(
                {
                    "dims": self.dims,
                    "params": {
                        "actor": num_params(self.agent.actor),
                        "critic": num_params(self.agent.critic),
                    },
                },
                allow_val_change=True,
            )

        # Counters. Evaluation episodes count towards ``episodes`` only.
        self.episodes = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.transitions = 0
        self.env_steps = 0
        self.successes: deque[float] = deque(maxlen=cfg.train.success_window)
        self.eval_successes: deque[float] = deque(maxlen=cfg.train.success_window)
        self._episode_start = time.monotonic()
        self._episode_stats: dict[str, list[float]] = defaultdict(list)
        self._stop_requested = False
        self.server.deterministic = self._is_eval(self.episodes + 1)

    def _resume(self, ref: str) -> None:
        if Path(ref).exists():
            self.agent.load_checkpoint(ref)
        else:
            self.agent.load_wandb_checkpoint(ref)
        logger.info("Resumed agent from %s at update step %d", ref, self.agent.update_step)

    # ------------------------------------------------------------------ server callbacks

    def _is_eval(self, episode: int) -> bool:
        every = self.train_cfg.eval_every_episodes
        return bool(every) and episode % every == 0

    def on_transition(self, t: Transition) -> None:
        stats = self._episode_stats
        stats["reward"].append(t.reward)
        features = t.features.to(self.agent.device).unsqueeze(0)
        noise = t.noise.to(self.agent.device).unsqueeze(0)
        with torch.no_grad():
            _, log_std = self.agent.actor(features)
            q = torch.stack([critic(features, noise) for critic in self.agent.critic])
        stats["noise_abs"].append(noise.abs().mean().item())
        stats["noise_norm"].append(noise.norm().item())
        stats["actor_std"].append(log_std.exp().mean().item())
        stats["q"].append(q.mean().item())

        if self.server.deterministic:
            return
        self.agent.add_transition(t.features, t.action, t.reward, t.next_features, t.done, noise=t.noise)
        self.transitions += 1
        if len(self.replay_buffer) >= self.train_cfg.learning_starts:
            t0 = time.monotonic()
            self.agent.update()
            stats["update_seconds"].append(time.monotonic() - t0)

    def on_episode_end(self, info: dict) -> None:
        evaluating = self.server.deterministic
        self.episodes += 1
        self.env_steps += info["steps"]
        if evaluating:
            self.eval_episodes += 1
            self.eval_successes.append(float(info["success"]))
        else:
            self.train_episodes += 1
            self.successes.append(float(info["success"]))

        self._log_episode(info, evaluating)
        logger.info(
            "episode %d (%s): success=%s steps=%d chunks=%d | buffer=%d updates=%d",
            self.episodes,
            "eval" if evaluating else "train",
            info["success"],
            info["steps"],
            info["chunks"],
            len(self.replay_buffer),
            self.agent.update_step,
        )

        every = self.cfg.checkpoint.every_episodes
        if not evaluating and every and self.train_episodes % every == 0:
            self.save_checkpoint(f"agent_ep{self.train_episodes:04d}.pt")
        if self._budget_reached():
            self._stop_requested = True
        self.server.deterministic = self._is_eval(self.episodes + 1)

    def on_reset(self) -> None:
        self._episode_stats = defaultdict(list)
        self._episode_start = time.monotonic()

    def _log_episode(self, info: dict, evaluating: bool) -> None:
        stats = self._episode_stats
        mean = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")  # noqa: E731
        section = "eval" if evaluating else "rollout"
        window = self.eval_successes if evaluating else self.successes
        metrics = {
            _EPISODE_METRIC: self.episodes,
            "env/train_episodes": self.train_episodes,
            "env/eval_episodes": self.eval_episodes,
            "env/transitions": self.transitions,
            "env/env_steps": self.env_steps,
            f"{section}/success": float(info["success"]),
            f"{section}/success_rate": float(np.mean(window)),
            f"{section}/return": float(np.sum(stats["reward"])),
            f"{section}/steps": info["steps"],
            f"{section}/chunks": info["chunks"],
            f"{section}/noise_abs": mean("noise_abs"),
            f"{section}/noise_norm": mean("noise_norm"),
            f"{section}/actor_std": mean("actor_std"),
            f"{section}/q": mean("q"),
            "train/update_step": self.agent.update_step,
            "train/updates_this_episode": len(stats["update_seconds"]),
            "train/alpha": self.agent.alpha.item(),
            "buffer/size": len(self.replay_buffer),
            "time/episode_seconds": time.monotonic() - self._episode_start,
            "time/chunk_seconds": float(np.mean(self.server.chunk_seconds)) if self.server.chunk_seconds else 0.0,
            "time/update_seconds": mean("update_seconds"),
        }
        if self.run is not None:
            self.run.log(metrics)

    def _budget_reached(self) -> bool:
        cfg = self.train_cfg
        if cfg.max_episodes is not None and self.train_episodes >= cfg.max_episodes:
            return True
        return cfg.max_transitions is not None and self.transitions >= cfg.max_transitions

    # ------------------------------------------------------------------ checkpointing

    def save_checkpoint(self, name: str, upload: bool = False, aliases: tuple[str, ...] = ()) -> None:
        path = self.agent.save_checkpoint(self.checkpoint_dir / name, upload=upload, aliases=aliases)
        logger.info("Saved checkpoint %s (update step %d%s)", path, self.agent.update_step, ", uploading" if upload else "")

    # ------------------------------------------------------------------ serving

    def serve(self) -> None:
        """Serve the RoboTwin client until the budget is reached, it disconnects, or a signal arrives.

        ``RoboTwinRLServer.run`` serves forever, so the loop is reproduced here
        with a stop condition, checked after each reply is sent.
        """
        import websockets

        server = self.server

        async def main() -> None:
            loop = asyncio.get_running_loop()
            stop = loop.create_future()

            def request_stop() -> None:
                if not stop.done():
                    stop.set_result(None)

            def on_signal(sig: signal.Signals) -> None:
                logger.info("Received %s, stopping after the current message", sig.name)
                self._stop_requested = True
                request_stop()

            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, on_signal, sig)

            async def handler(websocket) -> None:
                logger.info("Client connected: %s", websocket.remote_address)
                try:
                    async for message in websocket:
                        try:
                            reply = server.handle(json.loads(message))
                        except Exception as e:
                            logger.exception("Error processing message")
                            reply = {"type": ERROR, "code": ERR_INTERNAL, "message": str(e)}
                        await websocket.send(json.dumps(reply))
                        if self._stop_requested:
                            request_stop()
                            return
                except websockets.exceptions.ConnectionClosed:
                    pass
                logger.info("Client disconnected")
                if self.train_cfg.stop_on_disconnect:
                    request_stop()

            # Inference and updates block the event loop far past keepalive deadlines.
            async with websockets.serve(
                handler, self.cfg.server.host, self.cfg.server.port, max_size=MAX_MESSAGE_BYTES, ping_interval=None
            ):
                logger.info("DSRL training server on ws://%s:%d", self.cfg.server.host, self.cfg.server.port)
                await stop

        asyncio.run(main())

    def train(self) -> None:
        try:
            self.serve()
        finally:
            self.save_checkpoint("final.pt", upload=self.cfg.checkpoint.upload_final, aliases=("final",))
            if self.run is not None:
                self.run.summary.update(
                    {
                        "final/train_episodes": self.train_episodes,
                        "final/transitions": self.transitions,
                        "final/update_step": self.agent.update_step,
                        "final/success_rate": float(np.mean(self.successes)) if self.successes else 0.0,
                    }
                )
                self.run.finish()


@hydra.main(version_base="1.3", config_path="configs", config_name="train_dsrl")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
