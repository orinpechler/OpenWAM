"""Evaluate OpenWAM steered by a trained DSRL latent actor in RoboTwin.

Drop-in for the OpenWAM policy server (``scripts/deploy.sh``): it speaks the same
WebSocket protocol, so RoboTwin is driven by OpenWAM's own unchanged evaluation,
``benchmarks/robotwin/single_eval.sh`` (evaluation seeds, results in
``$ROBOTWIN_PATH/eval_result/``); ``jobs/test_wam_dsrl.job`` runs both. At each
chunk boundary the observation is encoded (``WAMEncoder``), the latent actor
picks the noise (its mean by default) and the WAM decodes it into the chunk.

Only the OpenWAM checkpoint and the actor are loaded: the DSRL checkpoint is
memory-mapped and only its ``actor`` weights are read (no critics, optimizers or
replay buffer). The actor architecture, encoder layer and WAM deploy settings
come from the training run's config, so the WAM runs exactly as in training:
    local checkpoint    <checkpoint dir>/{agent_ep<N>,final}.pt -> <checkpoint dir>/config.yaml
                        (older runs: <run dir>/checkpoints/agent.pt -> <run dir>/.hydra/config.yaml)
    wandb artifact      "<entity>/<project>/dsrl-agent-<run id>:final" -> config of the run that logged it
or ``--train-config`` to point at the config explicitly.

Usage:
    python dsrl/test_wam_dsrl.py --ckpt-dir <OpenWAM checkpoint dir> --policy-ckpt <final.pt | artifact> \
        [--port 8848] [--device cuda:0]
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from dsrl.data.robotwin.server import RoboTwinRLServer  # noqa: E402
from dsrl.models.latent_actor import LatentActor  # noqa: E402
from dsrl.models.wam_noise import steering_noise_layout  # noqa: E402
from dsrl.sac_agent import SACAgent  # noqa: E402
from dsrl.train_dsrl import build_engine  # noqa: E402

logger = logging.getLogger("test_wam_dsrl")


class SteeringPolicy:
    """The latent actor and the frozen WAM, with the ``act`` interface ``RoboTwinRLServer`` uses."""

    # Same steering as in training; these only need horizon / active_dims / base_noise /
    # executed_steps / wam / device.
    expand_noise = SACAgent.expand_noise
    decode = SACAgent.decode

    def __init__(
        self,
        actor: LatentActor,
        wam,
        horizon: int,
        executed_steps: int,
        active_dims: list[int] | None,
        base_noise: torch.Tensor | None,
        device: str | torch.device,
    ):
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.wam = wam
        self.horizon = horizon
        self.executed_steps = executed_steps
        self.active_dims = active_dims
        self.base_noise = None if base_noise is None else base_noise.to(device=self.device, dtype=torch.float32)
        self.chunks = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, wam_inputs, deterministic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        noise = self.actor.deterministic(obs) if deterministic else self.actor.sample(obs)[0]
        action = self.decode(wam_inputs, noise)
        # One line per chunk, to verify encoder -> actor -> WAM decode are all in the loop.
        self.chunks += 1
        logger.info(
            "chunk %d: obs |f|=%.2f | actor noise mean|w|=%.3f |w|=%.3f range=[%.2f, %.2f] | action %s range=[%.2f, %.2f]",
            self.chunks,
            obs.norm().item(),
            noise.abs().mean().item(),
            noise.norm().item(),
            noise.min().item(),
            noise.max().item(),
            tuple(action.shape),
            action.min().item(),
            action.max().item(),
        )
        return action, noise


def resolve_policy_checkpoint(ref: str, train_config: str | None) -> tuple[Path, DictConfig]:
    """Local path of the DSRL checkpoint and the config of the run that trained it."""
    path = Path(ref)
    if path.exists():
        if train_config:
            return path, OmegaConf.load(train_config)
        candidates = (path.parent / "config.yaml", path.parent.parent / ".hydra" / "config.yaml")
        for config_path in candidates:
            if config_path.exists():
                return path, OmegaConf.load(config_path)
        raise FileNotFoundError(f"training config not found at {' or '.join(map(str, candidates))}; pass --train-config")

    import wandb

    artifact = wandb.Api().artifact(ref, type="model")
    (path,) = Path(artifact.download()).glob("*.pt")
    cfg = OmegaConf.load(train_config) if train_config else OmegaConf.create(artifact.logged_by().config)
    return path, cfg


def load_actor(
    path: Path, cfg: DictConfig, input_dim: int, output_dim: int, active_dims: list[int] | None
) -> LatentActor:
    """Build the actor from the training config and load only its weights.

    ``active_dims`` are the noise dims steered with this WAM; they must be the
    ones the actor was trained to steer.
    """
    # mmap: only the actor's tensors are read from disk.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    trained_dims = state.get("active_dims")
    if trained_dims != active_dims:
        raise ValueError(
            f"the actor was trained to steer noise dims {trained_dims}, but this WAM's active dims are "
            f"{active_dims}; use the OpenWAM checkpoint it was trained with"
        )
    actor = instantiate(cfg.actor, input_dim=input_dim, output_dim=output_dim)
    actor.load_state_dict(state["actor"])
    logger.info("Loaded actor from %s (agent update step %s)", path, state.get("update_step"))
    return actor


def build_server(args: argparse.Namespace) -> RoboTwinRLServer:
    policy_path, cfg = resolve_policy_checkpoint(args.policy_ckpt, args.train_config)
    # The OpenWAM checkpoint and device come from the command line; the deploy settings from training.
    cfg.wam.ckpt_dir = args.ckpt_dir
    cfg.wam.ckpt_name = args.ckpt_name
    cfg.wam.device = args.device
    logger.info("WAM deploy settings from training: config=%s overrides=%s", cfg.wam.deploy_config, cfg.wam.overrides)

    engine = build_engine(cfg)
    wam = engine.architecture
    encoder = instantiate(cfg.encoder)(wam=wam)
    horizon = int(getattr(engine.cfg.inference, "num_frames", 49)) - 1
    executed_steps = cfg.agent.executed_steps or horizon

    active_dims, base_noise = steering_noise_layout(wam, horizon)
    noise_dim = wam.action_dim if active_dims is None else len(active_dims)

    actor = load_actor(policy_path, cfg, encoder.feature_dim, noise_dim, active_dims)
    policy = SteeringPolicy(actor, wam, horizon, executed_steps, active_dims, base_noise, device=args.device)
    logger.info(
        "Steering policy ready: horizon=%d executed_steps=%d noise_dim=%d (of %d) obs_dim=%d, %s actions",
        horizon,
        executed_steps,
        noise_dim,
        wam.action_dim,
        encoder.feature_dim,
        "sampled" if args.stochastic else "mean",
    )
    successes = []

    def log_episode(info: dict) -> None:
        successes.append(info["success"])
        logger.info(
            "episode %d: success=%s steps=%d chunks=%d | success rate %d/%d",
            len(successes),
            info["success"],
            info["steps"],
            info["chunks"],
            sum(successes),
            len(successes),
        )

    return RoboTwinRLServer(engine, policy, encoder, on_episode_end=log_episode, deterministic=not args.stochastic)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True, help="OpenWAM checkpoint directory.")
    parser.add_argument("--ckpt-name", default=None, help="checkpoint_step_*.safetensors in --ckpt-dir; default: latest.")
    parser.add_argument("--policy-ckpt", required=True, help="DSRL checkpoint (.pt) or wandb artifact ref.")
    parser.add_argument("--train-config", default=None, help="Training run config.yaml; default: found from --policy-ckpt.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind host.")
    parser.add_argument("--port", type=int, default=8848, help="WebSocket port.")
    parser.add_argument("--stochastic", action="store_true", help="Sample the latent noise instead of the actor's mean.")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed (only matters with --stochastic).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    build_server(args).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
