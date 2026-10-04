"""DSRL policy server for RoboTwin: OpenWAM's WebSocket protocol, driven by the DSRL agent.

Drop-in for ``openwam.deploy.server.PolicyServer`` from the RoboTwin side: the
same ``obs`` / ``reset`` / ``ping`` messages, plus one ``episode_end`` message
sent by ``dsrl/data/robotwin/client.py``:

    Client -> {"type": "episode_end", "success": bool, "obs": <obs payload>}
    Server -> {"type": "episode_end_ack"}

The server preprocesses observations exactly like ``PolicyServer``, and at each
chunk boundary encodes the state (``WAMEncoder``), samples latent noise from the
agent's actor and decodes it with the steered WAM. It returns one action per
``obs`` message and emits one chunk-level transition per executed chunk:

    reward = 1 for the chunk during which the task succeeded, else 0 (RoboTwin's
             success is sparse and ends the episode)
    done   = success; a step-limit cut-off is a truncation (done = False) and
             bootstraps from the final observation

The ``action`` stored with a transition is the chunk the WAM decoded (before the
gripper projection), even when the episode ended part-way through it, so it
matches what ``SACAgent.decode`` returns for the same state and noise.
"""

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dsrl.models.wam_encoder import WAMEncoder
from dsrl.models.wam_inputs import WAMInputsBuilder
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.server import (
    ACTION,
    ERR_INTERNAL,
    ERR_UNKNOWN_TYPE,
    ERROR,
    MAX_MESSAGE_BYTES,
    OBS,
    PING,
    PONG,
    RESET,
    RESET_ACK,
)

if TYPE_CHECKING:
    from dsrl.sac_agent import SACAgent
    from openwam.deploy.engine import JointInferenceEngine

logger = logging.getLogger(__name__)

EPISODE_END = "episode_end"
EPISODE_END_ACK = "episode_end_ack"


@dataclass
class Transition:
    """One executed chunk. ``obs`` / ``next_obs`` are preprocessed observations
    (``image``, ``prompt``, raw ``state``) from which ``WAMInputsBuilder``
    rebuilds the WAM inputs; ``features`` are the encoder's outputs for them."""

    obs: dict
    features: torch.Tensor  # (obs_dim,)
    noise: torch.Tensor  # (noise_dim,)
    action: np.ndarray  # (executed_steps, raw_action_dim)
    reward: float
    next_obs: dict
    next_features: torch.Tensor  # (obs_dim,)
    done: bool


@dataclass
class _Chunk:
    obs: dict
    features: torch.Tensor
    noise: torch.Tensor
    action: np.ndarray


class RoboTwinRLServer:
    """Acts in RoboTwin with the DSRL agent and hands out chunk-level transitions.

    ``on_transition(transition)`` is called once per executed chunk and
    ``on_episode_end(info)`` once per episode (``success``, ``steps``,
    ``chunks``); both run while the simulator waits for the reply, so they can
    store data and run agent updates. ``deterministic`` acts with the actor's
    mean (for evaluation).
    """

    def __init__(
        self,
        engine: "JointInferenceEngine",
        agent: "SACAgent",
        encoder: WAMEncoder,
        on_transition: Callable[[Transition], None] | None = None,
        on_episode_end: Callable[[dict], None] | None = None,
        deterministic: bool = False,
    ):
        self.engine = engine
        self.agent = agent
        self.encoder = encoder
        self.on_transition = on_transition
        self.on_episode_end = on_episode_end
        self.deterministic = deterministic
        self.build_inputs = WAMInputsBuilder(engine)
        self._preprocessor = ObsPreprocessor.from_cfg(engine.cfg, engine)
        self._binary_dims = tuple(getattr(engine.architecture, "binary_command_dims", ()) or ())

        self._actions: deque = deque()
        self._chunk: _Chunk | None = None
        self._steps = 0
        self._chunks = 0

    # ------------------------------------------------------------------ episode logic

    def reset(self) -> None:
        """Start a new episode; a chunk without ``episode_end`` is dropped."""
        self._actions.clear()
        self._chunk = None
        self._steps = 0
        self._chunks = 0

    def step(self, payload: dict) -> np.ndarray:
        """Next action for one ``obs`` message, starting a new chunk when the last one is used up."""
        obs = self._preprocessor.preprocess(dict(payload))
        if not self._actions:
            self._start_chunk(obs)
        self._steps += 1
        action = np.array(self._actions.popleft())
        # Same final gripper projection as WAMPolicy.predict_action.
        for d in self._binary_dims:
            action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def end_episode(self, success: bool, final_payload: dict) -> None:
        """Close the episode: emit the last chunk's transition, then report the episode."""
        if self._chunk is not None:
            final_obs = self._preprocessor.preprocess(dict(final_payload))
            final_features = self.encoder(**self.build_inputs(final_obs))
            self._emit(final_obs, final_features, reward=float(success), done=success)
        if self.on_episode_end is not None:
            self.on_episode_end({"success": success, "steps": self._steps, "chunks": self._chunks})
        self.reset()

    def _start_chunk(self, obs: dict) -> None:
        wam_inputs = self.build_inputs(obs)
        features = self.encoder(**wam_inputs)
        if self._chunk is not None:
            self._emit(obs, features, reward=0.0, done=False)
        action, noise = self.agent.act(features.unsqueeze(0), wam_inputs=[wam_inputs], deterministic=self.deterministic)
        self._chunk = _Chunk(obs=obs, features=features, noise=noise[0], action=action[0].cpu().numpy())
        self._actions.extend(self._chunk.action)
        self._chunks += 1

    def _emit(self, next_obs: dict, next_features: torch.Tensor, reward: float, done: bool) -> None:
        chunk = self._chunk
        if self.on_transition is not None:
            self.on_transition(
                Transition(
                    obs=chunk.obs,
                    features=chunk.features,
                    noise=chunk.noise,
                    action=chunk.action,
                    reward=reward,
                    next_obs=next_obs,
                    next_features=next_features,
                    done=done,
                )
            )

    # ------------------------------------------------------------------ protocol

    def handle(self, data: dict) -> dict:
        """Reply to one client message."""
        msg_type = data.get("type", OBS)
        if msg_type == OBS:
            t0 = time.monotonic()
            action = self.step(data)
            return {
                "type": ACTION,
                "action": action.tolist(),
                "step": self._steps,
                "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            }
        if msg_type == RESET:
            self.reset()
            return {"type": RESET_ACK}
        if msg_type == EPISODE_END:
            self.end_episode(bool(data["success"]), data["obs"])
            return {"type": EPISODE_END_ACK}
        if msg_type == PING:
            contract = getattr(self.engine.architecture, "repr_contract", None) or {}
            return {"type": PONG, **contract}
        return {"type": ERROR, "code": ERR_UNKNOWN_TYPE, "message": f"Unknown message type: {msg_type}"}

    def run(self, host: str = "0.0.0.0", port: int = 8848) -> None:
        """Serve one RoboTwin client over WebSocket until the process is stopped."""
        import websockets

        async def handler(websocket: Any) -> None:
            logger.info("Client connected: %s", websocket.remote_address)
            try:
                async for message in websocket:
                    try:
                        reply = self.handle(json.loads(message))
                    except Exception as e:
                        logger.exception("Error processing message")
                        reply = {"type": ERROR, "code": ERR_INTERNAL, "message": str(e)}
                    await websocket.send(json.dumps(reply))
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected")

        async def serve() -> None:
            # Inference and updates block the event loop far past keepalive deadlines.
            async with websockets.serve(handler, host, port, max_size=MAX_MESSAGE_BYTES, ping_interval=None):
                logger.info("DSRL RoboTwin server on ws://%s:%d", host, port)
                await asyncio.Future()

        asyncio.run(serve())
