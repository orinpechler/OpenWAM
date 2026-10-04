"""RoboTwin policy module for DSRL: OpenWAM's RoboTwin client plus an ``episode_end`` message.

RoboTwin loads it like any policy (``--policy_name dsrl.data.robotwin.client``); nothing
on the RoboTwin side changes. Acting is delegated to
``benchmarks/robotwin/openwam2robotwin_interface.py``. After each action, the
client reads RoboTwin's own episode state on the task object: ``eval_success``
(set by ``check_success()`` inside ``take_action``) and the step counter
against ``step_lim``. When the episode is over it sends

    {"type": "episode_end", "success": bool, "obs": <obs payload of the final observation>}

so ``dsrl/data/robotwin/server.py`` can close the last chunk's transition. RoboTwin's
loop stops right after that action, so this is the last message of the episode.

Needs the OpenWAM repo root and ``benchmarks/robotwin`` on ``PYTHONPATH``
(``dsrl/data/robotwin/rollout.sh`` sets both).
"""

import numpy as np

import openwam2robotwin_interface as base
from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from benchmarks.utils import client

EPISODE_END = "episode_end"

get_model = base.get_model
reset_model = base.reset_model


def _payload(TASK_ENV, model: base.ModelClient, observation: dict) -> dict:
    """Obs payload for ``observation``, built as ``ModelClient.step`` builds it."""
    cams = observation["observation"]
    state = base._extract_proprio(model, observation) if model._send_state else None
    left = cams.get("left_camera", {}).get("rgb")
    right = cams.get("right_camera", {}).get("rgb")
    return client.build_payload(
        head=client.encode_numpy_b64(cams["head_camera"]["rgb"]),
        left_wrist=client.encode_numpy_b64(left) if left is not None else None,
        right_wrist=client.encode_numpy_b64(right) if right is not None else None,
        prompt=format_prompt_for_inference(str(TASK_ENV.get_instruction())),
        state=None if state is None else [float(v) for v in np.asarray(state, dtype=np.float32).reshape(-1)],
    )


def eval(TASK_ENV, model: base.ModelClient, observation: dict) -> None:
    """Per-step callback for RoboTwin's eval loop: act, then report the episode end."""
    base.eval(TASK_ENV, model, observation)

    success = bool(TASK_ENV.eval_success)
    if success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
        final_obs = TASK_ENV.get_obs()
        model._client._roundtrip(
            {"type": EPISODE_END, "success": success, "obs": _payload(TASK_ENV, model, final_obs)},
            reconnect=False,
        )
