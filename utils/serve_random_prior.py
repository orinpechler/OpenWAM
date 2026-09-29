"""OpenWAM policy server that draws a fresh, recorded action prior per chunk.

Same CLI as ``scripts/deploy.py`` (all arguments are forwarded). Before the
server starts, ``BaseWAMArchitecture.generate`` is wrapped in memory — no repo
file is changed — so that every generated action chunk:

  * draws its action prior with seed ``PCA_SEED_OFFSET + run * 1000 + chunk``
    instead of the fixed default, so reruns from the same starting point differ
    but stay reproducible;
  * keeps the video noise on the fixed seed ``PCA_VIDEO_SEED`` (default 42, the
    stock default) for every chunk and rerun, so only the action prior varies;
  * saves its (32, 80) action prior to ``$PCA_OUT_DIR/run_<run>/chunk_<chunk>.npz``.

The current run id is read from the file ``$PCA_RUN_FILE`` (written by the job
before each rerun); the chunk counter restarts whenever that id changes.
Without ``PCA_RUN_FILE`` the server behaves exactly like the stock one.
"""

import functools
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from openwam.model.architectures.base import BaseWAMArchitecture  # noqa: E402

logger = logging.getLogger("serve_random_prior")

OUT_DIR = os.environ.get("PCA_OUT_DIR")
RUN_FILE = os.environ.get("PCA_RUN_FILE")
SEED_OFFSET = int(os.environ.get("PCA_SEED_OFFSET", "0"))
VIDEO_SEED = int(os.environ.get("PCA_VIDEO_SEED", "42"))

_state = {"run": None, "chunk": 0}


def _current_run():
    try:
        return int(Path(RUN_FILE).read_text().strip())
    except (OSError, ValueError):
        return None


def _patch_generate():
    orig_generate = BaseWAMArchitecture.generate

    @functools.wraps(orig_generate)
    def generate(self, *args, **kwargs):
        run = _current_run()
        if run is None:
            return orig_generate(self, *args, **kwargs)
        if run != _state["run"]:
            _state["run"], _state["chunk"] = run, 0
        chunk = _state["chunk"]
        _state["chunk"] += 1

        seed = SEED_OFFSET + run * 1000 + chunk
        kwargs["seed"] = seed
        # generate() uses `seed` twice: for the video noise (via the video
        # backbone's preprocess) and for the action prior. Pin the video one.
        vb = self.video_backbone
        orig_preprocess = vb.preprocess_input_for_inference
        vb.preprocess_input_for_inference = lambda **kw: orig_preprocess(**{**kw, "seed": VIDEO_SEED})
        try:
            result = orig_generate(self, *args, **kwargs)
        finally:
            del vb.preprocess_input_for_inference  # drop the instance override

        # Re-draw the prior exactly as base.py does: a fresh generator seeded with
        # `seed`, first draw of shape (1, action_num_frames - 1, action_dim).
        action_num_frames = int(kwargs.get("action_num_frames") or kwargs.get("num_frames", 49))
        prior = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        run_dir = Path(OUT_DIR) / f"run_{run:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / f"chunk_{chunk:03d}.npz", prior=prior[0].float().cpu().numpy(), seed=seed)
        logger.info(
            "[pca] run=%d chunk=%d action_seed=%d video_seed=%d prior=%s",
            run, chunk, seed, VIDEO_SEED, tuple(prior.shape[1:]),
        )
        return result

    BaseWAMArchitecture.generate = generate


if __name__ == "__main__":
    if RUN_FILE:
        if not OUT_DIR:
            raise SystemExit("PCA_OUT_DIR must be set together with PCA_RUN_FILE")
        _patch_generate()

    from openwam.deploy.server import main

    main()
