from typing import TYPE_CHECKING

from openwam.deploy.denoise_schedule import make_schedule

if TYPE_CHECKING:
    from openwam.deploy.engine import JointInferenceEngine


class WAMInputsBuilder:
    """``generate`` kwargs for one server-preprocessed observation, built as in deployment.

    Mirrors how ``JointInferenceEngine.generate`` turns a request into
    ``architecture.generate`` kwargs (deploy config, denoise schedule, proprio
    normalization, frame counts, and the engine's prompt / VACE / DiT caches), so
    the steered WAM sees exactly what the deployed one does. ``action_noise`` and
    ``decode_video`` are left out; the agent sets them. Keep in sync with
    ``openwam/deploy/engine.py``.

    ``obs`` is the output of ``ObsPreprocessor.preprocess``: ``image`` (PIL),
    ``prompt`` and, for proprio checkpoints, the raw ``state``.
    """

    def __init__(self, engine: "JointInferenceEngine", seed: int = 42):
        self.engine = engine
        self.seed = seed
        arch = engine.architecture
        inf_cfg = engine.cfg.inference

        shift = getattr(getattr(arch, "action_backbone", None), "shift_action", None)
        self.shift = 5.0 if shift is None else shift
        self.denoise_steps = inf_cfg.denoise_steps
        # The schedule only depends on the deploy config, so it is built once.
        self.schedule = make_schedule(
            getattr(inf_cfg, "denoise_mode", "sync"),
            video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler,
            num_steps=self.denoise_steps,
            shift=self.shift,
            shift_video=getattr(getattr(arch, "video_backbone", None), "shift_video", None),
            lead=getattr(inf_cfg, "lead_modality", "video"),
            alpha=getattr(inf_cfg, "variance_shift_alpha", 1.0),
            offset=getattr(inf_cfg, "linear_offset", 0.0),
        )
        self.action_num_frames = int(getattr(inf_cfg, "num_frames", 49))
        self.video_num_frames = int(getattr(inf_cfg, "video_num_frames", self.action_num_frames))
        self.height = getattr(inf_cfg, "height", 384)
        self.width = getattr(inf_cfg, "width", 320)

    def __call__(self, obs: dict) -> dict:
        engine = self.engine
        state = obs.get("state")
        kwargs = engine._filter_architecture_generate_kwargs(
            {
                "schedule": self.schedule,
                "prompt": obs.get("prompt", ""),
                "vace_video": None,
                "first_frame_image": [obs["image"]],
                "num_frames": self.video_num_frames,
                "action_num_frames": self.action_num_frames,
                "height": self.height,
                "width": self.width,
                "seed": self.seed,
                "tiled": True,
                "input_video_latents": None,
                "num_inference_steps": self.denoise_steps,
                "shift": self.shift,
                "dit_cache": engine._dit_cache,
                "profile": engine._profile,
                "vace_cache": engine._vace_cache,
                "prompt_embed_cache": engine._prompt_embed_cache,
                "proprio": engine.architecture.normalize_deploy_proprio(state) if state is not None else None,
                "cfg_scale": engine._cfg_scale,
                "cfg_merge": engine._cfg_merge,
            }
        )
        return kwargs
