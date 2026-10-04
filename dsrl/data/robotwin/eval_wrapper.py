"""OpenWAM's RoboTwin launcher (``benchmarks/robotwin/eval_policy_wrapper.py``) with per-episode videos off.

RoboTwin's ``eval_policy.py`` records an mp4 of every episode (ffmpeg fed every
step) whenever the task config sets ``eval_video_log: true``, as ``demo_clean``
and ``demo_randomized`` do. For training rollouts that is thousands of videos.
RoboTwin enables the recording by passing ``eval_video_save_dir`` in the args
each episode's ``setup_demo`` receives, so this launcher wraps ``eval_policy``
and drops that key; everything else (task config, seeds, results file) is
unchanged. ``ROBOTWIN_EVAL_VIDEO=1`` keeps the videos.

Takes the same arguments as ``eval_policy_wrapper.py`` and needs
``benchmarks/robotwin`` on ``PYTHONPATH`` (``dsrl/data/robotwin/rollout.sh`` sets it).
"""

import os

import eval_policy_wrapper as wrapper


def _without_videos(module) -> None:
    orig_eval_policy = module.eval_policy

    def eval_policy(task_name, task_env, args, *rest, **kwargs):
        args.pop("eval_video_save_dir", None)
        return orig_eval_policy(task_name, task_env, args, *rest, **kwargs)

    module.eval_policy = eval_policy


def _bootstrap_without_videos(bootstrap):
    def wrapped(*args, **kwargs):
        module = bootstrap(*args, **kwargs)
        _without_videos(module)
        print("[dsrl eval_wrapper] per-episode eval videos disabled (ROBOTWIN_EVAL_VIDEO=1 keeps them)")
        return module

    return wrapped


if __name__ == "__main__":
    if os.environ.get("ROBOTWIN_EVAL_VIDEO", "0") != "1":
        wrapper.bootstrap_robotwin_module = _bootstrap_without_videos(wrapper.bootstrap_robotwin_module)
    raise SystemExit(wrapper.main())
