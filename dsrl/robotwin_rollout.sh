#!/bin/bash
# Run RoboTwin episodes against the DSRL server (dsrl/robotwin_server.py).
#
# Same setup as benchmarks/robotwin/single_eval.sh, but RoboTwin loads the DSRL
# client (dsrl/robotwin_client.py), which also reports each episode's end and
# success to the server, and it runs NUM_EPISODES episodes. RoboTwin itself is
# used unchanged.
#
# Usage: bash dsrl/robotwin_rollout.sh <task_name> <task_config> <gpu_id> [port] [host]
# Env:   ROBOTWIN_PATH, ROBOTWIN_PYTHON (required, as for single_eval.sh)
#        NUM_EPISODES   episodes to run (default 100000, i.e. until stopped)
#        SEED           RoboTwin seed index; episodes use seeds from 100000 * (1 + SEED).
#                       Default 1 keeps training seeds apart from the evaluation seeds (SEED=0).

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash dsrl/robotwin_rollout.sh <task_name> <task_config> <gpu_id> [port] [host]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${REPO_ROOT}/benchmarks/robotwin"
ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:?ROBOTWIN_PYTHON must point to the RoboTwin env python}"

task_name="$1"
task_config="$2"
gpu_id="$3"
port="${4:-${ROBOTWIN_PORT:-8848}}"
host="${5:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
seed="${SEED:-1}"

runtime_config="$(mktemp "${TMPDIR:-/tmp}/dsrl_policy_config.XXXXXX.yml")"
trap 'rm -f "${runtime_config}"' EXIT
sed \
    -e "s/^host:.*/host: \"${host}\"/" \
    -e "s/^port:.*/port: ${port}/" \
    "${BENCH_DIR}/policy_config.yml" > "${runtime_config}"

# Same SAPIEN EGL setup as single_eval.sh.
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -z "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]]; then
    egl_json="$(dirname "$(dirname "${ROBOTWIN_PYTHON}")")/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    [[ -f "${egl_json}" ]] && export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_json}"
fi

export ROBOTWIN_PATH
export ROBOTWIN_TEST_NUM="${NUM_EPISODES:-100000}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${ROBOTWIN_PATH}:${BENCH_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

cd "${ROBOTWIN_PATH}"
echo "task         : ${task_name} (${task_config}), episodes=${ROBOTWIN_TEST_NUM}, seed=${seed}"
echo "server       : ws://${host}:${port}"

PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
"${ROBOTWIN_PYTHON}" "${BENCH_DIR}/eval_policy_wrapper.py" \
    --config    "${runtime_config}" \
    --overrides \
    --task_name        "${task_name}" \
    --task_config      "${task_config}" \
    --ckpt_setting     dsrl \
    --seed             "${seed}" \
    --policy_name      dsrl.robotwin_client
