#!/bin/bash
# Watchdog for the RoboTwin simulator client, run by the job scripts.
#
# The client can hang for good inside SAPIEN's ray-tracing denoiser (OIDN on CUDA:
# cuStreamSynchronize never returns, the process spins at 100% CPU and prints
# nothing). This script runs the client command in its own process group, tees its
# output to a per-attempt log and, when that log has not grown for
# SIM_STALL_TIMEOUT seconds, kills the group and starts the command again. The
# policy server keeps running and sees a new client connect; every episode starts
# with a reset, so no state carries over from the episode that hung.
#
# A restart resumes RoboTwin's seed loop after the last finished episode, read from
# its "Success rate: S/N ..., current seed: X" lines: ROBOTWIN_START_SEED=X+1 (see
# dsrl/data/robotwin/eval_wrapper.py) and the remaining episode count. The episode
# that hung is rerun on its own seed, so the evaluated seeds are the ones an
# uninterrupted run would use. Only stalls are restarted: a client that exits
# (finished or failed) ends the watchdog with its exit status. SIGTERM / SIGINT stop
# the client and the watchdog.
#
# Usage (from a job script, in the foreground or with &):
#   bash jobs/sim_watchdog.sh <log prefix> <episodes> bash dsrl/data/robotwin/rollout.sh <args...>
# Logs go to <log prefix>_client_try<N>.log; the output also goes to stdout. The
# command must read NUM_EPISODES and ROBOTWIN_START_SEED, as rollout.sh does.
# Env:
#   SIM_STALL_TIMEOUT  seconds without client output that count as a hang (default 600;
#                      an episode prints every step, about 1 s apart; the longest normal silence,
#                      start-up or expert checks between episodes, was about 2 min)
#   SIM_MAX_RESTARTS   restarts before giving up with status 124 (default 5)
#   SIM_POLL_INTERVAL  seconds between checks (default 30)

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash jobs/sim_watchdog.sh <log prefix> <episodes> <command> [args...]" >&2
    exit 1
fi

prefix="$1"
episodes="$2"
shift 2
stall="${SIM_STALL_TIMEOUT:-600}"
max_restarts="${SIM_MAX_RESTARTS:-5}"
poll="${SIM_POLL_INTERVAL:-30}"

attempt=0
done_eps=0
successes=0
start_seed=""
pid=""
pgid_file="$(mktemp "${TMPDIR:-/tmp}/sim_watchdog_pgid.XXXXXX")"

kill_group() {
    [[ -n "${pid}" ]] || return 0
    local pgid i
    pgid="$(cat "${pgid_file}" 2>/dev/null || true)"
    [[ -n "${pgid}" ]] || return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || return 0
    for i in {1..15}; do
        kill -0 -- "-${pgid}" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-${pgid}" 2>/dev/null || true
}
trap 'kill_group; exit 143' TERM INT
trap 'kill_group; rm -f "${pgid_file}"' EXIT

while true; do
    log="${prefix}_client_try${attempt}.log"
    echo "[WATCHDOG] attempt ${attempt}: $((episodes - done_eps)) episodes from seed ${start_seed:-<default>}, log ${log}"
    # setsid: the client (and its tee) get their own process group, which kill_group stops as a
    # whole. Its id is written to pgid_file, since setsid forks when the caller already leads a
    # group (job control on); -w keeps ${pid} alive and carrying the exit status in that case.
    : > "${pgid_file}"
    NUM_EPISODES="$((episodes - done_eps))" ROBOTWIN_START_SEED="${start_seed}" \
        setsid -w bash -c 'echo $$ > "$0"; log="$1"; shift; set -o pipefail; "$@" 2>&1 | tee "${log}"' \
        "${pgid_file}" "${log}" "$@" &
    pid=$!

    stalled=0
    while kill -0 "${pid}" 2>/dev/null; do
        # Backgrounded so the TERM trap runs without waiting out the sleep.
        sleep "${poll}" &
        wait $! || true
        kill -0 "${pid}" 2>/dev/null || break
        age=$(( $(date +%s) - $(stat -c %Y "${log}" 2>/dev/null || date +%s) ))
        if (( age > stall )); then
            stalled=1
            echo "[WATCHDOG] no client output for ${age}s, simulator hung: killing it"
            kill_group
            break
        fi
    done
    wait "${pid}" && status=0 || status=$?
    pid=""

    # RoboTwin's counts start at zero in every attempt; its last line covers the whole attempt.
    last="$(sed 's/\x1b\[[0-9;]*m//g' "${log}" | grep -a -o 'Success rate: [0-9]*/[0-9]*.*current seed: [0-9]*' | tail -n 1 || true)"
    if [[ "${last}" =~ Success\ rate:\ ([0-9]+)/([0-9]+).*current\ seed:\ ([0-9]+) ]]; then
        successes=$((successes + BASH_REMATCH[1]))
        done_eps=$((done_eps + BASH_REMATCH[2]))
        start_seed=$((BASH_REMATCH[3] + 1))
    fi
    echo "[WATCHDOG] total so far: ${successes}/${done_eps} episodes succeeded, ${attempt} restart(s)"

    if (( ! stalled )); then
        echo "[WATCHDOG] client exited with status ${status}"
        exit "${status}"
    fi
    if (( done_eps >= episodes )); then
        echo "[WATCHDOG] all ${episodes} episodes done before the hang"
        exit 0
    fi
    attempt=$((attempt + 1))
    if (( attempt > max_restarts )); then
        echo "[WATCHDOG] giving up after ${max_restarts} restarts" >&2
        exit 124
    fi
done
