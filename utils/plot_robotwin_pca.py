"""Plot the 2D PCA of RoboTwin action priors, per generated chunk.

Reads the output of jobs/robotwin_pca.job:
    <pca_dir>/run_<g>/chunk_<k>.npz   prior (32, 80) + seed
    <pca_dir>/run_<g>/result.txt      success=0|1, status=<exit code>

For each of the first --num-chunks chunks k, the priors of every finished rerun
that reached chunk k are projected with utils.pca.pca_2d, saved to
<out_dir>/pca_chunk_<k>.npz (run ids, 2D components, success), and plotted
twice: plain, and colored by whether the whole trajectory succeeded.

Usage (from the repo root):
    python -m utils.plot_robotwin_pca outputs/robotwin_pca/<array job id>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.pca import pca_2d  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#2a78d6"
SUCCESS = "#0ca30c"
FAILURE = "#d03b3b"


def load_runs(pca_dir: Path) -> dict:
    """Return {run id: success (bool)} for every rerun that finished."""
    runs = {}
    for run_dir in sorted(pca_dir.glob("run_*")):
        result = run_dir / "result.txt"
        if not result.is_file():
            continue
        fields = dict(line.split("=", 1) for line in result.read_text().split())
        if fields.get("status") != "0" or fields.get("success", "") == "":
            continue
        runs[int(run_dir.name.split("_")[1])] = int(fields["success"]) > 0
    return runs


def style_axes(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("PC 1", color=INK_SECONDARY)
    ax.set_ylabel("PC 2", color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def scatter(ax, xy: np.ndarray, color: str, marker: str = "o", label=None) -> None:
    # >= 8px markers with a surface-colored ring so overlapping dots stay legible.
    ax.scatter(xy[:, 0], xy[:, 1], s=64, c=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pca_dir", type=Path, help="outputs/robotwin_pca/<array job id>")
    parser.add_argument("--num-chunks", type=int, default=5, help="first K chunks to plot (default 5)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <pca_dir>/pca")
    args = parser.parse_args()

    out_dir = args.out_dir or args.pca_dir / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.pca_dir)
    print(f"{len(runs)} finished reruns ({sum(runs.values())} successful) in {args.pca_dir}")

    for k in range(args.num_chunks):
        ids = [g for g in runs if (args.pca_dir / f"run_{g:05d}" / f"chunk_{k:03d}.npz").is_file()]
        if len(ids) < 2:
            print(f"chunk {k}: {len(ids)} reruns reached this chunk, need >= 2 for PCA; skipping")
            continue
        priors = np.stack([np.load(args.pca_dir / f"run_{g:05d}" / f"chunk_{k:03d}.npz")["prior"] for g in ids])
        success = np.array([runs[g] for g in ids])
        comps = pca_2d(priors)
        np.savez(out_dir / f"pca_chunk_{k}.npz", runs=np.array(ids), components=comps, success=success)

        n_succ, n = int(success.sum()), len(ids)
        title = f"Action prior, chunk {k}: first 2 principal components ({n} reruns)"

        fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
        style_axes(ax, title)
        scatter(ax, comps, NEUTRAL)
        fig.tight_layout()
        fig.savefig(out_dir / f"pca_chunk_{k}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
        style_axes(ax, title)
        # Shape as well as color, so success is never encoded by color alone.
        scatter(ax, comps[~success], FAILURE, marker="X", label=f"Failure ({n - n_succ})")
        scatter(ax, comps[success], SUCCESS, marker="o", label=f"Success ({n_succ})")
        legend = ax.legend(frameon=False, loc="best", fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
        fig.tight_layout()
        fig.savefig(out_dir / f"pca_chunk_{k}_success.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

        print(f"chunk {k}: {n} reruns, {n_succ} successful -> {out_dir}/pca_chunk_{k}[_success].png")


if __name__ == "__main__":
    main()
