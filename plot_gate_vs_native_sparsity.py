"""Learned concept gate vs. each pipeline's own sparsity mechanism, for LF-CBM / UCBM / VLG-CBM.

The counterpart of plot_gate_vs_sparse_grid.py (DN-CBM: gate vs. L1 probe) for the other three
pipelines. Reads, per (model, dataset), the newest
  - <dataset>_gate_sweep_<stamp>.json   (train_cbm_seed_sweep.py --gated with a lambda_gate list), and
  - <dataset>_sparse_sweep_<stamp>.json (the baseline arm with a sparsity list: LF-CBM --lams,
    VLG-CBM --saga_lams -- GLM-SAGA elastic net -- and UCBM --lam_ws, final-layer weight elastic net)
written via `run_seed_sweeps.py ... --lambda-gates ...` / `... -a baseline --sparse-sweep`.

x is the number of concepts the model can use at all (log): open gates for the gate, and concepts
with any non-zero class weight for the sparse layer (UCBM: max |w| > 1e-3, since its Adam-trained
elastic net never zeroes a weight exactly -- the same threshold as DN-CBM's sweep_lambda_sparse.py).
y is test accuracy, mean +/- std over seeds.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
import numpy as np

from panel_style import use_paper_font
from plot_gate_sweep_vs_random_subset import DATASET_NAMES, GRID, INK, MODEL_NAMES, MUTED, ROOT, SWEEP_GLOBS

# Reference categorical palette slots 1-3 in fixed order, plus marker + line style per series.
GATE_SERIES = [
    # key in gate-sweep point, label, color, marker, linestyle
    ("gated", "Learned gate", "#2a78d6", "o", "-"),
    ("gated_refit", "Learned gate, refit", "#1baf7a", "^", ":"),
]
SPARSE_STYLE = ("Native sparse layer", "#eb6834", "s", "--")
SPARSE_ARM = {"lfcbm": "baseline", "ucbm": "baseline", "vlgcbm": "sparse"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(SWEEP_GLOBS), choices=list(SWEEP_GLOBS))
    p.add_argument("--datasets", nargs="+", default=["cifar100", "cub", "places365"])
    p.add_argument("--json", nargs=4, action="append", default=[], metavar=("KIND", "MODEL", "DATASET", "PATH"),
                   help="use PATH as the KIND (gate|sparse) sweep for (MODEL, DATASET) instead of the newest. Repeatable.")
    p.add_argument("--out", default=os.path.join(ROOT, "img", "gate_vs_native_sparsity.png"))
    return p.parse_args()


def newest(kind, model, dataset, overrides):
    if (kind, model, dataset) in overrides:
        return overrides[(kind, model, dataset)]
    pattern = SWEEP_GLOBS[model].replace("_gate_sweep_", f"_{kind}_sweep_").format(d=dataset)
    paths = glob.glob(os.path.join(ROOT, pattern))
    return max(paths, key=os.path.getmtime) if paths else None


def curve(points, key, count):
    """(concept count, acc mean, acc std) per sweep point for one arm, sorted by count."""
    rows = [(p[key][count], p[key]["acc_mean"], p[key]["acc_std"])
            for p in points if key in p and p[key].get(count, 0) > 0]
    rows.sort()
    return [np.array(c) for c in zip(*rows)] if rows else None


def draw(ax, xy, label, color, marker, ls):
    k, mean, std = xy
    ax.fill_between(k, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.plot(k, mean, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=6,
            markeredgecolor="white", markeredgewidth=1.2, label=label)


def main(args):
    use_paper_font(plt)
    overrides = {(k, m, d): p for k, m, d, p in args.json}
    nr, nc = len(args.models), len(args.datasets)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.9 * nr), squeeze=False)

    for i, model in enumerate(args.models):
        for j, dataset in enumerate(args.datasets):
            ax = axes[i][j]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8, labelcolor=INK)
            ax.grid(True, axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel(f"{MODEL_NAMES[model]}\ntest accuracy", fontsize=9, color=INK)
            if i == nr - 1:
                ax.set_xlabel("concepts used (log)", fontsize=9, color=INK)

            gate_path = newest("gate", model, dataset, overrides)
            sparse_path = newest("sparse", model, dataset, overrides)
            drawn, total = False, None
            if gate_path:
                with open(gate_path) as f:
                    gate = json.load(f)
                total = gate.get("n_concepts_total")
                print(f"{model}/{dataset} gate:   {os.path.relpath(gate_path, ROOT)}")
                for key, label, color, marker, ls in GATE_SERIES:
                    xy = curve(gate["points"], key, "n_open_mean")
                    if xy is not None:
                        draw(ax, xy, label, color, marker, ls); drawn = True
            if sparse_path:
                with open(sparse_path) as f:
                    sparse = json.load(f)
                total = total or sparse.get("n_concepts_total")
                print(f"{model}/{dataset} sparse: {os.path.relpath(sparse_path, ROOT)}")
                arm = SPARSE_ARM[model]
                count = "n_used_mean" if any("n_used_mean" in p.get(arm, {}) for p in sparse["points"]) else "n_open_mean"
                xy = curve(sparse["points"], arm, count)
                if xy is not None:
                    draw(ax, xy, *SPARSE_STYLE); drawn = True

            if not drawn:
                ax.text(0.5, 0.5, "no sweeps yet", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for side in ("left", "bottom"):
                    ax.spines[side].set_visible(False)
                continue
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if total:
                ax.text(0.98, 0.04, f"dictionary: {total}", transform=ax.transAxes, ha="right", va="bottom",
                        color=MUTED, fontsize=7.5)

    styles = [(c, m, ls) for _, _, c, m, ls in GATE_SERIES] + [SPARSE_STYLE[1:]]
    labels = [s[1] for s in GATE_SERIES] + [SPARSE_STYLE[0]]
    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=2, marker=m, markersize=6,
                          markeredgecolor="white") for c, m, ls in styles]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main(parse_args())
