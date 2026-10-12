"""Gated concept selection vs. a matched random subset, across lambda_gate, for LF-CBM / UCBM / VLG-CBM.

Reads the <dataset>_gate_sweep_<stamp>.json that each pipeline's train_cbm_seed_sweep.py writes when
run with --lambda_gates/--gate_lams and --random_control (launch them all with
`run_seed_sweeps.py -m lfcbm ucbm vlgcbm -a gated --random-control --lambda-gates ...`). At every
lambda_gate, K = gates > refit_tau, and two fresh linear heads are trained with the identical
cold-start recipe (gating.matched_random_refits):

  - gate-selected : the concepts the gate kept
  - random        : the same number |K| of concepts drawn uniformly at random (per seed)

so the gap between them is purely about *which* concepts the gate picked. The gated model as
trained (no refit) is drawn too, for reference. One panel per (model, dataset); x is |K| on a log
axis, y is test accuracy (mean +/- std over seeds).
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

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_GLOBS = {
    "lfcbm": "prelim/Label-free-CBM/seed_sweep_results/{d}_gate_sweep_*.json",
    "ucbm": "prelim/ucbm/save/RESULTS/{d}-*/classifier/*/seed_sweep_results/{d}_gate_sweep_*.json",
    "vlgcbm": "prelim/VLG-CBM/saved_models/{d}/seed_sweep_results/{d}_gate_sweep_*.json",
}
MODEL_NAMES = {"lfcbm": "LF-CBM", "ucbm": "UCBM", "vlgcbm": "VLG-CBM"}
DATASET_NAMES = {"cifar100": "CIFAR-100", "cub": "CUB", "places365": "Places365"}

# Reference categorical palette, slots 1-3 in fixed order; each series also has its own
# marker + line style so identity never rests on hue alone.
SERIES = [
    # key in sweep point, label, color, marker, linestyle
    ("gated_refit_cold", "Gate-selected K (refit)", "#2a78d6", "o", "-"),
    ("random_refit", "Random |K| concepts (refit)", "#eb6834", "s", "--"),
    ("gated", "Gated model, no refit", "#1baf7a", "^", ":"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(SWEEP_GLOBS), choices=list(SWEEP_GLOBS))
    p.add_argument("--datasets", nargs="+", default=["cifar100", "cub", "places365"])
    p.add_argument("--sweep-json", nargs=3, action="append", default=[], metavar=("MODEL", "DATASET", "PATH"),
                   help="use this sweep json for (MODEL, DATASET) instead of the newest one. Repeatable.")
    p.add_argument("--out", default=os.path.join(ROOT, "img", "gate_sweep_vs_random_subset.png"))
    return p.parse_args()


def find_sweep(model, dataset, overrides):
    if (model, dataset) in overrides:
        return overrides[(model, dataset)]
    paths = glob.glob(os.path.join(ROOT, SWEEP_GLOBS[model].format(d=dataset)))
    return max(paths, key=os.path.getmtime) if paths else None


def series_xy(points, key):
    """(K, acc_mean, acc_std) per lambda_gate for one arm, sorted by K. K is the matched set size
    (taken from random_refit, which is drawn at exactly |gate > tau|)."""
    rows = []
    for p in points:
        if key not in p or "random_refit" not in p:
            continue
        rows.append((p["random_refit"]["n_open_mean"], p[key]["acc_mean"], p[key]["acc_std"]))
    rows = [r for r in rows if r[0] > 0]
    rows.sort()
    return [np.array(c) for c in zip(*rows)] if rows else None


def main(args):
    use_paper_font(plt)
    overrides = {(m, d): p for m, d, p in args.sweep_json}
    nr, nc = len(args.models), len(args.datasets)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.9 * nr), squeeze=False)

    for i, model in enumerate(args.models):
        for j, dataset in enumerate(args.datasets):
            ax = axes[i][j]
            ax.set_facecolor("white")
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
                ax.set_xlabel("concepts kept |K| (log)", fontsize=9, color=INK)

            path = find_sweep(model, dataset, overrides)
            if path is None:
                ax.text(0.5, 0.5, "no sweep yet", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for side in ("left", "bottom"):
                    ax.spines[side].set_visible(False)
                continue
            with open(path) as f:
                sweep = json.load(f)
            print(f"{model}/{dataset}: {os.path.relpath(path, ROOT)}")

            for key, label, color, marker, ls in SERIES:
                xy = series_xy(sweep["points"], key)
                if xy is None:
                    continue
                k, mean, std = xy
                ax.fill_between(k, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                ax.plot(k, mean, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=6,
                        markeredgecolor="white", markeredgewidth=1.2, label=label)
            ax.set_xscale("log")
            # plain integer ticks (1, 2, 5 x 10^n) instead of mixed 4x10^1-style mathtext labels
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            total = sweep.get("n_concepts_total")
            if total:
                ax.text(0.98, 0.04, f"dictionary: {total}", transform=ax.transAxes, ha="right", va="bottom",
                        color=MUTED, fontsize=7.5)

    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=2, marker=m, markersize=6,
                          markeredgecolor="white") for _, _, c, m, ls in SERIES]
    fig.legend(handles, [s[1] for s in SERIES], loc="lower center", ncol=len(SERIES), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main(parse_args())
