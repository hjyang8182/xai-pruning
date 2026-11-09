"""Plot concept_influence.py: per model, the share of the baseline's influence carried by GCG's kept
concepts (contribution, flipped predictions, explanation slots) against the share of concepts they
are, which is what a random set of the same size would carry. Styled like plot_concept_reduction.py.

    python plot_concept_influence.py --json ../extra_results/data/concept_influence/concept_influence_<stamp>.json

Several --json files (e.g. one per dataset, each holding the same model) give one panel per file, titled
by dataset:

    python plot_concept_influence.py --models dncbm --json data/concept_influence/concept_influence_{cifar100,cub,places365}_*.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GCG = "#2a78d6"
BASE = "#9d9c97"
BASE_TXT = "#55544f"
INK = "#000000"
MODEL_NAMES = {"dncbm": "DN-CBM", "lfcbm": "LF-CBM", "vlgcbm": "VLG-CBM", "ucbm": "UCBM"}
DATASET_NAMES = {"cifar100": "CIFAR-100", "cub": "CUB", "places365": "Places365", "food101": "Food-101",
                 "oxford_pet": "Oxford-IIIT Pet", "imagenet": "ImageNet"}
ROWS = [("contribution", "Total |contribution|"),
        ("flip_rate", "Predictions flipped\nby removing it"),
        ("explain_rate", "Appearances in\nexplanations (NCC)")]


def load_panels(paths, models):
    """[(title, result)] in --json then --models order. One file: a panel per model. Several files:
    a panel per (file, model), titled by the dataset when the files are one model on several datasets."""
    panels = []
    for path in paths:
        with open(path) as f:
            res = json.load(f)["models"]
        for m in models:
            if m in res:
                panels.append((m, res[m]))
    by_dataset = len(paths) > 1 and len({m for m, _ in panels}) == 1
    return [(DATASET_NAMES.get(r.get("dataset"), r.get("dataset")) if by_dataset else MODEL_NAMES.get(m, m), r)
            for m, r in panels]


def main(args):
    panels = load_panels(args.json, args.models)
    fs = dict(title=16 * args.font_scale, panel=15 * args.font_scale, sub=12 * args.font_scale,
              name=13 * args.font_scale, value=12 * args.font_scale, tick=12 * args.font_scale)
    fig, axes = plt.subplots(1, len(panels), figsize=(args.width, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(ROWS))[::-1]
    for ax, (title, r) in zip(axes, panels):
        s = r["summary"]
        size_share = r["n_kept"] / r["n_concepts"]
        shares = [100 * s[k]["kept"]["share"]["mean"] for k, _ in ROWS]
        for yy in y[::2]:
            ax.axhspan(yy - 0.5, yy + 0.5, color="#f7f7f9", zorder=0)
        ax.barh(y, shares, height=0.6, color=GCG, zorder=2)
        ax.axvline(100 * size_share, color=BASE_TXT, linestyle="--", linewidth=1.2, zorder=3)
        for yy, v in zip(y, shares):
            ax.text(v + 2, yy, f"{v:.0f}%", va="center", ha="left", fontsize=fs["value"], color=INK, zorder=4)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(ROWS) - 0.5)
        ax.set_title(f"{title}\n", fontsize=fs["panel"], color=INK, loc="left", fontweight="bold")
        ax.text(0, 1.0, f"Keeps {r['n_kept']:,} of {r['n_concepts']:,} concepts ({100 * size_share:.0f}%, dashed)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=fs["sub"], color=BASE_TXT)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
        ax.tick_params(axis="x", colors=INK, length=0, labelsize=fs["tick"])
        ax.tick_params(axis="y", length=0, pad=4)
        ax.grid(axis="x", color="#eeede9", linewidth=0.8, zorder=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d6d5d0")
    axes[0].set_yticks(y, [label for _, label in ROWS], fontsize=fs["name"], color=INK)
    fig.supxlabel("Share of the baseline's total carried by GCG's kept concepts  "
                  "(dashed: share expected from a random set of the same size)", fontsize=fs["sub"], color=INK)
    fig.suptitle(args.title, fontsize=fs["title"], color=INK, x=0.01, ha="left")
    fig.tight_layout(w_pad=2.5)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    root, _ = os.path.splitext(args.out)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(root + ext, bbox_inches="tight", dpi=250)
    print(f"Saved {root}.{{pdf,png,svg}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["dncbm", "vlgcbm", "lfcbm"])
    p.add_argument("--title", default="On CUB, the concepts GCG keeps carry far more of the baseline's "
                                      "influence than their number suggests")
    p.add_argument("--width", type=float, default=14.0)
    p.add_argument("--font-scale", type=float, default=1.0)
    p.add_argument("--out", default="img/concept_influence_cub.pdf")
    main(p.parse_args())
