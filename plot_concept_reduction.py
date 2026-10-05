"""Per-model before/after: effective concept size as % of dictionary, baseline -> GCG.

One row per model (same layout as plot_effective_concept_size.py). Each row draws an
arrow from the ungated model (concepts with non-zero weight in at least one class) to
the GCG model (open gates after mask-and-refit), both as a share of that model's
dictionary so the rows share one axis; test accuracy is annotated at each end.

All numbers are 5-seed means on CIFAR-100, read from the seed-sweep JSONs listed in
SOURCES (edit those paths to point at newer runs). Styled to match
plot_effective_concept_size.py.

Usage:
    python plot_concept_reduction.py --out img/concept_reduction_cifar100.pdf
"""
import argparse
import json
import os
import statistics
import textwrap

from panel_style import fit_one_line, text_width_in, use_paper_font, wrap_to_width

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GCG = "#2a78d6"
BASE = "#9d9c97"
BASE_TXT = "#55544f"   # the grey of the dots is too light to read as text
INK = "#000000"
INK2 = "#000000"
# (label, baseline loader, gcg loader) -> each loader returns (n_concepts, test_acc_fraction, n_total).
# The gated side reads each pipeline's gated_sweep_final.json (the run a pipeline's seed sweep
# settles on) where one exists; LF-CBM has no such file yet, so it names its sweep json directly.
# Only the seed *means* are read -- the panel plots points, not spreads.
SOURCES = [
    ("DN-CBM",
     ("data/cifar100/local_explanation_size_20260920_020033.json", "n_union", "acc"),
     ("data/cifar100/model/gated_20260921_205510/probe_config.json",
      "refit_n_concepts_mean", "refit_accuracy_mean")),
    ("UCBM",
     ("prelim/ucbm/save/RESULTS/cifar100-resnet50_v2/classifier/concepts_1000_64/seed_sweep_results/"
      "cifar100_baseline_seed_sweep_2026_08_31_18_25.json", "baseline.n_open_mean", "baseline.acc_mean"),
     ("prelim/ucbm/save/RESULTS/cifar100-resnet50_v2/classifier/concepts_1000_64/seed_sweep_results/"
      "gated_sweep_final.json", "gated_refit.n_open_mean", "gated_refit.acc_mean")),
    ("LF-CBM",
     ("prelim/Label-free-CBM/seed_sweep_results/cifar100_seed_sweep_2026_09_02_09_22.json",
      "baseline.n_open_mean", "baseline.acc_mean"),
     ("prelim/Label-free-CBM/seed_sweep_results/cifar100_gated_seed_sweep_2026_09_20_01_47.json",
      "gated_refit.n_open_mean", "gated_refit.acc_mean")),
    ("VLG-CBM",
     ("prelim/VLG-CBM/saved_models/cifar100/seed_sweep_results/cifar100_dense_seed_sweep_2026_09_19_11_09.json",
      "dense.n_open_mean", "dense.acc_mean"),
     ("prelim/VLG-CBM/saved_models/cifar100/seed_sweep_results/gated_sweep_final.json",
      "gated_refit.n_open_mean", "gated_refit.acc_mean")),
]


def _get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    if isinstance(d, dict):            # e.g. open_gates_by_seed -> mean over seeds
        d = statistics.fmean(d.values())
    return float(d)


def _load(spec):
    path, n_key, acc_key = spec
    with open(path) as f:
        d = json.load(f)
    if "n_concepts_total" in d:
        total = _get(d, "n_concepts_total")
    else:                                  # DN-CBM probe_config.json nests it per seed
        total = next(iter(d["concept_usage_by_seed"].values()))["n_concepts_total"]
    return _get(d, n_key), 100 * _get(d, acc_key), float(total)


def _left_margin_in(labels, fontsize):
    """Inches to reserve for the y tick labels (model names) plus a little air."""
    return 0.10 + max(len(l) for l in labels) * fontsize * 0.0088


def plot(rows, out_path, title, width=5.9, font_scale=1.0,
         xlabel="Used concepts (% of dictionary)"):
    # Typography: one scale for the whole figure so `--font-scale` moves everything
    # together. Sizes are large relative to the width on purpose -- this panel is narrow
    # (it sits next to another one in a two-column figure), so the text has to stay
    # readable after the shrink to column width. Matches plot_effective_concept_size.py.
    FS_NAME  = 16 * font_scale   # model names (y tick labels)
    FS_AXIS  = 16 * font_scale   # x axis label
    FS_TICK  = 14 * font_scale   # x tick labels
    FS_VALUE = 16 * font_scale   # the accuracy annotation at each endpoint
    FS_LEG   = 15 * font_scale
    FS_TITLE = 16 * font_scale
    use_paper_font(plt)          # Spectral, or the closest serif available
    plt.rcParams.update({"font.size": FS_NAME})

    n = len(rows)

    # Title and legend are anchored to the *figure*, not the axes: left-aligned at the
    # axes they overhang the right edge, and a tight bounding box would then quietly
    # save the panel wider than `width`. Both are wrapped against the real figure width
    # further down, once there is a figure to measure text on; these are the starting
    # line counts for the header height.
    n_title_lines, legend_rows, n_xlabel_lines = 1, 1, 1

    # Each row is an arrow plus two endpoint annotations stacked above and below it,
    # so a row needs room for three lines of content.
    row_h  = 0.98 * font_scale
    legend_h = lambda: legend_rows * FS_LEG * 1.75 / 72 + 0.10

    def _foot_h():                       # x ticks, then the x label, then the legend
        return (0.34 * font_scale + n_xlabel_lines * FS_AXIS * 1.35 / 72 + legend_h())

    def _head_h():
        return n_title_lines * FS_TITLE * 1.35 / 72 + 0.12

    fig, ax = plt.subplots(figsize=(width, n * row_h + _head_h() + _foot_h()))

    # Everything below is measured against the space it really has. Text is not clipped,
    # so anything that overhangs would silently widen the saved panel past `width`.
    title_text = wrap_to_width(fig, title, FS_TITLE, width - 0.05) if title else ""
    title_lines = title_text.split("\n") if title_text else []
    n_title_lines = max(1, len(title_lines))
    legend_labels = ["Baseline", "With GCG"]
    # The legend sits under the x label, in one row -- this panel is wide, so it should
    # spend horizontal rather than vertical space.
    legend_ncol = 2 if text_width_in(
        fig, "   ".join(legend_labels), FS_LEG) + 1.2 <= width else 1
    legend_rows = 1 if legend_ncol == 2 else 2
    # The x label is centred on the *figure* and kept on one line: the axes is stretched
    # below to cover it instead of the label being wrapped.
    xlabel_text = xlabel
    n_xlabel_lines = 1
    fig.set_figheight(n * row_h + _head_h() + _foot_h())

    # What each dot means is carried by two things: its position on the axis (how much
    # of the dictionary the model uses) and its label (what that costs in task accuracy).
    # The GCG label also carries the change from the baseline, which is the comparison
    # the panel is actually making.
    axes_w_in = width * 0.995 - _left_margin_in([r[0] for r in rows], FS_NAME)
    # A dot can sit at 100% (a dense baseline uses the whole dictionary), so the axis runs
    # a little past both ends -- otherwise that marker is sliced in half by the spine.
    XPAD = 2.5
    xspan = 100 + 2 * XPAD

    ys = list(range(n))[::-1]  # first row on top

    for y, (label, (nb, ab, tb), (ng, ag, tg)) in zip(ys, rows):
        pb, pg = 100 * nb / tb, 100 * ng / tg

        ax.annotate(
            "",
            xy=(pg, y),
            xytext=(pb, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=GCG,
                linewidth=2,
                mutation_scale=14,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )

        ax.scatter([pb], [y], s=55, color=BASE, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([pg], [y], s=55, color=GCG, edgecolor="white", linewidth=1.2, zorder=3)

        # Baseline reads above its dot, GCG below its dot, each coloured like the dot it
        # belongs to (with the labels stacked rather than beside the dots, colour is what
        # says which endpoint a number belongs to). The label is centred on its dot but
        # slid back inside the axes when that would push it off the panel -- a label is
        # nearly as wide as the whole axis here, so near either end it has nowhere to go.
        def _label(px, text, dy, color):
            w = xspan * text_width_in(fig, text, FS_VALUE) / axes_w_in
            x = min(max(px - w / 2, -XPAD), max(-XPAD, 100 + XPAD - w))
            ax.annotate(
                text, (x, y),
                xytext=(0, dy), textcoords="offset points",
                ha="left", va="bottom" if dy > 0 else "top",
                fontsize=FS_VALUE, color=color,
            )

        pad = 9 * font_scale
        _label(pb, f"{ab:.1f}% accuracy", +pad, BASE_TXT)
        _label(pg, f"{ag:.1f}% accuracy ({ag - ab:+.1f}%)", -pad, GCG)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_NAME, color=INK)

    ax.set_ylim(-0.62, n - 0.38)
    ax.set_xlim(-XPAD, 100 + XPAD)

    # Labelled ticks every 20% with a gridline every 10%: the whole point of a dot's x
    # position is how far that model sits from a full dictionary, so the axis should be
    # easy to read off, but 11 labelled ticks collide even at this width.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
    ax.grid(axis="x", which="minor", color="#f4f3f0", linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # x label above the legend, legend at the very bottom of the figure.
    fig.supxlabel(xlabel_text, fontsize=FS_AXIS, color=INK2,
                  y=(legend_h() - 0.02) / fig.get_figheight())

    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=2)

    ax.grid(axis="x", color="#eeede9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", color=BASE, linestyle="none", markersize=10,
               label=legend_labels[0]),
        Line2D([], [], marker="o", color=GCG, linestyle="none", markersize=10,
               label=legend_labels[1]),
    ]

    fig_h_in = fig.get_figheight()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=legend_ncol,
        fontsize=FS_LEG,
        frameon=False,
        handlelength=1.0,
        labelspacing=0.35,
        columnspacing=1.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    if title_lines:
        fig.text(0.005, 0.995, "\n".join(title_lines),
                 fontsize=FS_TITLE, color=INK, ha="left", va="top")

    # tight_layout would fight the figure-level legend/title and can collapse the axes
    # at this width, so the margins are set explicitly from the font metrics above.
    fig.subplots_adjust(
        left=_left_margin_in([r[0] for r in rows], FS_NAME) / width,
        right=0.995,
        top=1 - _head_h() / fig_h_in,
        bottom=_foot_h() / fig_h_in,
    )

    root, _ = os.path.splitext(out_path)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 250}), (".svg", {})):
        fig.savefig(root + ext, bbox_inches="tight", **kw)
    plt.close(fig)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="GCG shrinks the concept set a model uses")
    p.add_argument("--width", type=float, default=5.9,
                   help="figure width in inches (default 5.9; the panel reads best wide)")
    p.add_argument("--xlabel", default="Used concepts (% of dictionary)",
                   help="x axis label; kept on one line (the axes is stretched to cover it)")
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="multiplies every font size (and the figure height with them)")
    args = p.parse_args()
    rows = []
    for label, base_spec, gcg_spec in SOURCES:
        b, g = _load(base_spec), _load(gcg_spec)
        rows.append((label, b, g))
        print(f"{label:8s} baseline {b[0]:>7,.0f}/{b[2]:.0f} ({100 * b[0] / b[2]:.0f}%) {b[1]:.2f}%  ->  "
              f"GCG {g[0]:>6,.0f}/{g[2]:.0f} ({100 * g[0] / g[2]:.0f}%) {g[1]:.2f}%  ({g[1] - b[1]:+.1f} pt)")
    root = plot(rows, args.out, args.title, width=args.width, font_scale=args.font_scale,
                xlabel=args.xlabel)
    print(f"saved {root}.pdf / {root}.png / {root}.svg")


if __name__ == "__main__":
    main()
