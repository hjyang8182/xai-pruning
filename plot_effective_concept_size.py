"""Left panel of the motivation figure: effective concept size of existing CBMs.

For each method, the number of concepts that appear (non-zero weight) in at least
one class's prediction — the global union — against the size of its concept
dictionary. One dataset per figure; reads the local_explanation_size_*.json files
written by each substrate's `concept_retention.py --analysis local`.

Usage:
    python plot_effective_concept_size.py --out img/effective_concept_size_cifar100.pdf \\
        "LF-CBM=prelim/Label-free-CBM/seed_sweep_results/cifar100_local_explanation_size_20260919_164207.json" \\
        "VLG-CBM=prelim/VLG-CBM/saved_models/cifar100/cifar100_cbm_baseline_2026_08_31_10_56_11/local_explanation_size_20260919_165259.json" \\
        "UCBM=prelim/ucbm/save/RESULTS/cifar100-resnet50_v2/classifier/concepts_1000_64/concept_retention_results/cifar100_local_explanation_size_20260919_164207.json" \\
        "DN-CBM=data/cifar100/local_explanation_size_20260920_020033.json"

Each positional arg is <label>=<json path or glob> (latest match is used).
"""

import argparse
import glob
import json
import os
import textwrap

from panel_style import fit_one_line, text_width_in, use_paper_font, wrap_to_width

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


USED = "#2a78d6"        # concepts used
DICT = "#e6e5e1"        # dictionary size, ghost bar
INK = "#000000"
INK2 = "#000000"
BUDGET = "#c0654b"


def _load(spec):
    label, pattern = spec.split("=", 1)

    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"nothing matched {pattern}")

    with open(paths[-1]) as f:
        r = json.load(f)

    return (
        label,
        int(round(r["n_union"])),
        int(r["n_concepts_total"]),
        paths[-1],
    )


def _left_margin_in(labels, fontsize):
    """Inches to reserve for the y tick labels (method names) plus a little air."""
    return 0.10 + max(len(l) for l in labels) * fontsize * 0.0088


def plot(rows, out_path, title, budget, width=3.6, font_scale=1.0, xlabel="Used concepts"):
    rows = sorted(rows, key=lambda r: r[1])

    labels = [r[0] for r in rows]
    used = [r[1] for r in rows]
    total = [r[2] for r in rows]

    n = len(rows)
    y = list(range(n))

    # Typography: one scale for the whole figure so `--font-scale` moves everything
    # together. The sizes are large relative to the figure width on purpose -- this
    # panel is narrow (it sits next to another one in a two-column figure), so the
    # text has to stay readable after the shrink to column width.
    FS_NAME  = 16 * font_scale   # method names (y tick labels)
    FS_AXIS  = 16 * font_scale   # x axis label
    FS_TICK  = 14 * font_scale   # x tick labels
    FS_VALUE = 14 * font_scale   # "used / total (pct)" per bar
    FS_LEG   = 15 * font_scale
    FS_TITLE = 16 * font_scale
    FS_NOTE  = 13 * font_scale   # concept-budget annotation
    use_paper_font(plt)          # Spectral, or the closest serif available
    plt.rcParams.update({"font.size": FS_NAME})

    # Each row is one bar plus the count annotation stacked above it. At this width the
    # annotation cannot sit *after* the bar (it is wider than the whole panel), so it
    # goes above, where the only cost is vertical space -- which is cheap here.
    # Title and legend are anchored to the *figure*, not the axes: left-aligned at the
    # axes they overhang the right edge, and a tight bounding box would then quietly
    # save the panel wider than `width`. Both are wrapped against the real figure width
    # further down, once there is a figure to measure text on; these are the starting
    # line counts for the header height.
    n_title_lines, legend_rows, n_xlabel_lines = 1, 1, 1

    row_h  = 0.60 * font_scale                 # inches per row (bar + its annotation)
    legend_h = lambda: legend_rows * FS_LEG * 1.75 / 72 + 0.10

    def _foot_h():                       # x ticks, then the x label, then the legend
        return (0.34 * font_scale + n_xlabel_lines * FS_AXIS * 1.35 / 72 + legend_h())

    def _head_h():
        return n_title_lines * FS_TITLE * 1.35 / 72 + 0.12

    fig, ax = plt.subplots(figsize=(width, n * row_h + _head_h() + _foot_h()))

    # Everything below is measured against the space it really has: the title and legend
    # get the full figure width, the bar annotations get what is left of it after the
    # method names. Text is not clipped, so anything that overhangs would silently widen
    # the saved panel past `width`.
    title_text = wrap_to_width(fig, title, FS_TITLE, width - 0.05) if title else ""
    title_lines = title_text.split("\n") if title_text else []
    n_title_lines = max(1, len(title_lines))
    legend_labels = ["Used for at least one sample", "Concept dictionary"]
    # The legend sits under the x label. One row if the two entries fit side by side on
    # the figure, two rows otherwise.
    legend_ncol = 2 if text_width_in(
        fig, "   ".join(legend_labels), FS_LEG) + 0.9 <= width else 1
    legend_rows = 1 if legend_ncol == 2 else 2
    # The x label is centred on the *figure* rather than the axes, and kept on one line:
    # the axes is stretched below to cover it instead of the label being wrapped.
    xlabel_text = xlabel
    n_xlabel_lines = 1
    fig.set_figheight(n * row_h + _head_h() + _foot_h())

    # Dictionary size -- background bar; effective concept size -- foreground bar.
    # Both sit in the lower half of their row so the annotation has room above.
    ax.barh(y, total, height=0.40, color=DICT, edgecolor="none", zorder=1)
    ax.barh(y, used, height=0.26, color=USED, edgecolor="none", zorder=2)

    xmax = max(total)

    # The annotation starts at x=0, so it has the axes width to live in. If the full
    # "used / total (pct)" does not fit at this width and font size, drop the total --
    # the ghost bar already shows it -- rather than overhang the panel. The format is
    # chosen once for the whole panel (the widest row decides), so the rows stay
    # comparable instead of some carrying the total and some not.
    label_room = width * 0.995 - _left_margin_in(labels, FS_NAME)
    formats = [
        lambda u, t: f"{u:,} / {t:,}  ({100 * u / t:.0f}%)",
        lambda u, t: f"{u:,} / {t:,} ({100 * u / t:.0f}%)",
        lambda u, t: f"{u:,} ({100 * u / t:.0f}%)",
        lambda u, t: f"{100 * u / t:.0f}%",
    ]
    widest = max(zip(used, total), key=lambda ut: len(formats[0](*ut)))
    fmt = formats[[f(*widest) for f in formats].index(
        fit_one_line(fig, [f(*widest) for f in formats], FS_VALUE, label_room))]
    value_labels = [fmt(u, t) for u, t in zip(used, total)]

    for yi, lab in zip(y, value_labels):
        ax.text(
            0.0,
            yi + 0.30,
            lab,
            va="bottom",
            ha="left",
            fontsize=FS_VALUE,
            color="black",
            zorder=3,
        )

    # Optional human-readable concept budget
    if budget:
        ax.axvline(budget, color=BUDGET, linewidth=1.2, linestyle=(0, (3, 2)), zorder=3)
        ax.text(
            budget + 0.015 * xmax,
            n - 0.52,
            f"what a reader can hold (~{budget})",
            fontsize=FS_NOTE,
            color=BUDGET,
            va="bottom",
            ha="left",
        )

    # Y axis
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_NAME, color="black")

    # Only the bars live on the x axis now, so barely any padding is needed.
    ax.set_xlim(0, xmax * 1.02)
    # The top row's annotation sits 0.30 data units above its bar and is drawn upward,
    # so the y limit has to clear it -- otherwise it spills out of the axes and into the
    # title. One data unit is `row_h` inches here (the axes is n data units tall).
    ax.set_ylim(-0.45, (n - 1) + 0.34 + FS_VALUE * 1.35 / 72 / row_h
                + (0.35 if budget else 0))


    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )

    # Few ticks: at this width and font size more than a couple of comma-formatted
    # numbers collide.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2, integer=True))

    # x label above the legend, legend at the very bottom of the figure.
    fig.supxlabel(xlabel_text, fontsize=FS_AXIS, color="black",
                  y=(legend_h() - 0.02) / fig.get_figheight())

    ax.tick_params(axis="x", colors="black", length=0, labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=2)

    # Grid
    ax.grid(axis="x", color="#eeede9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")

    # Legend
    from matplotlib.patches import Patch

    fig_h_in = fig.get_figheight()
    fig.legend(
        handles=[
            Patch(color=USED, label=legend_labels[0]),
            Patch(color=DICT, label=legend_labels[1]),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=legend_ncol,
        fontsize=FS_LEG,
        frameon=False,
        handlelength=1.5,
        handleheight=1.1,
        labelspacing=0.35,
        columnspacing=1.4,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelcolor="black",
    )

    if title_lines:
        fig.text(
            0.005, 0.995, "\n".join(title_lines),
            fontsize=FS_TITLE, color="black", ha="left", va="top",
        )

    # tight_layout would fight the figure-level legend/title and can collapse the axes
    # at this width, so the margins are set explicitly from the font metrics above.
    fig.subplots_adjust(
        left=_left_margin_in(labels, FS_NAME) / width,
        right=0.995,
        top=1 - _head_h() / fig_h_in,
        bottom=_foot_h() / fig_h_in,
    )

    # Save all formats
    root, _ = os.path.splitext(out_path)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 250}), (".svg", {})):
        fig.savefig(root + ext, bbox_inches="tight", **kw)
    plt.close(fig)

    return root


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "specs",
        nargs="+",
        help="<label>=<json glob>",
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--title",
        default="Existing CBMs rely on most of their dictionary",
    )

    p.add_argument(
        "--width",
        type=float,
        default=3.6,
        help="figure width in inches (default 3.6; the panel is meant to be crammed)",
    )

    p.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="multiplies every font size (and the figure height with them)",
    )

    p.add_argument(
        "--xlabel",
        default="Used concepts",
        help="x axis label; kept on one line (the axes is stretched to cover it)",
    )

    p.add_argument(
        "--budget",
        type=int,
        default=0,
        help="optional reference line for a human-readable concept "
             "budget (0 = off)",
    )

    args = p.parse_args()

    rows = [
        _load(s)
        for s in args.specs
    ]

    for label, u, t, path in rows:
        print(
            f"{label:8s} "
            f"{u:>5,} / {t:>5,} "
            f"({100 * u / t:.0f}%)  <- {path}"
        )

    root = plot(
        rows,
        args.out,
        args.title,
        args.budget,
        width=args.width,
        font_scale=args.font_scale,
        xlabel=args.xlabel,
    )

    print(
        f"saved {root}.pdf / "
        f"{root}.png / "
        f"{root}.svg"
    )


if __name__ == "__main__":
    main()