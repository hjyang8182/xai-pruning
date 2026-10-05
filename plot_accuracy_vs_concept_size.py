"""Right panel of the motivation figure: accuracy vs effective concept size with GCG.

Reads a lambda_gate sweep (sweep_lambda_gate.py -> lambda_gate_sweep.json: test accuracy
vs mean open gates across seeds) and draws test accuracy against the number of concepts
the gated model keeps, next to the ungated model as a reference point. Styled to match
plot_effective_concept_size.py so the two sit side by side.

Usage:
    python plot_accuracy_vs_concept_size.py --out img/accuracy_vs_concept_size_cifar100.pdf \\
        "DN-CBM=data/cifar100/lambda_gate_sweep.json" --baseline-size 7057
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834"]   # categorical slots 1-2, fixed order
BASE = "#9d9c97"                  # ungated reference point
INK = "#0b0b0b"
INK2 = "#52514e"


def _load(spec):
    label, path = spec.split("=", 1)
    with open(path) as f:
        r = json.load(f)
    pts = sorted(zip(r["open_gates"], r["test_accuracy"]))
    x = [p[0] for p in pts]
    y = [100 * p[1] for p in pts]
    return label, x, y, 100 * r["baseline_test_accuracy"]


def plot(entries, out_path, title, baseline_sizes, logx, xlabel):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    handles = []
    for i, (label, x, y, base_acc) in enumerate(entries):
        c = SERIES[i]
        (h,) = ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=5,
                       markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                       label=f"{label} + GCG", solid_joinstyle="round", solid_capstyle="round")
        handles.append(h)
        bs = baseline_sizes[i] if i < len(baseline_sizes) else None
        if bs:
            ax.axhline(base_acc, color=BASE, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
            hb = ax.scatter([bs], [base_acc], s=42, color=BASE, edgecolor="white", linewidth=1.2,
                            zorder=4, label=f"{label} (no GCG)")
            handles.append(hb)
            ax.annotate(f"{bs:,} concepts, {base_acc:.1f}%", (bs, base_acc), xytext=(0, -9),
                        textcoords="offset points", ha="right", va="top", fontsize=7.5, color=INK2)
        # call out the knee: the smallest open set within 1 point of the ungated model
        knee = next((j for j in range(len(x)) if y[j] >= base_acc - 1.0), None)
        if knee is not None:
            ax.annotate(f"{x[knee]:,.0f} concepts, {y[knee]:.1f}%", (x[knee], y[knee]),
                        xytext=(10, -14), textcoords="offset points", ha="left", va="top",
                        fontsize=7.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color="#b5b4af", linewidth=0.8, shrinkB=4))

    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlim(left=0)
    ax.set_ylim(bottom=50)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("Test accuracy", color=INK2)
    ax.tick_params(axis="both", colors=INK2, length=0)
    ax.grid(color="#eeede9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color("#d6d5d0")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7.5,
              frameon=False, handlelength=1.6, columnspacing=1.2, borderaxespad=0.2)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=22)
    fig.tight_layout()
    root, _ = os.path.splitext(out_path)
    for ext in (".pdf", ".svg"):
        fig.savefig(root + ext, bbox_inches="tight")
    fig.savefig(root + ".png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="<label>=<lambda_gate_sweep.json>")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="GCG keeps accuracy with far fewer concepts")
    p.add_argument("--baseline-size", type=int, nargs="*", default=[],
                   help="effective concept size of each ungated model (same order as specs), e.g. its global union")
    p.add_argument("--logx", action="store_true", help="log x-axis instead of linear")
    p.add_argument("--xlabel", default="Concepts with non-zero weight")
    args = p.parse_args()
    entries = [_load(s) for s in args.specs]
    for label, x, y, b in entries:
        print(f"{label}: baseline {b:.2f}%  " + "  ".join(f"{xi:.0f}={yi:.2f}" for xi, yi in zip(x, y)))
    root = plot(entries, args.out, args.title, args.baseline_size, args.logx, args.xlabel)
    print(f"saved {root}.pdf / {root}.png / {root}.svg")


if __name__ == "__main__":
    main()
