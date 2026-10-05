"""One figure + one table for "tau as a hyperparameter".

Takes the tau_selection_*.json files written by each substrate's
`concept_retention.py --analysis select-tau` (LF-CBM / VLG-CBM / UCBM) and draws,
per model, eval accuracy vs open concepts for the hard cut with the soft head kept
and for the hard cut + refit head, marking tau* (chosen on the selection fold) and
tau=0.5. Prints a markdown table of the same numbers.

Usage:
    python plot_tau_selection.py --out img/tau_selection_cifar100.png \\
        "LF-CBM=prelim/Label-free-CBM/seed_sweep_results/cifar100_tau_selection_*.json" ...
Each positional arg is  <panel label>=<json path or glob>  (latest match is used).
"""
import argparse
import glob
import json

import numpy as np

C_REFIT, C_NOFIT, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"


def _load(spec):
    label, pattern = spec.split("=", 1)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"nothing matched {pattern}")
    with open(paths[-1]) as f:
        return label, json.load(f), paths[-1]


def markdown_table(entries):
    lines = ["| model | #concepts | soft full acc | τ* | open @τ* | acc @τ* refit | acc @τ* hard cut | "
             "@0.5: open / hard cut / refit | mushy |",
             "|" + "---|" * 9]
    for label, r, _ in entries:
        ch, ref = r["chosen"], r["reference"]["0.5"]
        mode = "refit" if r["refit"] else "nofit"
        seeds = f" (×{r['n_seeds']})" if r.get("n_seeds", 1) > 1 and "per_seed_chosen" in r and r.get("n_folds") != r["n_seeds"] else ""
        lines.append(
            f"| {label}{seeds} | {r['n_concepts_total']} | {r['eval_acc_full']:.3f} | {ch['tau']:.2f} | "
            f"{ch['n_open']:.0f} | {ch.get('eval_acc_refit', float('nan')):.3f} | {ch['eval_acc_nofit']:.3f} | "
            f"{ref['n_open']:.0f} / {ref['eval_acc_nofit']:.3f} / {ref.get('eval_acc_refit', float('nan')):.3f} | "
            f"{r['bimodal_mushy_frac']:.2f} |")
    return "\n".join(lines)


def plot(entries, out_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2), squeeze=False)
    for j, (label, r, _) in enumerate(entries):
        ax = axes[0, j]
        rows = r["candidates"]
        n_open = np.array([c["n_open"] for c in rows], dtype=float)
        order = np.argsort(n_open)
        x = np.maximum(n_open[order], 0.8)
        ax.plot(x, [rows[i]["eval_acc_nofit"] for i in order], color=C_NOFIT, marker="o", markersize=3.5,
                linewidth=1.6, label="hard cut, soft head kept")
        if r["refit"]:
            ax.plot(x, [rows[i]["eval_acc_refit"] for i in order], color=C_REFIT, marker="o", markersize=3.5,
                    linewidth=2.0, label="hard cut + refit head")
        ax.axhline(r["eval_acc_full"], color=INK2, linestyle=":", linewidth=1.0, label="soft full model")
        ch, ref = r["chosen"], r["reference"]["0.5"]
        key = "eval_acc_refit" if r["refit"] else "eval_acc_nofit"
        ax.scatter([max(ch["n_open"], 0.8)], [ch[key]], s=90, facecolors="none", edgecolors=INK,
                   linewidths=1.6, zorder=5)
        ax.annotate(f"τ*={ch['tau']:.2f}\n{ch['n_open']:.0f} open, {ch[key]:.3f}",
                    (max(ch["n_open"], 0.8), ch[key]), xytext=(-8, -30), textcoords="offset points",
                    fontsize=8, color=INK, ha="right")
        ax.scatter([max(ref["n_open"], 0.8)], [ref["eval_acc_nofit"]], s=60, marker="s", facecolors="none",
                   edgecolors=INK2, linewidths=1.2, zorder=5)
        ax.annotate(f"τ=0.5: {ref['n_open']:.0f} open, {ref['eval_acc_nofit']:.3f}",
                    (max(ref["n_open"], 0.8), ref["eval_acc_nofit"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.7, r["n_concepts_total"] * 1.4)
        ax.set_ylim(0, max(0.05, min(1.0, r["eval_acc_full"] * 1.25)))
        ax.set_title(label, fontsize=11, color=INK)
        ax.set_xlabel("open concepts (gate > τ)", color=INK2)
        if j == 0:
            ax.set_ylabel("eval accuracy", color=INK2)
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2)
    fig.suptitle(title or "τ chosen by max CEA on a held-out fold; head refit on survivors", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="<label>=<json path or glob>")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--table-out", default=None)
    return p.parse_args()


def main(args):
    entries = [_load(s) for s in args.specs]
    for label, _, path in entries:
        print(f"{label}: {path}")
    md = markdown_table(entries)
    print()
    print(md)
    if args.table_out:
        with open(args.table_out, "w") as f:
            f.write(md + "\n")
    plot(entries, args.out, title=args.title)
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main(parse_args())
