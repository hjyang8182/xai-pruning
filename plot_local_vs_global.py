"""One figure + one table for the "sparse per class, dense overall" motivation.

Takes the local_explanation_size_*.json files written by each substrate's
`concept_retention.py --analysis local` (LF-CBM / VLG-CBM / UCBM, any dataset) and
draws, per model, the distribution of per-image explanation size next to NEC (the
per-class count the methods report) and the global union (the concept set a reader
must know to read the model). Also prints a markdown table of the same numbers.

Usage:
    python plot_local_vs_global.py --out img/local_vs_global.png \\
        "LF-CBM (CIFAR-100)=prelim/Label-free-CBM/seed_sweep_results/cifar100_local_explanation_size_*.json" \\
        "VLG-CBM (CIFAR-100)=prelim/VLG-CBM/saved_models/cifar100/.../local_explanation_size_*.json" \\
        "UCBM (CIFAR-100)=prelim/ucbm/save/RESULTS/.../cifar100_local_explanation_size_*.json"
Each positional arg is  <panel label>=<json path or glob>  (latest match is used).
"""
import argparse
import glob
import json
import os

import numpy as np

from explanation_size import _fmt_level, _int_bins


def _load(spec):
    label, pattern = spec.split("=", 1)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"nothing matched {pattern}")
    with open(paths[-1]) as f:
        return label, json.load(f), paths[-1]


def table_rows(entries, level):
    rows = []
    for label, r, _ in entries:
        km = r["k_mass"][level]
        nz = r["n_nonzero"]
        rows.append({
            "model": label, "J": r["n_concepts_total"], "acc": r["acc"],
            "NEC": r["nec"], "union": r["n_union"], "union_frac": r["union_frac"],
            "inter": r["n_intersection"],
            "k_mean": km["mean"], "k_med": km["median"], "k_p99": km["p99"], "k_max": km["max"],
            "nz_mean": nz["mean"], "nz_med": nz["median"], "nz_max": nz["max"],
            "local_union": r["local_union_mass"][level],
            # older JSONs predate local_union_nonzero; for dense activations it equals the global union
            "local_union_nz": r.get("local_union_nonzero", float("nan")),
        })
    return rows


def markdown_table(rows, level):
    hdr = (f"| model | #concepts | acc | NEC (per class) | global union | intersection | "
           f"per-image concepts @{level} mass: mean / median / p99 / max | per-image non-zero terms: mean / median / max | "
           f"union of per-image explanations @{level} mass | union of per-image non-zero terms |")
    lines = [hdr, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['J']} | {r['acc']:.3f} | {r['NEC']:.1f} | "
            f"{r['union']:.0f} ({100 * r['union_frac']:.0f}%) | {r['inter']:.0f} | "
            f"{r['k_mean']:.1f} / {r['k_med']:.0f} / {r['k_p99']:.0f} / {r['k_max']:.0f} | "
            f"{r['nz_mean']:.1f} / {r['nz_med']:.0f} / {r['nz_max']:.0f} | {r['local_union']:.0f} | {r['local_union_nz']:.0f} |")
    return "\n".join(lines)


def plot(entries, out_path, level, measure):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, (label, r, _) in zip(axes[0], entries):
        pe = r["per_example"]
        v = np.asarray(pe["k_mass"][level] if measure == "k_mass" else pe[measure], dtype=float)
        ax.hist(v, bins=_int_bins(v), color="#00376d", alpha=0.85, edgecolor="none",
                label="per-image explanation")
        ax.set_yscale("log")
        ax.axvline(r["nec"], color="#2e7d5b", linestyle="--", linewidth=1.4,
                   label=f"NEC = {r['nec']:.0f}")
        ax.axvline(v.max(), color="#c0654b", linestyle="-", linewidth=1.4,
                   label=f"largest image = {v.max():.0f}")
        ax.set_title(f"{label}\nglobal union {r['n_union']:.0f} / {r['n_concepts_total']} concepts "
                     f"({100 * r['union_frac']:.0f}%)", fontsize=10)
        if measure == "k_mass":
            ax.set_xlabel(f"# concepts covering {level} of predicted-class logit")
        elif measure == "n_nonzero":
            ax.set_xlabel("# non-zero concept terms in predicted-class logit")
        else:
            ax.set_xlabel("# concepts in minimal sufficient prefix")
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("# test images (log)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="<label>=<json glob>")
    p.add_argument("--out", required=True)
    p.add_argument("--level", type=float, default=0.9, help="which k_mass level to show")
    p.add_argument("--measure", choices=["k_mass", "n_nonzero", "k_suff"], default="k_mass")
    args = p.parse_args()
    level = _fmt_level(args.level)
    entries = [_load(s) for s in args.specs]
    for label, _, path in entries:
        print(f"{label}: {path}")
    rows = table_rows(entries, level)
    table = markdown_table(rows, level)
    print()
    print(table)
    plot(entries, args.out, level, args.measure)
    # Same table next to the figure, so the numbers survive without re-running (the JSONs are untracked)
    table_path = os.path.splitext(args.out)[0] + ".md"
    with open(table_path, "w") as f:
        f.write(table + "\n\nSources:\n" + "\n".join(f"- {label}: `{path}`" for label, _, path in entries) + "\n")
    print(f"\nSaved figure to {args.out} and table to {table_path}")


if __name__ == "__main__":
    main()
