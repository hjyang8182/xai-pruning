"""Accuracy vs. NCC_0.95 (De Santis et al., ICLR 2026) for GCG and each pipeline's native sparsity.

Inputs: the newest <dataset>_ncc_sweep_*.json of each pipeline (prelim/<pipeline>/ncc_sweep.py). Writes
  data/plot/acc_vs_ncc_<stamp>.{svg,png}  rows = LF-CBM / UCBM / VLG-CBM, columns = datasets; one point
                                          per lambda (mean over seeds, seed s.d. as error bars)
  data/plot/acc_at_ncc_<stamp>.{json,md}  the paper's Table 1 numbers per arm: accuracy at NCC=5 and
                                          NCC=avg (mean over NCC 5, 10, ..., 30)

"Accuracy at NCC=n" is the best mean accuracy among an arm's lambdas whose mean NCC is <= n (the
paper trains the head at several regularisation strengths and compares at equal NCC; the lambdas are
not tuned to land exactly on n, so this reads the accuracy-vs-NCC frontier at n). An arm with no
lambda at NCC <= n gets none.

    python plot_ncc.py               # NCC over all classes (the paper's definition)
    python plot_ncc.py --pred        # NCC of the predicted class only
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from plot_pipeline_grids import DATASET_NAMES, DATASETS, MODEL_NAMES, ROOT, load, newest  # noqa: E402

ARMS = {"gated": ("GCG (gated head)", "#4C72B0", "o", "-"),
        "gated_refit": ("GCG + mask-and-refit", "#8FB3E0", "s", "--"),
        "sparse": ("Native sparsity", "#DD8452", "^", "-")}
TARGETS = (5, 10, 15, 20, 25, 30)


def acc_at(points, n, key):
    ok = [p["acc_mean"] for p in points if p[key] <= n]
    return max(ok) if ok else None


def main(args):
    key = "ncc_pred_mean" if args.pred else "ncc_mean"
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    table = {}
    for i, model in enumerate(("lfcbm", "ucbm", "vlgcbm")):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i][j]
            path = newest(model, dataset, "ncc")
            ax.set_title(f"{MODEL_NAMES[model]} · {DATASET_NAMES[dataset]}", fontsize=13)
            if path is None:
                ax.text(0.5, 0.5, "no NCC sweep", ha="center", va="center", transform=ax.transAxes, color="#777777")
                continue
            d = load(path)
            row = {"ncc_sweep": os.path.relpath(path, ROOT), "max_examples": d.get("max_examples")}
            for arm, (label, color, marker, ls) in ARMS.items():
                # drawn in lambda order: the line follows the regularisation path (NCC need not be monotone in it)
                pts = sorted((p for p in d["points"] if p["arm"] == arm and p["acc_mean"] > args.min_acc),
                             key=lambda p: p["lam"])
                if not pts:
                    continue
                x = np.array([max(p[key], 0.5) for p in pts])
                y = np.array([100 * p["acc_mean"] for p in pts])
                ax.errorbar(x, y, yerr=[100 * p["acc_std"] for p in pts], xerr=[p[key.replace("_mean", "_std")] for p in pts],
                            color=color, marker=marker, linestyle=ls, linewidth=2, markersize=8,
                            markeredgecolor="white", markeredgewidth=1.2, capsize=0, elinewidth=1, label=label, zorder=3)
                at = {n: acc_at(pts, n, key) for n in TARGETS}
                row[arm] = {"acc_at_ncc5": at[5],
                            "acc_at_ncc_avg": float(np.mean(list(at.values()))) if all(v is not None for v in at.values()) else None,
                            "acc_at_ncc": {str(n): v for n, v in at.items()},
                            "min_ncc": float(min(p[key] for p in pts))}
            table[f"{model}/{dataset}"] = row
            ax.axvline(5, color="#999999", linestyle=":", linewidth=1.2, zorder=1)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(f"NCC$_{{0.95}}$ ({'predicted class' if args.pred else 'all classes'})")
            ax.set_ylabel("Test accuracy (%)")
            ax.grid(True, which="major", color="#E5E5E5", linewidth=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "data", "plot")
    tag = "_pred" if args.pred else ""
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out_dir, f"acc_vs_ncc{tag}_{stamp}.{ext}"), dpi=200, bbox_inches="tight")

    fmt = lambda v: "–" if v is None else f"{100 * v:.1f}"  # noqa: E731
    md = [f"Accuracy (%) at NCC_0.95 ({'predicted class' if args.pred else 'all classes'}) <= n (best lambda at or "
          f"under n; – = no lambda reaches it). avg = the paper's NCC=avg, the mean over n = 5..30, only when all exist.", "",
          "| Model | Dataset | Arm | min NCC | " + " | ".join(f"@{n}" for n in TARGETS) + " | avg |",
          "|---|---|---|---|" + "---|" * (len(TARGETS) + 1)]
    for name, row in table.items():
        model, dataset = name.split("/")
        for arm in ARMS:
            if arm not in row:
                continue
            r = row[arm]
            md.append(f"| {MODEL_NAMES[model]} | {DATASET_NAMES[dataset]} | {ARMS[arm][0]} | {r['min_ncc']:.1f} | "
                      + " | ".join(fmt(r["acc_at_ncc"][str(n)]) for n in TARGETS) + f" | {fmt(r['acc_at_ncc_avg'])} |")
    with open(os.path.join(out_dir, f"acc_at_ncc{tag}_{stamp}.json"), "w") as f:
        json.dump({"ncc_key": key, "targets": TARGETS, "rows": table}, f, indent=2)
    with open(os.path.join(out_dir, f"acc_at_ncc{tag}_{stamp}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {out_dir}/acc_vs_ncc{tag}_{stamp}.svg")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", action="store_true", help="NCC of the predicted class instead of all classes")
    p.add_argument("--min-acc", type=float, default=0.02, help="drop collapsed heads (chance-level accuracy)")
    main(p.parse_args())
