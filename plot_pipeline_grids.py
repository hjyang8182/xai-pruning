"""House-style grids (like data/plot/gate_vs_sparse_grid_*.svg) for the LF-CBM / UCBM / VLG-CBM results:

  gate_vs_native_sparsity_<model>  test accuracy vs. % of concepts used, GCG lambda_gate sweep against
                                   the pipeline's own sparsity sweep (LF-CBM / VLG-CBM: GLM-SAGA
                                   elastic net; UCBM: weight elastic net), one panel per dataset
  gate_vs_random_<model>           same axes, mask-and-refit on the gate-selected concepts against the
                                   same recipe on an equal number of random concepts, and (when the
                                   sweep has them) on the top-|K| by dense-head weight magnitude and the
                                   group-lasso support at |K|
  leakage_vs_random_<encoding>     CTL / ICL (Parisini et al.) of the GCG open set and the baseline's
                                   size-matched top-k, relative to random same-size concept sets

Inputs are the newest <dataset>_{gate,sparse}_sweep_*.json of each pipeline (run_seed_sweeps.py) and
the newest data/leakage/leakage_pipelines_*.json (concept_leakage_pipelines.py). The dashed baseline is
the pipeline's default sparse head; the dotted line marks the deployed GCG model (lambda_gate from
paper/tab_hyperparams.tex), taken from the sweep point at that lambda or a separate run at it.
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import visualise as V
from plot_gate_sweep_vs_random_subset import ROOT, SWEEP_GLOBS

MODEL_NAMES = {"lfcbm": "LF-CBM", "ucbm": "UCBM", "vlgcbm": "VLG-CBM"}
# the leakage figure also covers DN-CBM (export_dncbm_leakage_data.py); its sweep grids are plotted elsewhere
LEAKAGE_NAMES = {"dncbm": "DN-CBM", **MODEL_NAMES}
DATASET_NAMES = {"cifar100": "CIFAR-100", "cub": "CUB", "places365": "Places-365"}
SPARSE_LABELS = {"lfcbm": "(Original) GLM-SAGA Sparse Layer", "vlgcbm": "(Original) GLM-SAGA Sparse Layer",
                 "ucbm": "(Original) Elastic-Net Head"}
# stronger size-matched controls (gating.size_matched_refits), drawn on the gate-vs-random grid when present
CONTROL_LABELS = {"magnitude_refit": "Top-|K| by weight magnitude (refit)", "group_lasso_refit": "Group-lasso at |K| (refit)"}
SPARSE_ARM = {"lfcbm": "baseline", "ucbm": "baseline", "vlgcbm": "sparse"}
# deployed lambda_gate per (model, dataset): paper/tab_hyperparams.tex
DEPLOYED = {("lfcbm", "cifar100"): 1e-4, ("lfcbm", "cub"): 7e-4, ("lfcbm", "places365"): 5e-4,
            ("ucbm", "cifar100"): 1e-4, ("ucbm", "cub"): 5e-4, ("ucbm", "places365"): 3e-4,
            ("vlgcbm", "cifar100"): 5e-5, ("vlgcbm", "cub"): 5e-5, ("vlgcbm", "places365"): 1e-4}
DATASETS = ["cifar100", "cub", "places365"]


def newest(model, dataset, kind):
    paths = glob.glob(os.path.join(ROOT, SWEEP_GLOBS[model].replace("_gate_sweep_", f"_{kind}_sweep_").format(d=dataset)))
    return max(paths, key=os.path.getmtime) if paths else None


def load(path):
    with open(path) as f:
        return json.load(f)


def results_dir(gate_path):
    return os.path.dirname(gate_path)


# ---------------------------------------------------------------------------------------------
# Baseline accuracy (dashed line) and deployed operating point (dotted line)
# ---------------------------------------------------------------------------------------------
def lfcbm_baseline_from_probes(rdir, dataset, shared_from):
    """LF-CBM baseline accuracy from saved probes when the baseline sweep was stopped before writing
    its json: the session whose seeds were trained on (and saved) the same shared stage."""
    if not shared_from:
        return None
    shared_dir = os.path.join(ROOT, "prelim", "Label-free-CBM", shared_from)
    for session in sorted(glob.glob(os.path.join(rdir, f"{dataset}_baseline_seed_sweep_*")), reverse=True):
        seeds = [d for d in glob.glob(os.path.join(session, "seed*")) if os.path.basename(d)[4:].isdigit()]
        if not seeds:
            continue
        with open(os.path.join(seeds[0], "args.txt")) as f:
            a = json.load(f)
        if os.path.normpath(a.get("save_shared") or a.get("load_shared") or "") != os.path.normpath(shared_from):
            continue
        val_c = torch.load(os.path.join(shared_dir, "val_c.pt"), map_location="cpu")
        val_y = torch.load(os.path.join(shared_dir, "val_y.pt"), map_location="cpu")
        accs = []
        for d in seeds:
            W = torch.load(os.path.join(d, "W_g.pt"), map_location="cpu")
            b = torch.load(os.path.join(d, "b_g.pt"), map_location="cpu")
            accs.append(((val_c @ W.T + b).argmax(1) == val_y).float().mean().item())
        return float(np.mean(accs)), float(np.std(accs))
    return None


def baseline_acc(model, dataset, gate):
    """(mean, std) test accuracy of the pipeline's default sparse head, or None."""
    if model == "vlgcbm":
        s = gate.get("sparse")
        return (s["acc_mean"], s["acc_std"]) if s else None
    rdir = results_dir(gate["_path"])
    for path in sorted(glob.glob(os.path.join(rdir, f"{dataset}_baseline_seed_sweep_*.json")),
                       key=os.path.getmtime, reverse=True):
        d = load(path)
        if d.get("n_concepts_total") == gate["n_concepts_total"] and "baseline" in d:
            return d["baseline"]["acc_mean"], d["baseline"]["acc_std"]
    if model == "lfcbm":
        return lfcbm_baseline_from_probes(rdir, dataset, gate["points"][0].get("shared_from"))
    return None


def operating_point(model, dataset, gate):
    """Mean open-gate count of the deployed GCG model: the sweep point at the deployed lambda_gate,
    else the newest separate hard-gate run at it with the same concept dictionary."""
    lam = DEPLOYED.get((model, dataset))
    if lam is None:
        return None
    for p in gate["points"]:
        if math.isclose(p["lambda_gate"], lam, rel_tol=1e-6):
            return p["gated"]["n_open_mean"]
    key = "gate_lam" if model == "lfcbm" else "lambda_gate"
    for path in sorted(glob.glob(os.path.join(results_dir(gate["_path"]), f"{dataset}_gated_seed_sweep_*.json")),
                       key=os.path.getmtime, reverse=True):
        d = load(path)
        if (d.get(key) is not None and math.isclose(d[key], lam, rel_tol=1e-6) and d.get("gate_forward") == "hard"
                and d.get("n_concepts_total") == gate["n_concepts_total"] and "gated" in d):
            return d["gated"]["n_open_mean"]
    return None


# ---------------------------------------------------------------------------------------------
# Accuracy-vs-count grids
# ---------------------------------------------------------------------------------------------
def curve(points, arm, count, scale):
    rows = sorted((p[arm][count] * scale, p[arm]["acc_mean"], p[arm]["acc_std"])
                  for p in points if arm in p and p[arm].get(count, 0) > 0)
    return [list(c) for c in zip(*rows)] if rows else (None, None, None)


def model_panels(model, kind):
    panels = {}
    for dataset in DATASETS:
        gpath = newest(model, dataset, "gate")
        if gpath is None:
            continue
        gate = {**load(gpath), "_path": gpath}
        scale = 100.0 / gate["n_concepts_total"]
        if kind == "sparse":
            x, acc, std = curve(gate["points"], "gated", "n_open_mean", scale)
            spath = newest(model, dataset, "sparse")
            cmp = (None, None, None)
            if spath:
                sparse = load(spath)
                arm = SPARSE_ARM[model]
                count = "n_used_mean" if any("n_used_mean" in p.get(arm, {}) for p in sparse["points"]) else "n_open_mean"
                cmp = curve(sparse["points"], arm, count, scale)
        else:  # random-subset control: both arms refit with the same cold-start recipe at |K|
            x, acc, std = curve(gate["points"], "gated_refit_cold", "n_open_mean", scale)
            cmp = curve(gate["points"], "random_refit", "n_open_mean", scale)
        extra = []
        if kind == "random":
            for arm, label in CONTROL_LABELS.items():
                c = curve(gate["points"], arm, "n_open_mean", scale)
                if c[0] is not None:
                    extra.append((*c, label))
        base = baseline_acc(model, dataset, gate)
        op = operating_point(model, dataset, gate)
        panels[DATASET_NAMES[dataset]] = {
            "open_gates": x, "gated_acc": acc, "gated_acc_std": std,
            "cmp_x": cmp[0], "cmp_acc": cmp[1], "cmp_acc_std": cmp[2],
            "baseline_acc": base[0] if base else None, "baseline_acc_std": base[1] if base else None,
            "operating_point": op * scale if op is not None else None, "extra_curves": extra,
        }
        print(f"  {model}/{dataset}: gate {os.path.relpath(gpath, ROOT)}"
              + (f", baseline {base[0]:.4f}" if base else ", no baseline")
              + (f", deployed {op:.1f} concepts" if op is not None else ", no deployed run"))
    return panels


def plot_grids(models):
    for model in models:
        name = MODEL_NAMES[model]
        for kind, cmp_label, gated_label, label in (
                ("sparse", SPARSE_LABELS[model], f"{name} w/ GCG", f"gate_vs_native_sparsity_{model}"),
                ("random", "Random concepts (refit)", f"{name} w/ GCG (refit)", f"gate_vs_random_{model}")):
            print(f"[{label}]")
            panels = model_panels(model, kind)
            if not panels:
                continue
            path = V._plot_acc_vs_count_grid(panels, cmp_label, ncols=len(panels), frame_on_gated=(kind == "sparse"),
                                             plot_label=label, gated_label=gated_label,
                                             legend_fontsize=V.FS_LEGEND + (3 if kind == "sparse" else 0))
            print(f"  -> {path}")


# ---------------------------------------------------------------------------------------------
# Leakage relative to random same-size concept sets
# ---------------------------------------------------------------------------------------------
def plot_leakage(encoding):
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "leakage", "leakage_pipelines_*.json")), key=os.path.getmtime)
    if not paths:
        print("no data/leakage/leakage_pipelines_*.json; run concept_leakage_pipelines.py")
        return
    res = load(paths[-1])["models"]
    models = [m for m in LEAKAGE_NAMES if m in res]
    print(f"[leakage_vs_random_{encoding}] {os.path.relpath(paths[-1], ROOT)}")

    GATED_COLOR, BASE_COLOR = "#4C72B0", "#55A868"
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 2, figsize=(V.PANEL_W * 2 * max(1.0, len(models) / 3), V.PANEL_H + 0.9),
                             layout="constrained")
    for ax, (metric, title) in zip(axes, (("ctl", "Concepts-Task Leakage"), ("icl", "Interconcept Leakage"))):
        for i, model in enumerate(models):
            sets = res[model]["encodings"][encoding]["sets"]
            gated, topk = sets["gated"]["per_seed"], sets["baseline_topk"]["per_seed"]
            null_mean = np.array([g[f"null_{metric}_mean"] for g in gated])  # same null for both sets
            lo = np.mean(np.array([g[f"null_{metric}_lo"] for g in gated]) / null_mean)
            hi = np.mean(np.array([g[f"null_{metric}_hi"] for g in gated]) / null_mean)
            ax.fill_between([i - 0.32, i + 0.32], lo, hi, color="#555555", alpha=0.12, linewidth=0, zorder=1,
                            label="Random subsets (95%)" if i == 0 else None)
            for j, (per_seed, color, marker, lab) in enumerate(
                    ((topk, BASE_COLOR, "s", "Baseline top-k"), (gated, GATED_COLOR, "o", "w/ GCG"))):
                ratio = np.array([r[f"{metric}_mean"] for r in per_seed]) / null_mean
                ax.errorbar(i + (j - 0.5) * 0.3, ratio.mean(), yerr=ratio.std(), color=color, marker=marker,
                            markersize=V.MS + 2, markeredgecolor="white", markeredgewidth=V.MEW, linestyle="none",
                            elinewidth=V.LW_REF, capsize=4, zorder=3, label=lab if i == 0 else None)
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=V.LW_REF, zorder=2, label="Random mean")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([f"{LEAKAGE_NAMES[m]}\n({res[m]['encodings'][encoding]['sets']['gated']['n']:.0f}"
                            f"/{res[m]['n_concepts']})" for m in models])
        V._style_axes(ax, title)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylabel(f"{metric.upper()} / random subset", fontsize=V.FS_LABEL, fontweight="bold")
    fig.patch.set_facecolor("white")
    h, l = axes[0].get_legend_handles_labels()
    V._legend_below(fig, h, l)
    path = V._save_plot(None, f"leakage_vs_random_{encoding}", save_dir=os.path.join(V.DATA_PATH, "plot"))
    print(f"  -> {path}")


def main(args):
    if "grids" in args.what:
        plot_grids(args.models)
    if "leakage" in args.what:
        for encoding in args.encodings:
            plot_leakage(encoding)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    p.add_argument("--what", nargs="+", default=["grids", "leakage"], choices=["grids", "leakage"])
    p.add_argument("--encodings", nargs="+", default=["hard", "soft"], choices=["hard", "soft"])
    main(p.parse_args())
