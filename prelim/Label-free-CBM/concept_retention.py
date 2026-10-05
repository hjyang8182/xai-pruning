"""Threshold-free concept count for a trained Label-free-CBM final layer.

Ranks the model's concepts by contribution c_j = ||W_g[:,j]||_2 * mean|a_j| on the
standardised concept activations, keeps only the top-k, and reports the smallest k
that retains >= 95%/99% of full eval accuracy. See ../../concept_retention.py.

For arm C the learned gate is already folded into W_g (as saved), so it is NOT
passed again here -- gate_logits.pt is loaded only for the optional --ranking gate
comparison.

Rebuilds the eval-split concept activations the same way cbm.CBM_model.forward does
    proj_c = (target_features @ W_c.T - proj_mean) / proj_std
straight from the run's W_c.pt / proj_mean.pt / proj_std.pt and the backbone
target features already cached in --activation-dir, so no projection layer is
retrained. Point --load-dir at a single run dir (one W_g.pt), or pass it several
times / a glob-able parent to average over seeds.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from concept_retention import (  # noqa: E402
    retention_curve, mean_retention, plot_retention_curve,
    threshold_sweep, mean_threshold_sweep, plot_threshold_sensitivity,
    select_tau_crossfit, mean_select_tau, plot_tau_selection, select_tau_summary_line,
)
from explanation_size import (  # noqa: E402
    local_explanation_sizes, mean_explanation_report, plot_explanation_sizes,
    format_extreme_examples, summary_line,
)

import data_utils  # noqa: E402
import utils  # noqa: E402


def _run_dirs(patterns):
    dirs = []
    for p in patterns:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "W_g.pt")):
            dirs.append(p)
        else:
            dirs.extend(sorted(d for d in glob.glob(p) if os.path.exists(os.path.join(d, "W_g.pt"))))
    if not dirs:
        raise FileNotFoundError(f"No run dirs with W_g.pt matched {patterns}")
    return dirs


def load_eval_concepts(run_dir, activation_dir, device, split="eval", max_rows=None, seed=0):
    """(proj_c, y) for the run's eval split (or its train split with split="train"),
    rebuilt from cached backbone features. `max_rows` subsamples rows before the
    projection (the Places365 train features are ~15 GB)."""
    with open(os.path.join(run_dir, "args.txt")) as f:
        a = json.load(f)
    dataset = a["dataset"]
    if split == "train":
        d_val = dataset + "_train"
    else:
        d_val = dataset + ("_test" if dataset == "cub" else "_val")
    target_save_name, _, _ = utils.get_save_names(
        a["clip_name"], a["backbone"], a["feature_layer"], d_val, a["concept_set"], "avg", activation_dir)
    if not os.path.exists(target_save_name):
        raise FileNotFoundError(
            f"{target_save_name} not found; run train_cbm.py once for {dataset} so backbone "
            f"features for '{d_val}' are cached, or pass --activation-dir")
    feats = torch.load(target_save_name, map_location="cpu")
    y = torch.LongTensor(data_utils.get_targets_only(d_val))
    if max_rows is not None and feats.shape[0] > max_rows:
        idx = torch.as_tensor(np.sort(np.random.RandomState(seed).choice(feats.shape[0], max_rows, replace=False)))
        feats, y = feats[idx], y[idx]
    feats = feats.float()

    W_c = torch.load(os.path.join(run_dir, "W_c.pt"), map_location="cpu").float()
    proj_mean = torch.load(os.path.join(run_dir, "proj_mean.pt"), map_location="cpu").float()
    proj_std = torch.load(os.path.join(run_dir, "proj_std.pt"), map_location="cpu").float()

    with torch.no_grad():
        proj_c = feats @ W_c.T
        proj_c = (proj_c - proj_mean) / proj_std
    return proj_c, y


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load-dir", nargs="+", required=True,
                    help="Run dir(s) with W_g.pt/W_c.pt/proj_*.pt (glob-able parents ok); averaged over")
    p.add_argument("--activation-dir", default="saved_activations",
                    help="Where the cached backbone target features live")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--analysis", choices=["retention", "threshold", "local", "select-tau"], default="retention",
                    help="retention: contribution-ranked top-k retention_k. "
                         "threshold: sweep the gate open/closed threshold tau (needs gate_logits.pt). "
                         "local: per-image explanation size vs. global union (explanation_size.py)")
    p.add_argument("--objective", choices=["cea", "tol"], default="cea",
                    help="select-tau: pick tau maximising CEA on the selection split, or the largest "
                         "tau within --tol of the full model")
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.25, help="select-tau: CEA exponent")
    p.add_argument("--n-tau", type=int, default=25, help="select-tau: number of candidate thresholds")
    p.add_argument("--no-refit", action="store_true",
                    help="select-tau: do not refit the head on the survivors (hard cut only)")
    p.add_argument("--n-folds", type=int, default=2,
                    help="select-tau: the eval split is cut into this many folds; tau is chosen on one "
                         "and scored on the rest, for every fold (the soft head overfits train, so "
                         "selecting on a train hold-out would keep everything)")
    p.add_argument("--max-fit", type=int, default=None,
                    help="select-tau: subsample the fit split to at most this many examples")
    p.add_argument("--refit-epochs", type=int, default=20)
    p.add_argument("--refit-lr", type=float, default=1e-3)
    p.add_argument("--refit-batch-size", type=int, default=4096)
    p.add_argument("--mass-levels", type=float, nargs="+", default=[0.9],
                    help="local: coverage fractions for k_mass")
    p.add_argument("--n-extreme", type=int, default=20,
                    help="local: how many largest-explanation images to dump with concept names")
    p.add_argument("--gate-temperature", type=float, default=None,
                    help="T in sigmoid(gate_logits / T) for the gate (threshold sweep) or "
                         "the --ranking gate comparison. Default: the run's converged "
                         "temperature (gate_temperature_used in metrics.txt; 1.0 if absent)")
    p.add_argument("--retain-levels", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--ranking", choices=["contribution", "gate", "weight", "random"], default="contribution")
    p.add_argument("--n-random-trials", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--out-dir", default="seed_sweep_results")
    return p.parse_args()


def _trained_gate_temperature(run_dir):
    """The temperature the run's gate converged to (annealed runs end well below the
    starting T), so the sweep sees the same gate values the model was trained with.
    n_open@0.5 is T-invariant (sigmoid(0)=0.5) but every other tau is not."""
    try:
        with open(os.path.join(run_dir, "metrics.txt")) as f:
            return float(json.load(f).get("gate_temperature_used", 1.0))
    except (FileNotFoundError, ValueError, TypeError):
        return 1.0


def main(args):
    run_dirs = _run_dirs(args.load_dir)
    print(f"Averaging retention over {len(run_dirs)} run(s)")

    reports, dataset = [], None
    for i, run_dir in enumerate(run_dirs):
        proj_c, y = load_eval_concepts(run_dir, args.activation_dir, args.device)
        with open(os.path.join(run_dir, "args.txt")) as f:
            dataset = json.load(f)["dataset"]
        W_g = torch.load(os.path.join(run_dir, "W_g.pt"), map_location="cpu").float()
        b_g = torch.load(os.path.join(run_dir, "b_g.pt"), map_location="cpu").float()
        gl_path = os.path.join(run_dir, "gate_logits.pt")
        gate = None
        if os.path.exists(gl_path):
            T = args.gate_temperature if args.gate_temperature is not None else _trained_gate_temperature(run_dir)
            gate = torch.sigmoid(torch.load(gl_path, map_location="cpu").float() / T)
            print(f"  gate temperature T={T:g}")
        if args.analysis in ("threshold", "select-tau") and gate is None:
            raise FileNotFoundError(f"{run_dir} has no gate_logits.pt; {args.analysis} needs an arm-C gated run")

        if args.analysis == "retention":
            rep = retention_curve(
                W_g, b_g, proj_c, y,
                gate=gate, fold_gate=False,  # arm C W_g already has the gate baked in
                column_norm="l2", act_stat="meanabs",
                retain_levels=tuple(args.retain_levels), ranking=args.ranking,
                n_random_trials=args.n_random_trials, random_seed=args.random_seed + i,
            )
            rk = rep["retention_k"]
            print(f"[{os.path.basename(run_dir.rstrip('/'))}] acc_full={rep['acc_full']:.4f}  "
                  + "  ".join(f"k@{lvl:g}={rk[format(lvl, 'g')]}" for lvl in args.retain_levels)
                  + f"  / {rep['n_concepts_total']}")
        elif args.analysis == "local":
            rep = local_explanation_sizes(W_g, b_g, proj_c, y, gate=gate, fold_gate=False,
                                          mass_levels=tuple(args.mass_levels), n_extreme=args.n_extreme)
            print(summary_line(rep, os.path.basename(run_dir.rstrip('/'))[:10]))
        elif args.analysis == "select-tau":
            tr_c, tr_y = load_eval_concepts(run_dir, args.activation_dir, args.device, split="train",
                                            max_rows=args.max_fit, seed=args.random_seed + i)
            print(f"  refit on {len(tr_y)} train; select/eval {args.n_folds}-fold over {len(y)} eval")
            rep = select_tau_crossfit(
                W_g, b_g, gate, fold_gate=False,  # W_g already has the gate folded in
                fit=(tr_c, tr_y), evaluate=(proj_c, y), n_folds=args.n_folds, seed=args.random_seed + i,
                objective=args.objective, tol=args.tol, beta=args.beta,
                n_tau=args.n_tau, refit=not args.no_refit, device=args.device,
                refit_kwargs=dict(epochs=args.refit_epochs, lr=args.refit_lr, batch_size=args.refit_batch_size))
            del tr_c
            print(select_tau_summary_line(rep, os.path.basename(run_dir.rstrip('/'))))
        else:
            rep = threshold_sweep(W_g, b_g, proj_c, y, gate, fold_gate=False)
            print(f"[{os.path.basename(run_dir.rstrip('/'))}] acc_full={rep['acc_full']:.4f}  "
                  f"n_open@0.5={rep['n_open_at']['0.5']}  mushy={rep['bimodal_mushy_frac']:.2f}  "
                  f"plateau_width={rep['plateau_width']:.2f}  n_open(0.3-0.7)={rep['n_open_range_mid']}")
        reports.append(rep)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = {"retention": "concept_retention", "threshold": "gate_threshold_sensitivity",
           "local": "local_explanation_size", "select-tau": "tau_selection"}[args.analysis]

    if args.analysis == "select-tau":
        summary = mean_select_tau(reports)
        summary.update({"dataset": dataset, "load_dirs": run_dirs})
        plot_fn, plot_title = plot_tau_selection, f"LF-CBM threshold selection ({args.objective})"
        tail = select_tau_summary_line(summary, "mean")
    elif args.analysis == "local":
        summary = mean_explanation_report(reports)
        summary.update({"dataset": dataset, "load_dirs": run_dirs})
        plot_fn, plot_title = plot_explanation_sizes, "LF-CBM local explanation size"
        tail = summary_line(summary, "mean")
        concept_names, class_names = None, None
        cpath = os.path.join(run_dirs[0], "concepts.txt")
        if os.path.exists(cpath):
            with open(cpath) as f:
                concept_names = [ln.strip() for ln in f if ln.strip()]
        try:
            with open(data_utils.LABEL_FILES[dataset]) as f:
                class_names = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            class_names = None
        ext_path = os.path.join(out_dir, f"{dataset}_{tag}_extreme_{stamp}.txt")
        with open(ext_path, "w") as f:
            f.write(format_extreme_examples(summary, concept_names, class_names))
        print(f"Saved {len(summary['extreme'])} extreme examples to {ext_path}")
    elif args.analysis == "retention":
        summary = mean_retention(reports)
        summary.update({"dataset": dataset, "load_dirs": run_dirs,
                        "per_seed_retention_k": [r["retention_k"] for r in reports]})
        plot_fn, plot_title = plot_retention_curve, f"LF-CBM concept retention ({args.ranking})"
        tail = f"mean retention_k: {summary['retention_k']}  (+/- {summary['retention_k_std']})"
    else:
        summary = mean_threshold_sweep(reports)
        summary.update({"dataset": dataset, "load_dirs": run_dirs})
        plot_fn, plot_title = plot_threshold_sensitivity, "LF-CBM gate threshold sensitivity"
        tail = (f"mean n_open@0.5={summary['n_open_at']['0.5']:.0f}  "
                f"plateau_width={summary['plateau_width']:.2f}  mushy={summary['bimodal_mushy_frac']:.2f}")
    summary["analysis"] = args.analysis

    json_path = os.path.join(out_dir, f"{dataset}_{tag}_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    plot_path = os.path.join(out_dir, f"{dataset}_{tag}_{stamp}.png")
    plot_fn(summary, plot_path, title=plot_title, dataset_name=dataset)
    print(tail)
    print(f"Saved results to {json_path} and plot to {plot_path}")


if __name__ == "__main__":
    main(parse_args())
