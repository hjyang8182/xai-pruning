"""Threshold-free concept count for a trained UCBM gated (or baseline) classifier.

Ranks the classifier's concepts by contribution c_j = ||W[:,j]||_2 * mean|a_j| (the
gated post-selection signal fed into the linear head, so the learned gate is already
folded in), keeps only the top-k, and reports the smallest k that retains >= 95%/99%
of the full-model test accuracy. See ../../concept_retention.py for the method.

Operates on the concept-similarity activations already cached to disk by a prior
train_cbm.py run (save/RESULTS/<dataset>-<backbone>/concept_data/<concept_data>/
saved_test_activations/), exactly like sweep_lambda_gate.py -- no backbone forward
passes here. Point --run-dir at a directory containing classifier.pth, or at one
whose immediate subdirectories each contain a classifier.pth (per-seed runs from
train_cbm_seed_sweep.py --save_classifiers); results are averaged over whatever is found.
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

from concept_auxiliaries.concept_auxiliaries import MemTensorDataset  # noqa: E402
from constants import DATA_SETS, MODELS, RESULT_PATH  # noqa: E402
from data_loader import load_data  # noqa: E402
from plotter.plotter import Plotter  # noqa: E402
from ucbm import UCBM  # noqa: E402


def load_cached_activations(act_bank_path, split, n_expected, normalize=False, mean=None, std=None,
                            row_idx=None):
    """Stack the concept-similarity activations train_cbm.py cached for `split`
    ("train"/"test"; mirrors sweep_lambda_gate.load_cached_activations, inlined to
    avoid importing that module -- it pulls in DN-CBM's src.models). With
    `normalize`, pass the *train* mean/std for the test split (as UCBM.fit does);
    left None, MemTensorDataset derives them from the split itself."""
    dset = MemTensorDataset(os.path.join(act_bank_path, f"saved_{split}_activations"),
                            normalize=normalize, mean=mean, std=std)
    if len(dset) != n_expected:
        raise RuntimeError(
            f"Cached {split} activations at {act_bank_path} have {len(dset)} entries, expected "
            f"{n_expected}. Run train_cbm.py once for this dataset/backbone/concept_data first.")
    rows = range(len(dset)) if row_idx is None else row_idx
    return torch.stack([dset[i] for i in rows]), dset


def load_cached_test_activations(act_bank_path, n_expected, normalize=False, mean=None, std=None):
    return load_cached_activations(act_bank_path, "test", n_expected, normalize, mean, std)[0]


def _classifier_paths(run_dir):
    direct = os.path.join(run_dir, "classifier.pth")
    if os.path.exists(direct):
        return [direct]
    nested = sorted(glob.glob(os.path.join(run_dir, "*", "classifier.pth")))
    if not nested:
        raise FileNotFoundError(f"No classifier.pth in {run_dir} or its immediate subdirs")
    return nested


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dataset", type=str, default="cifar100", choices=DATA_SETS)
    p.add_argument("-b", "--backbone", type=str, default="resnet50_v2", choices=MODELS)
    p.add_argument("-c", "--concept_data", type=str, required=True,
                    help="Name of the concept data (must already have cached test activations)")
    p.add_argument("--run-dir", type=str, required=True,
                    help="Dir with classifier.pth (or per-seed subdirs each with one)")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cpu", "cuda"])
    p.add_argument("--normalize_concepts", action="store_true",
                    help="Z-score the cached activations per concept (must match how the run was trained)")
    p.add_argument("--analysis", choices=["retention", "threshold", "local", "select-tau"], default="retention",
                    help="retention: contribution-ranked top-k retention_k. "
                         "threshold: sweep the gate open/closed threshold tau (needs a gated run). "
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
                    help="select-tau: the test split is cut into this many folds; tau is chosen on one "
                         "and scored on the rest, for every fold")
    p.add_argument("--max-fit", type=int, default=None,
                    help="select-tau: subsample the train (refit) split to at most this many examples")
    p.add_argument("--refit-epochs", type=int, default=20)
    p.add_argument("--refit-lr", type=float, default=1e-3)
    p.add_argument("--refit-batch-size", type=int, default=4096)
    p.add_argument("--mass-levels", type=float, nargs="+", default=[0.9],
                    help="local: coverage fractions for k_mass")
    p.add_argument("--n-extreme", type=int, default=20,
                    help="local: how many largest-explanation images to dump")
    p.add_argument("--retain-levels", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--ranking", choices=["contribution", "gate", "weight", "random"], default="contribution")
    p.add_argument("--n-random-trials", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=0)
    return p.parse_args()


def main(args):
    plotter = Plotter(os.path.join(RESULT_PATH, f"{args.dataset}-{args.backbone}/"))
    act_bank_path = os.path.join(plotter.get_concept_data_path(), args.concept_data)
    if not os.path.exists(act_bank_path):
        raise AttributeError(f"Concept bank {args.concept_data} does not exist")

    test_data = load_data(args.dataset, False, reorder_idcs=True)
    test_labels = torch.tensor(test_data.targets)
    train_data = load_data(args.dataset, True, reorder_idcs=True)
    train_labels = torch.tensor(train_data.targets)
    # Normalisation stats come from the train cache (what UCBM.fit used), never the test split.
    train_stats = MemTensorDataset(os.path.join(act_bank_path, "saved_train_activations"),
                                   normalize=True) if args.normalize_concepts else None
    test_acts = load_cached_test_activations(
        act_bank_path, len(test_data), normalize=args.normalize_concepts,
        mean=train_stats.mean if train_stats else None, std=train_stats.std if train_stats else None)
    train_acts = None
    if args.analysis == "select-tau":
        row_idx = None
        if args.max_fit is not None and len(train_data) > args.max_fit:
            row_idx = np.sort(np.random.RandomState(args.random_seed).choice(len(train_data), args.max_fit, replace=False))
            train_labels = train_labels[torch.as_tensor(row_idx)]
        train_acts, _ = load_cached_activations(
            act_bank_path, "train", len(train_data), normalize=args.normalize_concepts,
            mean=train_stats.mean if train_stats else None, std=train_stats.std if train_stats else None,
            row_idx=row_idx)

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(
        plotter.get_classifier_path(), args.concept_data, args.run_dir)
    paths = _classifier_paths(run_dir)
    print(f"Found {len(paths)} classifier(s) under {run_dir}")

    reports = []
    for i, cls_path in enumerate(paths):
        ucbm = UCBM.load_from_file(os.path.dirname(cls_path), "classifier.pth",
                                   device=args.device, backbone_p=torch.nn.Identity())
        clf = ucbm._classifier.eval().to(args.device)
        with torch.no_grad():
            # forward returns (logits, gated, x); `gated` is the post-selection,
            # post-gate signal that actually multiplies clf.linear.weight.
            _, gated, _ = clf(test_acts.to(args.device))
        gated = gated.cpu()

        if args.analysis == "retention":
            rep = retention_curve(
                clf.linear.weight.detach(), clf.linear.bias.detach(),
                gated, test_labels, gate=None,
                column_norm="l2", act_stat="meanabs",
                retain_levels=tuple(args.retain_levels), ranking=args.ranking,
                n_random_trials=args.n_random_trials, random_seed=args.random_seed + i,
            )
            rk = rep["retention_k"]
            print(f"[{os.path.basename(os.path.dirname(cls_path))}] gated={clf.gated} "
                  f"acc_full={rep['acc_full']:.4f}  "
                  + "  ".join(f"k@{lvl:g}={rk[format(lvl, 'g')]}" for lvl in args.retain_levels)
                  + f"  / {rep['n_concepts_total']}")
        elif args.analysis == "local":
            # `gated` is the post-gate signal, so the gate is already in acts: fold_gate=False.
            rep = local_explanation_sizes(
                clf.linear.weight.detach(), clf.linear.bias.detach(), gated, test_labels,
                gate=(clf.gate_probs().detach() if clf.gated else None), fold_gate=False,
                mass_levels=tuple(args.mass_levels), n_extreme=args.n_extreme)
            print(summary_line(rep, os.path.basename(os.path.dirname(cls_path))[:10]))
        elif args.analysis == "select-tau":
            if not clf.gated:
                raise ValueError(f"{cls_path} is a baseline (ungated) classifier; select-tau needs a gated run")
            with torch.no_grad():
                gated_tr = torch.cat([clf(train_acts[j:j + 65536].to(args.device))[1].cpu()
                                      for j in range(0, len(train_acts), 65536)])
            print(f"  refit on {len(train_labels)} train; select/eval {args.n_folds}-fold over {len(test_labels)} test")
            # `gated` already has the gate baked in (fold_gate=False); the gate vector only picks columns.
            rep = select_tau_crossfit(
                clf.linear.weight.detach(), clf.linear.bias.detach(), clf.gate_probs().detach(),
                fold_gate=False, fit=(gated_tr, train_labels), evaluate=(gated, test_labels),
                n_folds=args.n_folds, seed=args.random_seed + i,
                objective=args.objective, tol=args.tol, beta=args.beta,
                n_tau=args.n_tau, refit=not args.no_refit, device=args.device,
                refit_kwargs=dict(epochs=args.refit_epochs, lr=args.refit_lr, batch_size=args.refit_batch_size))
            del gated_tr
            print(select_tau_summary_line(rep, os.path.basename(os.path.dirname(cls_path))))
        else:
            if not clf.gated:
                raise ValueError(f"{cls_path} is a baseline (ungated) classifier; threshold sweep needs a gated run")
            # `gated` already has the gate baked in, so fold_gate=False and the gate
            # vector is used only to pick which columns each tau closes.
            rep = threshold_sweep(clf.linear.weight.detach(), clf.linear.bias.detach(),
                                  gated, test_labels, clf.gate_probs().detach(), fold_gate=False)
            print(f"[{os.path.basename(os.path.dirname(cls_path))}] acc_full={rep['acc_full']:.4f}  "
                  f"n_open@0.5={rep['n_open_at']['0.5']}  mushy={rep['bimodal_mushy_frac']:.2f}  "
                  f"plateau_width={rep['plateau_width']:.2f}  n_open(0.3-0.7)={rep['n_open_range_mid']}")
        reports.append(rep)

    out_dir = os.path.join(plotter.get_classifier_path(), args.concept_data, "concept_retention_results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common = {"dataset": args.dataset, "backbone": args.backbone,
              "concept_data": args.concept_data, "run_dir": run_dir, "analysis": args.analysis}
    tag = {"retention": "concept_retention", "threshold": "gate_threshold_sensitivity",
           "local": "local_explanation_size", "select-tau": "tau_selection"}[args.analysis]

    if args.analysis == "select-tau":
        summary = mean_select_tau(reports)
        summary.update(common)
        plot_fn, plot_title = plot_tau_selection, f"UCBM threshold selection ({args.objective})"
        tail = select_tau_summary_line(summary, "mean")
    elif args.analysis == "local":
        summary = mean_explanation_report(reports)
        summary.update(common)
        plot_fn, plot_title = plot_explanation_sizes, "UCBM local explanation size"
        tail = summary_line(summary, "mean")
        # UCBM concepts are unsupervised (unnamed): dump indices + class names only
        ext_path = os.path.join(out_dir, f"{args.dataset}_{tag}_extreme_{stamp}.txt")
        with open(ext_path, "w") as f:
            f.write(format_extreme_examples(summary, None, list(getattr(test_data, "classes", []))))
        print(f"Saved {len(summary['extreme'])} extreme examples to {ext_path}")
    elif args.analysis == "retention":
        summary = mean_retention(reports)
        summary.update(common)
        summary["per_seed_retention_k"] = [r["retention_k"] for r in reports]
        plot_fn, plot_title = plot_retention_curve, f"UCBM concept retention ({args.ranking})"
        tail = f"mean retention_k: {summary['retention_k']}  (+/- {summary['retention_k_std']})"
    else:
        summary = mean_threshold_sweep(reports)
        summary.update(common)
        plot_fn, plot_title = plot_threshold_sensitivity, "UCBM gate threshold sensitivity"
        tail = (f"mean n_open@0.5={summary['n_open_at']['0.5']:.0f}  "
                f"plateau_width={summary['plateau_width']:.2f}  mushy={summary['bimodal_mushy_frac']:.2f}")

    json_path = os.path.join(out_dir, f"{args.dataset}_{tag}_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    plot_path = os.path.join(out_dir, f"{args.dataset}_{tag}_{stamp}.png")
    plot_fn(summary, plot_path, title=plot_title, dataset_name=f"{args.dataset}-{args.backbone}")
    print(tail)
    print(f"Saved results to {json_path} and plot to {plot_path}")


if __name__ == "__main__":
    main(parse_args())
