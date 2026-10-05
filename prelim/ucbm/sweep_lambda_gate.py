"""
Sweep lambda_gate for UCBM's learned per-concept gate and compare against the
ungated baseline, using CEA (Concept-Efficient Accuracy, Zhao et al. PS-CBM)
exactly as the DN-CBM sweep (../../sweep_lambda_gate.py) does it: CEA trades
accuracy off against how many concepts the model actually exposes, so a
gated run isn't just compared on accuracy but on accuracy-per-concept-used.

This mirrors the DN-CBM sweep's structure at the activation-tensor level
rather than going through UCBM.fit()/train_cbm.py: it operates directly on
the concept-similarity activations already cached to disk by a prior
train_cbm.py run (save/RESULTS/{dataset}-{backbone}/concept_data/{concept_data}/
saved_{train,test}_activations/), so no backbone forward passes are needed
here. A stratified validation split is carved out of the cached train
activations (mirroring the split scheme in train_cbm.py/sweep_lambda_gate.py
for DN-CBM) so lambda_gate is picked without touching the test set; test
accuracy is reported for completeness but never used for selection.

The per-batch loss composition below is intentionally a straight copy of
ucbm.py::UCBM.fit()'s non-cocostuff branch (lam_pi elastic-net on pi(x),
lam_w elastic-net on the linear weights, and -- only when gated -- lambda_gate
times sum(sigmoid(gate_logits)) added on top). If that loss changes in
ucbm.py, mirror the change here too.

Only single-label datasets are supported (CrossEntropyLoss); UCBM's
multilabel branch (e.g. cocostuff) isn't handled by this script.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from concept_auxiliaries.concept_auxiliaries import MemTensorDataset
from constants import DATA_SETS, MODELS, RESULT_PATH
from data_loader import load_data
from plotter.plotter import Plotter
from ucbm import Classifier, elastic_loss_activations, elastic_loss_weights, l0_loss, gate_temperature_schedule


def _mean_std(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def compute_cea(num_classes, num_concepts, acc, beta=0.25):
    # Concept-Efficient Accuracy, Zhao et al. (PS-CBM) -- identical formula to
    # src/metrics.py::compute_cea at the top level.
    k = np.ceil(np.log2(num_classes))
    return acc / (np.log(num_concepts) / np.log(k)) ** beta


def load_cached_activations(saved_activation_path, data_label, n_expected, normalize=False, mean=None, std=None):
    dset = MemTensorDataset(
        os.path.join(saved_activation_path, f'saved_{data_label}_activations'),
        normalize=normalize, mean=mean, std=std,
    )
    if len(dset) != n_expected:
        raise RuntimeError(
            f"Cached '{data_label}' activations at {saved_activation_path} have {len(dset)} entries, "
            f"expected {n_expected}. Run train_cbm.py once for this dataset/backbone/concept_data first "
            f"so the concept-similarity cache is fully populated.")
    return torch.stack([dset[i] for i in range(len(dset))]), dset


def few_shot_indices(labels, shots, seed):
    '''Indices selecting up to `shots` examples per class from `labels`
    (a 1-D LongTensor), drawn with a seed-derived RandomState so the
    subsample is reproducible and varies with the seed. Classes with fewer
    than `shots` examples are kept in full.'''
    rng = np.random.RandomState(seed)
    picked = []
    for c in torch.unique(labels).tolist():
        c_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        if len(c_idx) > shots:
            c_idx = rng.choice(c_idx, size=shots, replace=False)
        picked.append(c_idx)
    return np.sort(np.concatenate(picked))


def train_classifier(train_acts, train_labels, eval_acts, eval_labels, n_classes,
                      relu, scale_mode, bias_mode, dropout_p, k, gated,
                      lam_pi, lam_w, lambda_gate, lr, epochs, batch_size, device,
                      gate_temperature=1.0, gate_temperature_final=None, gate_forward="soft"):
    classifier = Classifier(
        train_acts.shape[1], n_classes, relu, scale_mode, bias_mode, dropout_p, k, gated=gated,
        gate_temperature=gate_temperature, gate_forward=gate_forward,
    ).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_acts, train_labels), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        classifier.train()
        if gated:
            classifier.gate_temperature = gate_temperature_schedule(
                epoch, epochs, gate_temperature, gate_temperature_final)
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred, after_gate, before_gate = classifier(X_batch)
            loss = loss_fn(y_pred, y_batch)

            if lam_pi != 0:
                if relu != "jumpReLU":
                    loss = loss + lam_pi * elastic_loss_activations(after_gate)
                else:
                    loss = loss + lam_pi * l0_loss(
                        before_gate, classifier._jumpReLU.log_threshold.exp(), classifier._jumpReLU.bandwidth)

            if lam_w != 0:
                loss = loss + lam_w * elastic_loss_weights(classifier.linear.weight)
            if gated and lambda_gate != 0:
                loss = loss + lambda_gate * classifier.gate_probs().sum()

            loss.backward()
            optimizer.step()
        if not gated:
            lr_scheduler.step()

    classifier.eval()
    with torch.no_grad():
        y_pred, _, _ = classifier(eval_acts.to(device))
        acc = (y_pred.argmax(dim=1) == eval_labels.to(device)).float().mean().item()

    if gated:
        n_open = int((classifier.gate_probs() > 0.5).sum().item())
    else:
        weight = classifier.linear.weight.data
        n_open = int((weight.abs() > 1e-5).any(dim=0).sum().item())

    return acc, classifier, n_open


def parse_args():
    parser = argparse.ArgumentParser(description='Sweep lambda_gate for UCBM and plot accuracy / gate sparsity vs. baseline, using CEA')
    parser.add_argument("-d", "--dataset", type=str, default="cifar100", choices=DATA_SETS)
    parser.add_argument("-b", "--backbone", type=str, default="resnet50_v2", choices=MODELS)
    parser.add_argument("-c", "--concept_data", type=str, required=True,
                         help="Name of the concept data to use (must already have cached train/test activations from a prior train_cbm.py run)")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cpu", "cuda"])

    parser.add_argument("--lambda-gates", type=float, nargs='+',
                         default=[1e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 5e-3, 1e-2])
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 10, 20],
                         help='Seeds to train each lambda_gate (and the baseline) with; results are averaged (mean/std) across these')
    parser.add_argument("--val-frac", type=float, default=0.1,
                         help='Fraction of the training set held out as a validation set for the sweep')
    parser.add_argument("--val-seed", type=int, default=1,
                         help='Seed used to carve out the validation split; kept fixed across lambda_gates/seeds so all runs are tuned on the same split')
    parser.add_argument("--train-shots", type=int, default=-1,
                         help='If > 0, sub-sample the probe training set to this many examples per class (few-shot) '
                              'before training every probe. The validation split is left full-size. On CIFAR100 (and '
                              'other easy-for-a-linear-probe datasets) the full-data probe saturates val accuracy at '
                              '~100%% regardless of how many gates are open, so CEA just rewards sparsity; starving the '
                              'probe of data makes val accuracy actually respond to the open-gate count. The subsample '
                              'is drawn per seed (so its variance shows up in the across-seed std) but is identical for '
                              "the baseline and every lambda_gate at that seed. -1 disables (full training set).")
    parser.add_argument("--beta", type=float, default=0.25,
                         help='Beta used for CEA; lambda_gate is selected by CEA, not raw val accuracy, so it directly trades off val accuracy against open-gate count')

    parser.add_argument("--normalize_concepts", action="store_true",
                         help="Z-score-normalize the cached concept-similarity activations per concept, "
                              "using train-set mean/std -- mirrors ucbm.py's UCBM(normalize=...). Must match "
                              "whatever --normalize_concepts setting train_cbm.py used to build this concept_data "
                              "cache, since the .pth files store raw (unnormalized) cosine similarities either way.")
    parser.add_argument("--relu", type=str, default="ReLU", choices=["no", "ReLU", "jumpReLU"])
    parser.add_argument("--scale_choose", type=str, default="learn", choices=["learn", "no"])
    parser.add_argument("--bias_choose", type=str, default="learn", choices=["learn", "no"])
    parser.add_argument("--k", type=int, default=-1, help="Top K concepts to keep, -1 to disable")
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--lam_pi", type=float, default=1e-4,
                         help="Elastic-net regularization strength on the concept selector's output pi(x), independent of gating")
    parser.add_argument("--lam_w", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gate_forward", choices=["soft", "hard"], default="soft",
                         help="gated arm: 'soft' multiplies by sigmoid(gate); 'hard' uses the mask 1[sigmoid>0.5] in the "
                              "forward pass with a straight-through gradient (same flag as train_cbm_seed_sweep.py)")
    parser.add_argument("--gate_temperature", type=float, default=1.0,
                         help="Temperature T dividing gate_logits before the sigmoid, for every gated run in the "
                              "sweep. T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. "
                              "Starting temperature of the anneal if --gate_temperature_final is also given.")
    parser.add_argument("--gate_temperature_final", type=float, default=None,
                         help="If given, anneal the gate temperature geometrically from --gate_temperature down "
                              "(or up) to this value over each gated run's training, instead of keeping it fixed.")
    return parser.parse_args()


def main(args):
    device = args.device
    plotter = Plotter(os.path.join(RESULT_PATH, f'{args.dataset}-{args.backbone}/'))
    act_bank_path = os.path.join(plotter.get_concept_data_path(), args.concept_data)
    if not os.path.exists(act_bank_path):
        raise AttributeError(f"Concept bank {args.concept_data} does not exist")

    train_data = load_data(args.dataset, True, reorder_idcs=True)
    test_data = load_data(args.dataset, False, reorder_idcs=True)
    n_classes = len(train_data.classes)
    assert not isinstance(train_data.targets[0], list), \
        "This sweep only supports single-label datasets (CrossEntropyLoss)."

    # Mean/std for normalization are computed once over the *entire* cached train set
    # (matching train_cbm.py/ucbm.py's UCBM.fit(), which normalizes using train-set
    # statistics before any validation split is carved out) and then reused as-is for
    # the test set, never recomputed on it -- mirrors UCBM.predict()/get_evaluation_metric().
    train_acts_full, train_dset = load_cached_activations(
        act_bank_path, "train", len(train_data), normalize=args.normalize_concepts)
    train_mean = train_dset.mean if args.normalize_concepts else None
    train_std = train_dset.std if args.normalize_concepts else None
    test_acts, _ = load_cached_activations(
        act_bank_path, "test", len(test_data), normalize=args.normalize_concepts,
        mean=train_mean, std=train_std)
    train_labels_full = torch.tensor(train_data.targets)
    test_labels = torch.tensor(test_data.targets)
    n_concepts_total = train_acts_full.shape[1]

    train_idx, val_idx = train_test_split(
        range(len(train_acts_full)), test_size=args.val_frac, random_state=args.val_seed,
        stratify=train_labels_full,
    )
    train_acts,  train_labels  = train_acts_full[train_idx], train_labels_full[train_idx]
    val_acts,    val_labels    = train_acts_full[val_idx],   train_labels_full[val_idx]

    # Few-shot: precompute one per-class subsample of the (post-val-split) train
    # pool per seed, shared by the baseline and every lambda_gate at that seed so
    # they're compared on identical data. None => use the full train pool.
    if args.train_shots > 0:
        shot_idx_by_seed = {s: few_shot_indices(train_labels, args.train_shots, s) for s in args.seeds}
        print(f"Few-shot: {args.train_shots} examples/class -> "
              f"{len(next(iter(shot_idx_by_seed.values())))} train rows "
              f"(from {len(train_acts)}); val split left at {len(val_acts)} rows")
    else:
        shot_idx_by_seed = None

    def seed_train_data(seed):
        if shot_idx_by_seed is None:
            return train_acts, train_labels
        idx = shot_idx_by_seed[seed]
        return train_acts[idx], train_labels[idx]

    common_kwargs = dict(
        relu=args.relu, scale_mode=args.scale_choose, bias_mode=args.bias_choose,
        dropout_p=args.dropout_p, k=args.k, lam_pi=args.lam_pi, lam_w=args.lam_w,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=device,
        gate_temperature=args.gate_temperature, gate_temperature_final=args.gate_temperature_final,
        gate_forward=args.gate_forward,
    )

    print(f"=== baseline (ungated), {len(args.seeds)} seed(s) ===")
    baseline_val_accs, baseline_test_accs, baseline_ceas, baseline_n_opens = [], [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        seed_train_acts, seed_train_labels = seed_train_data(seed)
        val_acc, classifier, n_open_seed = train_classifier(
            seed_train_acts, seed_train_labels, val_acts, val_labels, n_classes,
            gated=False, lambda_gate=0.0, **common_kwargs,
        )
        classifier.eval()
        with torch.no_grad():
            test_pred = classifier(test_acts.to(device))[0].argmax(dim=1).cpu()
        test_acc = (test_pred == test_labels).float().mean().item()
        # Baseline CEA used to always divide by the full dictionary size
        # (n_concepts_total), on the assumption that "baseline" means no
        # selector restricts the concept space. But common_kwargs threads
        # relu/k through to the baseline the same as every gated run, so
        # when --k is set the baseline is a TopK(k) selector, not a raw
        # pass-through -- using n_concepts_total there overstated its concept
        # count. train_classifier already returns the same nonzero-weight-
        # column count used elsewhere (see ucbm.py's get_info_dict) as
        # n_open_seed; use that so a restricted baseline's CEA reflects what
        # it actually exposes.
        cea = compute_cea(n_classes, max(n_open_seed, 1), val_acc, args.beta)
        print(f"[baseline] seed={seed} val_acc={val_acc:.4f} test_acc={test_acc:.4f} open_concepts={n_open_seed}/{n_concepts_total}")
        baseline_val_accs.append(val_acc)
        baseline_test_accs.append(test_acc)
        baseline_ceas.append(cea)
        baseline_n_opens.append(n_open_seed)
    baseline_val_acc,  baseline_val_acc_std  = _mean_std(baseline_val_accs)
    baseline_test_acc, baseline_test_acc_std = _mean_std(baseline_test_accs)
    baseline_cea,      baseline_cea_std      = _mean_std(baseline_ceas)
    baseline_n_open,   baseline_n_open_std   = _mean_std(baseline_n_opens)

    accs, accs_std = [], []
    test_accs, test_accs_std = [], []
    n_open, n_open_std = [], []
    cea_list, cea_std = [], []
    accs_seeds, test_accs_seeds, n_open_seeds, cea_seeds = [], [], [], []

    for lambda_gate in args.lambda_gates:
        print(f"=== lambda_gate={lambda_gate}, {len(args.seeds)} seed(s) ===")
        seed_accs, seed_test_accs, seed_n_open, seed_cea = [], [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed)
            seed_train_acts, seed_train_labels = seed_train_data(seed)
            val_acc, classifier, n_open_seed = train_classifier(
                seed_train_acts, seed_train_labels, val_acts, val_labels, n_classes,
                gated=True, lambda_gate=lambda_gate, **common_kwargs,
            )
            classifier.eval()
            with torch.no_grad():
                test_pred = classifier(test_acts.to(device))[0].argmax(dim=1).cpu()
            test_acc = (test_pred == test_labels).float().mean().item()
            cea = compute_cea(n_classes, max(n_open_seed, 1), val_acc, args.beta)
            print(f"[gated] seed={seed} val_acc={val_acc:.4f} test_acc={test_acc:.4f} open_gates={n_open_seed}/{n_concepts_total}")
            seed_accs.append(val_acc)
            seed_test_accs.append(test_acc)
            seed_n_open.append(n_open_seed)
            seed_cea.append(cea)

        mean, std = _mean_std(seed_accs);      accs.append(mean);      accs_std.append(std)
        mean, std = _mean_std(seed_test_accs); test_accs.append(mean); test_accs_std.append(std)
        mean, std = _mean_std(seed_n_open);    n_open.append(mean);    n_open_std.append(std)
        mean, std = _mean_std(seed_cea);       cea_list.append(mean);  cea_std.append(std)
        accs_seeds.append(seed_accs)
        test_accs_seeds.append(seed_test_accs)
        n_open_seeds.append(seed_n_open)
        cea_seeds.append(seed_cea)

    summary = {
        'dataset': args.dataset, 'backbone': args.backbone, 'concept_data': args.concept_data,
        'n_concepts_total': n_concepts_total, 'n_classes': n_classes,
        'lambda_gates': args.lambda_gates, 'seeds': args.seeds, 'beta': args.beta,
        'train_shots': args.train_shots, 'val_frac': args.val_frac,
        'gate_forward': args.gate_forward, 'epochs': args.epochs, 'lam_pi': args.lam_pi, 'lam_w': args.lam_w,
        'val_accuracy': accs, 'val_accuracy_std': accs_std, 'val_accuracy_per_seed': accs_seeds,
        'test_accuracy': test_accs, 'test_accuracy_std': test_accs_std, 'test_accuracy_per_seed': test_accs_seeds,
        'open_gates': n_open, 'open_gates_std': n_open_std, 'open_gates_per_seed': n_open_seeds,
        'cea': cea_list, 'cea_std': cea_std, 'cea_per_seed': cea_seeds,
        'baseline_val_accuracy': baseline_val_acc, 'baseline_val_accuracy_std': baseline_val_acc_std,
        'baseline_test_accuracy': baseline_test_acc, 'baseline_test_accuracy_std': baseline_test_acc_std,
        'baseline_cea': baseline_cea, 'baseline_cea_std': baseline_cea_std,
        'baseline_open_concepts': baseline_n_open, 'baseline_open_concepts_std': baseline_n_open_std,
    }

    results_dir = os.path.join(plotter.get_classifier_path(), args.concept_data, "lambda_gate_sweep_results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{args.dataset}_lambda_gate_sweep_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_path}")

    plotter.plot_lambda_gate_sweep(
        args.lambda_gates, cea_list, n_open, args.dataset, args.backbone,
        accs_std=cea_std, n_open_std=n_open_std,
        baseline_acc=baseline_cea, baseline_acc_std=baseline_cea_std,
        metric_name='CEA', split_label='val', save_dir=results_dir,
        accs2=accs, accs2_std=accs_std,
        baseline_acc2=baseline_val_acc, baseline_acc2_std=baseline_val_acc_std,
        metric2_name='Accuracy',
    )
    plotter.plot_lambda_gate_sweep(
        args.lambda_gates, test_accs, n_open, args.dataset, args.backbone,
        accs_std=test_accs_std, n_open_std=n_open_std,
        baseline_acc=baseline_test_acc, baseline_acc_std=baseline_test_acc_std,
        metric_name='Accuracy', split_label='test', save_dir=results_dir,
    )

    # Plotted separately from CEA: accuracy vs. open-gate count directly, so a
    # saturated validation split (all lambda_gates tying on accuracy) is
    # visible as flat points rather than being hidden inside a CEA ratio that
    # would still show a "winner" driven purely by sparsity.
    plotter.plot_pareto_frontier(
        n_open, accs, args.lambda_gates, args.dataset, args.backbone,
        open_gates_std=n_open_std, accs_std=accs_std,
        baseline_open_gates=baseline_n_open, baseline_acc=baseline_val_acc,
        baseline_acc_std=baseline_val_acc_std,
        metric_name='Accuracy', split_label='val', save_dir=results_dir,
    )
    plotter.plot_pareto_frontier(
        n_open, test_accs, args.lambda_gates, args.dataset, args.backbone,
        open_gates_std=n_open_std, accs_std=test_accs_std,
        baseline_open_gates=baseline_n_open, baseline_acc=baseline_test_acc,
        baseline_acc_std=baseline_test_acc_std,
        metric_name='Accuracy', split_label='test', save_dir=results_dir,
    )


if __name__ == "__main__":
    main(parse_args())
