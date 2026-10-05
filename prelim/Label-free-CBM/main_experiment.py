import sys
import os
import json
import datetime
import argparse
import statistics

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gated_probe import GatedProbe, gate_temperature_schedule

from train_cbm_seed_sweep import build_shared_concept_data

"""
Step 4 main experiment: arm A (GLM-SAGA, full regularisation path) vs arm B
(Adam/CE, no gate) vs arm C (Adam/CE + gate, swept over lambda_gate), 5 seeds
per setting, on one dataset per invocation.

Protocol:
  - Concept discovery, filtering, and the W_c projection layer are computed
    once (build_shared_concept_data, imported unmodified from
    train_cbm_seed_sweep.py) -- seed-independent, so untouched by seeding.
  - A stratified 10% holdout is carved out of the *training* split
    (val_frac/val_seed, fixed across all seeds/arms) and used ONLY for
    hyperparameter selection: which point on arm A's GLM-SAGA path counts as
    "best", and which lambda_gate is reported as arm C's headline number.
  - The dataset's actual test split (build_shared_concept_data's "val_c",
    which -- despite the name inherited from the rest of this codebase -- is
    CIFAR100(train=False) / the CUB test split, i.e. genuinely held out) is
    used ONLY for final reported accuracy: the table, both plots. It is never
    used for selection.
  - Arm A's regularisation path is the *actual* glm_saga path: max_lam is
    auto-computed from the training data (no metadata override), and epsilon
    sweeps down to a much weaker regularisation, giving a real range of
    sparsity levels rather than the single hand-picked operating point
    train_cbm.py uses for a production run.
"""


def count_open(weight, tol=1e-5):
    return int((weight.abs() > tol).any(dim=0).sum().item())


def eval_linear(weight, bias, X, y, device):
    with torch.no_grad():
        logits = X.to(device) @ weight.to(device).T + bias.to(device)
        return (logits.argmax(dim=1).cpu() == y).float().mean().item()


def train_arm_b(train_c, train_y, n_classes, epochs, batch_size, device, seed):
    torch.manual_seed(seed)
    probe = torch.nn.Linear(train_c.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_c, train_y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss_fn(probe(X_batch), y_batch).backward()
            optimizer.step()
    return probe.weight.data.cpu(), probe.bias.data.cpu()


def train_arm_c(train_c, train_y, n_classes, lambda_gate, epochs, batch_size, device, seed,
                 gate_temperature=1.0, gate_temperature_final=None):
    torch.manual_seed(seed)
    gate_probe = GatedProbe(train_c.shape[1], n_classes, gate_temperature=gate_temperature).to(device)
    optimizer = torch.optim.Adam(gate_probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_c, train_y), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        gate_probe.gate_temperature = gate_temperature_schedule(epoch, epochs, gate_temperature, gate_temperature_final)
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, gates_batch = gate_probe(X_batch)
            loss = loss_fn(logits, y_batch) + lambda_gate * gates_batch.sum()
            loss.backward()
            optimizer.step()
    gates = gate_probe.gate_probs().detach().cpu()
    W = gate_probe.linear.weight.data.cpu() * gates.unsqueeze(0)
    b = gate_probe.linear.bias.data.cpu()
    n_open = int((gates >= 0.5).sum().item())
    return W, b, n_open


def run_arm_a(train_c, train_y, holdout_c, holdout_y, test_c, test_y, n_classes,
              saga_batch_size, n_iters, path_k, path_epsilon, seed, device):
    torch.manual_seed(seed)
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=saga_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(holdout_c, holdout_y), batch_size=saga_batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_c, test_y), batch_size=saga_batch_size, shuffle=False)

    linear = torch.nn.Linear(train_c.shape[1], n_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    # No metadata override -> max_lam is auto-computed from train data (the
    # real "everything zeroed" endpoint), path_epsilon controls how far down
    # towards near-zero regularisation the path descends.
    out = glm_saga(linear, indexed_train_loader, 0.1, n_iters, 0.99, epsilon=path_epsilon, k=path_k,
                    val_loader=val_loader, test_loader=test_loader, do_zero=False,
                    n_ex=train_c.shape[0], n_classes=n_classes)
    path = out['path']
    for p in path:
        p['n_open'] = count_open(p['weight'])
    best = out['best']
    best_n_open = count_open(best['weight'])
    return path, best, best_n_open


def _mean_std(values):
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def main(args):
    if args.concept_set is None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)
    os.makedirs(args.results_dir, exist_ok=True)

    torch.manual_seed(args.concept_seed)
    shared = build_shared_concept_data(args)
    n_concepts = shared["train_c"].shape[1]
    n_classes = len(shared["classes"])
    test_c, test_y = shared["val_c"], shared["val_y"]  # the dataset's real test split
    print("Shared concept set: {} concepts, {} classes".format(n_concepts, n_classes))

    train_idx, holdout_idx = train_test_split(
        range(len(shared["train_y"])), test_size=args.val_frac, random_state=args.val_seed,
        stratify=shared["train_y"].numpy())
    train_c, train_y = shared["train_c"][train_idx], shared["train_y"][train_idx]
    holdout_c, holdout_y = shared["train_c"][holdout_idx], shared["train_y"][holdout_idx]
    print("train_inner={} holdout={} test={}".format(len(train_idx), len(holdout_idx), len(test_y)))

    seeds = list(range(args.n_seeds))
    seeds_A = list(range(args.n_seeds_A))

    # ---- Arm A: full regularisation path, n_seeds_A seeds (GLM-SAGA path
    # extraction is far more expensive per seed than B/C, so it gets its own,
    # usually smaller, seed count -- reported std reflects fewer seeds). ----
    armA_paths = []
    armA_best_acc, armA_best_open = [], []
    for seed in seeds_A:
        path, best, best_n_open = run_arm_a(
            train_c, train_y, holdout_c, holdout_y, test_c, test_y, n_classes,
            args.saga_batch_size, args.path_n_iters, args.path_k, args.path_epsilon, seed, args.device)
        armA_paths.append(path)
        armA_best_acc.append(best['metrics']['acc_test'])
        armA_best_open.append(best_n_open)
        print("[arm A] seed={} val-selected: test_acc={:.4f} open={}/{}".format(
            seed, best['metrics']['acc_test'], best_n_open, n_concepts))

    # ---- Arm B: 5 seeds ----
    armB_acc = []
    for seed in seeds:
        W, b = train_arm_b(train_c, train_y, n_classes, args.gate_epochs, args.probe_batch_size, args.device, seed)
        acc = eval_linear(W, b, test_c, test_y, args.device)
        armB_acc.append(acc)
        print("[arm B] seed={} test_acc={:.4f}".format(seed, acc))

    # ---- Arm C: sweep lambda_gate x 5 seeds ----
    # sweep[lg] = {'test_acc': [...], 'holdout_acc': [...], 'n_open': [...]}
    sweep = {lg: {'test_acc': [], 'holdout_acc': [], 'n_open': []} for lg in args.lambda_gates}
    for lambda_gate in args.lambda_gates:
        for seed in seeds:
            W, b, n_open = train_arm_c(train_c, train_y, n_classes, lambda_gate,
                                        args.gate_epochs, args.probe_batch_size, args.device, seed,
                                        gate_temperature=args.gate_temperature,
                                        gate_temperature_final=args.gate_temperature_final)
            test_acc = eval_linear(W, b, test_c, test_y, args.device)
            holdout_acc = eval_linear(W, b, holdout_c, holdout_y, args.device)
            sweep[lambda_gate]['test_acc'].append(test_acc)
            sweep[lambda_gate]['holdout_acc'].append(holdout_acc)
            sweep[lambda_gate]['n_open'].append(n_open)
            print("[arm C] lambda_gate={:g} seed={} test_acc={:.4f} holdout_acc={:.4f} open={}/{}".format(
                lambda_gate, seed, test_acc, holdout_acc, n_open, n_concepts))

    # Select the headline lambda_gate by mean HOLDOUT accuracy (never test).
    selected_lg = max(args.lambda_gates, key=lambda lg: statistics.fmean(sweep[lg]['holdout_acc']))
    print("Selected lambda_gate (by holdout acc): {:g}".format(selected_lg))

    # ---- Assemble table row ----
    armA_acc_mean, armA_acc_std = _mean_std(armA_best_acc)
    armA_open_mean, armA_open_std = _mean_std(armA_best_open)
    armB_acc_mean, armB_acc_std = _mean_std(armB_acc)
    armC_acc_mean, armC_acc_std = _mean_std(sweep[selected_lg]['test_acc'])
    armC_open_mean, armC_open_std = _mean_std(sweep[selected_lg]['n_open'])
    deltas = [c - b for c, b in zip(sweep[selected_lg]['test_acc'], armB_acc)]
    delta_mean, delta_std = _mean_std(deltas)

    table_row = {
        'dataset': args.dataset, 'n_concepts': n_concepts, 'n_classes': n_classes,
        'armA_acc_mean': armA_acc_mean, 'armA_acc_std': armA_acc_std,
        'armA_open_mean': armA_open_mean, 'armA_open_std': armA_open_std,
        'armB_acc_mean': armB_acc_mean, 'armB_acc_std': armB_acc_std,
        'armC_selected_lambda_gate': selected_lg,
        'armC_gate_temperature_start': args.gate_temperature, 'armC_gate_temperature_final': args.gate_temperature_final,
        'armC_acc_mean': armC_acc_mean, 'armC_acc_std': armC_acc_std,
        'armC_open_mean': armC_open_mean, 'armC_open_std': armC_open_std,
        'armC_open_pct_of_dict': 100 * armC_open_mean / n_concepts,
        'C_minus_B_delta_mean': delta_mean, 'C_minus_B_delta_std': delta_std,
    }
    print(json.dumps(table_row, indent=2))

    # ---- Sweep plot: test acc + open gates vs lambda_gate, error bars over seeds ----
    lgs = args.lambda_gates
    acc_mean = [statistics.fmean(sweep[lg]['test_acc']) for lg in lgs]
    acc_std = [statistics.pstdev(sweep[lg]['test_acc']) if args.n_seeds > 1 else 0.0 for lg in lgs]
    open_mean = [statistics.fmean(sweep[lg]['n_open']) for lg in lgs]
    open_std = [statistics.pstdev(sweep[lg]['n_open']) if args.n_seeds > 1 else 0.0 for lg in lgs]

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.errorbar(lgs, acc_mean, yerr=acc_std, color="#4C72B0", marker='o', capsize=3, label='Arm C test acc')
    ax1.axhline(armB_acc_mean, color="#555555", linestyle='--', label='Arm B test acc')
    ax1.axhspan(armB_acc_mean - armB_acc_std, armB_acc_mean + armB_acc_std, color="#555555", alpha=0.1)
    ax1.set_xscale('log')
    ax1.set_xlabel('lambda_gate')
    ax1.set_ylabel('test accuracy', color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.errorbar(lgs, open_mean, yerr=open_std, color="#DD8452", marker='D', linestyle='--', capsize=3, label='Arm C open gates')
    ax2.set_ylabel('open gates', color="#DD8452")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.08, 0.5))
    fig.suptitle('{}: arm C sweep vs lambda_gate ({} seeds)'.format(args.dataset, args.n_seeds))
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, '{}_sweep_plot.png'.format(args.dataset)), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Trade-off plot: accuracy vs concept count, arm C sweep + arm A path + arm B line ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    c_open_sorted = sorted(zip(open_mean, acc_mean, acc_std))
    xs = [o for o, _, _ in c_open_sorted]
    ys = [a for _, a, _ in c_open_sorted]
    es = [s for _, _, s in c_open_sorted]
    ax.errorbar(xs, ys, yerr=es, color="#4C72B0", marker='o', capsize=3, label='Arm C (swept lambda_gate)')

    # Average arm A's path across seeds by aligning on path index (each seed's
    # path has the same k and descends the same auto-computed max_lam ->
    # epsilon*max_lam schedule, so index i is the same nominal lambda across
    # seeds).
    path_len = min(len(p) for p in armA_paths)
    a_open = [statistics.fmean([armA_paths[s][i]['n_open'] for s in range(args.n_seeds_A)]) for i in range(path_len)]
    a_acc = [statistics.fmean([armA_paths[s][i]['metrics']['acc_test'] for s in range(args.n_seeds_A)]) for i in range(path_len)]
    a_sorted = sorted(zip(a_open, a_acc))
    ax.plot([o for o, _ in a_sorted], [a for _, a in a_sorted], color="#55A868", marker='s',
            markersize=4, label='Arm A (GLM-SAGA path)')

    ax.axhline(armB_acc_mean, color="#555555", linestyle='--', label='Arm B (no sparsity)')
    ax.axhspan(armB_acc_mean - armB_acc_std, armB_acc_mean + armB_acc_std, color="#555555", alpha=0.1)
    ax.set_xlabel('concept count (open / non-zero-weight)')
    ax.set_ylabel('test accuracy')
    ax.legend()
    fig.suptitle('{}: accuracy vs. concept count'.format(args.dataset))
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, '{}_tradeoff_plot.png'.format(args.dataset)), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Save everything ----
    out = {
        'table_row': table_row,
        'armA_best_per_seed': {'acc': armA_best_acc, 'n_open': armA_best_open},
        'armA_path_per_seed': [[{'lam': float(p['lam']), 'acc_test': p['metrics']['acc_test'], 'n_open': p['n_open']} for p in path]
                                for path in armA_paths],
        'armB_acc_per_seed': armB_acc,
        'armC_sweep': {str(lg): sweep[lg] for lg in lgs},
        'args': vars(args),
    }
    out_path = os.path.join(args.results_dir, '{}_main_experiment_{}.json'.format(
        args.dataset, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')))
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved results to {}'.format(out_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: main experiment (arm A/B/C, 5 seeds)")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--concept_set", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="clip_RN50")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--proj_batch_size", type=int, default=50000)
    parser.add_argument("--feature_layer", type=str, default="layer4")
    parser.add_argument("--activation_dir", type=str, default="saved_activations")
    parser.add_argument("--results_dir", type=str, default="main_experiment_results")
    parser.add_argument("--clip_cutoff", type=float, default=0.25)
    parser.add_argument("--proj_steps", type=int, default=1000)
    parser.add_argument("--proj_patience", type=int, default=1)
    parser.add_argument("--interpretability_cutoff", type=float, default=0.45)
    parser.add_argument("--no_filter", action="store_true")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--gate_epochs", type=int, default=100, help="Epochs for arm B/C probe training")
    parser.add_argument("--gate_temperature", type=float, default=1.0, help="Arm C only: temperature T dividing gate_logits before the sigmoid. T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. Starting temperature of the anneal if --gate_temperature_final is also given.")
    parser.add_argument("--gate_temperature_final", type=float, default=None, help="Arm C only: if given, anneal the gate temperature geometrically from --gate_temperature down (or up) to this value over training, instead of keeping it fixed.")
    parser.add_argument("--lambda-gates", type=float, nargs="+",
                         default=[1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1])
    parser.add_argument("--path_k", type=int, default=8, help="Number of points on arm A's GLM-SAGA regularisation path")
    parser.add_argument("--path_epsilon", type=float, default=0.02, help="min_lam = epsilon * max_lam for arm A's path")
    parser.add_argument("--path_n_iters", type=int, default=50, help="SAGA epoch budget per path point (reduced from train_cbm.py's 1000; points won't fully converge, but the accuracy trend across points is smooth even under this budget -- see timing probe)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Seeds for arms B and C")
    parser.add_argument("--n_seeds_A", type=int, default=2, help="Seeds for arm A (GLM-SAGA path extraction) -- kept lower than n_seeds since each seed costs path_k SAGA fits, far more expensive than B/C")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of the training split held out for hyperparameter selection")
    parser.add_argument("--val_seed", type=int, default=1, help="Fixed seed for the train/holdout split, independent of the n_seeds probe-training seeds")
    parser.add_argument("--concept_seed", type=int, default=0, help="Seed for the (seed-independent-in-principle) concept extraction stage")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
