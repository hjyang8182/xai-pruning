import argparse
import json
import os
import torch
from sklearn.model_selection import train_test_split
from src import config
from src.models import load_autoencoder, load_clip
from src.data import load_cifar100, load_food101, load_cub, load_oxford_pet, save_clip_features, load_clip_features, get_sae_acts
from src.train import train_probe, train_gated_probe, get_predictions, get_gated_predictions
from src.metrics import compute_cea
from gating import prune_and_refit, masked_accuracy, evaluate_head
from src.visualise import plot_lambda_gate_sweep
from train_cbm import DATASET_LOADERS, N_CLASSES

def _mean_std(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description='Sweep lambda_gate and plot resulting accuracy / gate sparsity')
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS.keys(), default='cifar100')
    parser.add_argument('--lambda-gates', type=float, nargs='+',
                         default=[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.PROBE_BATCH_SIZE)
    parser.add_argument('--lambda-sparse', type=float, default=config.LAMBDA_SPARSE)
    parser.add_argument('--gate-forward', choices=['soft', 'hard'], default='soft',
                         help='"soft" multiplies concepts by sigmoid(gate); "hard" uses the mask 1[sigmoid > 0.5] '
                              'with a straight-through gradient (see train_cbm.py). A hard sweep is written to '
                              'lambda_gate_sweep_hard.json so it does not overwrite the soft one')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20, 30, 40],
                         help='Seeds to train each lambda_gate with; results are averaged (mean/std) across these')
    parser.add_argument('--val-frac', type=float, default=0.1,
                         help='Fraction of the training set held out as a validation set for the sweep')
    parser.add_argument('--val-seed', type=int, default=config.SEED,
                         help='Seed used to carve out the validation split; kept fixed across lambda_gates/seeds so all runs are tuned on the same split')
    parser.add_argument('--refit-tau', type=float, default=0.5,
                         help='After training, keep concepts with gate > tau, zero the rest, and refit the linear '
                              'head on the survivors (same mask-and-refit stage as train_cbm.py). The "refit" '
                              'accuracy / CEA is that of the delivered hard-masked model')
    parser.add_argument('--refit-epochs', type=int, default=20)
    parser.add_argument('--no-refit', action='store_true', help='Skip the mask-and-refit stage')
    parser.add_argument('--beta', type=float, default=0.25,
                         help='Beta used for CEA (Concept-efficient Accuracy, Zhao et al. PS-CBM); lambda_gate is selected by CEA, not raw val accuracy, so it directly trades off val accuracy against open-gate count')
    return parser.parse_args()


def main(args):
    dataset   = args.dataset
    n_classes = N_CLASSES[dataset]
    SAVE      = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE  = os.path.join(SAVE, 'activations')
    os.makedirs(ACT_SAVE, exist_ok=True)

    train_dataset, test_dataset, _ = DATASET_LOADERS[dataset]()

    if not os.path.exists(os.path.join(ACT_SAVE, 'train_clip_features.pt')):
        clip_model, _ = load_clip()
        save_clip_features(clip_model, train_dataset, test_dataset, ACT_SAVE)
    train_feats, train_labels = load_clip_features(ACT_SAVE, 'train')
    test_feats,  test_labels  = load_clip_features(ACT_SAVE, 'test')

    train_acts_path = os.path.join(ACT_SAVE, 'train_sae_acts.pt')
    test_acts_path  = os.path.join(ACT_SAVE, 'test_sae_acts.pt')
    if not os.path.exists(train_acts_path):
        autoencoder = load_autoencoder(config.sae_device)
        train_acts = get_sae_acts(autoencoder, train_feats, config.sae_device)
        test_acts  = get_sae_acts(autoencoder, test_feats,  config.sae_device)
        torch.save(train_acts, train_acts_path)
        torch.save(test_acts,  test_acts_path)
    train_acts = torch.load(train_acts_path)
    test_acts  = torch.load(test_acts_path)

    # Carve a stratified validation split out of the training set to tune lambda_gate on, so
    # the held-out test set isn't used for model/hyperparameter selection.
    n = train_acts.shape[0]
    train_idx, val_idx = train_test_split(
        range(n), test_size=args.val_frac, random_state=args.val_seed, stratify=train_labels,
    )
    val_acts,   val_labels   = train_acts[val_idx],   train_labels[val_idx]
    train_acts, train_labels = train_acts[train_idx], train_labels[train_idx]

    baseline_accs_seeds, baseline_test_accs_seeds, baseline_cea_seeds = [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        acc, baseline_probe = train_probe(
            train_acts, train_labels, val_acts, val_labels, n_classes=n_classes,
            lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
            lambda_sparse=args.lambda_sparse,
        )
        test_preds = get_predictions(baseline_probe, test_acts)
        baseline_accs_seeds.append(acc)
        baseline_test_accs_seeds.append((test_preds == test_labels).float().mean().item())
        baseline_cea_seeds.append(compute_cea(n_classes, config.N_LEARNED_FEATURES, acc, args.beta))
    baseline_acc,      baseline_acc_std      = _mean_std(baseline_accs_seeds)
    baseline_test_acc, baseline_test_acc_std = _mean_std(baseline_test_accs_seeds)
    baseline_cea,      baseline_cea_std      = _mean_std(baseline_cea_seeds)

    # lambda_gate is selected using val CEA (Concept-efficient Accuracy, Zhao et al. PS-CBM;
    # see src/metrics.py) rather than raw val accuracy, since CEA trades val accuracy off against
    # open-gate count directly instead of picking the largest open set that doesn't hurt accuracy.
    # Test accuracy/CEA are computed purely for reporting and never used to pick a lambda_gate.
    # Each lambda_gate is trained across multiple seeds so the plots can show mean +/- std.
    accs, accs_std = [], []
    test_accs, test_accs_std = [], []
    n_open, n_open_std = [], []
    cea, cea_std = [], []
    accs_seeds, test_accs_seeds, n_open_seeds, cea_seeds = [], [], [], []
    # Mask-and-refit stage (gating.prune_and_refit): the deliverable at each lambda_gate is the
    # |K| = #(gate > tau) surviving concepts with the head refit on them, so record its val / test
    # accuracy and CEA alongside the soft-gated numbers. `masked_*` is the hard cut *before* the
    # refit, i.e. how far the soft gate is from a real mask.
    refit = {k: [] for k in ('masked_val', 'masked_test', 'refit_val', 'refit_test', 'refit_cea', 'refit_n')}
    refit_seeds = {k: [] for k in refit}
    for lambda_gate in args.lambda_gates:
        seed_accs, seed_test_accs, seed_n_open, seed_cea = [], [], [], []
        seed_refit = {k: [] for k in refit}
        for seed in args.seeds:
            torch.manual_seed(seed)
            acc, probe = train_gated_probe(
                train_acts, train_labels, val_acts, val_labels, n_classes=n_classes,
                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                lambda_sparse=args.lambda_sparse, lambda_gate=lambda_gate,
                gate_forward=args.gate_forward,
            )
            gates = torch.sigmoid(probe.gate_logits).detach()
            test_preds = get_gated_predictions(probe, test_acts)
            n_open_seed = (gates > 0.5).sum().item()
            seed_accs.append(acc)
            seed_test_accs.append((test_preds == test_labels).float().mean().item())
            seed_n_open.append(n_open_seed)
            seed_cea.append(compute_cea(n_classes, n_open_seed, acc, args.beta))

            if not args.no_refit:
                # GatedProbe stores W ungated (gate applied in forward) -> fold_gate=True.
                gate = probe.gate_probs().detach()
                W_lin, b_lin = probe.linear.weight, probe.linear.bias
                W_r, b_r, keep = prune_and_refit(
                    W_lin, b_lin, gate, train_acts, train_labels, tau=args.refit_tau, fold_gate=True,
                    epochs=args.refit_epochs, lr=args.lr, batch_size=args.batch_size,
                    device=config.device, seed=seed)
                n_keep = int(keep.sum())
                refit_val = evaluate_head(W_r, b_r, val_acts, val_labels, device=config.device)
                seed_refit['masked_val'].append(masked_accuracy(W_lin, b_lin, gate, val_acts, val_labels, tau=args.refit_tau))
                seed_refit['masked_test'].append(masked_accuracy(W_lin, b_lin, gate, test_acts, test_labels, tau=args.refit_tau))
                seed_refit['refit_val'].append(refit_val)
                seed_refit['refit_test'].append(evaluate_head(W_r, b_r, test_acts, test_labels, device=config.device))
                seed_refit['refit_cea'].append(compute_cea(n_classes, n_keep, refit_val, args.beta))
                seed_refit['refit_n'].append(n_keep)
                print(f'[lambda_gate {lambda_gate:g} seed {seed}] {n_keep} concepts @tau={args.refit_tau}: '
                      f'soft test {seed_test_accs[-1]:.4f}, hard cut {seed_refit["masked_test"][-1]:.4f} '
                      f'-> refit {seed_refit["refit_test"][-1]:.4f}')

        for k in refit:
            if seed_refit[k]:
                mean, std = _mean_std(seed_refit[k]); refit[k].append((mean, std)); refit_seeds[k].append(seed_refit[k])

        mean, std = _mean_std(seed_accs);      accs.append(mean);      accs_std.append(std)
        mean, std = _mean_std(seed_test_accs); test_accs.append(mean); test_accs_std.append(std)
        mean, std = _mean_std(seed_n_open);    n_open.append(mean);    n_open_std.append(std)
        mean, std = _mean_std(seed_cea);       cea.append(mean);       cea_std.append(std)
        accs_seeds.append(seed_accs)
        test_accs_seeds.append(seed_test_accs)
        n_open_seeds.append(seed_n_open)
        cea_seeds.append(seed_cea)

    suffix = '' if args.gate_forward == 'soft' else f'_{args.gate_forward}'
    results_path = os.path.join(SAVE, f'lambda_gate_sweep{suffix}.json')
    refit_out = {}
    if not args.no_refit:
        refit_out['refit_tau'] = args.refit_tau
        refit_out['refit_epochs'] = args.refit_epochs
        for k, name in (('masked_val', 'masked_accuracy'), ('masked_test', 'masked_test_accuracy'),
                        ('refit_val', 'refit_accuracy'), ('refit_test', 'refit_test_accuracy'),
                        ('refit_cea', 'refit_cea'), ('refit_n', 'refit_n_concepts')):
            refit_out[name] = [m for m, _ in refit[k]]
            refit_out[f'{name}_std'] = [s for _, s in refit[k]]
            refit_out[f'{name}_per_seed'] = refit_seeds[k]
    with open(results_path, 'w') as f:
        json.dump({
            'lambda_gates': args.lambda_gates, 'seeds': args.seeds, 'beta': args.beta,
            'gate_forward': args.gate_forward, 'epochs': args.epochs, 'lambda_sparse': args.lambda_sparse,
            'accuracy': accs, 'accuracy_std': accs_std, 'accuracy_per_seed': accs_seeds,
            'test_accuracy': test_accs, 'test_accuracy_std': test_accs_std, 'test_accuracy_per_seed': test_accs_seeds,
            'open_gates': n_open, 'open_gates_std': n_open_std, 'open_gates_per_seed': n_open_seeds,
            'cea': cea, 'cea_std': cea_std, 'cea_per_seed': cea_seeds,
            'baseline_accuracy': baseline_acc, 'baseline_accuracy_std': baseline_acc_std,
            'baseline_test_accuracy': baseline_test_acc, 'baseline_test_accuracy_std': baseline_test_acc_std,
            'baseline_cea': baseline_cea, 'baseline_cea_std': baseline_cea_std,
            **refit_out,
        }, f, indent=2)
    print(f'Saved results to {results_path}')

    plot_lambda_gate_sweep(
        args.lambda_gates, cea, n_open, dataset,
        accs_std=cea_std, n_open_std=n_open_std,
        baseline_acc=baseline_cea, baseline_acc_std=baseline_cea_std,
        test_accs=test_accs, test_accs_std=test_accs_std,
        baseline_test_acc=baseline_test_acc, baseline_test_acc_std=baseline_test_acc_std,
        val_metric_name='CEA',
        test_overlays=([(refit_out['refit_test_accuracy'], refit_out['refit_test_accuracy_std'], '#2B4A7A',
                         f'Gated, refit @tau={args.refit_tau}')] if refit_out else None),
    )


if __name__ == '__main__':
    main(parse_args())
