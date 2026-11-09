import argparse
import json
import os
from datetime import datetime

import torch
from src import config
from src.models import load_autoencoder, load_clip
from src.data import load_cifar100, load_food101, load_cub, load_oxford_pet, save_clip_features, load_clip_features, get_sae_acts
from src.train import train_probe
from src.metrics import count_used_concepts
from src.visualise import plot_lambda_sparse_sweep
from train_cbm import DATASET_LOADERS, N_CLASSES
from gating import prune_and_refit, evaluate_head


def _mean_std(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description='Sweep lambda_sparse and plot resulting baseline-probe accuracy / used-concept count')
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS.keys(), default='cifar100')
    parser.add_argument('--lambda-sparses', type=float, nargs='+',
                         default=[0.0, 1e-4, 3e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.PROBE_BATCH_SIZE)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20],
                         help='Seeds to train each lambda_sparse with; results are averaged (mean/std) across these')
    parser.add_argument('--used-concept-threshold', type=float, default=1e-3,
                         help='Weight magnitude above which a concept counts as "used" (see count_used_concepts in src/metrics.py)')
    parser.add_argument('--no-refit', action='store_true',
                        help='Skip the mask-and-refit stage (the original L1-only sweep)')
    parser.add_argument('--refit-epochs', type=int, default=20,
                        help='Mask-and-refit epochs: K = concepts used at --used-concept-threshold, the head is refit '
                             'on K with no L1 penalty -- the same stage (gating.prune_and_refit, lr = --lr) as the '
                             'gated arm in sweep_lambda_gate.py, so L1 and GCG are compared after the same refit')
    parser.add_argument('--refit-batch-size', type=int, default=None,
                        help='Refit batch size (default: --batch-size). Match the gated arm\'s refit: the DN-CBM GCG runs '
                             'use 64 on cifar100/cub and 4096 on places365 -- at 4096 a 20-epoch CUB refit is ~40 steps')
    parser.add_argument('--refit-cold', action='store_true',
                        help='Refit from a fresh init instead of warm-starting from the L1 weights on K')
    parser.add_argument('--no-plot', action='store_true', help='Skip the per-dataset sweep plot')
    parser.add_argument('--out-name', default=None,
                        help='Results file name under data/<dataset>/ (default: lambda_sparse_sweep_refit_<stamp>.json, '
                             'or lambda_sparse_sweep.json with --no-refit)')
    return parser.parse_args()


def main(args):
    dataset   = args.dataset
    n_classes = N_CLASSES[dataset]
    SAVE      = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE  = os.path.join(SAVE, 'activations')
    os.makedirs(ACT_SAVE, exist_ok=True)

    if not os.path.exists(os.path.join(ACT_SAVE, 'train_clip_features.pt')):
        # raw images are only needed to build the cache; with the cached features/activations copied over
        # (e.g. on another cluster) the dataset itself never has to be present
        train_dataset, test_dataset, _ = DATASET_LOADERS[dataset]()
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

    # Each lambda_sparse is trained across multiple seeds so the plot can show mean +/- std, in
    # line with sweep_lambda_gate.py. There's no hyperparameter selection happening here (unlike
    # sweep_lambda_gate.py, which picks a lambda_gate by validation CEA), so accuracy is measured
    # directly on the test set.
    accs, accs_std = [], []
    n_used, n_used_std = [], []
    accs_seeds, n_used_seeds = [], []
    refit = {'accuracy': [], 'accuracy_std': [], 'accuracy_per_seed': []}
    for lambda_sparse in args.lambda_sparses:
        seed_accs, seed_n_used, seed_refit = [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed)
            acc, probe = train_probe(
                train_acts, train_labels, test_acts, test_labels, n_classes=n_classes,
                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                lambda_sparse=lambda_sparse,
            )
            used = count_used_concepts(probe.linear.weight.detach(), threshold=args.used_concept_threshold)
            seed_accs.append(acc)
            seed_n_used.append(used)
            msg = f'[lambda_sparse={lambda_sparse:.0e}, seed {seed}] accuracy: {acc:.4f}, used concepts: {used}/{train_acts.shape[1]}'
            if not args.no_refit:
                # mask-and-refit: K = concepts with any |w| > threshold (the same K that `used` counts), head refit
                # on K without the L1 term -- what the gated arm gets, so the comparison isn't L1's shrinkage bias
                W = probe.linear.weight.detach().cpu()
                b = probe.linear.bias.detach().cpu()
                keep = (W.abs() > args.used_concept_threshold).any(dim=0)
                W_r, b_r, _ = prune_and_refit(W, b, keep.float(), train_acts, train_labels, keep=keep, fold_gate=False,
                                              epochs=args.refit_epochs, lr=args.lr,
                                              batch_size=args.refit_batch_size or args.batch_size, device=config.device, seed=seed, warm_start=not args.refit_cold)
                acc_r = evaluate_head(W_r, b_r, test_acts, test_labels, device=config.device)
                seed_refit.append(acc_r)
                msg += f' -> refit {acc_r:.4f}'
            print(msg, flush=True)

        mean, std = _mean_std(seed_accs);   accs.append(mean);   accs_std.append(std)
        mean, std = _mean_std(seed_n_used); n_used.append(mean); n_used_std.append(std)
        accs_seeds.append(seed_accs)
        n_used_seeds.append(seed_n_used)
        if seed_refit:
            mean, std = _mean_std(seed_refit)
            refit['accuracy'].append(mean); refit['accuracy_std'].append(std); refit['accuracy_per_seed'].append(seed_refit)

    name = args.out_name or ('lambda_sparse_sweep.json' if args.no_refit else
                             f"lambda_sparse_sweep_refit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_path = os.path.join(SAVE, name)
    with open(results_path, 'w') as f:
        json.dump({
            'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
            **({} if args.no_refit else {'refit': {**refit, 'epochs': args.refit_epochs, 'warm_start': not args.refit_cold,
                                                  'batch_size': args.refit_batch_size or args.batch_size}}),
            'lambda_sparses': args.lambda_sparses, 'seeds': args.seeds,
            'used_concept_threshold': args.used_concept_threshold,
            'accuracy': accs, 'accuracy_std': accs_std, 'accuracy_per_seed': accs_seeds,
            'used_concepts': n_used, 'used_concepts_std': n_used_std, 'used_concepts_per_seed': n_used_seeds,
        }, f, indent=2)
    print(f'Saved results to {results_path}')

    if args.no_plot:
        return
    plot_path = plot_lambda_sparse_sweep(args.lambda_sparses, accs, n_used, dataset,
                                          accs_std=accs_std, n_used_std=n_used_std)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main(parse_args())
