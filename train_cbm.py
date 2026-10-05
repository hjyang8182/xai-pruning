import argparse
import json
import os
from datetime import datetime
import torch
from sklearn.model_selection import train_test_split
from src import config
from src.models import load_autoencoder, load_clip
from src.data import (load_cifar100, load_food101, load_cub, load_oxford_pet, load_places365,
                       save_clip_features, load_clip_features, get_sae_acts, subsample_per_class)
from src.concepts import (load_vocab, save_vocab_embeddings, load_vocab_embeddings, name_concepts, load_concept_names)
from src.train import train_probe, train_gated_probe, get_predictions, get_gated_predictions
from src.metrics import concept_usage_report, mean_report
from gating import prune_and_refit, masked_accuracy, evaluate_head
from src.visualise import (plot_comparison, plot_gated_comparison, plot_relevant_concepts, plot_gate_distribution,
                            plot_training_curves)

DATASET_LOADERS   = {'cifar100': load_cifar100, 'food101': load_food101, 'cub': load_cub, 'oxford_pet': load_oxford_pet, 'places365': load_places365}
N_CLASSES         = {'cifar100': 100, 'food101': 101, 'cub': 200, 'oxford_pet': 37, 'places365': 365}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train and evaluate concept-bottleneck probes')
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS.keys(), default='cifar100')
    parser.add_argument('-g', '--gated', action='store_true', help='Train/evaluate the gated probe instead of the baseline probe')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.PROBE_BATCH_SIZE,
                         help='Probe minibatch over cached concept activations (not images)')
    parser.add_argument('--lambda-sparse', type=float, default=config.LAMBDA_SPARSE)
    parser.add_argument('--lambda-gate', type=float, default=config.LAMBDA_GATE, help='Only used with --gated')
    parser.add_argument('--gate-forward', choices=['soft', 'hard'], default='soft',
                        help='Only used with --gated: "soft" multiplies concepts by sigmoid(gate); "hard" uses the '
                             'mask 1[sigmoid > 0.5] in the forward pass with a straight-through gradient, so the '
                             'trained model is exactly the masked one and gates cannot drift below 0.5 for free.')
    parser.add_argument('--gate-init', type=float, default=2.0,
                         help='Initial value of every gate logit (sigmoid(2.0) ~= 0.88: all concepts start open; '
                              'negative starts them closed). Only used with --gated')
    parser.add_argument('--gate-init-std', type=float, default=0.0,
                         help='Std of Gaussian noise added to the initial gate logits (0 = identical start for '
                              'every concept). Only used with --gated')
    parser.add_argument('--refit-tau', type=float, default=0.5,
                        help='Only used with --gated: after training, keep concepts with gate > tau, drop the '
                             'rest, and refit the linear head on the survivors. The reported "refit" accuracy '
                             'is the accuracy of exactly that concept set (no gate at inference).')
    parser.add_argument('--refit-epochs', type=int, default=20)
    parser.add_argument('--no-refit', action='store_true', help='Skip the mask-and-refit stage')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20, 30, 40],
                         help='Seeds to train with; results are averaged (mean/std) across these')
    parser.add_argument('--subsample-per-class', type=int, default=None,
                         help='Train the probe on at most this many examples per class (deterministic '
                              'stratified subset of the cached activations). Big speedup on Places365 '
                              '(5000/class) at negligible probe-accuracy cost; leave unset to use all.')
    parser.add_argument('--acts-device', choices=['auto', 'cuda', 'cpu'], default='auto',
                         help='Where the cached activation matrices live during probe training. '
                              '"auto" keeps them GPU-resident when they fit, else pinned CPU.')
    parser.add_argument('--amp', action='store_true',
                         help='Run the probe forward/backward under bfloat16 autocast')
    parser.add_argument('--val-frac', type=float, default=0.2,
                         help='Fraction of the training set held out as a validation set (excluded from training, e.g. used to pick lambda_gate in sweep_lambda_gate.py)')
    parser.add_argument('--val-seed', type=int, default=config.SEED,
                         help='Seed used to carve out the validation split; match sweep_lambda_gate.py so the same rows are held out')
    return parser.parse_args(argv)


def mean_std(values):
    """(mean, std) of a list of per-seed scalars; std is 0 for a single seed."""
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def main(args):
    dataset    = args.dataset
    n_classes  = N_CLASSES[dataset]
    SAVE       = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE   = os.path.join(SAVE, 'activations')
    MODEL_DIR = os.path.join(SAVE, 'model')
    MODEL_NAME = f"{'gated' if args.gated else 'baseline'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    os.makedirs(ACT_SAVE, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # CLIP features
    if not os.path.exists(os.path.join(ACT_SAVE, 'train_clip_features.pt')):
        clip_model, _ = load_clip()
        train_dataset, test_dataset, _ = DATASET_LOADERS[dataset]()
        save_clip_features(clip_model, train_dataset, test_dataset, ACT_SAVE)
    train_feats, train_labels = load_clip_features(ACT_SAVE, 'train')
    test_feats,  test_labels  = load_clip_features(ACT_SAVE, 'test')

    # Vocab embeddings
    if not os.path.exists(config.EMB_PATH):
        clip_model, _ = load_clip()
        save_vocab_embeddings(clip_model, load_vocab())
    text_emb = load_vocab_embeddings()

    # SAE activations for dataset
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

    # Optional stratified subsample of the (cached, full) train activations. Deterministic
    # and seeded independently of --seeds, so the held-out rows are stable across runs and
    # reproducible by the lambda sweeps given the same --subsample-per-class.
    if args.subsample_per_class is not None:
        keep = subsample_per_class(train_labels, args.subsample_per_class, seed=config.SEED)
        train_acts, train_labels = train_acts[keep], train_labels[keep]
        print(f'Subsampled train to {len(keep)} examples (<= {args.subsample_per_class}/class)')

    # Hold out the same stratified validation split used to pick lambda_gate in
    # sweep_lambda_gate.py, so the final probes are never trained on rows that
    # informed hyperparameter selection.
    n = train_acts.shape[0]
    train_idx, _ = train_test_split(
        range(n), test_size=args.val_frac, random_state=args.val_seed, stratify=train_labels,
    )
    train_acts, train_labels = train_acts[train_idx], train_labels[train_idx]

    # Concept names
    if not os.path.exists(config.CSV_PATH):
        autoencoder = load_autoencoder(config.sae_device)
        name_concepts(autoencoder, text_emb, load_vocab())
    concept_names = load_concept_names()

    hyperparams = {
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lambda_sparse': args.lambda_sparse,
        'subsample_per_class': args.subsample_per_class,
        'train_examples': int(len(train_idx)),
        'amp': args.amp,
    }

    config_path = os.path.join(MODEL_SAVE_PATH, 'probe_config.json')

    if args.gated:
        hyperparams['lambda_gate'] = args.lambda_gate
        hyperparams['gate_forward'] = args.gate_forward
        hyperparams['gate_init'] = args.gate_init
        hyperparams['gate_init_std'] = args.gate_init_std
        hyperparams['refit_tau'] = args.refit_tau
        accs, open_gates, histories, usage = [], [], {}, []
        masked_accs, refit_accs, refit_n_concepts = [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed)
            probe_path = os.path.join(MODEL_SAVE_PATH, f'probe_seed{seed}.pt')
            gated_acc, gated_probe, history = train_gated_probe(
                train_acts, train_labels, test_acts, test_labels, n_classes=n_classes,
                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                lambda_sparse=args.lambda_sparse, lambda_gate=args.lambda_gate,
                track_history=True, acts_device=args.acts_device, amp=args.amp,
                gate_forward=args.gate_forward, gate_init=args.gate_init, gate_init_std=args.gate_init_std,
            )
            torch.save(gated_probe.state_dict(), probe_path)
            accs.append(gated_acc)
            histories[seed] = history
            gates = torch.sigmoid(gated_probe.gate_logits).detach()
            open_gates.append((gates > 0.5).sum().item())
            # GatedProbe.linear.weight is stored ungated (the gate is applied in forward),
            # so hand the gate over separately; test_acts are the pre-gate SAE code.
            usage.append(concept_usage_report(
                gated_probe.linear.weight, test_acts,
                gate=gated_probe.gate_probs().detach()))
            print(f'[seed {seed}] Gated accuracy: {gated_acc:.4f}, open gates: {open_gates[-1]}/{train_acts.shape[1]}')

            if not args.no_refit:
                # The deliverable: the |K| open concepts as a hard mask, head refit on them.
                # GatedProbe stores W ungated (gate applied in forward) -> fold_gate=True.
                gate = gated_probe.gate_probs().detach()
                W_lin, b_lin = gated_probe.linear.weight, gated_probe.linear.bias
                masked_acc = masked_accuracy(W_lin, b_lin, gate, test_acts, test_labels, tau=args.refit_tau)
                W_r, b_r, keep = prune_and_refit(
                    W_lin, b_lin, gate, train_acts, train_labels, tau=args.refit_tau, fold_gate=True,
                    epochs=args.refit_epochs, lr=args.lr, batch_size=args.batch_size,
                    device=config.device, acts_device=args.acts_device, seed=seed)
                refit_acc = evaluate_head(W_r, b_r, test_acts, test_labels, device=config.device)
                torch.save({'weight': W_r, 'bias': b_r, 'keep': keep, 'tau': args.refit_tau},
                           os.path.join(MODEL_SAVE_PATH, f'probe_seed{seed}_refit.pt'))
                masked_accs.append(masked_acc)
                refit_accs.append(refit_acc)
                refit_n_concepts.append(int(keep.sum()))
                print(f'[seed {seed}] {int(keep.sum())} concepts @tau={args.refit_tau}: '
                      f'hard cut {masked_acc:.4f} -> refit {refit_acc:.4f}')

        hyperparams['open_gates_by_seed'] = dict(zip(args.seeds, open_gates))
        # Number of concepts the gated model uses = open gates (sigmoid > 0.5).
        hyperparams['n_concepts_mean'], hyperparams['n_concepts_std'] = mean_std(open_gates)
        print(f'Open gates over {len(open_gates)} seeds: {hyperparams["n_concepts_mean"]:.1f} +/- '
              f'{hyperparams["n_concepts_std"]:.1f} / {train_acts.shape[1]}')
        if refit_accs:
            hyperparams['masked_accuracy_by_seed'] = dict(zip(args.seeds, masked_accs))
            hyperparams['refit_accuracy_by_seed'] = dict(zip(args.seeds, refit_accs))
            hyperparams['refit_accuracy_mean'], hyperparams['refit_accuracy_std'] = mean_std(refit_accs)
            # Survivors of the tau cut = the concept set the refit head actually uses.
            hyperparams['refit_n_concepts_by_seed'] = dict(zip(args.seeds, refit_n_concepts))
            hyperparams['refit_n_concepts_mean'], hyperparams['refit_n_concepts_std'] = mean_std(refit_n_concepts)
            print(f'Refit accuracy over {len(refit_accs)} seeds: {hyperparams["refit_accuracy_mean"]:.4f} +/- '
                  f'{hyperparams["refit_accuracy_std"]:.4f}  (soft gated: {torch.tensor(accs).mean().item():.4f}), '
                  f'{hyperparams["refit_n_concepts_mean"]:.1f} +/- {hyperparams["refit_n_concepts_std"]:.1f} concepts')
        hyperparams['concept_usage_by_seed'] = dict(zip(args.seeds, usage))
        hyperparams['concept_usage_mean'] = mean_report(usage)

        plot_relevant_concepts(gated_probe, concept_names, dataset)
        plot_gate_distribution(gated_probe, dataset)
        curves_path = plot_training_curves(histories, dataset, label='gated')
        print(f'Saved training curves to {curves_path}')

    else:
        accs, histories, usage = [], {}, []
        for seed in args.seeds:
            torch.manual_seed(seed)
            probe_path = os.path.join(MODEL_SAVE_PATH, f'probe_seed{seed}.pt')
            base_acc, base_probe, history = train_probe(
                train_acts, train_labels, test_acts, test_labels, n_classes=n_classes,
                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                lambda_sparse=args.lambda_sparse,
                track_history=True, acts_device=args.acts_device, amp=args.amp,
            )
            torch.save(base_probe.state_dict(), probe_path)
            accs.append(base_acc)
            histories[seed] = history
            usage.append(concept_usage_report(base_probe.linear.weight, test_acts))
            print(f'[seed {seed}] Baseline accuracy: {base_acc:.4f}, '
                  f'concepts used: {usage[-1]["n_used_union"]}/{train_acts.shape[1]}')

        hyperparams['concept_usage_by_seed'] = dict(zip(args.seeds, usage))
        hyperparams['concept_usage_mean'] = mean_report(usage)
        # No gate here: a concept is "used" if any class weight on it clears the usage
        # report's weight_threshold (same definition as n_used_union in the report).
        n_used = [u['n_used_union'] for u in usage]
        hyperparams['n_concepts_by_seed'] = dict(zip(args.seeds, n_used))
        hyperparams['n_concepts_mean'], hyperparams['n_concepts_std'] = mean_std(n_used)
        print(f'Concepts used over {len(n_used)} seeds: {hyperparams["n_concepts_mean"]:.1f} +/- '
              f'{hyperparams["n_concepts_std"]:.1f} / {train_acts.shape[1]}')

        curves_path = plot_training_curves(histories, dataset, label='baseline')
        print(f'Saved training curves to {curves_path}')

    acc_mean, acc_std = mean_std(accs)
    print(f'Accuracy over {len(args.seeds)} seeds: {acc_mean:.4f} +/- {acc_std:.4f}')

    hyperparams['seeds'] = args.seeds
    hyperparams['accuracy_mean'] = acc_mean
    hyperparams['accuracy_std'] = acc_std
    hyperparams['accuracy_by_seed'] = dict(zip(args.seeds, accs))
    with open(config_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f'Saved run to {MODEL_SAVE_PATH}')
    return MODEL_SAVE_PATH


if __name__ == '__main__':
    main(parse_args())
