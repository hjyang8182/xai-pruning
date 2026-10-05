import argparse
import json
import math
import os
import sys
from datetime import datetime
import torch
from src import config
from src.models import load_clip, RandomConceptLayer
from src.data import (
    load_cifar100, load_food101, load_cub, load_oxford_pet,
    save_clip_features, load_clip_features,
)
from src.train import train_probe, train_probes_shared_projection
from src.visualise import plot_random_concept_baseline
from train_cbm import DATASET_LOADERS, N_CLASSES


# Default --concept-counts: the open-gate count achieved at each lambda_gate in this dataset's
# sweep_lambda_gate.py run (data/<dataset>/lambda_gate_sweep.json's mean open_gates, one per
# lambda_gate), floored to the nearest int and de-duplicated - so the random-projection baseline is
# evaluated at exactly the concept counts the gated probe actually produced, rather than an
# arbitrary fraction schedule.
def concept_counts_from_gate_sweep(dataset):
    path = os.path.join(config.DATA_PATH, dataset, 'lambda_gate_sweep.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found; run sweep_lambda_gate.py for {dataset} first, or pass --concept-counts explicitly'
        )
    with open(path) as f:
        sweep = json.load(f)
    concept_counts = sorted({math.floor(g) for g in sweep['open_gates']})
    concept_counts.append(config.N_LEARNED_FEATURES)
    return concept_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage control: replace the trained SAE concept layer with an untrained, "
                     "randomly initialized one of matching dimensionality (Gaussian weights, frozen), "
                     "and train only the final linear probe on top. Accuracy here is meant to be "
                     "compared against a real-concept probe's accuracy at the same concept count "
                     "(e.g. from train_cbm.py / prune_concepts.py) elsewhere. If the random layer's "
                     "accuracy tracks the real one - especially as concept count grows - that's "
                     "evidence accuracy comes from having enough dimensions to fit any linear "
                     "function of CLIP space, not from what the concepts encode."
    )
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS.keys(), default='cifar100')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lambda-sparse', type=float, default=config.LAMBDA_SPARSE)
    parser.add_argument('--concept-counts', type=int, nargs='+', default=None,
                         help='Concept counts (dictionary sizes) to test; defaults to the open_gates '
                              'counts from this dataset\'s lambda_gate_sweep.json (see sweep_lambda_gate.py), '
                              'floored to the nearest int')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20, 30, 40],
                         help='Each seed pairs a fresh random-projection init with a fresh probe-training run')
    parser.add_argument('--amp', action='store_true',
                         help='bfloat16 autocast for the projection and probe. Roughly halves the time on '
                              'large datasets (places365); leave off to keep fp32 numerics')
    parser.add_argument('--share-projection', action='store_true',
                         help='Train every concept count together off one shared projection matmul per '
                              'minibatch (~2x faster, higher GPU utilisation). All counts then share a '
                              'shuffle order, so per-(seed, count) numbers shift slightly versus the '
                              'sequential path -- fine for a fresh sweep, not for extending an old one')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                         help='Physical GPU ids to spread --seeds over, one worker process each '
                              '(e.g. --gpus 0 3 4 5 6). Seeds are independent, so this is an exact '
                              'wall-clock win. Default: single process on the current device')
    parser.add_argument('--partial-out', type=str, default=None,
                         help=argparse.SUPPRESS)  # worker mode: dump this process's seeds and exit
    return parser.parse_args()


def run_workers(args, concept_counts):
    """One worker subprocess per GPU, seeds dealt round-robin, then merge their partials."""
    import subprocess, tempfile
    from collections import defaultdict

    assignments = defaultdict(list)
    for i, seed in enumerate(args.seeds):
        assignments[args.gpus[i % len(args.gpus)]].append(seed)

    with tempfile.TemporaryDirectory() as tmp:
        procs = []
        for gpu, seeds in assignments.items():
            out = os.path.join(tmp, f'gpu{gpu}.json')
            cmd = [sys.executable, os.path.abspath(__file__), '-d', args.dataset,
                   '--lr', str(args.lr), '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size), '--lambda-sparse', str(args.lambda_sparse),
                   '--concept-counts', *map(str, concept_counts),
                   '--seeds', *map(str, seeds), '--partial-out', out]
            if args.amp:
                cmd.append('--amp')
            if args.share_projection:
                cmd.append('--share-projection')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            print(f'GPU {gpu}: seeds {seeds}')
            procs.append((gpu, out, subprocess.Popen(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))))

        by_seed = {}
        for gpu, out, proc in procs:
            if proc.wait() != 0:
                raise RuntimeError(f'worker on GPU {gpu} failed with exit code {proc.returncode}')
            with open(out) as f:
                partial = json.load(f)
            for i, seed in enumerate(partial['seeds']):
                by_seed[seed] = {k: partial['random_acc_by_count'][str(k)][i] for k in concept_counts}

    # Workers finish out of order; re-key by seed so the per-count lists stay aligned with --seeds.
    return {k: [by_seed[seed][k] for seed in args.seeds] for k in concept_counts}


def run_seeds(args, concept_counts, train_feats, train_labels, test_feats, test_labels, n_classes):
    random_accs = {k: [] for k in concept_counts}
    for seed in args.seeds:
        # RandomConceptLayer draws its weight row by row from a seeded generator, so the k-row
        # layer is exactly the first k rows of the widest one at the same seed (verified by
        # torch.equal). Build the widest once and slice, rather than redrawing per count.
        widest = RandomConceptLayer(config.N_INPUT_FEATURES, max(concept_counts), seed=seed)
        W = widest.linear.weight.to(config.sae_device)
        common = dict(n_classes=n_classes, lr=args.lr, epochs=args.epochs,
                      batch_size=args.batch_size, lambda_sparse=args.lambda_sparse, amp=args.amp)

        if args.share_projection:
            torch.manual_seed(seed)
            trained = train_probes_shared_projection(
                train_feats, train_labels, test_feats, test_labels,
                lambda x: torch.relu(x @ W.T), concept_counts, seed=seed, **common)
            for k, (random_acc, _) in zip(concept_counts, trained):
                random_accs[k].append(random_acc)
                print(f'[seed {seed}] k={k:>5}  random acc={random_acc:.4f}')
            continue

        for k in concept_counts:
            # Project inside the training loop instead of materializing (n_examples, k)
            # activations: at places365's 1.8M examples that matrix reaches ~59GB at k=8192,
            # which then gets streamed from host memory once per epoch.
            W_k = W[:k]
            torch.manual_seed(seed)
            random_acc, _ = train_probe(
                train_feats, train_labels, test_feats, test_labels,
                project=lambda x, W_k=W_k: torch.relu(x @ W_k.T), n_concepts=k, **common)
            random_accs[k].append(random_acc)
            print(f'[seed {seed}] k={k:>5}  random acc={random_acc:.4f}')
    return random_accs


def main(args):
    dataset   = args.dataset
    n_classes = N_CLASSES[dataset]
    SAVE      = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE  = os.path.join(SAVE, 'activations')
    os.makedirs(ACT_SAVE, exist_ok=True)

    concept_counts = args.concept_counts or concept_counts_from_gate_sweep(dataset)

    if args.gpus:
        # The parent never touches CUDA or the feature files -- workers do all of it.
        random_accs = run_workers(args, concept_counts)
    else:
        train_dataset, test_dataset, _ = DATASET_LOADERS[dataset]()
        if not os.path.exists(os.path.join(ACT_SAVE, 'train_clip_features.pt')):
            clip_model, _ = load_clip()
            save_clip_features(clip_model, train_dataset, test_dataset, ACT_SAVE)
        train_feats, train_labels = load_clip_features(ACT_SAVE, 'train')
        test_feats,  test_labels  = load_clip_features(ACT_SAVE, 'test')

        random_accs = run_seeds(args, concept_counts, train_feats, train_labels,
                                test_feats, test_labels, n_classes)

    if args.partial_out:
        with open(args.partial_out, 'w') as f:
            json.dump({'seeds': args.seeds, 'random_acc_by_count': random_accs}, f)
        return

    random_mean = [sum(random_accs[k]) / len(random_accs[k]) for k in concept_counts]
    random_std  = [torch.tensor(random_accs[k]).std().item() if len(random_accs[k]) > 1 else 0.0 for k in concept_counts]

    for k, gm, gs in zip(concept_counts, random_mean, random_std):
        print(f'k={k:>5}  random={gm:.4f}+/-{gs:.4f}')

    results = {
        'dataset': dataset, 'concept_counts': concept_counts, 'seeds': args.seeds,
        'random_acc_mean': random_mean, 'random_acc_std': random_std,
        'random_acc_by_count': random_accs,
    }
    results_path = os.path.join(SAVE, f"random_concept_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {results_path}')

    plot_path = plot_random_concept_baseline(concept_counts, random_mean, random_std, dataset)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main(parse_args())
