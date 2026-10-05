import argparse
import json
import os
from collections import Counter
from datetime import datetime
import torch
from src import config
from src.models import GatedProbe, LinearProbe
from src.data import load_clip_features
from src.concepts import load_concept_names
from src.metrics import compute_cea, compute_nec, count_used_concepts
from src.visualise import plot_concept_stability
import numpy as np
from train_cbm import DATASET_LOADERS, N_CLASSES 

def parse_args():
    parser = argparse.ArgumentParser(description='Check stability of the open concept set across random seeds')
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS, default='oxford_pet')
    parser.add_argument('--gated-run', required=True,
                         help='data/<dataset>/model/<run> holding the probe_seed<seed>.pt (or probe.pt) '
                              'checkpoints to load, one per --seeds value')
    parser.add_argument('--baseline-run', default=None,
                         help='data/<dataset>/model/<run> holding baseline (ungated) probe_seed<seed>.pt '
                              '(or probe.pt) checkpoints, one per --seeds value, to compare accuracy/CEA '
                              'against the gated probes')
    parser.add_argument('--gate-threshold', type=float, default=0.5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20, 30, 40],
                         help='Seeds whose checkpoints to load; the open concept set is compared across these')
    parser.add_argument('--beta', '-b', type = float, default = 0.25, help = 'Beta value used for CEA computation')
    parser.add_argument('--nec-threshold', type=float, default=1e-3,
                         help='Absolute final-layer weight magnitude below which a concept counts as unused '
                              'for NEC (Number of Effective Concepts, arXiv:2408.01432); weights are never '
                              'exactly zero here since sparsity is only a soft L1 penalty, not proximal')
    return parser.parse_args()


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0

# A run dir holds one probe_seed<seed>.pt per seed trained by train_cbm.py; older runs may only
# have a single probe.pt.
def _probe_path(run_dir, seed):
    seed_path = os.path.join(run_dir, f'probe_seed{seed}.pt')
    if os.path.exists(seed_path):
        return seed_path
    single_path = os.path.join(run_dir, 'probe.pt')
    if os.path.exists(single_path):
        return single_path
    raise FileNotFoundError(f'No probe_seed{seed}.pt or probe.pt found in {run_dir}')


def main(args):
    dataset   = args.dataset
    n_classes = N_CLASSES[dataset]
    SAVE      = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE  = os.path.join(SAVE, 'activations')
    MODEL_DIR = os.path.join(SAVE, 'model')

    test_acts_path = os.path.join(ACT_SAVE, 'test_sae_acts.pt')
    if not os.path.exists(test_acts_path):
        raise FileNotFoundError(f'{test_acts_path} not found; run train_cbm.py for this dataset first')
    test_acts = torch.load(test_acts_path)
    _, test_labels = load_clip_features(ACT_SAVE, 'test')

    run_dir = args.gated_run if os.path.isabs(args.gated_run) else os.path.join(MODEL_DIR, args.gated_run)
    baseline_run_dir = None
    if args.baseline_run:
        baseline_run_dir = args.baseline_run if os.path.isabs(args.baseline_run) else os.path.join(MODEL_DIR, args.baseline_run)

    concept_names = load_concept_names()

    seeds = args.seeds
    open_gates_by_seed = {}
    open_sets_by_seed = {}
    accs = []
    cea_vals = []
    nec_vals = []
    baseline_accs = []
    baseline_cea_vals = []
    baseline_nec_vals = []
    baseline_used_concepts_vals = []
    for seed in seeds:
        probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
        probe.load_state_dict(torch.load(_probe_path(run_dir, seed), map_location='cpu'))
        probe.eval()
        with torch.no_grad():
            logits, gates = probe(test_acts)
            acc = (logits.argmax(dim=1) == test_labels).float().mean().item()
        gates = gates.detach()


        open_idx = (gates > args.gate_threshold).nonzero().squeeze(-1).tolist()
        n_concepts = len(open_idx)
        cea = compute_cea(n_classes, n_concepts, acc, args.beta)
        cea_vals.append(cea)
        nec_vals.append(compute_nec(probe.linear.weight.detach()[:, open_idx], args.nec_threshold))
        open_gates_by_seed[seed] = sorted(
            ((concept_names[i], gates[i].item()) for i in open_idx), key=lambda x: -x[1]
        )
        open_sets_by_seed[seed] = {concept_names[i] for i in open_idx}
        accs.append(acc)

        if baseline_run_dir:
            baseline_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
            baseline_probe.load_state_dict(torch.load(_probe_path(baseline_run_dir, seed), map_location='cpu'))
            baseline_probe.eval()
            with torch.no_grad():
                baseline_acc = (baseline_probe(test_acts).argmax(dim=1) == test_labels).float().mean().item()
            baseline_accs.append(baseline_acc)
            baseline_weight = baseline_probe.linear.weight.detach()
            baseline_used_concepts = count_used_concepts(baseline_weight, args.nec_threshold)
            baseline_used_concepts_vals.append(baseline_used_concepts)
            baseline_cea_vals.append(compute_cea(n_classes, baseline_used_concepts, baseline_acc, args.beta))
            baseline_nec_vals.append(compute_nec(baseline_weight, args.nec_threshold))

    for seed in seeds:
        txt_path = os.path.join(SAVE, f'open_concepts_seed_{seed}.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(f'{name}\tgate={val:.3f}' for name, val in open_gates_by_seed[seed]))
        print(f'Saved {len(open_gates_by_seed[seed])} open concepts to {txt_path}')

    n_seeds = len(seeds)
    freq = Counter()
    for s in seeds:
        freq.update(open_sets_by_seed[s])

    union = set(freq.keys())
    core = {c for c, n in freq.items() if n == n_seeds}
    only_once = {c for c, n in freq.items() if n == 1}

    pairwise = {}
    jaccards = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            si, sj = seeds[i], seeds[j]
            sim = jaccard(open_sets_by_seed[si], open_sets_by_seed[sj])
            pairwise[f'{si}-{sj}'] = sim
            jaccards.append(sim)
    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 1.0
    mean_cea = np.mean(cea_vals) if cea_vals else 0
    std_cea  = np.std(cea_vals, ddof=1) if len(cea_vals) > 1 else 0.0
    mean_nec = np.mean(nec_vals) if nec_vals else 0
    std_nec  = np.std(nec_vals, ddof=1) if len(nec_vals) > 1 else 0.0
    mean_baseline_acc = sum(baseline_accs)/len(baseline_accs) if baseline_accs else None
    mean_baseline_cea = np.mean(baseline_cea_vals) if baseline_cea_vals else None
    std_baseline_cea  = np.std(baseline_cea_vals, ddof=1) if len(baseline_cea_vals) > 1 else 0.0
    mean_baseline_nec = np.mean(baseline_nec_vals) if baseline_nec_vals else None
    std_baseline_nec  = np.std(baseline_nec_vals, ddof=1) if len(baseline_nec_vals) > 1 else 0.0

    print(f'\nDataset: {dataset}   gated_run={args.gated_run}   seeds={seeds}')
    print(f'Accuracy per seed: {[round(a, 4) for a in accs]}')
    print(f'Open gates per seed: {[len(open_sets_by_seed[s]) for s in seeds]}')
    print(f'Union of open concepts across seeds: {len(union)}')
    print(f'Core concepts (open in ALL {n_seeds} seeds): {len(core)}')
    print(f'Concepts open in only 1 seed: {len(only_once)}')
    print(f'Mean pairwise Jaccard similarity: {mean_jaccard:.3f}')
    print(f'Mean CEA: {mean_cea:.4f} +/- {std_cea:.4f}')
    print(f'Mean NEC: {mean_nec:.2f} +/- {std_nec:.2f}')
    if baseline_run_dir:
        print(f'\nBaseline run: {args.baseline_run}')
        print(f'Baseline accuracy per seed: {[round(a, 4) for a in baseline_accs]}')
        print(f'Mean baseline accuracy: {mean_baseline_acc:.4f}')
        print(f'Baseline used concepts per seed: {baseline_used_concepts_vals} (out of {config.N_LEARNED_FEATURES})')
        print(f'Mean baseline CEA: {mean_baseline_cea:.4f} +/- {std_baseline_cea:.4f}')
        print(f'Mean baseline NEC: {mean_baseline_nec:.2f} +/- {std_baseline_nec:.2f}')
    print('\nCore concepts:')
    for c in sorted(core):
        print(f'  {c}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(SAVE, f'concept_stability_{timestamp}.json')
    results = {
        'dataset': dataset, 'gated_run': args.gated_run,
        'seeds': seeds, 'gate_threshold': args.gate_threshold,
        'accuracy_by_seed': accs,
        'n_open_by_seed': {str(s): len(open_sets_by_seed[s]) for s in seeds},
        'n_union': len(union),
        'n_core': len(core),
        'n_only_once': len(only_once),
        'pairwise_jaccard': pairwise,
        'mean_jaccard': mean_jaccard,
        'mean_cea': mean_cea,
        'std_cea': std_cea,
        'nec_threshold': args.nec_threshold,
        'mean_nec': mean_nec,
        'std_nec': std_nec,
    }
    if baseline_run_dir:
        results.update({
            'baseline_run': args.baseline_run,
            'baseline_accuracy_by_seed': baseline_accs,
            'baseline_used_concepts_by_seed': baseline_used_concepts_vals,
            'mean_baseline_cea': mean_baseline_cea,
            'std_baseline_cea': std_baseline_cea,
            'mean_baseline_nec': mean_baseline_nec,
            'std_baseline_nec': std_baseline_nec,
        })
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {results_path}')

    plot_concept_stability(freq, n_seeds, mean_jaccard, dataset)


if __name__ == '__main__':
    main(parse_args())
