import argparse
import glob
import json
import os
import re
from datetime import datetime
import torch
from src import config
from src.models import GatedProbe, LinearProbe
from src.train import topk_prediction_change_rate
from src.visualise import plot_explanation_fidelity

N_CLASSES     = {'cifar100': 100, 'food101': 101, 'cub': 200, 'oxford_pet': 37}
DEFAULT_TOP_KS = [1, 2, 3, 5, 10, 20, 50]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure explanation fidelity: for each test image, prune the probe's "
                    "top-k contributing concepts (same ranking used by explain_image) down to "
                    "just those k, zero out every other concept, and re-run the probe. Reports "
                    "the percentage of predictions that change as a function of k."
    )
    parser.add_argument('-d', '--dataset', choices=N_CLASSES.keys(), default='oxford_pet')
    parser.add_argument('--gated-run', required=True,
                         help='data/<dataset>/model/<run> holding the gated probe_seed*.pt (or probe.pt) checkpoints')
    parser.add_argument('--baseline-run', default=None,
                         help='data/<dataset>/model/<run> holding the baseline (ungated) probe_seed*.pt '
                              '(or probe.pt) checkpoints to report alongside the gated probe; omit to only '
                              'measure the gated probe')
    parser.add_argument('--top-ks', type=int, nargs='+', default=DEFAULT_TOP_KS,
                         help='Number of top contributing concepts to keep when pruning each explanation')
    return parser.parse_args()


# A run dir holds one probe_seed<seed>.pt per seed trained by train_cbm.py; older runs may only
# have a single probe.pt.
def _probe_paths(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, 'probe_seed*.pt')))
    if not paths:
        single = os.path.join(run_dir, 'probe.pt')
        paths = [single] if os.path.exists(single) else []
    if not paths:
        raise FileNotFoundError(f'No probe_seed*.pt or probe.pt found in {run_dir}')
    return paths


def _probe_paths_by_seed(run_dir):
    by_seed = {}
    for path in _probe_paths(run_dir):
        m = re.search(r'probe_seed(\d+)\.pt$', os.path.basename(path))
        key = int(m.group(1)) if m else 'single'
        by_seed[key] = path
    return by_seed


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

    gated_run_dir = args.gated_run if os.path.isabs(args.gated_run) else os.path.join(MODEL_DIR, args.gated_run)
    gated_by_seed = _probe_paths_by_seed(gated_run_dir)
    seeds = sorted(gated_by_seed, key=lambda s: (isinstance(s, str), s))

    baseline_by_seed = None
    if args.baseline_run:
        baseline_run_dir = args.baseline_run if os.path.isabs(args.baseline_run) else os.path.join(MODEL_DIR, args.baseline_run)
        baseline_by_seed = _probe_paths_by_seed(baseline_run_dir)
        seeds = sorted(set(gated_by_seed) & set(baseline_by_seed), key=lambda s: (isinstance(s, str), s))
        if not seeds:
            raise ValueError(
                f"No matching seeds between gated run '{args.gated_run}' ({sorted(gated_by_seed, key=str)}) "
                f"and baseline run '{args.baseline_run}' ({sorted(baseline_by_seed, key=str)}); "
                f"they must be trained with the same --seeds"
            )

    gated_rates, baseline_rates = [], []
    for seed in seeds:
        gated_probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
        gated_probe.load_state_dict(torch.load(gated_by_seed[seed], map_location='cpu'))
        gated_probe.eval()
        seed_gated_rates = [topk_prediction_change_rate(gated_probe, test_acts, k) for k in args.top_ks]
        gated_rates.append(seed_gated_rates)

        msg = f'[gated seed {seed}] ' + '  '.join(f'k={k}: {r*100:.1f}%' for k, r in zip(args.top_ks, seed_gated_rates))

        if baseline_by_seed is not None:
            baseline_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
            baseline_probe.load_state_dict(torch.load(baseline_by_seed[seed], map_location='cpu'))
            baseline_probe.eval()
            seed_baseline_rates = [topk_prediction_change_rate(baseline_probe, test_acts, k) for k in args.top_ks]
            baseline_rates.append(seed_baseline_rates)
            msg += '  |  ' + '  '.join(f'[baseline] k={k}: {r*100:.1f}%' for k, r in zip(args.top_ks, seed_baseline_rates))

        print(msg)

    gated_rates = torch.tensor(gated_rates)
    gated_mean, gated_std = gated_rates.mean(0).tolist(), gated_rates.std(0).tolist()

    baseline_mean = baseline_std = None
    if baseline_rates:
        baseline_rates = torch.tensor(baseline_rates)
        baseline_mean, baseline_std = baseline_rates.mean(0).tolist(), baseline_rates.std(0).tolist()

    print(f'\n{"k":>5}  {"gated change rate":>20}' + ('  baseline change rate' if baseline_mean else ''))
    for i, k in enumerate(args.top_ks):
        line = f'{k:>5}  {gated_mean[i]*100:>18.1f}%'
        if baseline_mean:
            line += f'  {baseline_mean[i]*100:>20.1f}%'
        print(line)

    results = {
        'dataset': dataset, 'gated_run': args.gated_run, 'baseline_run': args.baseline_run,
        'seeds': [s if s != 'single' else None for s in seeds], 'top_ks': args.top_ks,
        'gated_change_rate_mean': gated_mean, 'gated_change_rate_std': gated_std,
        'baseline_change_rate_mean': baseline_mean, 'baseline_change_rate_std': baseline_std,
    }
    results_path = os.path.join(SAVE, f"explanation_fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {results_path}')

    plot_explanation_fidelity(args.top_ks, gated_mean, gated_std, dataset,
                               baseline_mean=baseline_mean, baseline_std=baseline_std)


if __name__ == '__main__':
    main(parse_args())
