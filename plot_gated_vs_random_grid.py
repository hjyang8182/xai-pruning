"""Re-plot the gated_vs_random figure (gated probe test accuracy vs. open-concept count from
lambda_gate_sweep.json, against an untrained random projection of the same size from
random_concept_baseline.py, plus the ungated all-concepts baseline) for several datasets as one
row of panels with a shared accuracy axis and a single legend, instead of the one-figure-per-
dataset SVG that plot_gate_sweep_vs_random.py writes.

Reads data/<dataset>/lambda_gate_sweep.json and the newest random_concept_baseline_*.json under
data/<dataset>/ for each dataset. Concept counts are shown as a percentage of the SAE dictionary.
"""
import argparse
import glob
import json
import os

from src.config import DATA_PATH, N_LEARNED_FEATURES
from src.visualise import plot_gated_vs_random_grid

DISPLAY_NAMES = {'cifar100': 'CIFAR-100', 'cub': 'CUB', 'places365': 'Places365', 'imagenet': 'ImageNet'}
# The deployed gated run per dataset (data/<dataset>/model/<run>), whose mean open-gate count is
# drawn as the operating point: the hard-forward runs reported in tab_hyperparams.tex
# (CIFAR-100 lambda_gate 5e-4, CUB 3e-3, Places-365 1e-4).
DEFAULT_GATED_RUNS = {
    'cifar100': 'gated_20260920_134925',
    'cub': 'gated_20260920_135741',
    'places365': 'gated_20260921_213151',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=['cifar100', 'cub', 'places365'])
    parser.add_argument('--random-json', nargs='*', default=None,
                        help='explicit random_concept_baseline json per dataset (same order as --datasets); '
                             'default: newest random_concept_baseline_*.json in data/<dataset>/')
    parser.add_argument('--gated-runs', nargs='*', default=None,
                        help='gated_<stamp> run dir per dataset (same order as --datasets) to mark as the '
                             'operating point; default: DEFAULT_GATED_RUNS. Pass "-" to skip a dataset.')
    parser.add_argument('--no-operating-point', action='store_true', help='omit the deployed-probe lines')
    parser.add_argument('--no-sharey', action='store_true', help='give each panel its own accuracy range')
    parser.add_argument('--absolute-counts', action='store_true',
                        help='x-axis as absolute concept counts rather than %% of the dictionary')
    return parser.parse_args()


def _latest_random_json(dataset):
    paths = sorted(glob.glob(os.path.join(DATA_PATH, dataset, 'random_concept_baseline_*.json')))
    if not paths:
        raise FileNotFoundError(f'no random_concept_baseline_*.json under {os.path.join(DATA_PATH, dataset)}')
    return paths[-1]


def main(args):
    random_paths = args.random_json or [_latest_random_json(d) for d in args.datasets]
    if len(random_paths) != len(args.datasets):
        raise ValueError('--random-json needs one path per dataset')
    scale = 1.0 if args.absolute_counts else 100.0 / N_LEARNED_FEATURES
    gated_runs = args.gated_runs or [DEFAULT_GATED_RUNS.get(d, '-') for d in args.datasets]
    if len(gated_runs) != len(args.datasets):
        raise ValueError('--gated-runs needs one entry per dataset')

    results = {}
    for dataset, random_path, gated_run in zip(args.datasets, random_paths, gated_runs):
        sweep_path = os.path.join(DATA_PATH, dataset, 'lambda_gate_sweep.json')
        with open(sweep_path) as f:
            sweep = json.load(f)
        with open(random_path) as f:
            rand = json.load(f)
        print(f'{dataset}: {sweep_path} + {random_path}')
        operating_point = None
        if gated_run != '-' and not args.no_operating_point:
            with open(os.path.join(DATA_PATH, dataset, 'model', gated_run, 'probe_config.json')) as f:
                gates = json.load(f)['open_gates_by_seed'].values()
            operating_point = scale * sum(gates) / len(gates)
            print(f'  operating point ({gated_run}): {operating_point:.1f}{"%" if scale != 1.0 else " concepts"}')

        # lambda_gate_sweep.json is ordered by ascending lambda_gate, i.e. descending open gates;
        # sort ascending so the line reads left-to-right.
        order = sorted(range(len(sweep['open_gates'])), key=lambda i: sweep['open_gates'][i])
        results[DISPLAY_NAMES.get(dataset, dataset)] = {
            'open_gates': [scale * sweep['open_gates'][i] for i in order],
            'gated_acc': [sweep['test_accuracy'][i] for i in order],
            'gated_acc_std': [sweep['test_accuracy_std'][i] for i in order],
            'cmp_x': [scale * c for c in rand['concept_counts']],
            'cmp_acc': rand['random_acc_mean'], 'cmp_acc_std': rand.get('random_acc_std'),
            'baseline_acc': sweep.get('baseline_test_accuracy'),
            'baseline_acc_std': sweep.get('baseline_test_accuracy_std'),
            'operating_point': operating_point,
        }

    out = plot_gated_vs_random_grid(results, ncols=len(results), sharey=not args.no_sharey,
                                    counts_as_pct=not args.absolute_counts)
    print(f'Saved plot to {out}')


if __name__ == '__main__':
    main(parse_args())
