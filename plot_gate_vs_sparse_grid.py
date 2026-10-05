"""Re-plot the gate_vs_sparse_sweep figure (gated probe test accuracy vs. open-concept count from
lambda_gate_sweep.json, against the L1 sparse probe's accuracy vs. used-concept count from
lambda_sparse_sweep.json, plus the ungated all-concepts baseline) for several datasets as one row
of panels with a shared accuracy axis and a single legend, instead of the one-figure-per-dataset
SVG that plot_gate_sweep_vs_sparse_sweep.py writes.

Reads data/<dataset>/lambda_gate_sweep.json and data/<dataset>/lambda_sparse_sweep.json for each
dataset. Concept counts are shown as a percentage of the SAE dictionary; each panel is framed on
the gated sweep's range (L1 drives the count to ~0 at chance accuracy, which would otherwise take
up most of a log axis).
"""
import argparse
import json
import os

from src.config import DATA_PATH, N_LEARNED_FEATURES
from src.visualise import plot_gate_vs_sparse_grid
from plot_gated_vs_random_grid import DEFAULT_GATED_RUNS, DISPLAY_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=['cifar100', 'cub', 'places365'])
    parser.add_argument('--gated-runs', nargs='*', default=None,
                        help='gated_<stamp> run dir per dataset (same order as --datasets) to mark as the '
                             'operating point; default: DEFAULT_GATED_RUNS. Pass "-" to skip a dataset.')
    parser.add_argument('--no-operating-point', action='store_true', help='omit the deployed-probe lines')
    parser.add_argument('--no-sharey', action='store_true', help='give each panel its own accuracy range')
    parser.add_argument('--absolute-counts', action='store_true',
                        help='x-axis as absolute concept counts rather than %% of the dictionary')
    return parser.parse_args()


def _sorted_by(xs, *ys):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [[seq[i] for i in order] for seq in (xs, *ys)]


def main(args):
    scale = 1.0 if args.absolute_counts else 100.0 / N_LEARNED_FEATURES
    gated_runs = args.gated_runs or [DEFAULT_GATED_RUNS.get(d, '-') for d in args.datasets]
    if len(gated_runs) != len(args.datasets):
        raise ValueError('--gated-runs needs one entry per dataset')

    results = {}
    for dataset, gated_run in zip(args.datasets, gated_runs):
        gate_path = os.path.join(DATA_PATH, dataset, 'lambda_gate_sweep.json')
        sparse_path = os.path.join(DATA_PATH, dataset, 'lambda_sparse_sweep.json')
        with open(gate_path) as f:
            gate = json.load(f)
        with open(sparse_path) as f:
            sparse = json.load(f)
        print(f'{dataset}: {gate_path} + {sparse_path}')

        operating_point = None
        if gated_run != '-' and not args.no_operating_point:
            with open(os.path.join(DATA_PATH, dataset, 'model', gated_run, 'probe_config.json')) as f:
                gates = json.load(f)['open_gates_by_seed'].values()
            operating_point = scale * sum(gates) / len(gates)
            print(f'  operating point ({gated_run}): {operating_point:.1f}{"%" if scale != 1.0 else " concepts"}')

        # Both JSONs are ordered by ascending lambda, i.e. descending count; sort ascending so
        # each line reads left-to-right.
        open_gates, gated_acc, gated_std = _sorted_by(gate['open_gates'], gate['test_accuracy'], gate['test_accuracy_std'])
        used, sparse_acc, sparse_std = _sorted_by(sparse['used_concepts'], sparse['accuracy'], sparse['accuracy_std'])
        results[DISPLAY_NAMES.get(dataset, dataset)] = {
            'open_gates': [scale * g for g in open_gates], 'gated_acc': gated_acc, 'gated_acc_std': gated_std,
            'cmp_x': [scale * u for u in used], 'cmp_acc': sparse_acc, 'cmp_acc_std': sparse_std,
            'baseline_acc': gate.get('baseline_test_accuracy'),
            'baseline_acc_std': gate.get('baseline_test_accuracy_std'),
            'operating_point': operating_point,
        }

    out = plot_gate_vs_sparse_grid(results, ncols=len(results), sharey=not args.no_sharey,
                                   counts_as_pct=not args.absolute_counts)
    print(f'Saved plot to {out}')


if __name__ == '__main__':
    main(parse_args())
