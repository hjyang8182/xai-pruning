"""Re-plot the gated_sweep_test figure (lambda_gate sweep: gated test accuracy, the retrained
random-subset overlay, the all-concepts baseline, and the used-concept count) for several datasets
as one row of panels with a single shared legend, instead of the one-figure-per-dataset SVG that
sweep_lambda_gate.py / concept_ablation.py --mode random-retrain write at run time.

Reads data/<dataset>/lambda_gate_sweep.json for each dataset; the random-subset curve comes from
the random_retrain block that concept_ablation.py --mode random-retrain folds into that file.
"""
import argparse
import json
import os

from src.config import DATA_PATH, N_LEARNED_FEATURES
from src.visualise import plot_lambda_gate_sweep_grid

DISPLAY_NAMES = {'cifar100': 'CIFAR-100', 'cub': 'CUB', 'places365': 'Places365', 'imagenet': 'ImageNet'}
RANDOM_COLOR = '#55A868'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=['cifar100', 'cub', 'places365'])
    parser.add_argument('--json', nargs='*', default=None,
                        help='explicit lambda_gate_sweep.json per dataset (same order as --datasets); '
                             'default: data/<dataset>/lambda_gate_sweep.json')
    parser.add_argument('--no-sharey', action='store_true', help='give each panel its own accuracy range')
    parser.add_argument('--no-random', action='store_true', help='omit the random-subset (retrained) overlay')
    parser.add_argument('--absolute-counts', action='store_true',
                        help='plot used concepts as absolute counts rather than %% of the dictionary')
    return parser.parse_args()


def main(args):
    paths = args.json or [os.path.join(DATA_PATH, d, 'lambda_gate_sweep.json') for d in args.datasets]
    if len(paths) != len(args.datasets):
        raise ValueError('--json needs one path per dataset')

    # The SAE dictionary is the same size for every DN-CBM dataset (the sweep JSON doesn't record it).
    scale = 1.0 if args.absolute_counts else 100.0 / N_LEARNED_FEATURES
    results = {}
    for dataset, path in zip(args.datasets, paths):
        with open(path) as f:
            sweep = json.load(f)
        print(f'{dataset}: {path}')
        overlays = []
        rr = sweep.get('random_retrain', {})
        if not args.no_random and 'random_subset_acc_mean_by_lambda' in rr:
            overlays.append((rr['random_subset_acc_mean_by_lambda'], rr.get('random_subset_acc_std_by_lambda'),
                             RANDOM_COLOR, 'Random subset'))
        elif not args.no_random:
            print(f'  (no random_retrain block by lambda in {path}; run concept_ablation.py --mode random-retrain)')
        results[DISPLAY_NAMES.get(dataset, dataset)] = {
            'lambda_gates': sweep['lambda_gates'],
            'accs': sweep['test_accuracy'], 'accs_std': sweep.get('test_accuracy_std'),
            'n_open': [scale * o for o in sweep['open_gates']],
            'n_open_std': [scale * s for s in sweep['open_gates_std']] if sweep.get('open_gates_std') else None,
            'baseline_acc': sweep.get('baseline_test_accuracy'), 'baseline_acc_std': sweep.get('baseline_test_accuracy_std'),
            'overlays': overlays,
        }

    out = plot_lambda_gate_sweep_grid(results, ncols=len(results), sharey=not args.no_sharey,
                                      n_open_as_pct=not args.absolute_counts)
    print(f'Saved plot to {out}')


if __name__ == '__main__':
    main(parse_args())
