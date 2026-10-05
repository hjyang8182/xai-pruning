"""Re-plot saved ranked-order concept-ablation results (concept_ablation.py --mode ranked-order)
for several datasets as one row of panels with a shared y-axis and a single legend, instead of the
one-figure-per-dataset SVG that concept_ablation.py writes at run time.

Reads the newest concept_ablation_ranked_*.json under data/<dataset>/ for each dataset and maps
its most-/least-selected-first curves onto the Open-first/Closed-first/Random legend used by the
open-closed ablation figures.
"""
import argparse
import glob
import json
import os

from src.config import DATA_PATH
from src.visualise import plot_concept_ablation_grid

DISPLAY_NAMES = {'cifar100': 'CIFAR-100', 'cub': 'CUB', 'places365': 'Places365', 'imagenet': 'ImageNet'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=['cifar100', 'cub', 'places365'])
    parser.add_argument('--json', nargs='*', default=None,
                        help='explicit results JSON per dataset (same order as --datasets); '
                             'default: newest concept_ablation_ranked_*.json in data/<dataset>/')
    return parser.parse_args()


def _latest_ranked_json(dataset):
    paths = sorted(glob.glob(os.path.join(DATA_PATH, dataset, 'concept_ablation_ranked_*.json')))
    if not paths:
        raise FileNotFoundError(f'no concept_ablation_ranked_*.json under {os.path.join(DATA_PATH, dataset)}')
    return paths[-1]


def main(args):
    paths = args.json or [_latest_ranked_json(d) for d in args.datasets]
    if len(paths) != len(args.datasets):
        raise ValueError('--json needs one path per dataset')

    results = {}
    for dataset, path in zip(args.datasets, paths):
        with open(path) as f:
            r = json.load(f)
        print(f'{dataset}: {path}')
        results[DISPLAY_NAMES.get(dataset, dataset)] = {
            'fractions': r['fractions'],
            'open_acc_mean': r['most_selected_first_acc_mean'], 'open_acc_std': r['most_selected_first_acc_std'],
            'closed_acc_mean': r['least_selected_first_acc_mean'], 'closed_acc_std': r['least_selected_first_acc_std'],
            'random_acc_mean': r['random_order_acc_mean'], 'random_acc_std': r['random_order_acc_std'],
        }

    out = plot_concept_ablation_grid(
        results, ncols=len(results), sharey=True,
        xlabel='Concepts Zeroed (%)', plot_label='concept_ablation_ranked_grid',
    )
    print(f'Saved plot to {out}')


if __name__ == '__main__':
    main(parse_args())
