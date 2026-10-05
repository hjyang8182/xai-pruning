import argparse
import glob
import json
import os
from src import config
from src.visualise import plot_random_concept_baseline

DATASETS = ['cifar100', 'food101', 'cub', 'oxford_pet']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot an existing random_concept_baseline.py run (test accuracy vs. concept "
                     "count for an untrained random-projection probe) without retraining anything."
    )
    parser.add_argument('-d', '--dataset', choices=DATASETS, default='cifar100')
    parser.add_argument('--random-run', default=None,
                         help='random_concept_baseline_<timestamp>.json filename under data/<dataset>/ to use; '
                              'defaults to the latest')
    parser.add_argument('--gated-run', default=None,
                         help='gated_<timestamp> dir name under data/<dataset>/model/ to mark as the operating '
                              'point (vertical line at the mean open-gate count from its probe_config.json)')
    return parser.parse_args()


def _latest_random_run(save_dir):
    paths = sorted(glob.glob(os.path.join(save_dir, 'random_concept_baseline_*.json')))
    if not paths:
        raise FileNotFoundError(f'No random_concept_baseline_*.json found in {save_dir}; run random_concept_baseline.py first')
    return paths[-1]


def _operating_point_from_gated_run(gated_run_dir):
    config_path = os.path.join(gated_run_dir, 'probe_config.json')
    with open(config_path) as f:
        probe_config = json.load(f)
    open_gates = probe_config['open_gates_by_seed'].values()
    return sum(open_gates) / len(open_gates)


def main(args):
    SAVE = os.path.join(config.DATA_PATH, args.dataset)

    random_path = os.path.join(SAVE, args.random_run) if args.random_run else _latest_random_run(SAVE)
    with open(random_path) as f:
        results = json.load(f)

    print(f'Random baseline ({os.path.basename(random_path)}): ' +
          '  '.join(f'{k}={a:.4f}' for k, a in zip(results['concept_counts'], results['random_acc_mean'])))

    operating_point = None
    if args.gated_run:
        gated_run_dir = os.path.join(SAVE, 'model', args.gated_run)
        operating_point = _operating_point_from_gated_run(gated_run_dir)
        print(f'Operating point ({args.gated_run}): {operating_point:.1f} concepts')

    plot_path = plot_random_concept_baseline(
        results['concept_counts'], results['random_acc_mean'], results['random_acc_std'], args.dataset,
        operating_point=operating_point,
    )
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main(parse_args())
