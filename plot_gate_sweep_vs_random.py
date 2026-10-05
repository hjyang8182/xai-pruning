import argparse
import glob
import json
import os
from src import config
from src.visualise import plot_gated_vs_random
DATASETS = ['cifar100', 'food101', 'cub', 'oxford_pet', 'places365']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay an existing sweep_lambda_gate.py run (lambda_gate_sweep.json: gated "
                     "probe test accuracy vs. mean open gates) with an existing "
                     "random_concept_baseline.py run (test accuracy vs. concept count) for the same "
                     "dataset, to see whether the gated probe's learned concepts beat an untrained "
                     "random projection of the same size. Does not retrain anything - both inputs "
                     "must already exist on disk."
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

    sweep_path = os.path.join(SAVE, 'lambda_gate_sweep.json')
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f'{sweep_path} not found; run sweep_lambda_gate.py for {args.dataset} first')
    with open(sweep_path) as f:
        sweep = json.load(f)

    random_path = os.path.join(SAVE, f'{args.random_run}.json') if args.random_run else _latest_random_run(SAVE)
    with open(random_path) as f:
        random_results = json.load(f)

    # lambda_gate_sweep.json is ordered by ascending lambda_gate, i.e. descending open_gates;
    # sort ascending by open_gates so the plotted line reads left-to-right.
    order = sorted(range(len(sweep['open_gates'])), key=lambda i: sweep['open_gates'][i])
    open_gates     = [sweep['open_gates'][i] for i in order]
    gated_test_acc = [sweep['test_accuracy'][i] for i in order]
    gated_test_std = [sweep['test_accuracy_std'][i] for i in order]

    print(f'Gated sweep ({os.path.basename(sweep_path)}): ' +
          '  '.join(f'{g:.0f}={a:.4f}' for g, a in zip(open_gates, gated_test_acc)))
    print(f'Random baseline ({os.path.basename(random_path)}): ' +
          '  '.join(f'{k}={a:.4f}' for k, a in zip(random_results['concept_counts'], random_results['random_acc_mean'])))

    operating_point = None
    if args.gated_run:
        gated_run_dir = os.path.join(SAVE, 'model', args.gated_run)
        operating_point = _operating_point_from_gated_run(gated_run_dir)
        print(f'Operating point ({args.gated_run}): {operating_point:.1f} concepts')

    plot_path = plot_gated_vs_random(
        open_gates, gated_test_acc, gated_test_std,
        random_results['concept_counts'], random_results['random_acc_mean'], random_results['random_acc_std'],
        args.dataset,
        baseline_test_acc=sweep.get('baseline_test_accuracy'),
        baseline_test_acc_std=sweep.get('baseline_test_accuracy_std'),
        operating_point=operating_point,
    )
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main(parse_args())
