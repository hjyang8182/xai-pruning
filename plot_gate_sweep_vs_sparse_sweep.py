import argparse
import json
import os
from src import config
from src.visualise import plot_gate_vs_sparse_sweep

DATASETS = ['cifar100', 'food101', 'cub', 'oxford_pet', 'places365']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay an existing sweep_lambda_gate.py run (lambda_gate_sweep.json: test "
                     "accuracy vs. mean open gates) with an existing sweep_lambda_sparse.py run "
                     "(lambda_sparse_sweep.json: test accuracy vs. mean used concepts) for the same "
                     "dataset, to compare how gating vs. L1 sparsity trade off concept count against "
                     "accuracy. Does not retrain anything - both inputs must already exist on disk."
    )
    parser.add_argument('-d', '--dataset', choices=DATASETS, default='cifar100')
    return parser.parse_args()


def main(args):
    SAVE = os.path.join(config.DATA_PATH, args.dataset)

    gate_path = os.path.join(SAVE, 'lambda_gate_sweep.json')
    if not os.path.exists(gate_path):
        raise FileNotFoundError(f'{gate_path} not found; run sweep_lambda_gate.py for {args.dataset} first')
    with open(gate_path) as f:
        gate_sweep = json.load(f)

    sparse_path = os.path.join(SAVE, 'lambda_sparse_sweep.json')
    if not os.path.exists(sparse_path):
        raise FileNotFoundError(f'{sparse_path} not found; run sweep_lambda_sparse.py for {args.dataset} first')
    with open(sparse_path) as f:
        sparse_sweep = json.load(f)

    # Both json files are ordered by ascending lambda, i.e. descending concept count; sort
    # ascending by concept count so each plotted line reads left-to-right.
    gate_order = sorted(range(len(gate_sweep['open_gates'])), key=lambda i: gate_sweep['open_gates'][i])
    open_gates     = [gate_sweep['open_gates'][i] for i in gate_order]
    gated_test_acc = [gate_sweep['test_accuracy'][i] for i in gate_order]
    gated_test_std = [gate_sweep['test_accuracy_std'][i] for i in gate_order]

    sparse_order = sorted(range(len(sparse_sweep['used_concepts'])), key=lambda i: sparse_sweep['used_concepts'][i])
    used_concepts   = [sparse_sweep['used_concepts'][i] for i in sparse_order]
    sparse_test_acc = [sparse_sweep['accuracy'][i] for i in sparse_order]
    sparse_test_std = [sparse_sweep['accuracy_std'][i] for i in sparse_order]

    print(f'Gated sweep ({os.path.basename(gate_path)}): ' +
          '  '.join(f'{g:.0f}={a:.4f}' for g, a in zip(open_gates, gated_test_acc)))
    print(f'Sparse sweep ({os.path.basename(sparse_path)}): ' +
          '  '.join(f'{u:.0f}={a:.4f}' for u, a in zip(used_concepts, sparse_test_acc)))

    plot_path = plot_gate_vs_sparse_sweep(
        open_gates, gated_test_acc, gated_test_std,
        used_concepts, sparse_test_acc, sparse_test_std,
        args.dataset,
        baseline_test_acc=gate_sweep.get('baseline_test_accuracy'),
        baseline_test_acc_std=gate_sweep.get('baseline_test_accuracy_std'),
    )
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main(parse_args())
