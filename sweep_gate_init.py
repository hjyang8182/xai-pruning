"""Ablation: how does the gate's initialisation affect the gated DN-CBM probe?

Every concept's gate logit starts at the same constant (default +2.0, sigmoid ~= 0.88, i.e.
"all open"). This sweeps that constant -- including negative values, where every concept
starts *closed* and must be opened by the task gradient -- and optionally a noisy start, and
trains a full gated run per setting with train_cbm.py (same lambda_gate / epochs / seeds /
refit as the paper run), then summarises accuracy (gated, hard-cut, refit) and open-gate
count per initialisation.

    python sweep_gate_init.py -d cifar100 --lambda-gate 5e-4 --gate-forward hard --epochs 100 \\
        --inits -4 -2 0 2 4 --noisy-std 2 --seeds 0 10 20

Writes data/<dataset>/gate_init_sweep_<stamp>.json and a figure via src.visualise.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np

import train_cbm
from src import config
from src.visualise import plot_gate_init_sweep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-d', '--dataset', default='cifar100')
    p.add_argument('--inits', type=float, nargs='+', default=[-4, -2, 0, 2, 4],
                   help='constant initial gate logits to try')
    p.add_argument('--noisy-std', type=float, nargs='*', default=[2.0],
                   help='also run gate_init=0 with N(0, std) noise for each std given ([] to skip)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20])
    p.add_argument('--lambda-gate', type=float, default=5e-4)
    p.add_argument('--lambda-sparse', type=float, default=1e-4)
    p.add_argument('--gate-forward', choices=['soft', 'hard'], default='hard')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--refit-tau', type=float, default=0.5)
    p.add_argument('--acts-device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--amp', action='store_true')
    return p.parse_args()


def _train_args(args, gate_init, gate_init_std):
    """argparse.Namespace for train_cbm.main mirroring its CLI defaults, one gated run."""
    ns = train_cbm.parse_args([])
    ns.dataset, ns.gated = args.dataset, True
    ns.lr, ns.epochs, ns.batch_size = args.lr, args.epochs, args.batch_size
    ns.lambda_sparse, ns.lambda_gate = args.lambda_sparse, args.lambda_gate
    ns.gate_forward, ns.refit_tau = args.gate_forward, args.refit_tau
    ns.seeds, ns.acts_device, ns.amp = list(args.seeds), args.acts_device, args.amp
    ns.gate_init, ns.gate_init_std = float(gate_init), float(gate_init_std)
    return ns


def main(args):
    settings = [(v, 0.0) for v in args.inits] + [(0.0, s) for s in (args.noisy_std or [])]
    rows = []
    for gate_init, std in settings:
        label = f'{gate_init:+g}' + (f' + N(0,{std:g})' if std > 0 else '')
        print(f'\n=== gate_init = {label} ===')
        run_dir = train_cbm.main(_train_args(args, gate_init, std))
        with open(os.path.join(run_dir, 'probe_config.json')) as f:
            cfg = json.load(f)
        og = list(cfg['open_gates_by_seed'].values())
        acc = list(cfg['accuracy_by_seed'].values())
        row = {
            'gate_init': gate_init, 'gate_init_std': std, 'label': label, 'run_dir': run_dir,
            'open_gates_mean': float(np.mean(og)), 'open_gates_std': float(np.std(og)),
            'accuracy_mean': float(np.mean(acc)), 'accuracy_std': float(np.std(acc)),
        }
        if 'refit_accuracy_by_seed' in cfg:
            ra = list(cfg['refit_accuracy_by_seed'].values())
            ma = list(cfg['masked_accuracy_by_seed'].values())
            row.update({'refit_accuracy_mean': float(np.mean(ra)), 'refit_accuracy_std': float(np.std(ra)),
                        'masked_accuracy_mean': float(np.mean(ma)), 'masked_accuracy_std': float(np.std(ma))})
        rows.append(row)
        print(f'  open gates {row["open_gates_mean"]:.1f} +/- {row["open_gates_std"]:.1f}   '
              f'acc {row["accuracy_mean"]:.4f} +/- {row["accuracy_std"]:.4f}'
              + (f'   refit {row["refit_accuracy_mean"]:.4f}' if 'refit_accuracy_mean' in row else ''))

    summary = {'dataset': args.dataset, 'seeds': args.seeds, 'lambda_gate': args.lambda_gate,
               'lambda_sparse': args.lambda_sparse, 'gate_forward': args.gate_forward,
               'epochs': args.epochs, 'refit_tau': args.refit_tau, 'rows': rows}
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(config.DATA_PATH, args.dataset, f'gate_init_sweep_{stamp}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {out}')
    print(f'{"init":>14s} {"open gates":>18s} {"gated acc":>18s} {"refit acc":>18s}')
    for r in rows:
        print(f'{r["label"]:>14s} {r["open_gates_mean"]:9.1f} +/- {r["open_gates_std"]:5.1f} '
              f'{r["accuracy_mean"]:9.4f} +/- {r["accuracy_std"]:6.4f} '
              + (f'{r["refit_accuracy_mean"]:9.4f} +/- {r["refit_accuracy_std"]:6.4f}' if 'refit_accuracy_mean' in r else ''))
    plot_gate_init_sweep(summary, args.dataset)


if __name__ == '__main__':
    main(parse_args())
