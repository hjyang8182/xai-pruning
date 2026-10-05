"""DN-CBM entry point for explanation_size.py: per-image explanation size vs. global union
for the SAE-concept linear probes trained by train_cbm.py, written in the same JSON shape
as the LF-CBM / UCBM / VLG-CBM `concept_retention.py --analysis local` so
plot_local_vs_global.py can take it as one more panel/row.

    python dncbm_explanation_size.py -d cub                 # latest baseline_* run, all seeds
    python dncbm_explanation_size.py -d cub --run gated_20260805_160756

For a gated run the gate is folded into the weight (GatedProbe stores it separately):
hard 1[sigmoid > 0.5] for hard-forward checkpoints, the soft sigmoid otherwise, matching
what forward() multiplies the activation by. Output: data/<dataset>/local_explanation_size_<stamp>.json
"""
import argparse
import glob
import json
import os
from datetime import datetime

import torch

from src import config
from src.data import load_clip_features
from src.models import LinearProbe, GatedProbe
from gating import hard_gate_ste
from explanation_size import local_explanation_sizes, mean_explanation_report, plot_explanation_sizes, summary_line
from train_cbm import N_CLASSES


def _latest_run(model_dir, prefix):
    # probe_config.json is written at the end of train_cbm.py, so its presence marks a completed run
    runs = sorted(d for d in os.listdir(model_dir) if d.startswith(prefix) and os.path.exists(os.path.join(model_dir, d, 'probe_config.json')))
    if not runs:
        raise FileNotFoundError(f"No '{prefix}*' run with probes in {model_dir}")
    return runs[-1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-d', '--dataset', required=True, choices=sorted(N_CLASSES))
    p.add_argument('--run', default=None, help='data/<dataset>/model/<run>; defaults to the latest baseline_* run')
    p.add_argument('--mass-levels', type=float, nargs='+', default=[0.9])
    p.add_argument('--n-extreme', type=int, default=20)
    p.add_argument('--weight-threshold', type=float, default=1e-5,
                   help='|w| above this counts as a live weight. The soft L1 penalty never drives weights to '
                        'exactly zero; on places365 the baseline probe parks ~94%% of its weights at ~1e-4 '
                        'while the ~0.04%% that matter are >0.1, so use e.g. 1e-2 there.')
    return p.parse_args()


def main(args):
    save = os.path.join(config.DATA_PATH, args.dataset)
    model_dir = os.path.join(save, 'model')
    run = args.run or _latest_run(model_dir, 'baseline_')
    run_dir = os.path.join(model_dir, run)
    gated = os.path.basename(run).startswith('gated_')
    n_classes = N_CLASSES[args.dataset]

    acts = torch.load(os.path.join(save, 'activations', 'test_sae_acts.pt'), map_location='cpu')
    _, labels = load_clip_features(os.path.join(save, 'activations'), 'test')

    paths = sorted(glob.glob(os.path.join(run_dir, 'probe_seed*.pt'))) or [os.path.join(run_dir, 'probe.pt')]
    reports = []
    for path in paths:
        probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes) if gated else LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        probe.load_state_dict(torch.load(path, map_location='cpu'))
        probe.eval()
        gate = None
        if gated:
            with torch.no_grad():
                g = probe.gate_probs()
                gate = hard_gate_ste(g) if probe.hard_forward else g
        rep = local_explanation_sizes(probe.linear.weight.detach(), probe.linear.bias.detach(), acts, labels,
                                      gate=gate, fold_gate=True, mass_levels=tuple(args.mass_levels), n_extreme=args.n_extreme,
                                      weight_threshold=args.weight_threshold)
        print(summary_line(rep, os.path.basename(path)[:10]))
        reports.append(rep)

    summary = mean_explanation_report(reports)
    summary.update({'dataset': args.dataset, 'load_dirs': [run_dir], 'analysis': 'local', 'gated': gated,
                    'weight_threshold': args.weight_threshold})
    print(summary_line(summary, 'mean'))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(save, f'local_explanation_size_{stamp}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=1)
    plot_explanation_sizes(summary, out.replace('.json', '.png'), title=f'DN-CBM local explanation size ({run})', dataset_name=args.dataset)
    print(f'Saved results to {out}')


if __name__ == '__main__':
    main(parse_args())
