"""
Error analysis comparing a baseline probe and a gated probe on CIFAR-100: reports
standard (fine, 100-class) accuracy alongside group/superclass (20-class) accuracy
for each, plus which superclasses shifted the most (positively or negatively)
between the two.

Does not retrain anything -- it loads cached SAE test activations + the saved
probe checkpoints produced by train_cbm.py
(data/<dataset>/model/<run>/probe_seed*.pt or probe.pt).

Usage:
    python eval_group_accuracy.py --baseline-name baseline_20260729_140210 --gated-name gated_20260729_150342
"""
import argparse
import glob
import json
import os
import torch

from src import config
from src.data import load_clip_features, cifar100_fine_to_coarse, CIFAR100_COARSE_LABEL_NAMES
from src.models import LinearProbe, GatedProbe
from src.train import get_predictions, get_gated_predictions, group_accuracy
from src.visualise import plot_superclass_confusion_matrix, plot_superclass_accuracy, plot_superclass_delta


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline vs. gated probe superclass accuracy on CIFAR-100')
    parser.add_argument('--baseline-name', required=True, help='e.g. baseline_20260729_140210')
    parser.add_argument('--gated-name', required=True, help='e.g. gated_20260729_150342')
    parser.add_argument('--dataset', default='cifar100', help='Only cifar100 has a superclass mapping defined')
    parser.add_argument('--top-confusions', type=int, default=10, help='How many confused superclass pairs to print per model')
    parser.add_argument('--top-deltas', type=int, default=10, help='How many superclasses to print, ranked by |delta|')
    return parser.parse_args()


def load_probe(path, gated, n_concepts, n_classes):
    probe_cls = GatedProbe if gated else LinearProbe
    probe = probe_cls(n_concepts, n_classes)
    probe.load_state_dict(torch.load(path, map_location='cpu'))
    return probe.eval()


def mean_std(xs):
    t = torch.tensor(xs)
    return t.mean().item(), t.std().item() if len(xs) > 1 else 0.0


def evaluate_model(model_name, gated, dataset, test_acts, test_labels, coarse_labels, n_super):
    model_dir = os.path.join('data', dataset, 'model', model_name)
    n_concepts = test_acts.shape[1]

    probe_paths = sorted(glob.glob(os.path.join(model_dir, 'probe_seed*.pt')))
    if not probe_paths:
        single = os.path.join(model_dir, 'probe.pt')
        probe_paths = [single] if os.path.exists(single) else []
    if not probe_paths:
        raise FileNotFoundError(f'No probe_seed*.pt or probe.pt found in {model_dir}')

    fine_accs, group_accs = [], []
    cm = torch.zeros(n_super, n_super, dtype=torch.long)  # rows: true, cols: predicted

    for path in probe_paths:
        probe = load_probe(path, gated, n_concepts, n_classes=100)
        preds = get_gated_predictions(probe, test_acts) if gated else get_predictions(probe, test_acts)

        fine_acc = (preds == test_labels).float().mean().item()
        grp_acc = group_accuracy(preds, test_labels, cifar100_fine_to_coarse)
        fine_accs.append(fine_acc)
        group_accs.append(grp_acc)
        print(f'[{model_name}/{os.path.basename(path)}] fine acc: {fine_acc:.4f}, group (superclass) acc: {grp_acc:.4f}')

        coarse_preds = cifar100_fine_to_coarse(preds)
        flat_idx = coarse_labels * n_super + coarse_preds
        cm += torch.bincount(flat_idx, minlength=n_super * n_super).reshape(n_super, n_super)

    fine_mean, fine_std = mean_std(fine_accs)
    grp_mean, grp_std = mean_std(group_accs)
    superclass_total = cm.sum(dim=1)
    superclass_correct = cm.diag()
    per_superclass_acc = {
        CIFAR100_COARSE_LABEL_NAMES[c]: (superclass_correct[c] / superclass_total[c]).item()
        for c in range(n_super)
    }

    return {
        'model_dir': model_dir, 'probe_paths': probe_paths,
        'fine_accs': fine_accs, 'group_accs': group_accs,
        'fine_mean': fine_mean, 'fine_std': fine_std,
        'grp_mean': grp_mean, 'grp_std': grp_std,
        'per_superclass_acc': per_superclass_acc, 'cm': cm,
    }


def report_and_save(model_name, result, dataset, top_confusions):
    print(f'\n=== {model_name} ===')
    print(f'  Fine accuracy:  {result["fine_mean"]:.4f} +/- {result["fine_std"]:.4f}')
    print(f'  Group accuracy: {result["grp_mean"]:.4f} +/- {result["grp_std"]:.4f}')

    print('  Per-superclass accuracy:')
    for name, acc in sorted(result['per_superclass_acc'].items(), key=lambda kv: kv[1]):
        print(f'    {name:<32s} {acc:.4f}')

    cm = result['cm']
    n_super = cm.shape[0]
    off_diag = cm.clone()
    off_diag.fill_diagonal_(0)
    top_idx = torch.argsort(off_diag.flatten(), descending=True)[:top_confusions]
    print(f'  Top {top_confusions} true->predicted superclass confusions:')
    for idx in top_idx.tolist():
        true_c, pred_c = idx // n_super, idx % n_super
        n = off_diag[true_c, pred_c].item()
        if n == 0:
            break
        print(f'    {CIFAR100_COARSE_LABEL_NAMES[true_c]:<32s} -> {CIFAR100_COARSE_LABEL_NAMES[pred_c]:<32s} {n}')

    cm_path = plot_superclass_confusion_matrix(cm, CIFAR100_COARSE_LABEL_NAMES, dataset, model_name)
    acc_path = plot_superclass_accuracy(result['per_superclass_acc'], dataset, model_name)
    print(f'  Saved confusion matrix plot to {cm_path}')
    print(f'  Saved per-superclass accuracy plot to {acc_path}')

    out_path = os.path.join(result['model_dir'], 'group_accuracy.json')
    with open(out_path, 'w') as f:
        json.dump({
            'fine_accuracy_mean': result['fine_mean'], 'fine_accuracy_std': result['fine_std'],
            'group_accuracy_mean': result['grp_mean'], 'group_accuracy_std': result['grp_std'],
            'fine_accuracy_by_probe': dict(zip(result['probe_paths'], result['fine_accs'])),
            'group_accuracy_by_probe': dict(zip(result['probe_paths'], result['group_accs'])),
            'per_superclass_accuracy': result['per_superclass_acc'],
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': CIFAR100_COARSE_LABEL_NAMES,
        }, f, indent=2)
    print(f'  Saved results to {out_path}')


def main(args):
    if args.dataset != 'cifar100':
        raise ValueError('Superclass mapping is only defined for cifar100')

    SAVE = os.path.join(config.DATA_PATH, args.dataset)
    ACT_SAVE = os.path.join(SAVE, 'activations')
    test_acts = torch.load(os.path.join(ACT_SAVE, 'test_sae_acts.pt'))
    _, test_labels = load_clip_features(ACT_SAVE, 'test')
    coarse_labels = cifar100_fine_to_coarse(test_labels)
    n_super = len(CIFAR100_COARSE_LABEL_NAMES)

    baseline = evaluate_model(args.baseline_name, False, args.dataset, test_acts, test_labels, coarse_labels, n_super)
    gated = evaluate_model(args.gated_name, True, args.dataset, test_acts, test_labels, coarse_labels, n_super)

    report_and_save(args.baseline_name, baseline, args.dataset, args.top_confusions)
    report_and_save(args.gated_name, gated, args.dataset, args.top_confusions)

    deltas = {
        name: gated['per_superclass_acc'][name] - baseline['per_superclass_acc'][name]
        for name in CIFAR100_COARSE_LABEL_NAMES
    }
    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)

    print(f'\n=== Superclasses ranked by |delta| (gated - baseline) ===')
    for name, delta in ranked[:args.top_deltas]:
        b, g = baseline['per_superclass_acc'][name], gated['per_superclass_acc'][name]
        print(f'  {name:<32s} baseline={b:.4f}  gated={g:.4f}  delta={delta:+.4f}')

    delta_path = plot_superclass_delta(baseline['per_superclass_acc'], gated['per_superclass_acc'],
                                        args.dataset, args.baseline_name, args.gated_name, top_n=args.top_deltas)
    print(f'\nSaved superclass delta plot to {delta_path}')

    delta_json_path = os.path.join(gated['model_dir'], f'superclass_delta_vs_{args.baseline_name}.json')
    with open(delta_json_path, 'w') as f:
        json.dump({
            'baseline_model': args.baseline_name,
            'gated_model': args.gated_name,
            'delta_by_superclass': deltas,
            'ranked_by_abs_delta': [
                {'superclass': name, 'baseline_acc': baseline['per_superclass_acc'][name],
                 'gated_acc': gated['per_superclass_acc'][name], 'delta': delta}
                for name, delta in ranked
            ],
        }, f, indent=2)
    print(f'Saved delta comparison to {delta_json_path}')


if __name__ == '__main__':
    main(parse_args())
