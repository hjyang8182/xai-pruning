import argparse
import glob
import json
import os
import re
from datetime import datetime
import torch
from src import config
from src.data import load_clip_features
from src.models import GatedProbe, LinearProbe
from src.train import train_probe
from src.visualise import plot_concept_ablation, plot_random_retrain_sweep, plot_lambda_gate_sweep
from train_cbm import DATASET_LOADERS, N_CLASSES
from concept_retention import (
    retention_curve, mean_retention, plot_retention_curve,
    threshold_sweep, mean_threshold_sweep, plot_threshold_sensitivity,
)

ABLATE_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two modes. 'open-closed' (default): identify a gated probe's open/closed "
                     "concept split (ranked by gate value) and ablate those concepts on a separate "
                     "baseline probe, at increasing fractions, with a random-ablation control. "
                     "'retention': threshold-free -- rank one probe's concepts by contribution "
                     "c_j = ||W[:,j]||_2 * mean|a_j| (gate folded in), keep only the top-k, and "
                     "report the smallest k that retains >= 95%/99% of full test accuracy."
    )
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS, default='cifar100')
    parser.add_argument('--mode',
                         choices=['open-closed', 'ranked-order', 'retention', 'threshold', 'random-retrain'],
                         default='open-closed',
                         help="'ranked-order': drop concepts in gate-score order over the whole dictionary "
                              "-- most-selected first, least-selected first, and random order -- so all "
                              "three curves remove the same count at each point (unlike 'open-closed', "
                              "which takes a fraction of each set separately). "
                              "'threshold': sweep the gate open/closed threshold tau for a gated run and "
                              "report how much n_open and accuracy depend on where tau is put. "
                              "'random-retrain': for each concept count N in a sweep_lambda_gate.py run "
                              "(N = mean #(gate>0.5) per lambda_gate), retrain fresh probes on N random "
                              "concepts and overlay their test accuracy on the gated_sweep_test plot "
                              "(gated curve + baseline read straight from lambda_gate_sweep.json). "
                              "No --gated-run needed.")
    parser.add_argument('--sweep-json',
                         help="random-retrain: path to the sweep_lambda_gate.py lambda_gate_sweep.json "
                              "whose per-lambda_gate open_gates counts are swept "
                              "(default: data/<dataset>/lambda_gate_sweep.json)")
    parser.add_argument('--concept-counts', type=int, nargs='+', default=None,
                         help="random-retrain: concept counts N to retrain at; overrides the counts "
                              "read from --sweep-json")
    parser.add_argument('--retrain-random-trials', type=int, default=5,
                         help='random-retrain: fresh random concept subsets (each fully retrained) '
                              'to average per concept count')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='random-retrain: probe LR')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='random-retrain: probe epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                         help='random-retrain: probe minibatch size (small, like train_cbm.py -- a big '
                              'batch underfits the probe at a fixed epoch count on smaller datasets)')
    parser.add_argument('--lambda-sparse', type=float, default=config.LAMBDA_SPARSE)
    parser.add_argument('--gated-run',
                         help='data/<dataset>/model/<run> holding the gated probe_seed*.pt (or probe.pt) '
                              'files. open-closed: used only to determine the open/closed concept split. '
                              'retention: the probe whose concepts are ranked and masked (its own '
                              'weights + gate; --baseline-run is ignored).')
    parser.add_argument('--baseline-run',
                         help='data/<dataset>/model/<run> holding the baseline (ungated) probe_seed*.pt '
                              '(or probe.pt) files that get ablated and evaluated (open-closed mode only)')
    parser.add_argument('--gate-threshold', type=float, default=0.5,
                         help='open-closed: sigmoid(gate_logits) threshold separating open from closed concepts')
    parser.add_argument('--gate-temperature', type=float, default=1.0,
                         help='Temperature T in sigmoid(gate_logits / T); set to the annealed final T '
                              'if the gated run was trained with a gate-temperature schedule')
    parser.add_argument('--fractions', type=float, nargs='+', default=ABLATE_FRACTIONS,
                         help='open-closed: fractions of each set to ablate')
    parser.add_argument('--retain-levels', type=float, nargs='+', default=[0.95, 0.99],
                         help='retention: accuracy-retention thresholds (as a fraction of full acc)')
    parser.add_argument('--ranking', choices=['contribution', 'gate', 'weight', 'random'],
                         default='contribution', help='retention: how concepts are ranked before the top-k sweep')
    parser.add_argument('--n-random-trials', type=int, default=5,
                         help='Number of random concept subsets to sample and average for the random control')
    parser.add_argument('--random-seed', type=int, default=0,
                         help='Seed for the random-ablation / random-subset control sampling')
    return parser.parse_args()


# A run dir holds one probe_seed<seed>.pt per seed trained by train_cbm.py; older runs may only
# have a single probe.pt.
def _probe_paths(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, 'probe_seed*.pt')))
    if not paths:
        single = os.path.join(run_dir, 'probe.pt')
        paths = [single] if os.path.exists(single) else []
    if not paths:
        raise FileNotFoundError(f'No probe_seed*.pt or probe.pt found in {run_dir}')
    return paths


# Maps each probe file to the seed it was trained with (parsed from the filename), so gated and
# baseline runs can be paired by seed rather than by list position. A bare probe.pt (no seed in
# the name) is keyed as 'single'.
def _probe_paths_by_seed(run_dir):
    by_seed = {}
    for path in _probe_paths(run_dir):
        m = re.search(r'probe_seed(\d+)\.pt$', os.path.basename(path))
        key = int(m.group(1)) if m else 'single'
        by_seed[key] = path
    return by_seed


def evaluate(probe, acts, labels, ablate_idx):
    acts = acts.clone()
    if ablate_idx:
        acts[:, ablate_idx] = 0.0
    with torch.no_grad():
        out = probe(acts)
        logits = out[0] if isinstance(out, tuple) else out
        return (logits.argmax(dim=1) == labels).float().mean().item()


# Ablate `frac` of `ranked_idx`, taken from the front - open concepts are ranked by descending
# gate value (most confidently open first), closed by ascending (most confidently closed first),
# so the prefix always holds the concepts most representative of that set.
def ablate_prefix(ranked_idx, frac):
    n = round(frac * len(ranked_idx))
    return ranked_idx[:n]


# Ablates `n` concepts chosen uniformly at random from all `n_concepts`, averaged over
# `n_trials` draws, as a control for whether the open-set ablation curve reflects something
# about those specific concepts rather than just the count removed.
def evaluate_random(probe, acts, labels, n_concepts, n, n_trials, generator):
    if n == 0:
        return evaluate(probe, acts, labels, [])
    accs = []
    for _ in range(n_trials):
        idx = torch.randperm(n_concepts, generator=generator)[:n].tolist()
        accs.append(evaluate(probe, acts, labels, idx))
    return sum(accs) / len(accs)


def run_ranked_order(args, test_acts, test_labels, MODEL_DIR, SAVE, dataset, n_classes):
    """Drop concepts in gate-score order across the *whole* dictionary, three orders over the
    same 8192 concepts: most-selected first (descending gate), least-selected first (ascending),
    and a random order.

    Unlike the open-closed mode, every curve walks the same set from 0% to 100%, so a point on
    the x-axis means the same number of concepts removed for all three and the curves are
    directly comparable. The open-closed mode instead took a fraction of each set separately,
    which at fraction f removed ~n_open concepts for the open curve but ~n_closed (an order of
    magnitude more on cifar100) for the closed one.

    Ablates the baseline probe when --baseline-run is given (does the gate's ranking transfer to
    a probe trained without it?), otherwise the gated probe itself.
    """
    if not args.gated_run:
        raise ValueError('ranked-order mode needs --gated-run (it supplies the selection scores)')
    gated_run_dir = args.gated_run if os.path.isabs(args.gated_run) else os.path.join(MODEL_DIR, args.gated_run)
    gated_by_seed = _probe_paths_by_seed(gated_run_dir)
    seeds = sorted(gated_by_seed, key=lambda s: (isinstance(s, str), s))

    baseline_by_seed = None
    if args.baseline_run:
        baseline_run_dir = args.baseline_run if os.path.isabs(args.baseline_run) else os.path.join(MODEL_DIR, args.baseline_run)
        baseline_by_seed = _probe_paths_by_seed(baseline_run_dir)
        seeds = sorted(set(gated_by_seed) & set(baseline_by_seed), key=lambda s: (isinstance(s, str), s))
        if not seeds:
            raise ValueError(f"No matching seeds between '{args.gated_run}' and '{args.baseline_run}'")

    n_concepts = config.N_LEARNED_FEATURES
    desc_accs, asc_accs, rand_accs, n_open_by_seed = [], [], [], []
    for seed in seeds:
        gated_probe = GatedProbe(n_concepts, n_classes)
        gated_probe.load_state_dict(torch.load(gated_by_seed[seed], map_location='cpu'))
        gated_probe.eval()

        probe = gated_probe
        if baseline_by_seed:
            probe = LinearProbe(n_concepts, n_classes)
            probe.load_state_dict(torch.load(baseline_by_seed[seed], map_location='cpu'))
            probe.eval()

        gates = torch.sigmoid(gated_probe.gate_logits.detach() / args.gate_temperature)
        order_desc = torch.argsort(gates, descending=True).tolist()
        order_asc = order_desc[::-1]
        n_open_by_seed.append(int((gates > args.gate_threshold).sum().item()))

        seed_str = seed if seed != 'single' else 0
        generator = torch.Generator().manual_seed(args.random_seed + seed_str)

        desc_accs.append([evaluate(probe, test_acts, test_labels, ablate_prefix(order_desc, f)) for f in args.fractions])
        asc_accs.append([evaluate(probe, test_acts, test_labels, ablate_prefix(order_asc, f)) for f in args.fractions])
        # A random *order*, not an independent draw per fraction: prefixes of one permutation, so
        # the random curve is nested the way the two ranked ones are.
        trials = []
        for _ in range(args.n_random_trials):
            perm = torch.randperm(n_concepts, generator=generator).tolist()
            trials.append([evaluate(probe, test_acts, test_labels, ablate_prefix(perm, f)) for f in args.fractions])
        rand_accs.append(torch.tensor(trials).mean(0).tolist())

        print(f'[seed {seed}] open={n_open_by_seed[-1]}/{n_concepts}  unablated acc={desc_accs[-1][0]:.4f}')

    desc_t, asc_t, rand_t = torch.tensor(desc_accs), torch.tensor(asc_accs), torch.tensor(rand_accs)
    desc_mean, desc_std = desc_t.mean(0).tolist(), desc_t.std(0).tolist()
    asc_mean, asc_std = asc_t.mean(0).tolist(), asc_t.std(0).tolist()
    rand_mean, rand_std = rand_t.mean(0).tolist(), rand_t.std(0).tolist()

    for frac, dm, ds, am, as_, rm, rs in zip(args.fractions, desc_mean, desc_std, asc_mean, asc_std, rand_mean, rand_std):
        print(f'{frac * 100:5.1f}% of dictionary ablated: most-selected-first={dm:.4f}+/-{ds:.4f}  '
              f'least-selected-first={am:.4f}+/-{as_:.4f}  random-order={rm:.4f}+/-{rs:.4f}')

    results = {
        'dataset': dataset, 'mode': 'ranked-order', 'gated_run': args.gated_run,
        'baseline_run': args.baseline_run, 'ablated_probe': 'baseline' if baseline_by_seed else 'gated',
        'gate_threshold': args.gate_threshold, 'gate_temperature': args.gate_temperature,
        'seeds': [s if s != 'single' else None for s in seeds], 'n_open_by_seed': n_open_by_seed,
        'n_concepts': n_concepts, 'fractions': args.fractions,
        'n_random_trials': args.n_random_trials, 'random_seed': args.random_seed,
        'most_selected_first_acc_mean': desc_mean, 'most_selected_first_acc_std': desc_std,
        'least_selected_first_acc_mean': asc_mean, 'least_selected_first_acc_std': asc_std,
        'random_order_acc_mean': rand_mean, 'random_order_acc_std': rand_std,
    }
    results_path = os.path.join(SAVE, f"concept_ablation_ranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {results_path}')

    plot_path = plot_concept_ablation(
        args.fractions, desc_mean, asc_mean, desc_std, asc_std, dataset,
        random_mean=rand_mean, random_std=rand_std,
        labels=('High-gate first', 'Low-gate first', 'Random'),
        xlabel='Dictionary ablated (%)', plot_label='concept_ablation_ranked',
    )
    print(f'Saved plot to {plot_path}')


def _load_probe(path, n_concepts, n_classes):
    """Load probe_seed*.pt, returning (probe, is_gated). A GatedProbe checkpoint has a
    'gate_logits' key; a bare LinearProbe checkpoint does not."""
    sd = torch.load(path, map_location='cpu')
    is_gated = any(k.endswith('gate_logits') or k == 'gate_logits' for k in sd)
    probe = (GatedProbe if is_gated else LinearProbe)(n_concepts, n_classes)
    probe.load_state_dict(sd)
    probe.eval()
    return probe, is_gated


def run_retention(args, test_acts, test_labels, MODEL_DIR, SAVE, dataset, n_classes):
    """Threshold-free concept count: rank one probe's concepts by contribution, keep the
    top-k, report the smallest k that retains >= each --retain-level of full test accuracy."""
    run = args.gated_run or args.baseline_run
    if run is None:
        raise ValueError("retention mode needs --gated-run (or --baseline-run) pointing at the probe run")
    run_dir = run if os.path.isabs(run) else os.path.join(MODEL_DIR, run)
    by_seed = _probe_paths_by_seed(run_dir)

    reports = []
    for seed in sorted(by_seed, key=lambda s: (isinstance(s, str), s)):
        probe, is_gated = _load_probe(by_seed[seed], config.N_LEARNED_FEATURES, n_classes)
        # GatedProbe.linear.weight is stored ungated (the gate is applied in forward), so
        # hand the gate over separately and let retention_curve fold it into W and acts.
        gate = (torch.sigmoid(probe.gate_logits.detach() / args.gate_temperature)
                if is_gated else None)
        rep = retention_curve(
            probe.linear.weight.detach(), probe.linear.bias.detach(),
            test_acts, test_labels, gate=gate,
            column_norm='l2', act_stat='meanabs',
            retain_levels=tuple(args.retain_levels), ranking=args.ranking,
            n_random_trials=args.n_random_trials, random_seed=args.random_seed + (seed if seed != 'single' else 0),
        )
        reports.append(rep)
        rk = rep['retention_k']
        print(f"[seed {seed}] gated={is_gated} acc_full={rep['acc_full']:.4f}  "
              + "  ".join(f"k@{lvl:g}={rk[format(lvl,'g')]}" for lvl in args.retain_levels)
              + f"  / {rep['n_concepts_total']}")

    summary = mean_retention(reports)
    summary.update({'dataset': dataset, 'run': run, 'mode': 'retention',
                    'per_seed_retention_k': [r['retention_k'] for r in reports]})
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(SAVE, f'concept_retention_{stamp}.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    plot_path = os.path.join(SAVE, f'concept_retention_{stamp}.png')
    plot_retention_curve(summary, plot_path, title=f'Concept retention ({args.ranking})', dataset_name=dataset)
    print(f'mean retention_k: {summary["retention_k"]}  (+/- {summary["retention_k_std"]})')
    print(f'Saved results to {results_path} and plot to {plot_path}')


def run_threshold(args, test_acts, test_labels, MODEL_DIR, SAVE, dataset, n_classes):
    """Sweep the gate open/closed threshold tau for a gated run: n_open(tau) and
    accuracy(tau) with concepts whose gate <= tau hard-zeroed. A flat n_open
    plateau => the count is threshold-robust; a steep slope through 0.5 => it isn't."""
    run = args.gated_run or args.baseline_run
    if run is None:
        raise ValueError("threshold mode needs --gated-run pointing at a gated probe run")
    run_dir = run if os.path.isabs(run) else os.path.join(MODEL_DIR, run)

    reports = []
    for seed, path in sorted(_probe_paths_by_seed(run_dir).items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
        probe, is_gated = _load_probe(path, config.N_LEARNED_FEATURES, n_classes)
        if not is_gated:
            raise ValueError(f"{path} is a baseline (ungated) probe; threshold sweep needs a gated run")
        gate = torch.sigmoid(probe.gate_logits.detach() / args.gate_temperature)
        rep = threshold_sweep(probe.linear.weight.detach(), probe.linear.bias.detach(),
                              test_acts, test_labels, gate, fold_gate=True)
        reports.append(rep)
        print(f"[seed {seed}] acc_full={rep['acc_full']:.4f}  n_open@0.5={rep['n_open_at']['0.5']}  "
              f"mushy={rep['bimodal_mushy_frac']:.2f}  plateau_width={rep['plateau_width']:.2f}  "
              f"n_open(0.3-0.7)={rep['n_open_range_mid']}")

    summary = mean_threshold_sweep(reports)
    summary.update({'dataset': dataset, 'run': run, 'mode': 'threshold'})
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(SAVE, f'gate_threshold_sensitivity_{stamp}.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    plot_path = os.path.join(SAVE, f'gate_threshold_sensitivity_{stamp}.png')
    plot_threshold_sensitivity(summary, plot_path, title='Gate threshold sensitivity', dataset_name=dataset)
    print(f"mean n_open@0.5={summary['n_open_at']['0.5']:.0f}  plateau_width={summary['plateau_width']:.2f}  "
          f"mushy={summary['bimodal_mushy_frac']:.2f}")
    print(f'Saved results to {results_path} and plot to {plot_path}')


def run_random_retrain(args, train_acts, train_labels, test_acts, test_labels,
                        MODEL_DIR, SAVE, dataset, n_classes):
    """For each concept count N in a sweep_lambda_gate.py run (N = mean #(gate>0.5) at each
    lambda_gate), retrain fresh probes from scratch on N concepts drawn uniformly at random
    and overlay their mean test accuracy on the gated_sweep_test plot -- the gated curve and
    the all-concepts baseline there are read straight from lambda_gate_sweep.json, unchanged.

    A random-subset curve that meets the gated curve means the gate is only fixing a
    dimensionality; a persistent gap means its learned concept selection carries information.
    (No --gated-run and no ranking: the gate's own selection is already the gated curve.)"""
    sweep_path = args.sweep_json or os.path.join(config.DATA_PATH, dataset, 'lambda_gate_sweep.json')
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f'{sweep_path} not found; run sweep_lambda_gate.py for {dataset} first or pass --sweep-json')
    with open(sweep_path) as f:
        sweep = json.load(f)

    if args.concept_counts:
        counts = sorted({c for c in args.concept_counts if 1 <= c <= config.N_LEARNED_FEATURES})
        counts_by_lambda = None
    else:
        # one N per lambda_gate row so the random-subset curve lays over the gated_sweep_test
        # plot aligned by lambda_gate.
        counts_by_lambda = [max(1, min(int(round(g)), config.N_LEARNED_FEATURES))
                             for g in sweep['open_gates']]
        counts = sorted(set(counts_by_lambda))
    if not counts:
        raise ValueError('no concept counts to sweep')
    print(f'retraining random subsets at concept counts: {counts}')

    n_trials = args.retrain_random_trials
    g = torch.Generator().manual_seed(args.random_seed)
    rand_mean, rand_std = [], []
    for c in counts:
        trials = []
        for t in range(n_trials):
            idx = torch.randperm(config.N_LEARNED_FEATURES, generator=g)[:c]
            torch.manual_seed(args.random_seed * 1000 + t)
            acc, _ = train_probe(
                train_acts[:, idx], train_labels, test_acts[:, idx], test_labels, n_classes=n_classes,
                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, lambda_sparse=args.lambda_sparse)
            trials.append(acc)
        tt = torch.tensor(trials)
        rand_mean.append(tt.mean().item())
        rand_std.append(tt.std().item() if n_trials > 1 else 0.0)
        print(f"N={c:>5}  random-subset={rand_mean[-1]:.4f} +/- {rand_std[-1]:.4f}  "
              f"(over {n_trials} subsets)")

    base_m = sweep.get('baseline_test_accuracy')
    base_s = sweep.get('baseline_test_accuracy_std')
    if base_m is not None:
        print(f"baseline (all {config.N_LEARNED_FEATURES}, from sweep): {base_m:.4f}")

    by_count = dict(zip(counts, zip(rand_mean, rand_std)))
    rand_m_by_l = rand_s_by_l = None
    if counts_by_lambda is not None:
        rand_m_by_l = [by_count[c][0] for c in counts_by_lambda]
        rand_s_by_l = [by_count[c][1] for c in counts_by_lambda]

    summary = {
        'dataset': dataset, 'mode': 'random-retrain', 'sweep_json': sweep_path,
        'concept_counts': counts, 'retrain_random_trials': n_trials, 'random_seed': args.random_seed,
        'epochs': args.epochs, 'lr': args.lr, 'lambda_sparse': args.lambda_sparse,
        'random_subset_acc_mean': rand_mean, 'random_subset_acc_std': rand_std,
        'baseline_test_acc': base_m, 'baseline_test_acc_std': base_s,
    }
    if counts_by_lambda is not None:
        summary.update({'concept_counts_by_lambda': counts_by_lambda,
                        'random_subset_acc_mean_by_lambda': rand_m_by_l,
                        'random_subset_acc_std_by_lambda': rand_s_by_l})
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(SAVE, f'random_retrain_ablation_{stamp}.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Fold the random-subset sweep back into lambda_gate_sweep.json.
    sweep['random_retrain'] = {k: summary[k] for k in (
        'concept_counts', 'random_subset_acc_mean', 'random_subset_acc_std',
        'retrain_random_trials', 'random_seed', 'epochs', 'lr', 'lambda_sparse')}
    if counts_by_lambda is not None:
        sweep['random_retrain'].update({
            'concept_counts_by_lambda': counts_by_lambda,
            'random_subset_acc_mean_by_lambda': rand_m_by_l,
            'random_subset_acc_std_by_lambda': rand_s_by_l})
    with open(sweep_path, 'w') as f:
        json.dump(sweep, f, indent=2)
    print(f'Merged random_retrain block into {sweep_path}')

    plot_path = plot_random_retrain_sweep(
        counts, None, None, rand_mean, rand_std, base_m, base_s, dataset)
    print(f'Saved results to {results_path} and standalone plot to {plot_path}')

    # Regenerate gated_sweep_{val,test}: the random-subset curve is overlaid on the test-split
    # accuracy axis; everything else comes straight from lambda_gate_sweep.json.
    if counts_by_lambda is not None and 'test_accuracy' in sweep:
        plot_lambda_gate_sweep(
            sweep['lambda_gates'], sweep['cea'], sweep['open_gates'], dataset,
            accs_std=sweep.get('cea_std'), n_open_std=sweep.get('open_gates_std'),
            baseline_acc=sweep.get('baseline_cea'), baseline_acc_std=sweep.get('baseline_cea_std'),
            test_accs=sweep['test_accuracy'], test_accs_std=sweep.get('test_accuracy_std'),
            baseline_test_acc=base_m, baseline_test_acc_std=base_s,
            val_metric_name='CEA',
            test_overlays=[(rand_m_by_l, rand_s_by_l, '#55A868', 'Random subset')],
        )
        print(f'Regenerated gated_sweep_test plot with random-subset overlay for {dataset}')


def main(args):
    dataset   = args.dataset
    n_classes = N_CLASSES[dataset]
    SAVE      = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE  = os.path.join(SAVE, 'activations')
    MODEL_DIR = os.path.join(SAVE, 'model')

    test_acts_path = os.path.join(ACT_SAVE, 'test_sae_acts.pt')
    if not os.path.exists(test_acts_path):
        raise FileNotFoundError(f'{test_acts_path} not found; run train_cbm.py for this dataset first')
    test_acts = torch.load(test_acts_path)
    _, test_labels = load_clip_features(ACT_SAVE, 'test')

    if args.mode == 'ranked-order':
        run_ranked_order(args, test_acts, test_labels, MODEL_DIR, SAVE, dataset, n_classes)
        return

    if args.mode == 'retention':
        run_retention(args, test_acts, test_labels, MODEL_DIR, SAVE, dataset, n_classes)
        return
    if args.mode == 'threshold':
        run_threshold(args, test_acts, test_labels, MODEL_DIR, SAVE, dataset, n_classes)
        return
    if args.mode == 'random-retrain':
        train_acts_path = os.path.join(ACT_SAVE, 'train_sae_acts.pt')
        if not os.path.exists(train_acts_path):
            raise FileNotFoundError(f'{train_acts_path} not found; run train_cbm.py for {dataset} first')
        train_acts = torch.load(train_acts_path)
        _, train_labels = load_clip_features(ACT_SAVE, 'train')
        run_random_retrain(args, train_acts, train_labels, test_acts, test_labels,
                           MODEL_DIR, SAVE, dataset, n_classes)
        return

    if not args.gated_run or not args.baseline_run:
        raise ValueError("open-closed mode needs both --gated-run and --baseline-run")
    gated_run_dir    = args.gated_run if os.path.isabs(args.gated_run) else os.path.join(MODEL_DIR, args.gated_run)
    baseline_run_dir = args.baseline_run if os.path.isabs(args.baseline_run) else os.path.join(MODEL_DIR, args.baseline_run)
    gated_by_seed    = _probe_paths_by_seed(gated_run_dir)
    baseline_by_seed = _probe_paths_by_seed(baseline_run_dir)

    seeds = sorted(set(gated_by_seed) & set(baseline_by_seed), key=lambda s: (isinstance(s, str), s))
    if not seeds:
        raise ValueError(
            f"No matching seeds between gated run '{args.gated_run}' ({sorted(gated_by_seed, key=str)}) "
            f"and baseline run '{args.baseline_run}' ({sorted(baseline_by_seed, key=str)}); "
            f"they must be trained with the same --seeds"
        )

    open_accs, closed_accs, random_accs, n_open_by_seed = [], [], [], []
    for seed in seeds:
        gated_probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
        gated_probe.load_state_dict(torch.load(gated_by_seed[seed], map_location='cpu'))
        gated_probe.eval()

        baseline_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        baseline_probe.load_state_dict(torch.load(baseline_by_seed[seed], map_location='cpu'))
        baseline_probe.eval()

        gates = torch.sigmoid(gated_probe.gate_logits).detach()
        open_idx   = (gates > args.gate_threshold).nonzero().squeeze(-1).tolist()
        closed_idx = (gates <= args.gate_threshold).nonzero().squeeze(-1).tolist()
        open_ranked   = sorted(open_idx,   key=lambda i: -gates[i].item())
        closed_ranked = sorted(closed_idx, key=lambda i:  gates[i].item())
        n_open_by_seed.append(len(open_idx))

        seed_str = seed if seed != 'single' else 0
        generator = torch.Generator().manual_seed(args.random_seed + seed_str)

        seed_open_accs   = [evaluate(baseline_probe, test_acts, test_labels, ablate_prefix(open_ranked, f))   for f in args.fractions]
        seed_closed_accs = [evaluate(baseline_probe, test_acts, test_labels, ablate_prefix(closed_ranked, f)) for f in args.fractions]
        seed_random_accs = [
            evaluate_random(baseline_probe, test_acts, test_labels, config.N_LEARNED_FEATURES,
                             round(f * len(open_ranked)), args.n_random_trials, generator)
            for f in args.fractions
        ]
        open_accs.append(seed_open_accs)
        closed_accs.append(seed_closed_accs)
        random_accs.append(seed_random_accs)
        print(f'[gated seed {seed}] open={len(open_idx)} closed={len(closed_idx)} '
              f'[baseline seed {seed}] unablated acc={seed_open_accs[0]:.4f}')

    open_accs, closed_accs, random_accs = torch.tensor(open_accs), torch.tensor(closed_accs), torch.tensor(random_accs)
    open_mean, open_std     = open_accs.mean(0).tolist(),   open_accs.std(0).tolist()
    closed_mean, closed_std = closed_accs.mean(0).tolist(), closed_accs.std(0).tolist()
    random_mean, random_std = random_accs.mean(0).tolist(), random_accs.std(0).tolist()

    for frac, om, ostd, cm, cstd, rm, rstd in zip(
        args.fractions, open_mean, open_std, closed_mean, closed_std, random_mean, random_std
    ):
        print(f'{frac * 100:5.1f}% ablated: open-set acc={om:.4f}+/-{ostd:.4f}  '
              f'closed-set acc={cm:.4f}+/-{cstd:.4f}  random acc={rm:.4f}+/-{rstd:.4f}')

    results = {
        'dataset': dataset, 'gated_run': args.gated_run, 'baseline_run': args.baseline_run,
        'gate_threshold': args.gate_threshold, 'seeds': [s if s != 'single' else None for s in seeds],
        'n_open_by_seed': n_open_by_seed, 'fractions': args.fractions,
        'n_random_trials': args.n_random_trials, 'random_seed': args.random_seed,
        'open_acc_mean': open_mean, 'open_acc_std': open_std,
        'closed_acc_mean': closed_mean, 'closed_acc_std': closed_std,
        'random_acc_mean': random_mean, 'random_acc_std': random_std,
    }
    results_path = os.path.join(SAVE, f"concept_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {results_path}')

    plot_concept_ablation(args.fractions, open_mean, closed_mean, open_std, closed_std, dataset,
                           random_mean=random_mean, random_std=random_std)


if __name__ == '__main__':
    main(parse_args())
