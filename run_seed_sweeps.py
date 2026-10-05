#!/usr/bin/env python
"""One entry point for the multi-seed baseline / gated sweeps of all four CBM pipelines.

    python run_seed_sweeps.py                                   # everything, 5 seeds, both arms
    python run_seed_sweeps.py -m vlgcbm ucbm -d cub             # subset
    python run_seed_sweeps.py --arms gated --gate-forward soft  # gated only, soft gate
    python run_seed_sweeps.py --lambda-gate ucbm:cub=3e-4 --extra ucbm "--epochs 100"
    python run_seed_sweeps.py --dry-run                         # just print the commands

Each (model, dataset, arm) is launched as a subprocess in that pipeline's own conda env and
working directory, exactly as you would run it by hand; the per-model flag recipes live in
RECIPES below and are the single thing to edit when a hyperparameter changes. Every run's
stdout+stderr is teed to sweep_logs/<stamp>/<model>_<dataset>_<arm>.log, and a
manifest.json + a final summary table (acc mean/std, open concepts, refit acc) are written
from whatever results JSON each pipeline produced.

Where the pipelines write their own results (unchanged by this script):
  dncbm   data/<dataset>/model/{baseline,gated}_<stamp>/probe_config.json
  lfcbm   prelim/Label-free-CBM/seed_sweep_results/<dataset>_<arm>_seed_sweep_<stamp>.json
  ucbm    prelim/ucbm/save/RESULTS/<dataset>-<backbone>/classifier/<concepts>/seed_sweep_results/
  vlgcbm  prelim/VLG-CBM/saved_models/<dataset>/seed_sweep_results/
"""
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDA_ENVS = Path(os.environ.get('CONDA_ENVS_DIR', Path.home() / '.conda' / 'envs'))
LOG_ROOT = ROOT / 'sweep_logs'

DEFAULT_SEEDS = [0, 10, 20, 30, 40]

# ---------------------------------------------------------------------------------------------
# How each pipeline is driven. `arms` maps our arm name -> the flags that select it; a model
# with `joint_arms` takes both arms in one invocation (VLG-CBM's --arms sparse gated).
# ---------------------------------------------------------------------------------------------
MODELS = {
    'dncbm': dict(
        cwd=ROOT, env='xai_env', script='train_cbm.py',
        seeds_flag='--seeds', lambda_gate_flag='--lambda-gate', gate_forward_flag='--gate-forward',
        arms={'baseline': [], 'gated': ['-g']}, common=[],
    ),
    'lfcbm': dict(
        cwd=ROOT / 'prelim' / 'Label-free-CBM', env='lfcbm', script='train_cbm_seed_sweep.py',
        seeds_flag='--seeds', lambda_gate_flag='--gate_lam', gate_forward_flag='--gate_forward',
        arms={'baseline': [], 'gated': ['-g']}, common=['--save_probes'],
    ),
    'ucbm': dict(
        cwd=ROOT / 'prelim' / 'ucbm', env='ucbm', script='train_cbm_seed_sweep.py',
        seeds_flag='--seeds', lambda_gate_flag='--lambda_gate', gate_forward_flag='--gate_forward',
        arms={'baseline': [], 'gated': ['-g']}, common=['--save_classifiers'],
    ),
    'vlgcbm': dict(
        cwd=ROOT / 'prelim' / 'VLG-CBM', env='vlg-cbm', script='train_cbm_seed_sweep.py',
        seeds_flag='--seeds', lambda_gate_flag='--lambda_gate', gate_forward_flag='--gate_forward',
        arms={'baseline': 'sparse', 'gated': 'gated'}, joint_arms=True, common=['--save_probes'],
    ),
}

# ---------------------------------------------------------------------------------------------
# Per-(model, dataset) recipes: the flags you'd type by hand for that pipeline, plus the
# lambda_gate used for the gated arm. `gated_args` are appended only to the gated arm.
# Values are the ones from the most recent runs of each pipeline; override per run with
# --lambda-gate model:dataset=VALUE and --extra model "...".
# ---------------------------------------------------------------------------------------------
RECIPES = {
    'dncbm': {
        'cifar100':   dict(args=['-d', 'cifar100'], lambda_gate=1e-4),
        'cub':        dict(args=['-d', 'cub'], lambda_gate=7e-4),
        'places365':  dict(args=['-d', 'places365', '--epochs', '150'], lambda_gate=1e-4),
        'oxford_pet': dict(args=['-d', 'oxford_pet'], lambda_gate=1e-2),
        'food101':    dict(args=['-d', 'food101'], lambda_gate=1e-4),
    },
    'lfcbm': {
        'cifar100':  dict(args=['--dataset', 'cifar100', '--concept_set', 'data/concept_sets/cifar100_filtered.txt'],
                          gated_args=['--gate_epochs', '150'], lambda_gate=5e-4),
        'cub':       dict(args=['--dataset', 'cub', '--backbone', 'resnet18_cub',
                               '--concept_set', 'data/concept_sets/cub_filtered.txt',
                               '--feature_layer', 'features.final_pool', '--clip_cutoff', '0.26',
                               '--n_iters', '5000', '--lam', '0.0002'],
                          gated_args=['--gate_epochs', '150'], lambda_gate=5e-4),
        'places365': dict(args=['--dataset', 'places365', '--backbone', 'resnet50',
                               '--concept_set', 'data/concept_sets/places365_filtered.txt',
                               '--clip_cutoff', '0.28', '--n_iters', '80', '--lam', '0.0003'],
                          gated_args=['--gate_epochs', '150'], lambda_gate=5e-4),
    },
    'ucbm': {
        'cifar100':  dict(args=['-d', 'cifar100', '-b', 'resnet50_v2', '-c', 'concepts_1000_64',
                               '--normalize_concepts', '--dropout_p', '0.1'],
                          lambda_gate=1e-4),
        'cub':       dict(args=['-d', 'cub', '-b', 'cub_rn18', '-c', 'concepts_200_64',
                               '--normalize_concepts', '--epochs', '150', '--scale_choose', 'no', '--k', '66'],
                          lambda_gate=5e-4),
        'places365': dict(args=['-d', 'places365', '-b', 'places365_rn18', '-c', 'concepts_3650_64'],
                          gated_args=['--lam_pi', '0'], lambda_gate=3e-4),
    },
    'vlgcbm': {
        'cifar100':  dict(args=['--config', 'configs/cifar100.json', '--dense_lr', '1e-4'], lambda_gate=1e-4),
        'cub':       dict(args=['--config', 'configs/cub.json', '--saga_n_iters', '1000'], lambda_gate=5e-5),
        'places365': dict(args=['--config', 'configs/places365.json', '--dense_lr', '1e-4',
                               '--saga_n_iters', '200', '--val_split', '0.1'],
                          lambda_gate=1e-4),
    },
}

ALL_DATASETS = sorted({d for m in RECIPES.values() for d in m})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-m', '--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.add_argument('-d', '--datasets', nargs='+', choices=ALL_DATASETS, default=['cifar100', 'cub', 'places365'],
                   help='(model, dataset) pairs without a recipe are skipped with a note')
    p.add_argument('-a', '--arms', nargs='+', choices=['baseline', 'gated'], default=['baseline', 'gated'])
    p.add_argument('-s', '--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--gate-forward', choices=['soft', 'hard'], default='hard',
                   help='Gate forward pass for every gated arm (STE hard mask by default)')
    p.add_argument('--lambda-gate', action='append', default=[], metavar='MODEL[:DATASET]=VALUE',
                   help='Override the recipe lambda_gate, e.g. "ucbm:cub=3e-4" or "lfcbm=1e-4" (all its datasets). Repeatable.')
    p.add_argument('--extra', nargs=2, action='append', default=[], metavar=('MODEL', 'ARGS'),
                   help='Extra flags appended verbatim to every run of MODEL, e.g. --extra ucbm "--epochs 100". Repeatable.')
    p.add_argument('--gpu', default=None, help='Value for CUDA_VISIBLE_DEVICES (default: inherit)')
    p.add_argument('--no-share', action='store_true',
                   help='LF-CBM: do not cache the seed-independent stage from the baseline run for the gated run')
    p.add_argument('--stop-on-error', action='store_true', help='Abort the sweep at the first failed run (default: keep going, report at the end)')
    p.add_argument('--dry-run', action='store_true', help='Print the commands and exit')
    return p.parse_args()


def parse_lambda_overrides(items):
    out = {}
    for item in items:
        key, _, val = item.partition('=')
        if not val:
            sys.exit(f'--lambda-gate expects MODEL[:DATASET]=VALUE, got {item!r}')
        model, _, dataset = key.partition(':')
        if model not in MODELS:
            sys.exit(f'--lambda-gate: unknown model {model!r}')
        out[(model, dataset or None)] = float(val)
    return out


def lambda_gate_for(overrides, model, dataset, recipe):
    return overrides.get((model, dataset), overrides.get((model, None), recipe['lambda_gate']))


def python_for(model):
    exe = CONDA_ENVS / MODELS[model]['env'] / 'bin' / 'python'
    if not exe.exists():
        sys.exit(f'{model}: interpreter {exe} not found (set CONDA_ENVS_DIR if your envs live elsewhere)')
    return str(exe)


def build_runs(args, lambda_overrides, extra_by_model, stamp):
    """Expand the requested grid into an ordered list of run dicts (one subprocess each)."""
    runs = []
    for model in args.models:
        spec = MODELS[model]
        for dataset in args.datasets:
            recipe = RECIPES[model].get(dataset)
            if recipe is None:
                print(f'[skip] {model}/{dataset}: no recipe in RECIPES')
                continue
            lam = lambda_gate_for(lambda_overrides, model, dataset, recipe)
            base = [python_for(model), spec['script'], *recipe['args'], spec['seeds_flag'],
                    *map(str, args.seeds), *spec['common'], *extra_by_model.get(model, [])]
            gated_flags = [spec['lambda_gate_flag'], repr(lam), spec['gate_forward_flag'], args.gate_forward,
                           *recipe.get('gated_args', [])]

            if spec.get('joint_arms'):
                arm_names = [spec['arms'][a] for a in args.arms]
                cmd = base + ['--arms', *arm_names]
                if 'gated' in args.arms:
                    cmd += gated_flags
                runs.append(dict(model=model, dataset=dataset, arm='+'.join(args.arms), cmd=cmd,
                                 cwd=spec['cwd'], lambda_gate=lam if 'gated' in args.arms else None))
                continue

            shared_dir = None
            if model == 'lfcbm' and not args.no_share and args.arms == ['baseline', 'gated']:
                shared_dir = f'seed_sweep_results/shared_{dataset}_{stamp}'
            for arm in args.arms:
                cmd = base + spec['arms'][arm]
                if arm == 'gated':
                    cmd += gated_flags
                if shared_dir:
                    cmd += ['--save_shared', shared_dir] if arm == 'baseline' else ['--load_shared', shared_dir]
                runs.append(dict(model=model, dataset=dataset, arm=arm, cmd=cmd, cwd=spec['cwd'],
                                 lambda_gate=lam if arm == 'gated' else None, shared_dir=shared_dir))
    return runs


def run_one(run, log_path, env):
    """Stream the subprocess's combined output to the terminal and the log; return rc + 'Saved results to' path."""
    saved = None
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(run['cmd'], cwd=run['cwd'], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            m = re.search(r'Saved results to (\S+)', line)
            if m:
                saved = m.group(1)
        proc.wait()
    return proc.returncode, saved


# ---------------------------------------------------------------------------------------------
# Result collection: each pipeline writes a differently shaped JSON; flatten to one row per arm.
# ---------------------------------------------------------------------------------------------
def find_dncbm_result(dataset, arm, started_at):
    cands = [p for p in glob.glob(str(ROOT / 'data' / dataset / 'model' / f'{arm}_*' / 'probe_config.json'))
             if os.path.getmtime(p) >= started_at]
    return max(cands, key=os.path.getmtime) if cands else None


def summarise_result(model, run, result_path):
    rows = []
    if not result_path or not os.path.exists(result_path):
        return rows
    with open(result_path) as f:
        d = json.load(f)
    if model == 'dncbm':
        arm = run['arm']
        gates = d.get('open_gates_by_seed')
        n_open = d.get('n_concepts_mean', sum(gates.values()) / len(gates) if gates else None)
        rows.append(dict(arm=arm, acc_mean=d.get('accuracy_mean'), acc_std=d.get('accuracy_std'), n_open=n_open))
        if 'refit_accuracy_mean' in d:
            rows.append(dict(arm='gated_refit', acc_mean=d['refit_accuracy_mean'], acc_std=d.get('refit_accuracy_std'),
                             n_open=d.get('refit_n_concepts_mean', n_open)))
        return rows
    for arm in ('baseline', 'sparse', 'dense', 'gated', 'gated_refit'):
        if isinstance(d.get(arm), dict) and 'acc_mean' in d[arm]:
            rows.append(dict(arm=arm, acc_mean=d[arm]['acc_mean'], acc_std=d[arm].get('acc_std'), n_open=d[arm].get('n_open_mean')))
    return rows


def fmt(x, nd=4):
    return '-' if x is None else f'{x:.{nd}f}'


def main():
    args = parse_args()
    lambda_overrides = parse_lambda_overrides(args.lambda_gate)
    extra_by_model = {}
    for model, extra in args.extra:
        if model not in MODELS:
            sys.exit(f'--extra: unknown model {model!r}')
        extra_by_model.setdefault(model, []).extend(shlex.split(extra))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    runs = build_runs(args, lambda_overrides, extra_by_model, stamp)
    if not runs:
        sys.exit('nothing to run')

    print(f'\n{len(runs)} run(s), seeds {args.seeds}, gate_forward={args.gate_forward}\n')
    for i, run in enumerate(runs, 1):
        rel = os.path.relpath(run['cwd'], ROOT) or '.'
        lam = f'  (lambda_gate={run["lambda_gate"]})' if run['lambda_gate'] is not None else ''
        print(f'[{i:2d}] {run["model"]}/{run["dataset"]}/{run["arm"]}{lam}\n     cd {rel} && {shlex.join(run["cmd"])}')
    if args.dry_run:
        return

    log_dir = LOG_ROOT / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    manifest = dict(stamp=stamp, seeds=args.seeds, gate_forward=args.gate_forward, runs=[])
    manifest_path = log_dir / 'manifest.json'
    table = []
    failed_shared = set()

    for i, run in enumerate(runs, 1):
        tag = f'{run["model"]}_{run["dataset"]}_{run["arm"].replace("+", "_")}'
        log_path = log_dir / f'{tag}.log'
        print(f'\n{"=" * 100}\n[{i}/{len(runs)}] {tag}   (log: {os.path.relpath(log_path, ROOT)})\n{"=" * 100}')

        # LF-CBM gated arm falls back to rebuilding the shared stage if the baseline run that
        # was supposed to cache it failed.
        if run.get('shared_dir') and run['arm'] == 'gated' and run['shared_dir'] in failed_shared:
            print('baseline run failed, so rebuilding the shared stage instead of --load_shared')
            k = run['cmd'].index('--load_shared')
            del run['cmd'][k:k + 2]

        t0 = time.time()
        rc, saved = run_one(run, log_path, env)
        dt = time.time() - t0

        if run['model'] == 'dncbm':
            saved = find_dncbm_result(run['dataset'], run['arm'], t0)
        elif saved and not os.path.isabs(saved):
            saved = str(run['cwd'] / saved)

        entry = dict(model=run['model'], dataset=run['dataset'], arm=run['arm'], cmd=shlex.join(run['cmd']),
                     cwd=str(run['cwd']), returncode=rc, seconds=round(dt), log=str(log_path), result=saved,
                     lambda_gate=run['lambda_gate'])
        manifest['runs'].append(entry)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        if rc != 0:
            print(f'\n!!! {tag} FAILED (rc={rc}) after {dt / 60:.1f} min -- see {log_path}')
            if run.get('shared_dir') and run['arm'] == 'baseline':
                failed_shared.add(run['shared_dir'])
            if args.stop_on_error:
                break
            table.append(dict(model=run['model'], dataset=run['dataset'], arm=run['arm'], acc_mean=None, acc_std=None,
                              n_open=None, status='FAILED'))
            continue

        print(f'\n{tag} done in {dt / 60:.1f} min -> {saved}')
        rows = summarise_result(run['model'], run, saved)
        if not rows:
            table.append(dict(model=run['model'], dataset=run['dataset'], arm=run['arm'], acc_mean=None,
                              acc_std=None, n_open=None, status='ok (no result json found)'))
        for r in rows:
            table.append(dict(model=run['model'], dataset=run['dataset'], **r, status='ok'))

    manifest['summary'] = table
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n{"=" * 100}\nSUMMARY  (seeds {args.seeds}, gate_forward={args.gate_forward})\n{"=" * 100}')
    print(f'{"model":8s} {"dataset":11s} {"arm":12s} {"acc":>8s} {"std":>8s} {"n_open":>8s}  status')
    for r in table:
        print(f'{r["model"]:8s} {r["dataset"]:11s} {r["arm"]:12s} {fmt(r["acc_mean"]):>8s} {fmt(r["acc_std"]):>8s} '
              f'{fmt(r["n_open"], 1):>8s}  {r["status"]}')
    print(f'\nmanifest: {manifest_path}')
    n_failed = sum(r['status'] == 'FAILED' for r in table)
    sys.exit(1 if n_failed else 0)


if __name__ == '__main__':
    main()
