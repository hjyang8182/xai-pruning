"""Redo a UCBM gate sweep's matched random control from its saved classifiers, on the right signal.

The sweep's random control (UCBM.prune_and_refit, random_control=True) refit on the post-gate
concept signal divided by the soft gate. Under gate_forward="hard" the closed concepts are exactly
zero after the gate, so they stay zero: a random |K| draw only had signal on the concepts it
happened to share with K (~|K|^2/N of them), which made the random arm far too weak on CIFAR-100
and Places-365 (big dictionaries, small K). This re-scores, for every saved
gated_lam<v>_seed_sweep_*/seed<s>/classifier.pth, on the true pre-gate signal (the selection
output -- relu / TopK -- with the gate switched off, UCBM.get_pre_gate_activations):

  gated_refit_cold   fresh head on K = gate > tau
  random_refit       fresh head on |K| concepts uniformly from the whole dictionary (corrected)
  random_live_refit  fresh head on |K| concepts uniformly from the live ones (non-zero pre-gate
                     signal on at least one training image), so dead concepts can't be drawn

and, with --arms magnitude group_lasso, the stronger size-matched controls on the same signal
(gating.size_matched_refits: top-|K| by a dense head's weight magnitude, and the group-lasso path at
|K|; every other arm is then carried over from --gate-sweep),

and, with --arms threshold, threshold selection (gating.threshold_refits: a fresh head on K_t = {gate
logit > t} for every cut in gating.LOGIT_THRESHOLDS, as threshold_<t>_refit, each row also recording how
many gate logits sit within gating.NEAR_ZERO of the 0.5 boundary),

all trained exactly like the sweep's arms (gating.prune_and_refit, cold start, the sweep's refit
epochs, the classifier's batch size, lr 1e-3, gating.matched_random_refits' seed offset). The sweep's
own random arm is kept as "random_refit_zeroed". Writes <dataset>_gate_sweep_<stamp>.json next to
the original (same schema, plus "random_control_rerun_of"), so plotting picks it up as the newest.

    python rerun_random_control.py -d cifar100 -b resnet50_v2 -c concepts_1000_64 \\
        --gate-sweep save/RESULTS/cifar100-resnet50_v2/classifier/concepts_1000_64/seed_sweep_results/cifar100_gate_sweep_2026_10_06_15_12.json
"""
import argparse
import glob
import json
import os
import statistics
from datetime import datetime

import numpy as np
import torch

from constants import RESULT_PATH
from data_loader import load_data
from model_loader import get_model
from plotter.plotter import Plotter
from ucbm import UCBM, _evaluate_head, _prune_and_refit
from gating import LOGIT_THRESHOLDS, SIZE_CONTROLS, size_matched_refits, threshold_arm, threshold_refits

RANDOM_ARMS = ("gated_refit_cold", "random_refit", "random_live_refit")


def summarise(rows):
    out = {"seeds": [r["seed"] for r in rows]}
    for key in ("acc", "n_open", "n_live", "n_k_live", "overlap_with_gate", "n_near_zero"):
        if key not in rows[0]:
            continue
        vals = [r[key] for r in rows]
        out[f"{key}_per_seed"] = vals
        out[f"{key}_mean"] = statistics.fmean(vals)
        out[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    # dense refit over the kept set: NEC = |set|
    out["nec_per_seed"], out["nec_mean"], out["nec_std"] = out["n_open_per_seed"], out["n_open_mean"], out["n_open_std"]
    return out


def draw(pool, k, seed):
    """k indices drawn uniformly from the bool mask `pool` (gating.matched_random_refits' seeding)."""
    idx = pool.nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed + 7919)
    keep = torch.zeros_like(pool)
    keep[idx[torch.randperm(len(idx), generator=g)[:min(k, len(idx))]]] = True
    return keep


def seed_dirs(session):
    dirs = [d for d in glob.glob(os.path.join(session, "seed*")) if os.path.basename(d)[4:].isdigit()
            and os.path.exists(os.path.join(d, "classifier.pth"))]
    return sorted(dirs, key=lambda d: int(os.path.basename(d)[4:]))


def main(args):
    act_bank_path = os.path.join(Plotter(os.path.join(RESULT_PATH, f"{args.dataset}-{args.backbone}/")).get_concept_data_path(),
                                 args.concept_data)
    train_data = load_data(args.dataset, True, reorder_idcs=True)
    test_data = load_data(args.dataset, False, reorder_idcs=True)
    y_train, y_test = torch.tensor(train_data.targets), torch.tensor(test_data.targets)
    backbone = get_model(args.backbone, args.device).g
    with open(args.gate_sweep) as f:
        sweep = json.load(f)
    epochs, tau = args.epochs or sweep.get("refit_epochs", 20), sweep.get("refit_tau", 0.5)

    controls = [a for a in args.arms if a in SIZE_CONTROLS]
    points = []
    for point in sweep["points"]:
        lam = point["lambda_gate"]
        if args.lams and not any(np.isclose(lam, l) for l in args.lams):
            continue
        arms = ((list(RANDOM_ARMS) if "random" in args.arms else []) + [f"{c}_refit" for c in controls]
                + ([threshold_arm(t) for t in LOGIT_THRESHOLDS] if "threshold" in args.arms else []))
        rows = {arm: [] for arm in arms}
        for seed_dir in seed_dirs(point["session_dir"]):
            seed = int(os.path.basename(seed_dir)[4:])
            if args.seeds and seed not in args.seeds:
                continue
            ph = UCBM.load_from_file(seed_dir, "classifier.pth", device=args.device, backbone_p=backbone)
            gate = ph.get_gate_values().detach().cpu().float()
            pre_train = ph.get_pre_gate_activations(train_data, act_bank_path, "train")
            pre_test = ph.get_pre_gate_activations(test_data, act_bank_path, "test")
            W, b = ph._classifier.linear.weight.detach(), ph._classifier.linear.bias.detach()
            keep = gate > tau
            live = (pre_train != 0).any(0)
            k = int(keep.sum())
            info = {"seed": seed, "n_live": int(live.sum()), "n_k_live": int((keep & live).sum())}
            msg = f"lam {lam:g} seed {seed}: |K|={k} ({info['n_k_live']} live), live {info['n_live']}/{len(live)}"
            if "random" in args.arms:
                masks = {"gated_refit_cold": keep, "random_refit": draw(torch.ones_like(keep), k, seed),
                         "random_live_refit": draw(live, k, seed)}
                for arm, mask in masks.items():
                    W_r, b_r, m = _prune_and_refit(W, b, gate, pre_train, y_train, keep=mask, tau=tau, fold_gate=True,
                                                   epochs=epochs, lr=1e-3, batch_size=ph._batch_size,
                                                   device=args.device, seed=seed, warm_start=False)
                    rows[arm].append({**info, "acc": _evaluate_head(W_r, b_r, pre_test, y_test, device=args.device),
                                      "n_open": int(m.sum())})
                old, old_seeds = point.get("random_refit", {}), point.get("random_refit", {}).get("seeds", sweep.get("seeds", []))
                old_acc = old["acc_per_seed"][old_seeds.index(seed)] if seed in old_seeds else float("nan")
                msg += (f"  gate-selected {rows['gated_refit_cold'][-1]['acc']:.4f}  random {old_acc:.4f} (zeroed) -> "
                        f"{rows['random_refit'][-1]['acc']:.4f}  random-live {rows['random_live_refit'][-1]['acc']:.4f}")
            if controls:
                ctrl = size_matched_refits(pre_train, y_train, pre_test, y_test, [k], methods=controls, epochs=epochs,
                                           lr=1e-3, batch_size=ph._batch_size, device=args.device, seed=seed)
                for c in controls:
                    r = ctrl[c][k]
                    rows[f"{c}_refit"].append({**info, "acc": r["acc"], "n_open": r["n_concepts"],
                                               "overlap_with_gate": int((r["keep"] & keep).sum())})
                    msg += f"  {c} {r['acc']:.4f} (overlap {rows[f'{c}_refit'][-1]['overlap_with_gate']})"
            if "threshold" in args.arms:
                # the gate is sigmoid(logits / T): cut on its logit so the thresholds sit where the other pipelines' do
                logit = torch.logit(gate.double().clamp(1e-12, 1 - 1e-12)).float()
                res, n_near = threshold_refits(pre_train, y_train, pre_test, y_test, logit, epochs=epochs, lr=1e-3,
                                               batch_size=ph._batch_size, device=args.device, seed=seed)
                info_t = {**info, "n_near_zero": n_near}
                for t, r in res.items():
                    rows[threshold_arm(t)].append({**info_t, "acc": r["acc"], "n_open": r["n_concepts"]})
                msg += f"  near-0 logits {n_near}; " + " ".join(f"{t:g}:{r['n_concepts']}/{r['acc']:.4f}" for t, r in res.items())
            print(msg, flush=True)
            del pre_train, pre_test
        if any(rows.values()):
            new = {**point}
            if "random" in args.arms:
                new["random_refit_zeroed"] = point.get("random_refit_zeroed", point.get("random_refit"))
            new.update({arm: summarise(r) for arm, r in rows.items()})
            points.append(new)

    out = {**sweep, "points": points}
    if "random" in args.arms:
        out.update({"random_control_rerun_of": os.path.abspath(args.gate_sweep), "random_control_refit_epochs": epochs})
    if controls:
        out.update({"size_controls_of": os.path.abspath(args.gate_sweep), "size_controls": controls,
                    "size_controls_refit_epochs": epochs})
    if "threshold" in args.arms:
        out.update({"threshold_refits_of": os.path.abspath(args.gate_sweep), "logit_thresholds": list(LOGIT_THRESHOLDS),
                    "threshold_refit_epochs": epochs})
    path = args.out or os.path.join(os.path.dirname(args.gate_sweep), "{}_gate_sweep_{}.json".format(
        sweep["dataset"], datetime.now().strftime("%Y_%m_%d_%H_%M")))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved lambda_gate sweep to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("-b", "--backbone", required=True)
    p.add_argument("-c", "--concept_data", required=True)
    p.add_argument("--gate-sweep", required=True, help="the sweep's <dataset>_gate_sweep_<stamp>.json")
    p.add_argument("--epochs", type=int, default=None, help="refit epochs (default: the sweep's refit_epochs)")
    p.add_argument("--lams", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=["random"], choices=["random", *SIZE_CONTROLS, "threshold"],
                   help="random: redo gate-selected / random / random-live cold refits; magnitude / group_lasso: add "
                        "those size-matched controls (other arms are carried over from --gate-sweep)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
