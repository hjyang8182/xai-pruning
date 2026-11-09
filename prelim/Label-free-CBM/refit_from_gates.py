"""Re-run an LF-CBM gate sweep's cold-start refits (gate-selected vs. matched random) from its saved gates.

train_cbm_seed_sweep.py --random_control refits both cold-start arms for --refit_epochs (20) at
--gate_batch_size (64). On CUB that is ~1.9k Adam steps, not quite converged, and the random arm
(weaker features) lags furthest: at lambda_gate 1e-3 the gate-vs-random gap is 1.3 points at 20
epochs and 0.8 at 300. This reloads the sweep's shared stage (shared_from: train_c / val_c, the
exact head inputs) and, for every point's session_dir/seed<s>/{W_g,b_g,gate_logits}.pt, redoes

  gated_refit_cold  fresh head on K = sigmoid(gate_logits) > tau   } gating.matched_random_refits,
  random_refit      fresh head on |K| uniformly random concepts    } identical recipe

with --epochs. With --arms magnitude group_lasso it instead adds the stronger size-matched controls
(gating.size_matched_refits: top-|K| by a dense head's weight magnitude, and the group-lasso path at
|K|), same recipe, keeping every arm already in the input json. With --arms threshold, threshold selection:
a fresh head on K_t = {gate logit > t} for every cut t in gating.LOGIT_THRESHOLDS (threshold_<t>_refit),
plus gate_logit_stats (how many logits sit within gating.NEAR_ZERO of the 0.5 boundary). The gated arm and the warm-start
gated_refit are kept as they were. Writes
<dataset>_gate_sweep_<stamp>.json next to the original (same schema, plus "refit_rerun_of" and the
new recipe), so plot_pipeline_grids.py picks it up as the newest sweep.

    python refit_from_gates.py --gate-sweep seed_sweep_results/cub_gate_sweep_2026_10_06_09_16.json --epochs 300
"""
import argparse
import json
import os
import statistics
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gating import (SIZE_CONTROLS, LOGIT_THRESHOLDS, matched_random_refits, size_matched_refits,  # noqa: E402
                    threshold_arm, threshold_refits)


def summarise(accs, ks, seeds):
    std = lambda v: statistics.pstdev(v) if len(v) > 1 else 0.0  # noqa: E731
    return {"seeds": seeds, "acc_per_seed": accs, "acc_mean": statistics.fmean(accs), "acc_std": std(accs),
            "n_open_per_seed": ks, "n_open_mean": statistics.fmean(ks), "n_open_std": std(ks)}


def main(cli):
    with open(cli.gate_sweep) as f:
        sweep = json.load(f)
    shared_dir = sweep["points"][0]["shared_from"]
    d = {k: torch.load(os.path.join(shared_dir, f"{k}.pt"), map_location="cpu") for k in ("train_c", "train_y", "val_c", "val_y")}
    tau = sweep.get("refit_tau", 0.5)
    print(f"shared {shared_dir}: train {tuple(d['train_c'].shape)}, eval {tuple(d['val_c'].shape)}; "
          f"{cli.epochs} epochs, batch {cli.batch_size}, lr {cli.lr}", flush=True)

    seeds_of = [[s for s in point["seeds"] if not cli.seeds or s in cli.seeds] for point in sweep["points"]]
    run_dirs = {(i, seed): os.path.join(point["session_dir"], f"seed{seed}")
                for i, point in enumerate(sweep["points"]) for seed in seeds_of[i]}
    gates, logits_of = {}, {}
    for (i, seed), run in run_dirs.items():
        assert sweep["points"][i]["shared_from"] == shared_dir, "every point of a sweep shares one shared stage"
        # sigmoid on the training device, as the sweep computed it: hard-STE logits that chatter at
        # ~1e-7 around 0 can land on either side of 0.5 depending on CPU vs. GPU rounding
        logits = torch.load(os.path.join(run, "gate_logits.pt"), map_location="cpu").float()
        gates[(i, seed)] = torch.sigmoid(logits.to(cli.device)).cpu()
        logits_of[(i, seed)] = logits
    recipe = dict(epochs=cli.epochs, lr=cli.lr, batch_size=cli.batch_size, device=cli.device)

    controls = [a for a in cli.arms if a in SIZE_CONTROLS]
    size_ctrl = {}
    for seed in sorted({s for _, s in run_dirs}):
        if controls:
            ks = [int((g > tau).sum()) for (i, s), g in gates.items() if s == seed]
            size_ctrl[seed] = size_matched_refits(d["train_c"], d["train_y"], d["val_c"], d["val_y"], ks,
                                                  methods=controls, seed=seed, **recipe)

    points = []
    for i, point in enumerate(sweep["points"]):
        rows = {}
        for seed in seeds_of[i]:
            gate_keep = gates[(i, seed)] > tau
            k = int(gate_keep.sum())
            msg = f"lam {point['lambda_gate']:g} seed {seed} |K|={k}:"
            if "cold_random" in cli.arms:
                W = torch.load(os.path.join(run_dirs[(i, seed)], "W_g.pt"), map_location="cpu")
                b = torch.load(os.path.join(run_dirs[(i, seed)], "b_g.pt"), map_location="cpu")
                ctrl = matched_random_refits(W, b, gates[(i, seed)], d["train_c"], d["train_y"], d["val_c"], d["val_y"],
                                             tau=tau, fold_gate=False, seed=seed, **recipe)
                for arm, key in (("gated_refit_cold", "gated_cold"), ("random_refit", "random")):
                    rows.setdefault(arm, ([], []))
                    rows[arm][0].append(ctrl[key]["acc"])
                    rows[arm][1].append(ctrl[key]["n_concepts"])
                j = point["seeds"].index(seed)
                msg += (f"  gate-selected {point['gated_refit_cold']['acc_per_seed'][j]:.4f}->{ctrl['gated_cold']['acc']:.4f}"
                        f"  random {point['random_refit']['acc_per_seed'][j]:.4f}->{ctrl['random']['acc']:.4f}")
            if "threshold" in cli.arms:
                res, n_near = threshold_refits(d["train_c"], d["train_y"], d["val_c"], d["val_y"], logits_of[(i, seed)],
                                               seed=seed, **recipe)
                rows.setdefault("gate_logit_stats", ([], []))
                rows["gate_logit_stats"][0].append(n_near)
                rows["gate_logit_stats"][1].append(k)
                for t, r in res.items():
                    rows.setdefault(threshold_arm(t), ([], []))
                    rows[threshold_arm(t)][0].append(r["acc"])
                    rows[threshold_arm(t)][1].append(r["n_concepts"])
                msg += f"  near-0 logits {n_near}; " + " ".join(f"{t:g}:{r['n_concepts']}/{r['acc']:.4f}" for t, r in res.items())
            for method in controls:
                r = size_ctrl[seed][method][k]
                arm = f"{method}_refit"
                rows.setdefault(arm, ([], [], []))
                rows[arm][0].append(r["acc"])
                rows[arm][1].append(r["n_concepts"])
                rows[arm][2].append(int((r["keep"] & gate_keep).sum()))
                msg += f"  {method} {r['acc']:.4f} (overlap {rows[arm][2][-1]})"
            print(msg, flush=True)
        new = {**point}
        for arm, r in rows.items():
            if arm == "gate_logit_stats":  # (n |logit| < NEAR_ZERO, n open at 0.5) per seed
                new[arm] = {"seeds": seeds_of[i], "n_near_zero_per_seed": r[0], "n_open_per_seed": r[1],
                            "n_near_zero_mean": statistics.fmean(r[0]), "n_open_mean": statistics.fmean(r[1])}
                continue
            new[arm] = summarise(r[0], r[1], seeds_of[i])
            if len(r) == 3:
                new[arm]["overlap_with_gate_per_seed"] = r[2]
        points.append(new)

    out = {**sweep, "points": points}
    if "cold_random" in cli.arms:
        out.update({"refit_rerun_of": os.path.abspath(cli.gate_sweep), "cold_refit_epochs": cli.epochs,
                    "cold_refit_batch_size": cli.batch_size, "cold_refit_lr": cli.lr})
    if "threshold" in cli.arms:
        out.update({"threshold_refits_of": os.path.abspath(cli.gate_sweep), "logit_thresholds": list(LOGIT_THRESHOLDS),
                    "threshold_recipe": {"epochs": cli.epochs, "batch_size": cli.batch_size, "lr": cli.lr}})
    if controls:
        out.update({"size_controls_of": os.path.abspath(cli.gate_sweep), "size_controls": controls,
                    "size_controls_recipe": {"epochs": cli.epochs, "batch_size": cli.batch_size, "lr": cli.lr}})
    path = cli.out or os.path.join(os.path.dirname(cli.gate_sweep), "{}_gate_sweep_{}.json".format(
        sweep["dataset"], datetime.now().strftime("%Y_%m_%d_%H_%M")))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved lambda_gate sweep to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate-sweep", required=True, help="the sweep's <dataset>_gate_sweep_<stamp>.json")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--arms", nargs="+", default=["cold_random"], choices=["cold_random", *SIZE_CONTROLS, "threshold"],
                   help="cold_random: redo gate-selected + random cold refits; magnitude / group_lasso: add those "
                        "size-matched controls (other arms are carried over from --gate-sweep)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="only these seeds (run seeds on separate GPUs, then merge_seed_parts.py)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
