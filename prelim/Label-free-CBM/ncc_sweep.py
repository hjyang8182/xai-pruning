"""NCC_tau (explanation_size.number_of_contributing_concepts) for every head of an LF-CBM gate sweep and
native-sparsity sweep, so accuracy can be compared at equal NCC as De Santis et al. (ICLR 2026) do.

Arms, per (lambda, seed), each scored on the sweep's shared eval set (shared_from/val_c: the exact head
inputs) with its own accuracy recomputed on the same images:
  gated        the gated head W_g (hard gate already folded in)
  gated_refit  the warm-start mask-and-refit head on K (W_refit, zero off keep_refit)
  sparse       the native GLM-SAGA head of the sparse sweep (W_g of each baseline_lam<v> session)

    python ncc_sweep.py --dataset cub
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from explanation_size import number_of_contributing_concepts, summarise_ncc  # noqa: E402

RESULTS = "seed_sweep_results"


def newest(pattern):
    paths = sorted(glob.glob(os.path.join(RESULTS, pattern)), key=os.path.getmtime)
    return paths[-1] if paths else None


def main(cli):
    gate_path = cli.gate_sweep or newest(f"{cli.dataset}_gate_sweep_*.json")
    sparse_path = cli.sparse_sweep or newest(f"{cli.dataset}_sparse_sweep_*.json")
    with open(gate_path) as f:
        gate = json.load(f)
    with open(sparse_path) as f:
        sparse = json.load(f)
    shared_dir = gate["points"][0]["shared_from"]
    assert all(p["shared_from"] == shared_dir for p in gate["points"] + sparse["points"]), "one shared stage"
    val_c = torch.load(os.path.join(shared_dir, "val_c.pt"), map_location="cpu")
    val_y = torch.load(os.path.join(shared_dir, "val_y.pt"), map_location="cpu")
    score = dict(labels=val_y, tau=cli.tau, max_examples=cli.max_examples, seed=0, device=cli.device)
    print(f"{cli.dataset}: eval {tuple(val_c.shape)}, tau {cli.tau}, "
          f"{cli.max_examples or 'all'} images\n  gate   {gate_path}\n  sparse {sparse_path}", flush=True)

    def run(arm, lam, point, reported_key, heads):
        rows = []
        for j, seed in enumerate(point["seeds"]):
            run_dir = os.path.join(point["session_dir"], f"seed{seed}")
            W, b, acts = heads(run_dir)
            r = number_of_contributing_concepts(W, acts, b, **score)
            reported = point.get(reported_key, {}).get("acc_per_seed") if reported_key else None
            rows.append({**r, "seed": seed, "reported_acc": reported[j] if reported else None})
        s = summarise_ncc(rows)
        print(f"  {arm:11s} lam {lam:<7g} acc {s['acc_mean']:.4f} (reported {s.get('reported_acc_mean', float('nan')):.4f})  "
              f"NCC {s['ncc_mean']:.2f}  NCC_pred {s['ncc_pred_mean']:.2f}  used {s['n_concepts_used_mean']:.0f}", flush=True)
        return {"arm": arm, "lam": lam, **s}

    def gated(run_dir):
        return (torch.load(os.path.join(run_dir, "W_g.pt"), map_location="cpu"),
                torch.load(os.path.join(run_dir, "b_g.pt"), map_location="cpu"), val_c)

    def refit(run_dir):
        W = torch.load(os.path.join(run_dir, "W_refit.pt"), map_location="cpu")
        keep = torch.load(os.path.join(run_dir, "keep_refit.pt"), map_location="cpu").bool()
        acts = val_c if W.shape[1] == val_c.shape[1] else val_c[:, keep]  # full-width (zeros off K) or K-only
        return W, torch.load(os.path.join(run_dir, "b_refit.pt"), map_location="cpu"), acts

    points = []
    for p in gate["points"]:
        points.append(run("gated", p["lambda_gate"], p, "gated", gated))
        if os.path.exists(os.path.join(p["session_dir"], f"seed{p['seeds'][0]}", "W_refit.pt")):
            points.append(run("gated_refit", p["lambda_gate"], p, "gated_refit", refit))
    for p in sparse["points"]:
        points.append(run("sparse", p["lam"], p, "baseline", gated))

    out = {"model": "lfcbm", "dataset": cli.dataset, "tau": cli.tau, "max_examples": cli.max_examples,
           "gate_sweep": os.path.abspath(gate_path), "sparse_sweep": os.path.abspath(sparse_path), "points": points}
    path = os.path.join(RESULTS, f"{cli.dataset}_ncc_sweep_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved NCC sweep to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["cub", "cifar100", "places365"])
    p.add_argument("--gate-sweep", default=None, help="default: newest <dataset>_gate_sweep_*.json")
    p.add_argument("--sparse-sweep", default=None, help="default: newest <dataset>_sparse_sweep_*.json")
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--max-examples", type=int, default=None, help="score a fixed random subset of eval images")
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
