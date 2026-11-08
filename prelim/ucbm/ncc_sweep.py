"""NCC_tau (explanation_size.number_of_contributing_concepts) for every head of a UCBM gate sweep and
native-sparsity (elastic-net) sweep, so accuracy can be compared at equal NCC as De Santis et al.
(ICLR 2026) do.

Arms, per (lambda, seed), each scored on the test set with its own accuracy recomputed on the same images:
  gated        the gated classifier on its exact head input (_get_concept_activations: selection, then the
               hard gate mask)
  gated_refit  the warm-start mask-and-refit head (classifier_refit.pt) on the signal it was fit on
               (UCBM.prune_and_refit: post-gate activations divided by the gate)
  sparse       the elastic-net baseline classifier of each baseline_lam<v> session

    python ncc_sweep.py -d cub -b cub_rn18 -c concepts_200_64
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import torch

from constants import RESULT_PATH
from data_loader import load_data
from model_loader import get_model
from plotter.plotter import Plotter
from rerun_random_control import seed_dirs
from ucbm import UCBM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from explanation_size import number_of_contributing_concepts, summarise_ncc  # noqa: E402


def newest(results, pattern):
    paths = sorted(glob.glob(os.path.join(results, pattern)), key=os.path.getmtime)
    return paths[-1] if paths else None


def main(args):
    run_root = os.path.join(RESULT_PATH, f"{args.dataset}-{args.backbone}/")
    act_bank_path = os.path.join(Plotter(run_root).get_concept_data_path(), args.concept_data)
    results = os.path.join(run_root, "classifier", args.concept_data, "seed_sweep_results")
    gate_path = args.gate_sweep or newest(results, f"{args.dataset}_gate_sweep_*.json")
    sparse_path = args.sparse_sweep or newest(results, f"{args.dataset}_sparse_sweep_*.json")
    with open(gate_path) as f:
        gate = json.load(f)
    with open(sparse_path) as f:
        sparse = json.load(f)
    test_data = load_data(args.dataset, False, reorder_idcs=True)
    y_test = torch.tensor(test_data.targets)
    backbone = get_model(args.backbone, args.device).g
    score = dict(labels=y_test, tau=args.tau, max_examples=args.max_examples, seed=0, device=args.device)
    print(f"{args.dataset}: test {len(y_test)}, tau {args.tau}, {args.max_examples or 'all'} images\n"
          f"  gate   {gate_path}\n  sparse {sparse_path}", flush=True)

    def run(arm, lam, point, reported_key):
        reported = point.get(reported_key, {})
        rep_seeds = reported.get("seeds", point.get("seeds", []))
        rows = []
        for seed_dir in seed_dirs(point["session_dir"]):
            seed = int(os.path.basename(seed_dir)[4:])
            ph = UCBM.load_from_file(seed_dir, "classifier.pth", device=args.device, backbone_p=backbone)
            post = ph._get_concept_activations(test_data, act_bank_path, "test")
            if arm == "gated_refit":
                head = torch.load(os.path.join(seed_dir, "classifier_refit.pt"), map_location="cpu")
                W, b = head["weight"], head["bias"]
                acts = post / ph.get_gate_values().detach().cpu().float().clamp_min(1e-12)
            else:
                W, b = ph._classifier.linear.weight.detach().cpu(), ph._classifier.linear.bias.detach().cpu()
                acts = post
            r = number_of_contributing_concepts(W, acts, b, **score)
            rep = reported.get("acc_per_seed")
            rows.append({**r, "seed": seed,
                         "reported_acc": rep[rep_seeds.index(seed)] if rep and seed in rep_seeds else None})
            del post, acts
        s = summarise_ncc(rows)
        print(f"  {arm:11s} lam {lam:<7g} acc {s['acc_mean']:.4f} (reported {s.get('reported_acc_mean', float('nan')):.4f})  "
              f"NCC {s['ncc_mean']:.2f}  NCC_pred {s['ncc_pred_mean']:.2f}  used {s['n_concepts_used_mean']:.0f}", flush=True)
        return {"arm": arm, "lam": lam, **s}

    points = []
    for p in gate["points"]:
        points.append(run("gated", p["lambda_gate"], p, "gated"))
        if os.path.exists(os.path.join(seed_dirs(p["session_dir"])[0], "classifier_refit.pt")):
            points.append(run("gated_refit", p["lambda_gate"], p, "gated_refit"))
    for p in sparse["points"]:
        points.append(run("sparse", p["lam_w"], p, "baseline"))

    out = {"model": "ucbm", "dataset": args.dataset, "tau": args.tau, "max_examples": args.max_examples,
           "gate_sweep": os.path.abspath(gate_path), "sparse_sweep": os.path.abspath(sparse_path), "points": points}
    path = os.path.join(results, f"{args.dataset}_ncc_sweep_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved NCC sweep to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("-b", "--backbone", required=True)
    p.add_argument("-c", "--concept_data", required=True)
    p.add_argument("--gate-sweep", default=None, help="default: newest <dataset>_gate_sweep_*.json")
    p.add_argument("--sparse-sweep", default=None, help="default: newest <dataset>_sparse_sweep_*.json")
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--max-examples", type=int, default=None, help="score a fixed random subset of test images")
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
