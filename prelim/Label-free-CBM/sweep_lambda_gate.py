import sys
import os
import json
import datetime
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gated_probe import GatedProbe, gate_temperature_schedule

from train_cbm_seed_sweep import build_shared_concept_data

"""
Step 2 calibration: find a usable lambda_gate range for arm C on LF-CBM's
dense, CLIP-similarity-derived concept activations, which are a very
different substrate from the sparse SAE activations the gate was originally
designed for.

Reuses build_shared_concept_data from train_cbm_seed_sweep.py for concept
discovery, filtering, and the W_c projection layer -- computed once, seed/arm
independent -- so this script never touches that stage itself (arm A's
published mechanism), only what's layered on top of it (arm B/C probe
training).

For each lambda_gate value, trains arm B once (same optimiser/epochs, no
gate -- the same-optimiser control) and arm C once, and reports:
  - CE loss and the gate penalty (lambda_gate * sum(sigmoid(gate_logits)))
    at the start (first batch, untrained) and end (final epoch) of arm C
    training, so their relative magnitude is visible.
  - Final test accuracy for arm B and arm C.
  - Final open-gate count (sigmoid(gate) >= 0.5) for arm C.
  - A histogram of final sigmoid(gate) values for arm C, reported as text
    (mass near 0 / diffuse middle / mass near 1) and saved as a plot --
    this is the key diagnostic for whether a fixed 0.5 threshold is
    meaningful on this substrate.
"""


def train_arm_b(train_c, train_y, val_c, val_y, n_classes, epochs, batch_size, device):
    probe = torch.nn.Linear(train_c.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_c, train_y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(probe(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        acc = (probe(val_c.to(device)).argmax(dim=1).cpu() == val_y).float().mean().item()
    return acc


def ce_and_gate_terms(gate_probe, loss_fn, train_c, train_y, lambda_gate, device, n_batches=None):
    """Mean CE over (a subset of) train_c/train_y, plus the gate penalty at
    the probe's current gate_logits -- used for the start/end magnitude
    comparison. The gate term doesn't depend on the batch (gates_batch is
    the raw (n_concepts,) vector), only CE does, so CE is averaged over a
    few batches for a stable estimate."""
    gate_probe.eval()
    with torch.no_grad():
        n = train_c.shape[0] if n_batches is None else min(train_c.shape[0], n_batches * 256)
        idx = torch.randperm(train_c.shape[0])[:n]
        logits, gates = gate_probe(train_c[idx].to(device))
        ce = loss_fn(logits, train_y[idx].to(device)).item()
        gate_term = (lambda_gate * gates.sum()).item()
    gate_probe.train()
    return ce, gate_term


def train_arm_c(train_c, train_y, val_c, val_y, n_classes, lambda_gate, epochs, batch_size, device,
                 gate_temperature=1.0, gate_temperature_final=None, gate_forward="soft"):
    gate_probe = GatedProbe(train_c.shape[1], n_classes, gate_temperature=gate_temperature,
                            gate_forward=gate_forward).to(device)
    optimizer = torch.optim.Adam(gate_probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_c, train_y), batch_size=batch_size, shuffle=True)

    start_ce, start_gate_term = ce_and_gate_terms(gate_probe, loss_fn, train_c, train_y, lambda_gate, device)

    for epoch in range(epochs):
        gate_probe.gate_temperature = gate_temperature_schedule(epoch, epochs, gate_temperature, gate_temperature_final)
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, gates_batch = gate_probe(X_batch)
            loss = loss_fn(logits, y_batch) + lambda_gate * gates_batch.sum()
            loss.backward()
            optimizer.step()

    end_ce, end_gate_term = ce_and_gate_terms(gate_probe, loss_fn, train_c, train_y, lambda_gate, device)

    gate_probe.eval()
    with torch.no_grad():
        acc = (gate_probe(val_c.to(device))[0].argmax(dim=1).cpu() == val_y).float().mean().item()
        final_gates = gate_probe.gate_probs().detach().cpu()

    return {
        'test_acc': acc,
        'start_ce': start_ce, 'start_gate_term': start_gate_term,
        'end_ce': end_ce, 'end_gate_term': end_gate_term,
        'n_open': int((final_gates >= 0.5).sum().item()),
        'n_concepts': final_gates.numel(),
        'gates': final_gates,
    }


def histogram_summary(gates, n_bins=20):
    counts, edges = np.histogram(gates.numpy(), bins=n_bins, range=(0, 1))
    near_0 = (gates < 0.1).float().mean().item()
    near_1 = (gates > 0.9).float().mean().item()
    middle = 1.0 - near_0 - near_1
    return {
        'bin_edges': edges.tolist(), 'bin_counts': counts.tolist(),
        'frac_near_0_lt_0.1': near_0, 'frac_middle_0.1_to_0.9': middle, 'frac_near_1_gt_0.9': near_1,
        'mean': gates.mean().item(), 'std': gates.std().item(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: coarse lambda_gate calibration for LF-CBM arm C")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--concept_set", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="clip_RN50")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for saving/loading backbone+CLIP activations")
    parser.add_argument("--probe_batch_size", type=int, default=256, help="Batch size for arm B/C probe training")
    parser.add_argument("--proj_batch_size", type=int, default=50000)
    parser.add_argument("--feature_layer", type=str, default="layer4")
    parser.add_argument("--activation_dir", type=str, default="saved_activations")
    parser.add_argument("--results_dir", type=str, default="lambda_gate_calibration")
    parser.add_argument("--clip_cutoff", type=float, default=0.25)
    parser.add_argument("--proj_steps", type=int, default=1000)
    parser.add_argument("--proj_patience", type=int, default=1)
    parser.add_argument("--interpretability_cutoff", type=float, default=0.45)
    parser.add_argument("--no_filter", action="store_true")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs for arm B/C probe training (coarse calibration, not final quality)")
    parser.add_argument("--lambda-gates", type=float, nargs="+",
                         default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate_forward", choices=["soft", "hard"], default="soft",
                         help="'soft' multiplies by sigmoid(gate); 'hard' uses the mask 1[sigmoid>0.5] in the forward "
                              "pass with a straight-through gradient (same flag as train_cbm_seed_sweep.py)")
    parser.add_argument("--gate_temperature", type=float, default=1.0,
                         help="Temperature T dividing gate_logits before the sigmoid, for every arm-C run in the "
                              "sweep. T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. "
                              "Starting temperature of the anneal if --gate_temperature_final is also given.")
    parser.add_argument("--gate_temperature_final", type=float, default=None,
                         help="If given, anneal the gate temperature geometrically from --gate_temperature down "
                              "(or up) to this value over each arm-C run's training, instead of keeping it fixed.")
    return parser.parse_args()


def main(args):
    if args.concept_set is None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)
    os.makedirs(args.results_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    shared = build_shared_concept_data(args)
    n_concepts = shared["train_c"].shape[1]
    n_classes = len(shared["classes"])
    print("Shared concept set: {} concepts, {} classes".format(n_concepts, n_classes))

    torch.manual_seed(args.seed)
    armB_acc = train_arm_b(shared["train_c"], shared["train_y"], shared["val_c"], shared["val_y"],
                            n_classes, args.epochs, args.probe_batch_size, args.device)
    print("[arm B] test_acc={:.4f}".format(armB_acc))

    results = {'dataset': args.dataset, 'n_concepts': n_concepts, 'n_classes': n_classes,
               'epochs': args.epochs, 'armB_test_acc': armB_acc, 'lambda_gates': [],
               'gate_temperature_start': args.gate_temperature, 'gate_temperature_final': args.gate_temperature_final,
               'gate_forward': args.gate_forward}

    for lambda_gate in args.lambda_gates:
        torch.manual_seed(args.seed)
        r = train_arm_c(shared["train_c"], shared["train_y"], shared["val_c"], shared["val_y"],
                         n_classes, lambda_gate, args.epochs, args.probe_batch_size, args.device,
                         gate_temperature=args.gate_temperature, gate_temperature_final=args.gate_temperature_final,
                         gate_forward=args.gate_forward)
        hist = histogram_summary(r['gates'])

        print("[arm C] lambda_gate={:g} test_acc={:.4f} open={}/{} "
              "CE(start->end)={:.4f}->{:.4f} gate_term(start->end)={:.6f}->{:.6f} "
              "gate dist: <0.1={:.3f} mid={:.3f} >0.9={:.3f}".format(
                  lambda_gate, r['test_acc'], r['n_open'], r['n_concepts'],
                  r['start_ce'], r['end_ce'], r['start_gate_term'], r['end_gate_term'],
                  hist['frac_near_0_lt_0.1'], hist['frac_middle_0.1_to_0.9'], hist['frac_near_1_gt_0.9']))

        plt.figure(figsize=(6, 3.5))
        plt.hist(r['gates'].numpy(), bins=20, range=(0, 1), color="#4C72B0")
        plt.axvline(0.5, color="red", linestyle="--", label="threshold")
        plt.xlabel("sigmoid(gate)")
        plt.ylabel("count")
        plt.title("lambda_gate={:g}: {}/{} open".format(lambda_gate, r['n_open'], r['n_concepts']))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, "gate_hist_{}_{:.0e}.png".format(args.dataset, lambda_gate)), dpi=150)
        plt.close()

        results['lambda_gates'].append({
            'lambda_gate': lambda_gate, 'test_acc': r['test_acc'],
            'n_open': r['n_open'], 'n_concepts': r['n_concepts'],
            'start_ce': r['start_ce'], 'start_gate_term': r['start_gate_term'],
            'end_ce': r['end_ce'], 'end_gate_term': r['end_gate_term'],
            'histogram': hist,
        })

    out_path = os.path.join(args.results_dir, "{}_lambda_gate_calibration_{}.json".format(
        args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to {}".format(out_path))


if __name__ == "__main__":
    main(parse_args())
