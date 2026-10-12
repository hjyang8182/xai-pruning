"""Export UCBM's CUB concept activations + probe concept sets for ../../concept_leakage_pipelines.py.

Writes one .npz with, keyed by CUB image id (images.txt), the cached raw concept similarities
train_cbm.py stored under concept_data/<concepts>/saved_{train,test}_activations (the concept
layer's output, before the classifier's normalisation / ReLU / TopK), the concept names (UCBM's
concepts are unnamed prototypes, so just their indices), and for every saved seed the baseline
classifier weight and the gated open set (sigmoid(gate_logits) > 0.5).
"""
import argparse
import glob
import os

import numpy as np
import torch

from constants import DATA_DIR, RESULT_PATH
from data_loader import load_data
from sweep_lambda_gate import load_cached_activations


def image_ids(dataset, cub_root):
    root = os.path.join(cub_root, "images")
    with open(os.path.join(cub_root, "images.txt")) as f:
        rel_to_id = {rel: int(i) for i, rel in (line.split() for line in f)}
    return np.array([rel_to_id[os.path.relpath(p, root)] for p, _ in dataset.samples]), np.array(dataset.targets)


def seed_dirs(session_dir):
    dirs = [d for d in glob.glob(os.path.join(session_dir, "seed*")) if os.path.basename(d)[4:].isdigit()]
    return sorted(dirs, key=lambda d: int(os.path.basename(d)[4:]))


def state(run_dir):
    return torch.load(os.path.join(run_dir, "classifier.pth"), map_location="cpu", weights_only=False)["model_state_dict"]


def main(args):
    concept_dir = os.path.join(RESULT_PATH, f"{args.dataset}-{args.backbone}", "concept_data", args.concept_data)
    train_data = load_data(args.dataset, True, reorder_idcs=True)
    test_data = load_data(args.dataset, False, reorder_idcs=True)
    train_acts, _ = load_cached_activations(concept_dir, "train", len(train_data))
    test_acts, _ = load_cached_activations(concept_dir, "test", len(test_data))
    train_ids, train_y = image_ids(train_data, DATA_DIR[args.dataset])
    test_ids, test_y = image_ids(test_data, DATA_DIR[args.dataset])

    base = {int(os.path.basename(d)[4:]): state(d)["linear.weight"].numpy() for d in seed_dirs(args.baseline_dir)}
    gated = {int(os.path.basename(d)[4:]): (torch.sigmoid(state(d)["gate_logits"]) > 0.5).numpy()
             for d in seed_dirs(args.gated_dir)}

    n = train_acts.shape[1]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, model="ucbm", concepts=np.array([f"concept_{i:03d}" for i in range(n)]),
        train_acts=train_acts.numpy(), train_ids=train_ids, train_y=train_y,
        test_acts=test_acts.numpy(), test_ids=test_ids, test_y=test_y,
        baseline_seeds=np.array(sorted(base)), baseline_W=np.stack([base[s] for s in sorted(base)]),
        gated_seeds=np.array(sorted(gated)), gated_keep=np.stack([gated[s] for s in sorted(gated)]),
        baseline_dir=args.baseline_dir, gated_dir=args.gated_dir)
    print(f"{n} concepts, {len(train_ids)} train / {len(test_ids)} test images, "
          f"baseline seeds {sorted(base)}, gated seeds {sorted(gated)} "
          f"(open: {[int(gated[s].sum()) for s in sorted(gated)]}) -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-d", "--dataset", default="cub")
    p.add_argument("-b", "--backbone", default="cub_rn18")
    p.add_argument("-c", "--concept_data", default="concepts_200_64")
    p.add_argument("--baseline-dir", required=True, help="baseline seed-sweep session dir (seed*/classifier.pth)")
    p.add_argument("--gated-dir", required=True, help="gated seed-sweep session dir (seed*/classifier.pth)")
    p.add_argument("--out", default=os.path.join("..", "..", "data", "leakage", "ucbm_cub.npz"))
    main(p.parse_args())
