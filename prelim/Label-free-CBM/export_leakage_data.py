"""Export LF-CBM's CUB concept activations + probe concept sets for ../../concept_leakage_pipelines.py.

Writes one .npz with, keyed by CUB image id (images.txt), the per-image concept activations of
the shared stage (train_c = CUB train, val_c = CUB test -- LF-CBM's "val" split for cub *is* the
official test split), the concept names, and for every saved seed the baseline GLM-SAGA weight
(W_g.pt) and the gated open set (gate_logits.pt: sigmoid > 0.5).
"""
import argparse
import glob
import os

import numpy as np
import torch
from torchvision import datasets

import data_utils


def image_ids(split_dir, cub_root):
    """CUB image ids in torchvision ImageFolder order for a CUB_200_2011/{train,test} folder."""
    with open(os.path.join(cub_root, "images.txt")) as f:
        rel_to_id = {rel: int(i) for i, rel in (line.split() for line in f)}
    folder = datasets.ImageFolder(split_dir)
    ids = [rel_to_id[os.path.relpath(p, split_dir)] for p, _ in folder.imgs]
    return np.array(ids), np.array(folder.targets)


def seed_dirs(session_dir):
    dirs = [d for d in glob.glob(os.path.join(session_dir, "seed*")) if os.path.basename(d)[4:].isdigit()]
    return sorted(dirs, key=lambda d: int(os.path.basename(d)[4:]))


def main(args):
    shared = {k: torch.load(os.path.join(args.shared, f"{k}.pt"), map_location="cpu")
              for k in ("train_c", "train_y", "val_c", "val_y")}
    with open(os.path.join(args.shared, "concepts.txt")) as f:
        concepts = [c for c in f.read().split("\n") if c]

    cub_root = os.path.join(data_utils.DATA_ROOT, "CUB_200_2011")
    train_ids, train_t = image_ids(data_utils.DATASET_ROOTS["cub_train"], cub_root)
    test_ids, test_t = image_ids(data_utils.DATASET_ROOTS["cub_test"], cub_root)
    # the shared stage was built from these same ImageFolders; check the order really lines up
    assert np.array_equal(train_t, shared["train_y"].numpy()) and np.array_equal(test_t, shared["val_y"].numpy())

    base = {int(os.path.basename(d)[4:]): torch.load(os.path.join(d, "W_g.pt"), map_location="cpu").numpy()
            for d in seed_dirs(args.baseline_dir)}
    gated = {int(os.path.basename(d)[4:]): (torch.sigmoid(torch.load(os.path.join(d, "gate_logits.pt"),
                                                                     map_location="cpu")) > 0.5).numpy()
             for d in seed_dirs(args.gated_dir)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, model="lfcbm", concepts=np.array(concepts),
        train_acts=shared["train_c"].numpy(), train_ids=train_ids, train_y=train_t,
        test_acts=shared["val_c"].numpy(), test_ids=test_ids, test_y=test_t,
        baseline_seeds=np.array(sorted(base)), baseline_W=np.stack([base[s] for s in sorted(base)]),
        gated_seeds=np.array(sorted(gated)), gated_keep=np.stack([gated[s] for s in sorted(gated)]),
        baseline_dir=args.baseline_dir, gated_dir=args.gated_dir)
    print(f"{len(concepts)} concepts, {len(train_ids)} train / {len(test_ids)} test images, "
          f"baseline seeds {sorted(base)}, gated seeds {sorted(gated)} "
          f"(open: {[int(gated[s].sum()) for s in sorted(gated)]}) -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared", required=True, help="seed_sweep_results/shared_cub_<stamp> (--save_shared dir)")
    p.add_argument("--baseline-dir", required=True, help="baseline seed-sweep session dir (seed*/W_g.pt)")
    p.add_argument("--gated-dir", required=True, help="gated seed-sweep session dir (seed*/gate_logits.pt)")
    p.add_argument("--out", default=os.path.join("..", "..", "data", "leakage", "lfcbm_cub.npz"))
    main(p.parse_args())
