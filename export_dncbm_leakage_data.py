"""Export DN-CBM's CUB concept activations + probe concept sets for concept_leakage_pipelines.py.

The DN-CBM counterpart of prelim/<pipeline>/export_leakage_data.py: writes data/leakage/dncbm_cub.npz
with the raw (unmerged) SAE activations for CUB train/test keyed by CUB image id, the SAE concept
names (+ each name's alignment with its dictionary vector, and the null threshold), and for every saved seed the baseline LinearProbe weight and the gated open set
(sigmoid(gate_logits) > 0.5). The cached activations follow images.txt order within each split
(checked against image_class_labels.txt), so the ids are read straight from the CUB metadata.

    python export_dncbm_leakage_data.py --baseline-run baseline_20260731_042914 --gated-run gated_20260920_135741
"""
import argparse
import os

import numpy as np
import torch

from src import config
from src.concepts import load_concept_names
from src.data import load_clip_features
from concept_accuracy import _probe_paths, pick_alignment_threshold


def split_ids(cub_root, train):
    """CUB image ids and 0-based class labels of one split, in images.txt order."""
    def read(name):
        with open(os.path.join(cub_root, name)) as f:
            return dict(line.split()[:2] for line in f)
    split, labels = read("train_test_split.txt"), read("image_class_labels.txt")
    with open(os.path.join(cub_root, "images.txt")) as f:
        ids = [line.split()[0] for line in f if split[line.split()[0]] == ("1" if train else "0")]
    return np.array(ids, dtype=np.int64), np.array([int(labels[i]) - 1 for i in ids])


def name_alignment(concepts, percentile):
    """Cosine of each SAE dictionary vector with the CLIP text embedding of its assigned name, and the
    --align-percentile of the shuffled-pairing null (concept_accuracy.filter_and_merge_concepts'
    filter): a name below it is no better a description of its concept than a random name."""
    import torch.nn.functional as F
    from src.concepts import embed_texts
    from src.models import load_autoencoder, load_clip
    clip_model, _ = load_clip()
    dic_vec = F.normalize(load_autoencoder(config.sae_device).decoder.weight.detach().cpu().squeeze(), dim=0)
    name_emb = F.normalize(embed_texts(clip_model, concepts), dim=1)
    alignment = (dic_vec.T * name_emb).sum(dim=1)
    return alignment.numpy(), float(pick_alignment_threshold(dic_vec, name_emb, percentile=percentile))


def seed_of(path):
    return int(os.path.basename(path)[len("probe_seed"):-len(".pt")])


def main(args):
    save = os.path.join(config.DATA_PATH, "cub")
    act_dir, model_dir = os.path.join(save, "activations"), os.path.join(save, "model")
    out = {}
    for split in ("train", "test"):
        acts = torch.load(os.path.join(act_dir, f"{split}_sae_acts.pt"), map_location="cpu").numpy()
        ids, y = split_ids(args.cub_root, split == "train")
        _, cached_y = load_clip_features(act_dir, split)
        assert len(ids) == len(acts) and np.array_equal(np.asarray(cached_y), y), f"{split}: row order != images.txt"
        out[f"{split}_acts"], out[f"{split}_ids"], out[f"{split}_y"] = acts, ids, y

    base = {seed_of(p): torch.load(p, map_location="cpu")["linear.weight"].numpy()
            for p in _probe_paths(os.path.join(model_dir, args.baseline_run))}
    gated = {seed_of(p): (torch.sigmoid(torch.load(p, map_location="cpu")["gate_logits"]) > 0.5).numpy()
             for p in _probe_paths(os.path.join(model_dir, args.gated_run))}
    concepts = load_concept_names(os.path.join(save, "concept_names.csv"))
    alignment, align_threshold = name_alignment(concepts, args.align_percentile)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, model="dncbm", concepts=np.array(concepts), **out,
        baseline_seeds=np.array(sorted(base)), baseline_W=np.stack([base[s] for s in sorted(base)]),
        gated_seeds=np.array(sorted(gated)), gated_keep=np.stack([gated[s] for s in sorted(gated)]),
        baseline_dir=args.baseline_run, gated_dir=args.gated_run,
        name_alignment=alignment, name_alignment_threshold=align_threshold)
    print(f"{len(concepts)} concepts, {len(out['train_ids'])} train / {len(out['test_ids'])} test images, "
          f"baseline seeds {sorted(base)}, gated seeds {sorted(gated)} "
          f"(open: {[int(gated[s].sum()) for s in sorted(gated)]}); {int((alignment > align_threshold).sum())} "
          f"names aligned with their dictionary vector (> {align_threshold:.4f}) -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-run", default="baseline_20260731_042914", help="data/cub/model/<run>")
    p.add_argument("--gated-run", default="gated_20260920_135741",
                   help="data/cub/model/<run>; default is the deployed hard-STE gate at lambda_gate 3e-3")
    p.add_argument("--align-percentile", type=float, default=95,
                   help="shuffled-pairing null percentile for the name_alignment_threshold stored in the npz")
    p.add_argument("--cub-root", default=os.path.join("dataset", "CUB_200_2011"))
    p.add_argument("--out", default=os.path.join("data", "leakage", "dncbm_cub.npz"))
    main(p.parse_args())
