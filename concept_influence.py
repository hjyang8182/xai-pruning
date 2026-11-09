"""How much of the baseline's influence do GCG's kept concepts carry, versus their share of the dictionary?

For a DN-CBM dataset, the ungated baseline probe (W, b) is scored on the test set, and every concept j gets
three influence totals over the test images (all for the baseline's predicted class yhat):

  contribution : sum_i |a_ij W[yhat_i, j]|                     (absolute, so +/- terms can't cancel)
  flip_rate    : #images whose prediction changes when only a_ij is zeroed
  explain_rate : #images whose NCC_tau explanation contains j  (the fewest top-|contribution| concepts
                 covering tau of the predicted-class total, as in explanation_size.number_of_contributing_concepts)

The kept share of each is sum_{j in K} m_j / sum_j m_j, with K = GCG's open set (gate > 0.5) for the same seed.
A random set of |K| concepts carries |K|/n_concepts in expectation (the dashed line in
plot_concept_influence.py); `random` also reports that share for --n-random actual draws.

    python concept_influence.py --datasets cifar100 cub places365 food101 oxford_pet
"""
import argparse
import glob
import json
import os
import statistics
from datetime import datetime

import torch

from src import config
from src.data import load_clip_features
from src.models import GatedProbe, LinearProbe
from train_cbm import N_CLASSES

METRICS = ("contribution", "flip_rate", "explain_rate")
# (baseline run, gated run) per dataset under data/<dataset>/model; the gated runs are the ones
# plot_gated_vs_random_grid.DEFAULT_GATED_RUNS deploys, the baselines the 5-seed runs at the same lambda_sparse.
RUNS = {
    "cifar100": ("baseline_20260729_140210", "gated_20260920_134925"),
    "cub": ("baseline_20260731_042914", "gated_20260920_135741"),
    "places365": ("baseline_20260829_232042", "gated_20260921_213151"),
    "food101": ("baseline_20260729_155142", "gated_20260805_194616"),
    "oxford_pet": ("baseline_20260803_051929", "gated_20260805_155623"),
}


def seed_paths(run_dir):
    paths = [p for p in glob.glob(os.path.join(run_dir, "probe_seed*.pt")) if not p.endswith("_refit.pt")]
    return {int(os.path.basename(p)[len("probe_seed"):-len(".pt")]): p for p in paths}


def kept_set(path, n_classes):
    state = torch.load(path, map_location="cpu")
    probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
    missing, _ = probe.load_state_dict(state, strict=False)
    if "gate_logits" in missing:
        raise KeyError(f"{path} has no gate_logits")
    with torch.no_grad():
        return probe.gate_probs() > 0.5


@torch.no_grad()
def influence(W, b, acts, *, tau, chunk, device):
    """Per-concept totals of METRICS for the probe (W, b) over acts (n, J); also the test predictions."""
    W, b = W.to(device), b.to(device)
    J = W.shape[1]
    tot = {m: torch.zeros(J, dtype=torch.float64, device=device) for m in METRICS}
    preds = []
    for i in range(0, len(acts), chunk):
        a = acts[i:i + chunk].to(device).float()
        z = a @ W.T + b
        yhat = z.argmax(1)
        preds.append(yhat.cpu())
        u = (a * W[yhat]).abs()                                   # (B, J) |contribution| to z_yhat
        tot["contribution"] += u.sum(0).double()
        # zero one concept at a time: z - a_ij W[:, j]; only concepts active in image i can flip it,
        # and the SAE codes are sparse, so test just the (i, j) pairs with a_ij != 0
        rows, cols = a.nonzero(as_tuple=True)
        z_drop = z[rows] - a[rows, cols, None] * W.T[cols]        # (P, C)
        flipped = z_drop.argmax(-1) != yhat[rows]
        tot["flip_rate"].scatter_add_(0, cols[flipped], torch.ones(int(flipped.sum()), dtype=torch.float64,
                                                                    device=device))
        order = u.argsort(1, descending=True)
        cum = u.gather(1, order).cumsum(1)
        total = cum[:, -1:]
        k = (cum < tau * total * (1 - 1e-6)).sum(1) + 1          # NCC_tau per image
        k = torch.where(total[:, 0] > 0, k, torch.zeros_like(k))
        member = torch.arange(J, device=device)[None] < k[:, None]
        tot["explain_rate"].scatter_add_(0, order[member], torch.ones(int(member.sum()), dtype=torch.float64,
                                                                       device=device))
    return {m: v.cpu() for m, v in tot.items()}, torch.cat(preds)


def share_stats(vals):
    return {"per_seed": vals, "mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}


def run_dataset(dataset, args):
    base_run, gated_run = RUNS[dataset]
    save = os.path.join(config.DATA_PATH, dataset)
    acts = torch.load(os.path.join(save, "activations", "test_sae_acts.pt"), map_location="cpu")
    _, labels = load_clip_features(os.path.join(save, "activations"), "test")
    labels = torch.as_tensor(labels)
    n_classes = N_CLASSES[dataset]
    base, gated = seed_paths(os.path.join(save, "model", base_run)), seed_paths(os.path.join(save, "model", gated_run))
    seeds = sorted(set(base) & set(gated))
    if not seeds:
        raise RuntimeError(f"{dataset}: no seed shared by {base_run} and {gated_run}")
    gen = torch.Generator().manual_seed(0)
    per = {m: {"kept": [], "random": []} for m in METRICS}
    n_kept, accs = [], []
    for s in seeds:
        state = torch.load(base[s], map_location="cpu")
        probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        probe.load_state_dict(state)
        tot, preds = influence(probe.linear.weight, probe.linear.bias, acts, tau=args.tau,
                               chunk=args.chunk, device=args.device)
        keep = kept_set(gated[s], n_classes)
        k = int(keep.sum())
        n_kept.append(k)
        accs.append((preds == labels).float().mean().item())
        rand = [torch.randperm(len(keep), generator=gen)[:k] for _ in range(args.n_random)]
        for m in METRICS:
            v = tot[m]
            per[m]["kept"].append(float(v[keep].sum() / v.sum()))
            per[m]["random"].append(statistics.fmean(float(v[r].sum() / v.sum()) for r in rand))
        print(f"[{dataset} seed {s}] |K| {k}/{len(keep)} ({100 * k / len(keep):.1f}%)  baseline acc {accs[-1]:.4f}  "
              + "  ".join(f"{m} {100 * per[m]['kept'][-1]:.1f}% (random {100 * per[m]['random'][-1]:.1f}%)"
                          for m in METRICS))
    return {
        "dataset": dataset, "baseline_run": base_run, "gated_run": gated_run, "seeds": seeds,
        "n_concepts": int(acts.shape[1]), "n_kept": round(statistics.fmean(n_kept)), "n_kept_per_seed": n_kept,
        "baseline_acc": share_stats(accs), "n_test": int(len(acts)),
        "summary": {m: {"kept": {"share": share_stats(per[m]["kept"])},
                        "random": {"share": share_stats(per[m]["random"])}} for m in METRICS},
    }


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for dataset in args.datasets:
        res = run_dataset(dataset, args)
        out = os.path.join(args.out_dir, f"concept_influence_{dataset}_{stamp}.json")
        with open(out, "w") as f:
            json.dump({"args": vars(args), "models": {"dncbm": res}}, f, indent=2)
        print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--tau", type=float, default=0.95, help="NCC coverage for explain_rate")
    p.add_argument("--n-random", type=int, default=100, help="random |K| sets per seed for the random share")
    p.add_argument("--chunk", type=int, default=256, help="test images per batch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=os.path.join(config.DATA_PATH, "concept_influence"))
    main(p.parse_args())
