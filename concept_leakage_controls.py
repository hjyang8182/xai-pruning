"""Is GCG's higher concepts-task leakage more than its concepts' information content predicts?

concept_leakage_pipelines.py compares the gated set with uniformly random concept sets. On DN-CBM
the gate keeps concepts that fire far more often than the baseline's top-k (median 35% vs 5% of
images), and a variable with more entropy can share more bits with anything, so a higher mean CTL
there may only say "denser concepts". Two controls, same matching / encodings / test images:

  ctl_norm   CTL with each side normalised by its own entropy instead of the label's:
             max(0, I(c_hat; y) / H(c_hat) - I(c; y) / H(c)), the excess share of the predicted
             concept's information that is about the class (ICL is already normalised this way)
  matched    the random reference is entropy-matched: each draw takes, per H(c_hat)-quantile
             stratum, as many concepts as the set has there (--n-strata strata over the matched
             concepts; with replacement only in a stratum the set over-fills)

Each set reports its CTL / ctl_norm / ICL percentiles vs. both the uniform and the matched draws.

    python concept_leakage_controls.py --models dncbm
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np

from concept_leakage import concepts_task_leakage, entropy
from concept_leakage_pipelines import MODELS, ROOT, encode, get_matching, icl_matrix, load_cub_attributes

METRICS = ("ctl", "ctl_norm", "icl")


def set_metrics(idx, vecs, icl):
    sub = icl[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
    sub = sub[~np.isnan(sub)]
    return [float(vecs["ctl"][idx].mean()), float(np.nanmean(vecs["ctl_norm"][idx])),
            float(sub.mean()) if sub.size else np.nan]


def matched_draw(idx, strata, rng):
    out = []
    for s, count in zip(*np.unique(strata[idx], return_counts=True)):
        pool = np.flatnonzero(strata == s)
        out.append(rng.choice(pool, count, replace=count > len(pool)))
    return np.concatenate(out)


def compare(idx, vecs, icl, strata, h_pred, n_random, rng):
    own = set_metrics(idx, vecs, icl)
    nulls = {"uniform": [rng.choice(len(h_pred), len(idx), replace=False) for _ in range(n_random)],
             "matched": [matched_draw(idx, strata, rng) for _ in range(n_random)]}
    out = {"n_matched": int(len(idx)), "h_pred_mean": float(h_pred[idx].mean()),
           **{m: own[i] for i, m in enumerate(METRICS)}}
    for name, draws in nulls.items():
        vals = np.array([set_metrics(dr, vecs, icl) for dr in draws])
        out[f"null_{name}_h_pred_mean"] = float(np.mean([h_pred[dr].mean() for dr in draws]))
        for i, m in enumerate(METRICS):
            v = vals[:, i][~np.isnan(vals[:, i])]
            out[f"null_{name}_{m}_mean"] = float(v.mean())
            out[f"{m}_pct_vs_{name}"] = float((v < own[i]).mean() * 100)
    return out


def analyse(path, attrs, attr_names, args):
    d = dict(np.load(path, allow_pickle=True))
    model = str(d["model"])
    matched, _, _ = get_matching(d, attrs, attr_names, args)
    order = list(matched)
    pos = {k: i for i, k in enumerate(order)}
    fire = (d["train_acts"][:, order] > 0).mean(0)
    base_norm = np.linalg.norm(d["baseline_W"], axis=1).mean(0)
    print(f"\n=== {model} [{args.matching} matching]: {len(order)} matched concepts")
    result = {"model": model, "matching": args.matching, "n_matched": len(order), "encodings": {}}
    for encoding in args.encodings:
        predicted, truth = encode(d, attrs, matched, encoding, args.n_bins)
        ctl = concepts_task_leakage(predicted, truth, d["test_y"])
        h_y = entropy(np.unique(d["test_y"], return_inverse=True)[1])
        h_pred = np.array([entropy(predicted[k]) for k in order])
        h_true = np.array([entropy(truth[k]) for k in order])
        with np.errstate(divide="ignore", invalid="ignore"):
            ctl_norm = np.maximum(0.0, np.array([ctl[k]["nmi_pred_task"] for k in order]) * h_y / h_pred
                                  - np.array([ctl[k]["nmi_true_task"] for k in order]) * h_y / h_true)
        ctl_norm[(h_pred <= 0) | (h_true <= 0)] = np.nan
        vecs = {"ctl": np.array([ctl[k]["ctl"] for k in order]), "ctl_norm": ctl_norm}
        icl = icl_matrix(predicted, truth, order)
        edges = np.quantile(h_pred, np.linspace(0, 1, args.n_strata + 1)[1:-1])
        strata = np.searchsorted(edges, h_pred, side="right")
        rng = np.random.default_rng(args.seed)
        rows = {"gated": [], "baseline_topk": []}
        for keep in d["gated_keep"]:
            open_set = np.flatnonzero(keep)
            sets = {"gated": open_set, "baseline_topk": np.argsort(-base_norm)[:len(open_set)]}
            for name, members in sets.items():
                idx = np.array([pos[k] for k in members if k in pos], dtype=np.int64)
                r = compare(idx, vecs, icl, strata, h_pred, args.n_random, rng)
                r["fire_rate_median"] = float(np.median(fire[idx]))
                rows[name].append(r)
        agg = {name: {key: float(np.nanmean([r[key] for r in rs])) for key in rs[0]} | {"per_seed": rs}
               for name, rs in rows.items()}
        agg["population"] = {"h_pred_mean": float(h_pred.mean()), "fire_rate_median": float(np.median(fire)),
                             **{m: float(np.nanmean(v)) for m, v in vecs.items()}}
        result["encodings"][encoding] = agg
        print(f"  [{encoding}] population: H(c_hat) {h_pred.mean():.3f}  CTL {vecs['ctl'].mean():.4f}  "
              f"ctl_norm {np.nanmean(ctl_norm):.4f}")
        for name in rows:
            a = agg[name]
            print(f"  [{encoding}] {name:13s} H(c_hat) {a['h_pred_mean']:.3f} (matched null {a['null_matched_h_pred_mean']:.3f})")
            for m in METRICS:
                print(f"  [{encoding}]   {m:8s} {a[m]:.4f}  uniform {a[f'null_uniform_{m}_mean']:.4f} "
                      f"(pct {a[f'{m}_pct_vs_uniform']:.0f})  entropy-matched {a[f'null_matched_{m}_mean']:.4f} "
                      f"(pct {a[f'{m}_pct_vs_matched']:.0f})")
    return result


def main(args):
    attrs, attr_names = load_cub_attributes(args.cub_root)
    results = {"args": vars(args), "models": {}}
    for model in args.models:
        results["models"][model] = analyse(os.path.join(args.data_dir, f"{model}_cub.npz"), attrs, attr_names, args)
    out = os.path.join(args.data_dir, f"leakage_controls{'_name' if args.matching == 'name' else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "leakage"))
    p.add_argument("--cub-root", default=os.path.join(ROOT, "dataset", "CUB_200_2011"))
    p.add_argument("--encodings", nargs="+", default=["hard", "soft"], choices=["hard", "soft"])
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--matching", default="data", choices=["data", "name"], help="concept_leakage_pipelines.get_matching")
    p.add_argument("--match-percentile", type=float, default=95)
    p.add_argument("--n-shuffles", type=int, default=5)
    p.add_argument("--n-strata", type=int, default=20, help="H(c_hat) quantile strata for the matched null")
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
