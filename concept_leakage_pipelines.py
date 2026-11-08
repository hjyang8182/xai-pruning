"""Concepts-task leakage (CTL) and interconcept leakage (ICL), Parisini et al. (arXiv:2504.14094),
for the DN-CBM / LF-CBM / UCBM / VLG-CBM CUB models: sparse baseline vs. gated (GCG) concept sets.

DN-CBM's input comes from export_dncbm_leakage_data.py (its older percentile-matching analysis is
concept_leakage.py). This reuses its MI estimators and score definitions
(plug-in, Miller-Madow corrected, on discretised variables) and differs in two places:

  * Input: the .npz each pipeline's export_leakage_data.py writes (concept activations for CUB
    train/test keyed by CUB image id, plus every seed's baseline weight and gated open set), so
    nothing here depends on a pipeline's own code or conda env.
  * Concept -> attribute matching (--matching data, the default) is data-driven, the same for every pipeline (UCBM's concepts are
    unnamed): on the *train* images, each concept is paired with the CUB attribute its activation
    ranks best -- max |z| of the Mann-Whitney statistic (AUROC standardised by its null variance,
    so the ~50 attributes present in <1% of images, whose raw AUROC is very noisy, don't win by
    chance) -- and kept only if that beats the `--match-percentile` of the same max-|z| under
    shuffled attribute rows. All scores are then
    taken on the *test* images, so the matching never sees the data it is scored on.
    --matching name instead pairs each named concept (LF-CBM / VLG-CBM phrases, DN-CBM words) with the
    attribute whose description it is closest to in CLIP text space (concept_accuracy_pipelines.py);
    the data-driven matching keeps every LF-CBM / UCBM / VLG-CBM concept, so it barely filters.

Sets compared, per seed (CTL/ICL are properties of the concept layer, so the probes differ only
through which concepts they use):
  population   - every matched concept (the chance reference)
  baseline     - concepts with any non-zero class weight in the baseline head
  baseline_topk- the baseline's top-|open| concepts by weight-column norm, size-matched to the
                 gated open set of the same seed (as concept_leakage.py pairs them)
  gated        - the gate's open set (sigmoid(gate) > 0.5)

Two encodings of the predicted concept: `soft` (zero bin + quantile bins, as concept_leakage.py)
and `hard` (binarised at the train threshold that best separates its matched attribute, so the
predicted and true variables have the same cardinality and the MI bias difference cancels).

Chance reference: for each seed, `--n-random` random subsets of the matched concepts, of the gated
set's size, give a null distribution of mean CTL / mean ICL; every set reports the percentile of
its own mean within it (50 = no different from picking concepts at random).
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
from scipy.stats import rankdata

from concept_leakage import concepts_task_leakage, discretise_soft, entropy, mutual_information

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ("dncbm", "lfcbm", "ucbm", "vlgcbm")


def load_cub_attributes(cub_root):
    """(max_img_id + 1, 312) binary attribute matrix indexed by CUB image id, and attribute names."""
    with open(os.path.join(os.path.dirname(os.path.normpath(cub_root)), "attributes.txt")) as f:  # dataset/attributes.txt
        names = [line.split(" ", 1)[1].strip() for line in f]
    rows = np.loadtxt(os.path.join(cub_root, "attributes", "image_attribute_labels.txt"),
                      usecols=(0, 1, 2), dtype=np.int64)
    attrs = np.zeros((rows[:, 0].max() + 1, len(names)), dtype=np.int8)
    attrs[rows[:, 0], rows[:, 1] - 1] = rows[:, 2]
    return attrs, names


def auroc_matrix(acts, attrs):
    """(|z|, AUROC) of every concept (column of acts) for every attribute, via rank sums: z is the
    Mann-Whitney U standardised by its tie-corrected null s.d.
    sqrt(n_pos n_neg / 12 * ((n + 1) - sum_t (t^3 - t) / (n (n - 1)))). Ties get average ranks:
    DN-CBM's SAE activations are mostly exact zeros, and breaking those ties by row order (CUB rows
    are sorted by class) would make even never-firing concepts look attribute-specific. A concept
    that is constant gets z = 0 (never matched)."""
    n = acts.shape[0]
    ranks = rankdata(acts, method="average", axis=0)
    tie_term = np.array([((t := np.unique(col, return_counts=True)[1].astype(np.float64)) ** 3 - t).sum()
                         for col in acts.T]) / (n * (n - 1))  # 0 for tie-free (continuous) columns
    pos = attrs.sum(0).astype(np.float64)
    neg = n - pos
    rank_sum = attrs.T.astype(np.float64) @ ranks  # (n_attr, n_concepts)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = rank_sum - pos[:, None] * (pos[:, None] + 1) / 2
        auc = u / (pos[:, None] * neg[:, None])
        z = (u - pos[:, None] * neg[:, None] / 2) / np.sqrt(pos[:, None] * neg[:, None] / 12 * ((n + 1) - tie_term[None, :]))
    return np.nan_to_num(np.abs(z)).T, np.nan_to_num(auc, nan=0.5).T  # (n_concepts, n_attr) each


def match_concepts(acts, attrs, percentile, n_shuffles, seed):
    """Data-driven concept -> attribute matching on train images (see module docstring)."""
    score, auc = auroc_matrix(acts, attrs)
    best_attr, best = score.argmax(1), score.max(1)
    rng = np.random.default_rng(seed)
    null = np.concatenate([auroc_matrix(acts, attrs[rng.permutation(len(attrs))])[0].max(1)
                           for _ in range(n_shuffles)])
    threshold = float(np.percentile(null, percentile))
    matched = {int(k): int(best_attr[k]) for k in np.flatnonzero(best > threshold)}
    match_auc = {k: float(auc[k, a]) for k, a in matched.items()}
    return matched, threshold, match_auc


def get_matching(d, attrs, attr_names, args):
    """(matched {concept: attribute}, threshold, AUROC per matched pair) by --matching: `data` (match_concepts)
    or `name` (concept_accuracy_pipelines.name_matching: CLIP text similarity of the concept phrase to the
    attribute description vs. a Places-365-phrase null; needs named concepts, so not UCBM)."""
    train_attrs = attrs[d["train_ids"]]
    if args.matching == "data":
        return match_concepts(d["train_acts"], train_attrs, args.match_percentile, args.n_shuffles, args.seed)
    if str(d["model"]) == "ucbm":
        raise ValueError("UCBM's concepts are unnamed (concept_000, ...): use --matching data")
    from concept_accuracy_pipelines import name_matching  # lazy: it imports this module, and CLIP
    from src.models import load_clip
    clip_model, _ = load_clip()
    matched, threshold, _ = name_matching(clip_model, d["concepts"], attr_names, args.match_percentile)
    ks = list(matched)
    _, auc = auroc_matrix(d["train_acts"][:, ks], train_attrs)
    return matched, threshold, {k: float(auc[i, matched[k]]) for i, k in enumerate(ks)}


def hard_threshold(x, truth):
    """Threshold + direction on train activations maximising balanced accuracy vs. the attribute."""
    qs = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 49)))
    best = (-1.0, qs[0], 1)
    for t in qs:
        pred = x > t
        for sign, p in ((1, pred), (-1, ~pred)):
            tpr = (p & (truth == 1)).sum() / max(1, (truth == 1).sum())
            tnr = (~p & (truth == 0)).sum() / max(1, (truth == 0).sum())
            if (tpr + tnr) / 2 > best[0]:
                best = ((tpr + tnr) / 2, t, sign)
    return best[1], best[2]


def encode(d, attrs, matched, encoding, n_bins):
    """Predicted (discretised activation) and true (matched attribute) codes on the test images."""
    test_attrs, train_attrs = attrs[d["test_ids"]], attrs[d["train_ids"]]
    predicted, truth = {}, {}
    for k, a in matched.items():
        x = d["test_acts"][:, k]
        if encoding == "hard":
            t, sign = hard_threshold(d["train_acts"][:, k], train_attrs[:, a])
            predicted[k] = ((x > t) if sign > 0 else (x <= t)).astype(np.int64)
        else:
            predicted[k] = discretise_soft(x, n_bins)
        truth[k] = test_attrs[:, a].astype(np.int64)
    return predicted, truth


def icl_matrix_loop(predicted, truth, order):
    """Pairwise ICL (concept_leakage.interconcept_leakage's definition) for every pair in `order`,
    as a symmetric matrix with NaN where undefined (a constant variable on either side). Reference
    implementation; icl_matrix computes the same thing in blocks."""
    h_pred = np.array([entropy(predicted[k]) for k in order])
    h_true = np.array([entropy(truth[k]) for k in order])
    m = np.full((len(order), len(order)), np.nan)
    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            if min(h_pred[i], h_pred[j], h_true[i], h_true[j]) <= 0:
                continue
            nmi_pred = mutual_information(predicted[a], predicted[b]) / np.sqrt(h_pred[i] * h_pred[j])
            nmi_true = mutual_information(truth[a], truth[b]) / np.sqrt(h_true[i] * h_true[j])
            m[i, j] = m[j, i] = max(0.0, nmi_pred - nmi_true)
    return m


def pairwise_mi(codes, block=64):
    """Miller-Madow MI (concept_leakage.mutual_information) between every pair of discrete columns,
    with each pair's joint counts read off a one-hot Gram matrix, so DN-CBM's thousands of matched
    concepts take seconds instead of a ~10^7-pair Python loop. Returns (MI matrix, entropies)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense = [np.unique(c, return_inverse=True)[1] for c in codes]
    n, k, m = len(dense[0]), len(dense), max(int(c.max()) + 1 for c in dense)
    onehot = torch.zeros(n, k, m, device=device)
    onehot.scatter_(2, torch.as_tensor(np.stack(dense, 1), device=device).unsqueeze(2), 1.0)
    onehot = onehot.view(n, k * m)
    h = np.array([entropy(c) for c in codes])

    def plogp_entropy(counts):  # counts (..., cells) -> Miller-Madow entropy in bits over the last axis
        p = counts / n
        nz = (counts > 0).sum(-1)
        return -(torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p))).sum(-1) + (nz - 1) / (2 * n * np.log(2))

    mi = np.empty((k, k))
    for s in range(0, k, block):
        e = min(k, s + block)
        joint = (onehot[:, s * m:e * m].T @ onehot).view(e - s, m, k, m).permute(0, 2, 1, 3)
        h_joint = plogp_entropy(joint.reshape(e - s, k, m * m).double()).cpu().numpy()
        mi[s:e] = np.maximum(0.0, h[s:e, None] + h[None, :] - h_joint)
    return mi, h


def icl_matrix(predicted, truth, order):
    """Same matrix as icl_matrix_loop (NaN on the diagonal and where any entropy is zero)."""
    mi_pred, h_pred = pairwise_mi([predicted[k] for k in order])
    mi_true, h_true = pairwise_mi([truth[k] for k in order])
    with np.errstate(divide="ignore", invalid="ignore"):
        icl = np.maximum(0.0, mi_pred / np.sqrt(np.outer(h_pred, h_pred)) - mi_true / np.sqrt(np.outer(h_true, h_true)))
    undefined = (h_pred <= 0) | (h_true <= 0)
    icl[undefined, :] = icl[:, undefined] = np.nan
    np.fill_diagonal(icl, np.nan)
    return icl


def set_means(idx, ctl_vec, icl):
    """Mean CTL and mean pairwise ICL of a set of row indices into the matched-concept order."""
    if len(idx) == 0:
        return None, None
    sub = icl[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
    sub = sub[~np.isnan(sub)]
    return float(ctl_vec[idx].mean()), (float(sub.mean()) if sub.size else None)


def set_stats(concepts, pos, ctl_vec, icl, null=None):
    idx = np.array([pos[k] for k in concepts if k in pos], dtype=np.int64)
    out = {"n": int(len(concepts)), "n_matched": int(len(idx))}
    out["ctl_mean"], out["icl_mean"] = set_means(idx, ctl_vec, icl)
    if null is not None:
        for key, col in (("ctl_mean", 0), ("icl_mean", 1)):
            if out[key] is not None:
                out[key.replace("_mean", "_pct_vs_random")] = float((null[:, col] < out[key]).mean() * 100)
            vals = null[:, col][~np.isnan(null[:, col])]
            if vals.size:
                name = key.replace("_mean", "")
                out[f"null_{name}_mean"] = float(vals.mean())
                out[f"null_{name}_lo"], out[f"null_{name}_hi"] = (float(q) for q in np.percentile(vals, [2.5, 97.5]))
    return out


def mean_std(values):
    v = [x for x in values if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def analyse(path, attrs, attr_names, args):
    d = dict(np.load(path, allow_pickle=True))
    model = str(d["model"])
    matched, threshold, match_auc = get_matching(d, attrs, attr_names, args)
    print(f"\n=== {model} [{args.matching} matching]: {len(matched)}/{d['train_acts'].shape[1]} concepts matched "
          f"(score > {threshold:.4f}, null {args.match_percentile}th pct; matched AUROC median "
          f"{np.median([abs(a - 0.5) + 0.5 for a in match_auc.values()]) if matched else float('nan'):.3f})")

    base_norm = np.linalg.norm(d["baseline_W"], axis=1).mean(0)  # mean over baseline seeds, per concept
    base_used = np.flatnonzero((np.abs(d["baseline_W"]) > 1e-5).any(axis=(0, 1)))
    result = {"model": model, "npz": os.path.relpath(path, ROOT), "matching": args.matching, "match_threshold": threshold,
              "n_concepts": int(d["train_acts"].shape[1]), "n_matched": len(matched),
              "matched": {str(k): {"attribute": attr_names[a], "auroc": match_auc[k]} for k, a in matched.items()},
              "encodings": {}}
    for encoding in args.encodings:
        predicted, truth = encode(d, attrs, matched, encoding, args.n_bins)
        ctl = concepts_task_leakage(predicted, truth, d["test_y"])
        order = list(matched)
        pos = {k: i for i, k in enumerate(order)}
        ctl_vec = np.array([ctl[k]["ctl"] for k in order])
        icl = icl_matrix(predicted, truth, order)
        rng = np.random.default_rng(args.seed)
        per_seed = {"baseline_topk": [], "gated": []}
        for s, keep in zip(d["gated_seeds"], d["gated_keep"]):
            open_set = np.flatnonzero(keep)
            topk = np.argsort(-base_norm)[:len(open_set)]
            n_draw = len([k for k in open_set if k in pos])
            null = np.array([[v if v is not None else np.nan
                              for v in set_means(rng.choice(len(order), n_draw, replace=False), ctl_vec, icl)]
                             for _ in range(args.n_random)])
            per_seed["gated"].append(set_stats(open_set, pos, ctl_vec, icl, null))
            per_seed["baseline_topk"].append(set_stats(topk, pos, ctl_vec, icl, null))
        sets = {"population": set_stats(order, pos, ctl_vec, icl),
                "baseline": set_stats(base_used, pos, ctl_vec, icl)}
        for name, rows in per_seed.items():
            agg = {"per_seed": rows}
            for key in ("n", "n_matched", "ctl_mean", "icl_mean", "ctl_pct_vs_random", "icl_pct_vs_random",
                        "null_ctl_mean", "null_ctl_lo", "null_ctl_hi", "null_icl_mean", "null_icl_lo", "null_icl_hi"):
                agg[key], agg[key + "_std"] = mean_std([r.get(key) for r in rows])
            sets[name] = agg
        result["encodings"][encoding] = {"sets": sets,
                                         "ctl_per_concept": {str(k): v for k, v in ctl.items()}}
        print(f"  [{encoding}] {'set':14s} {'n':>6s} {'matched':>8s} {'CTL':>8s} {'pct':>5s} {'ICL':>8s} {'pct':>5s}")
        for name, st in sets.items():
            fmt = lambda v, nd=4: "-" if v is None else f"{v:.{nd}f}"
            print(f"  [{encoding}] {name:14s} {st['n']:>6.1f} {st['n_matched']:>8.1f} "
                  f"{fmt(st.get('ctl_mean')):>8s} {fmt(st.get('ctl_pct_vs_random'), 0):>5s} "
                  f"{fmt(st.get('icl_mean')):>8s} {fmt(st.get('icl_pct_vs_random'), 0):>5s}")
    return result


def main(args):
    attrs, attr_names = load_cub_attributes(args.cub_root)
    results = {"args": vars(args), "models": {}}
    for model in args.models:
        path = os.path.join(args.data_dir, f"{model}_cub.npz")
        if not os.path.exists(path):
            print(f"[skip] {model}: {path} not found (run prelim/<pipeline>/export_leakage_data.py)")
            continue
        results["models"][model] = analyse(path, attrs, attr_names, args)
    os.makedirs(args.data_dir, exist_ok=True)
    out = os.path.join(args.data_dir, f"leakage_pipelines{'_name' if args.matching == 'name' else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "leakage"))
    p.add_argument("--cub-root", default=os.path.join(ROOT, "dataset", "CUB_200_2011"))
    p.add_argument("--encodings", nargs="+", default=["hard", "soft"], choices=["hard", "soft"])
    p.add_argument("--n-bins", type=int, default=8, help="soft encoding: nonzero quantile bins (+1 zero bin)")
    p.add_argument("--matching", default="data", choices=["data", "name"], help="concept -> attribute matching (see get_matching)")
    p.add_argument("--match-percentile", type=float, default=95)
    p.add_argument("--n-shuffles", type=int, default=5, help="attribute-row shuffles for the matching null")
    p.add_argument("--n-random", type=int, default=1000, help="random same-size subsets per seed (chance reference)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
