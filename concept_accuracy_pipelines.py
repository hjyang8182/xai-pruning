"""Concept accuracy / Jaccard of LF-CBM and VLG-CBM concepts against CUB's ground-truth attributes,
for the gate's open set (GCG) vs. size-matched alternatives -- the pipeline counterpart of
concept_accuracy.py (DN-CBM).

Input: the data/leakage/<model>_cub.npz each pipeline's export_leakage_data.py writes (concept names,
CUB train/test activations keyed by image id, baseline weights and gated open sets per seed).

Concept -> attribute matching, two ways (each reported separately):
  name  LF-CBM / VLG-CBM concepts are GPT phrases ("a black and white barred tail"), so each phrase is
        matched to the CUB attribute whose description ("barred tail pattern") is most similar by
        CLIP RN50 text embedding, kept if that similarity beats the --match-percentile of the same
        best-match score for Places-365 LF-CBM concept phrases (scene concepts, unrelated to birds)
  term  DN-CBM's protocol (concept_accuracy.py), for one-word concepts: the concept is matched to the
        attribute *term* ("blue", "striped", "cone") it is most similar to by CLIP text embedding, and its
        ground truth is the OR of every attribute sharing that term ("blue anywhere on the bird"); kept if
        the similarity beats the --match-percentile of the same best-match score for random words from
        the CLIP-Dissect vocab (concept_accuracy.pick_match_threshold's null)
  data  concept_leakage_pipelines.match_concepts: max Mann-Whitney |z| over the 312 attributes on the
        train images vs. a shuffled-row null (what the leakage analysis uses)

A concept whose export carries `name_alignment` (DN-CBM: cosine of its SAE dictionary vector with the
CLIP embedding of its name) is only matchable if that beats the export's null threshold
(concept_accuracy.filter_and_merge_concepts' filter), unless --no-align-filter.

Then, as concept_accuracy.py does: a per-concept activation threshold tuned for balanced accuracy on
a stratified slice of the train images, predicted presence on the test images, and per set
  mean_accuracy  per-concept agreement of predicted vs. true presence, averaged over the set's
                 matched concepts
  mean_jaccard   per-image Jaccard of the predicted-present vs. truly-present matched concepts

Sets, per gated seed (|K| = the gate's open-set size):
  gated          the gate's open set
  baseline_topk  top-|K| by the baseline head's weight-column norm (mean over baseline seeds) -- the
                 magnitude-top-|K| control
  group_lasso    the size-|K| support of the group-lasso path on the train activations
                 (gating.group_lasso_keeps) -- the l1-at-matched-size control
  random         --n-random uniform |K|-subsets (the chance reference; every set reports its
                 percentile among them)
plus a paired per-image test of gated vs. each control (concept_accuracy.paired_jaccard_test).

    python concept_accuracy_pipelines.py --models lfcbm vlgcbm
    python concept_accuracy_pipelines.py --models dncbm --matchings term data
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from concept_accuracy import (concept_mean_accuracy, paired_jaccard_test, per_image_jaccard, predict_concept_set,
                              tune_concept_thresholds)
from concept_leakage_pipelines import load_cub_attributes, match_concepts
from gating import group_lasso_keeps
from src.concepts import embed_texts, load_vocab
from src.models import load_clip

ROOT = os.path.dirname(os.path.abspath(__file__))
NULL_PHRASES = os.path.join(ROOT, "prelim", "Label-free-CBM", "data", "concept_sets", "places365_filtered.txt")
CONTROLS = ("baseline_topk", "group_lasso")


def attribute_description(name):
    """has_wing_color::blue -> 'blue wing color'; has_size::small_(5_-_9_in) -> 'small (5 - 9 in) size'."""
    category, term = name.split("::", 1)
    return f"{term.replace('_', ' ')} {category.removeprefix('has_').replace('_', ' ')}"


def name_matching(clip_model, concepts, attr_names, percentile):
    attr_emb = F.normalize(embed_texts(clip_model, [attribute_description(a) for a in attr_names]), dim=1)
    sims = F.normalize(embed_texts(clip_model, list(concepts)), dim=1) @ attr_emb.T
    with open(NULL_PHRASES) as f:
        null_phrases = [line.strip() for line in f if line.strip()]
    null_best = (F.normalize(embed_texts(clip_model, null_phrases), dim=1) @ attr_emb.T).max(1).values
    threshold = float(torch.quantile(null_best, percentile / 100))
    best, idx = sims.max(1)
    matched = {k: [int(idx[k])] for k in range(len(concepts)) if best[k] > threshold}
    return matched, threshold, {k: float(best[k]) for k in matched}


def term_matching(clip_model, concepts, attr_names, percentile, n_null=500, seed=0):
    """concept_accuracy.match_concepts_to_attributes: concept -> every attribute sharing its best term."""
    term_of = [a.split("::", 1)[1].replace("_", " ") for a in attr_names]
    terms = sorted(set(term_of))
    ids_of = {t: [i for i, u in enumerate(term_of) if u == t] for t in terms}
    term_emb = F.normalize(embed_texts(clip_model, terms), dim=1)
    vocab = load_vocab()
    words = [vocab[i] for i in torch.randperm(len(vocab), generator=torch.Generator().manual_seed(seed))[:n_null].tolist()]
    null_best = (F.normalize(embed_texts(clip_model, words), dim=1) @ term_emb.T).max(1).values
    threshold = float(torch.quantile(null_best, percentile / 100))
    best, idx = (F.normalize(embed_texts(clip_model, list(concepts)), dim=1) @ term_emb.T).max(1)
    matched = {k: ids_of[terms[int(idx[k])]] for k in range(len(concepts)) if best[k] > threshold}
    return matched, threshold, {k: float(best[k]) for k in matched}


def score_set(members, to_attr, predicted, test_attrs):
    """(mean accuracy, mean Jaccard, per-image Jaccard, n matched) of a concept set."""
    members = [int(k) for k in members if int(k) in predicted]
    if not members:
        return None, None, None, 0
    jaccard, n = per_image_jaccard(members, to_attr, predicted, test_attrs)
    acc, _ = concept_mean_accuracy(members, to_attr, predicted, test_attrs)
    return acc, float(jaccard.mean()), jaccard, n


def mean_of(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def analyse(path, attrs, attr_names, clip_model, args):
    d = dict(np.load(path, allow_pickle=True))
    model = str(d["model"])
    train_acts, test_acts = torch.from_numpy(d["train_acts"]).float(), torch.from_numpy(d["test_acts"]).float()
    train_attrs = torch.from_numpy(attrs[d["train_ids"]]).float()
    test_attrs = torch.from_numpy(attrs[d["test_ids"]]).float()
    n_concepts = train_acts.shape[1]
    base_norm = np.linalg.norm(d["baseline_W"], axis=1).mean(0)
    out = {"model": model, "npz": os.path.relpath(path, ROOT), "n_concepts": n_concepts, "matchings": {}}
    aligned = None
    if "name_alignment" in d and args.align_filter:
        aligned = d["name_alignment"] > float(d["name_alignment_threshold"])
        out["name_alignment_threshold"], out["n_aligned"] = float(d["name_alignment_threshold"]), int(aligned.sum())
        print(f"{model}: {aligned.sum()}/{n_concepts} concept names aligned with their concept "
              f"(> {float(d['name_alignment_threshold']):.4f})")

    gl_cache = {}
    for matching in args.matchings:
        if matching == "name":
            matched, threshold, score = name_matching(clip_model, d["concepts"], attr_names, args.match_percentile)
        elif matching == "term":
            matched, threshold, score = term_matching(clip_model, d["concepts"], attr_names, args.match_percentile,
                                                      seed=args.seed)
        else:
            matched, threshold, auc = match_concepts(d["train_acts"], attrs[d["train_ids"]], args.match_percentile,
                                                     args.n_shuffles, args.seed)
            matched = {k: [a] for k, a in matched.items()}
            score = {k: abs(a - 0.5) + 0.5 for k, a in auc.items()}
        n_before = len(matched)
        filtered = aligned is not None and matching != "data"  # the alignment check is about names
        if filtered:
            matched = {k: a for k, a in matched.items() if aligned[k]}
        to_attr = {k: [a + 1 for a in ids] for k, ids in matched.items()}  # concept_accuracy.py's 1-based attribute ids
        thresholds = tune_concept_thresholds(train_acts, train_attrs, to_attr, torch.from_numpy(d["train_y"]),
                                             val_frac=args.val_frac, seed=args.seed)
        predicted = predict_concept_set(test_acts, thresholds)
        print(f"\n=== {model} [{matching} matching]: {len(matched)}/{n_concepts} concepts matched "
              f"(threshold {threshold:.4f}"
              + (f"; {n_before - len(matched)} more dropped by the name-alignment filter" if filtered else "")
              + f"), {len(predicted)} with a tuned activation threshold")
        examples = sorted(matched, key=lambda k: -score[k])[:args.n_examples]
        pop_acc, pop_jac, _, pop_n = score_set(range(n_concepts), to_attr, predicted, test_attrs)
        res = {"threshold": threshold, "n_matched": len(matched), "n_scorable": len(predicted),
               "n_dropped_by_alignment": n_before - len(matched) if filtered else None,
               "population": {"n_matched": pop_n, "mean_accuracy": pop_acc, "mean_jaccard": pop_jac},
               "examples": [{"concept": str(d["concepts"][k]), "attribute": " | ".join(attr_names[a] for a in matched[k]),
                             "score": score[k]} for k in examples],
               "matched": {str(k): [attr_names[a] for a in ids] for k, ids in matched.items()}, "per_seed": []}

        rng = np.random.default_rng(args.seed)
        for seed, keep in zip(d["gated_seeds"], d["gated_keep"]):
            gated = np.flatnonzero(keep)
            k = len(gated)
            if k not in gl_cache:
                gl_cache[k] = np.flatnonzero(group_lasso_keeps(train_acts, torch.from_numpy(d["train_y"]), [k],
                                                               seed=args.seed)[k].numpy())
            sets = {"gated": gated, "baseline_topk": np.argsort(-base_norm)[:k], "group_lasso": gl_cache[k]}
            null = np.array([[v if v is not None else np.nan for v in
                              score_set(rng.choice(n_concepts, k, replace=False), to_attr, predicted, test_attrs)[:2]]
                             for _ in range(args.n_random)])
            row = {"seed": int(seed), "k": k}
            per_image = {}
            for name, members in sets.items():
                acc, jac, per_image[name], n = score_set(members, to_attr, predicted, test_attrs)
                row[name] = {"n_matched": n, "mean_accuracy": acc, "mean_jaccard": jac,
                             "overlap_with_gate": int(len(np.intersect1d(members, gated)))}
                for key, col in (("mean_accuracy", 0), ("mean_jaccard", 1)):
                    if row[name][key] is not None:
                        vals = null[:, col][~np.isnan(null[:, col])]
                        row[name][f"{key}_pct_vs_random"] = float((vals < row[name][key]).mean() * 100)
            row["random"] = {"mean_accuracy": float(np.nanmean(null[:, 0])), "mean_jaccard": float(np.nanmean(null[:, 1]))}
            for name in CONTROLS:
                if per_image["gated"] is not None and per_image[name] is not None:
                    row[f"gated_vs_{name}"] = paired_jaccard_test(per_image[name], per_image["gated"])
            res["per_seed"].append(row)
            print(f"  seed {seed} |K|={k}: " + "  ".join(
                f"{name} acc {row[name]['mean_accuracy'] or float('nan'):.3f} J {row[name]['mean_jaccard'] or float('nan'):.3f} "
                f"({row[name]['n_matched']} matched)" for name in sets)
                  + f"  random acc {row['random']['mean_accuracy']:.3f} J {row['random']['mean_jaccard']:.3f}")
        res["summary"] = {name: {key: mean_of([r[name] for r in res["per_seed"]], key)
                                 for key in ("mean_accuracy", "mean_jaccard", "n_matched", "mean_jaccard_pct_vs_random",
                                             "mean_accuracy_pct_vs_random", "overlap_with_gate")}
                          for name in ("gated", *CONTROLS)}
        res["summary"]["random"] = {key: mean_of([r["random"] for r in res["per_seed"]], key)
                                    for key in ("mean_accuracy", "mean_jaccard")}
        for name in CONTROLS:
            tests = [r[f"gated_vs_{name}"] for r in res["per_seed"] if f"gated_vs_{name}" in r]
            if tests:
                res["summary"][f"gated_vs_{name}"] = {
                    "mean_diff": mean_of(tests, "mean_diff"),
                    "seeds_ci_excludes_0": sum(t["bootstrap_ci95_mean_diff"][0] > 0 or t["bootstrap_ci95_mean_diff"][1] < 0
                                               for t in tests)}
        out["matchings"][matching] = res
    return out


def main(args):
    attrs, attr_names = load_cub_attributes(args.cub_root)
    clip_model, _ = load_clip() if {"name", "term"} & set(args.matchings) else (None, None)
    results = {"args": vars(args), "models": {}}
    for model in args.models:
        results["models"][model] = analyse(os.path.join(args.data_dir, f"{model}_cub.npz"), attrs, attr_names,
                                           clip_model, args)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"concept_accuracy_pipelines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["lfcbm", "vlgcbm"], choices=["dncbm", "lfcbm", "ucbm", "vlgcbm"])
    p.add_argument("--matchings", nargs="+", default=["name", "data"], choices=["name", "term", "data"])
    p.add_argument("--no-align-filter", dest="align_filter", action="store_false",
                   help="keep concepts whose name is not aligned with their dictionary vector (DN-CBM)")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "leakage"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "concept_accuracy"))
    p.add_argument("--cub-root", default=os.path.join(ROOT, "dataset", "CUB_200_2011"))
    p.add_argument("--match-percentile", type=float, default=95)
    p.add_argument("--n-shuffles", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--n-examples", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
