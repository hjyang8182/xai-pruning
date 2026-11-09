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
  dict  DN-CBM only, part-aware: the concept's SAE dictionary vector (a direction in CLIP image space,
        which is what DN-CBM names concepts from) is matched to the CUB attribute whose full description
        ("blue wing color", in --dict-template) has the most similar CLIP text embedding, so the body
        part is kept rather than OR-ed away as in `term`. Kept if that similarity beats the
        --dict-null-percentile (100 = all) of the same concept's similarities to the Places-365 phrases
        (in the same template) -- a per-concept null, since dictionary vectors differ in how close they
        sit to text overall. The name-alignment filter is not applied (the concept's name is not used).
  data  concept_leakage_pipelines.match_concepts: max Mann-Whitney |z| over the 312 attributes on the
        train images vs. a shuffled-row null (what the leakage analysis uses)

A concept whose export carries `name_alignment` (DN-CBM: cosine of its SAE dictionary vector with the
CLIP embedding of its name) is only matchable if that beats the export's null threshold
(concept_accuracy.filter_and_merge_concepts' filter), unless --no-align-filter.

Ground truth (--labels): CUB's per-image annotations (default), or `class` -- the standard CUB CBM's
cleaned labels (Koh et al. 2020): per-species majority vote over the train images, every image taking its
species' value, restricted to Koh et al.'s 112 attributes. Matching then only considers
those attributes.

Then, as concept_accuracy.py does: a per-concept activation threshold tuned for balanced accuracy on
a stratified slice of the train images, predicted presence on the test images, and per set
  mean_accuracy  per-concept agreement of predicted vs. true presence, averaged over the set's
                 matched concepts
  mean_jaccard   per-image Jaccard of the predicted-present vs. truly-present matched concepts
                 (depends on how many matched concepts the set holds -- compare equal-size sets only)
  mean_balanced_accuracy  per-concept mean of TPR and TNR, averaged over the set: size-independent,
                 so the one to use for GCG vs. the (much larger) baseline set

Sets, per gated seed (|K| = the gate's open-set size):
  gated          the gate's open set
  baseline_topk  top-|K| by the baseline head's weight-column norm (mean over baseline seeds) -- the
                 magnitude-top-|K| control
  group_lasso    the size-|K| support of the group-lasso path on the train activations
                 (gating.group_lasso_keeps) -- the l1-at-matched-size control
  baseline       (--controls baseline) every concept the baseline head uses (any |w| > 1e-5)
  random         --n-random uniform |K|-subsets (the chance reference; every set reports its
                 percentile among them)
plus a paired per-image test of gated vs. each control (concept_accuracy.paired_jaccard_test).

    python concept_accuracy_pipelines.py --models lfcbm vlgcbm
    python concept_accuracy_pipelines.py --models dncbm --matchings term data
    python concept_accuracy_pipelines.py --models dncbm --matchings dict     # part-aware DN-CBM matching
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
# The 112 CUB attributes (0-based) of the standard CUB CBM (Koh et al. 2020), as published in the
# concept-embedding-model code (SELECTED_CONCEPTS). Recomputing the >=10-species rule on the full CUB train
# split gives 109, 106 of them in this list -- the rest sit on the cutoff and depend on Koh et al.'s train/val split.
KOH_112 = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64,
           69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145,
           149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, 202,
           203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 254,
           259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]


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


def dict_matching(clip_model, dic_vec, attr_names, template, null_percentile):
    """dic_vec (d, n_concepts), unit columns -> {concept: [attr]}, the global null summary, and scores."""
    attr_emb = F.normalize(embed_texts(clip_model, [template.format(attribute_description(a)) for a in attr_names]), dim=1)
    with open(NULL_PHRASES) as f:
        null_phrases = [line.strip() for line in f if line.strip()]
    null_emb = F.normalize(embed_texts(clip_model, [template.format(p) for p in null_phrases]), dim=1)
    sims, null_sims = (attr_emb @ dic_vec).T, (null_emb @ dic_vec).T          # (n_concepts, 312), (n_concepts, n_null)
    thresh = torch.quantile(null_sims, null_percentile / 100, dim=1)
    best, idx = sims.max(1)
    # part margin: best attribute vs the best attribute with the same value on a different body part
    # (e.g. blue wing vs blue belly) -- small margins mean the part is a weak call
    value = [a.split("::", 1)[1] for a in attr_names]
    part_margin = {}
    matched = {}
    for k in torch.nonzero(best > thresh).squeeze(1).tolist():
        a = int(idx[k])
        others = [j for j, v in enumerate(value) if v == value[a] and j != a]
        part_margin[k] = float(best[k] - sims[k, others].max()) if others else float("nan")
        matched[k] = [a]
    return matched, float(thresh.mean()), {k: float(best[k] - thresh[k]) for k in matched}, part_margin


def concept_balanced_accuracy(members, to_attr, predicted, test_attrs):
    """Mean over the set's concepts of each concept's balanced accuracy (mean of TPR and TNR) on the
    test images -- a per-concept score, so unlike the per-image Jaccard it doesn't depend on how many
    concepts the set holds, and unlike plain accuracy it isn't won by staying off for rare attributes.
    Concepts whose attribute is constant on the test images are skipped."""
    scores = []
    for c in members:
        truth = test_attrs[:, [i - 1 for i in to_attr[c]]].any(dim=1)
        pred = predicted[c]
        pos, neg = int(truth.sum()), int((~truth).sum())
        if pos and neg:
            scores.append(0.5 * (float((pred & truth).sum()) / pos + float((~pred & ~truth).sum()) / neg))
    return float(np.mean(scores)) if scores else None


def score_set(members, to_attr, predicted, test_attrs):
    """(mean accuracy, mean Jaccard, mean balanced accuracy, per-image Jaccard, n matched) of a concept set."""
    members = [int(k) for k in members if int(k) in predicted]
    if not members:
        return None, None, None, None, 0
    jaccard, n = per_image_jaccard(members, to_attr, predicted, test_attrs)
    acc, _ = concept_mean_accuracy(members, to_attr, predicted, test_attrs)
    return acc, float(jaccard.mean()), concept_balanced_accuracy(members, to_attr, predicted, test_attrs), jaccard, n


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
        elif matching == "data":
            matched, threshold, auc = match_concepts(d["train_acts"], attrs[d["train_ids"]], args.match_percentile,
                                                     args.n_shuffles, args.seed)
            matched = {k: [a] for k, a in matched.items()}
            score = {k: abs(a - 0.5) + 0.5 for k, a in auc.items()}
        else:  # dict
            if model != "dncbm":
                print(f"{model}: dict matching needs SAE dictionary vectors (DN-CBM only), skipped")
                continue
            from src.models import load_autoencoder
            dic_vec = F.normalize(load_autoencoder("cpu").decoder.weight.detach().cpu().squeeze(), dim=0)
            assert dic_vec.shape[1] == n_concepts, (dic_vec.shape, n_concepts)
            matched, threshold, score, part_margin = dict_matching(clip_model, dic_vec, attr_names, args.dict_template,
                                                                   args.dict_null_percentile)
        n_before = len(matched)
        filtered = aligned is not None and matching in ("name", "term")  # the alignment check is about names
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
        pop_acc, pop_jac, pop_bal, _, pop_n = score_set(range(n_concepts), to_attr, predicted, test_attrs)
        res = {"threshold": threshold, "n_matched": len(matched), "n_scorable": len(predicted),
               "n_dropped_by_alignment": n_before - len(matched) if filtered else None,
               "population": {"n_matched": pop_n, "mean_accuracy": pop_acc, "mean_jaccard": pop_jac,
                              "mean_balanced_accuracy": pop_bal},
               "examples": [{"concept": str(d["concepts"][k]), "attribute": " | ".join(attr_names[a] for a in matched[k]),
                             "score": score[k]} for k in examples],
               "matched": {str(k): [attr_names[a] for a in ids] for k, ids in matched.items()}, "per_seed": []}
        if matching == "dict":
            margins = np.array([part_margin[k] for k in matched if not np.isnan(part_margin[k])])
            categories = {}
            for ids in matched.values():
                c = attr_names[ids[0]].split("::", 1)[0]
                categories[c] = categories.get(c, 0) + 1
            res.update(template=args.dict_template, null_percentile=args.dict_null_percentile,
                       n_distinct_attributes=len({ids[0] for ids in matched.values()}),
                       categories=dict(sorted(categories.items(), key=lambda kv: -kv[1])),
                       part_margin={"median": float(np.median(margins)) if len(margins) else None,
                                    "frac_above_0.01": float((margins > 0.01).mean()) if len(margins) else None})
            print(f"  {res['n_distinct_attributes']} distinct attributes; categories {list(res['categories'].items())[:6]}; "
                  f"part margin median {res['part_margin']['median']}")

        rng = np.random.default_rng(args.seed)
        controls = tuple(args.controls)
        metrics = (("mean_accuracy", 0), ("mean_jaccard", 1), ("mean_balanced_accuracy", 2))
        for i, (seed, keep) in enumerate(zip(d["gated_seeds"], d["gated_keep"])):
            gated = np.flatnonzero(keep)
            k = len(gated)
            sets = {"gated": gated}
            if "baseline" in controls:  # every concept the baseline head uses (any non-zero class weight)
                sets["baseline"] = np.flatnonzero((np.abs(d["baseline_W"][i % len(d["baseline_W"])]) > 1e-5).any(0))
            if "baseline_topk" in controls:
                sets["baseline_topk"] = np.argsort(-base_norm)[:k]
            if "group_lasso" in controls:
                if k not in gl_cache:
                    gl_cache[k] = np.flatnonzero(group_lasso_keeps(train_acts, torch.from_numpy(d["train_y"]), [k],
                                                                   seed=args.seed)[k].numpy())
                sets["group_lasso"] = gl_cache[k]
            null = np.array([[v if v is not None else np.nan for v in
                              score_set(rng.choice(n_concepts, k, replace=False), to_attr, predicted, test_attrs)[:3]]
                             for _ in range(args.n_random)])
            row = {"seed": int(seed), "k": k}
            per_image = {}
            for name, members in sets.items():
                acc, jac, bal, per_image[name], n = score_set(members, to_attr, predicted, test_attrs)
                row[name] = {"n_concepts": int(len(members)), "n_matched": n, "mean_accuracy": acc, "mean_jaccard": jac,
                             "mean_balanced_accuracy": bal, "overlap_with_gate": int(len(np.intersect1d(members, gated)))}
                for key, col in metrics:
                    if row[name][key] is not None:
                        vals = null[:, col][~np.isnan(null[:, col])]
                        row[name][f"{key}_pct_vs_random"] = float((vals < row[name][key]).mean() * 100)
            row["random"] = {key: float(np.nanmean(null[:, col])) for key, col in metrics}
            for name in controls:
                if per_image["gated"] is not None and per_image[name] is not None:
                    row[f"gated_vs_{name}"] = paired_jaccard_test(per_image[name], per_image["gated"])
            res["per_seed"].append(row)
            print(f"  seed {seed} |K|={k}: " + "  ".join(
                f"{name} acc {row[name]['mean_accuracy'] or float('nan'):.3f} J {row[name]['mean_jaccard'] or float('nan'):.3f} "
                f"bal {row[name]['mean_balanced_accuracy'] or float('nan'):.3f} ({row[name]['n_matched']}/{row[name]['n_concepts']} matched)"
                for name in sets)
                  + f"  random acc {row['random']['mean_accuracy']:.3f} J {row['random']['mean_jaccard']:.3f} "
                    f"bal {row['random']['mean_balanced_accuracy']:.3f}")
        res["summary"] = {name: {key: mean_of([r[name] for r in res["per_seed"]], key)
                                 for key in ("mean_accuracy", "mean_jaccard", "mean_balanced_accuracy", "n_concepts",
                                             "n_matched", "mean_jaccard_pct_vs_random", "mean_accuracy_pct_vs_random",
                                             "mean_balanced_accuracy_pct_vs_random", "overlap_with_gate")}
                          for name in ("gated", *controls)}
        res["summary"]["random"] = {key: mean_of([r["random"] for r in res["per_seed"]], key) for key, _ in metrics}
        for name in controls:
            tests = [r[f"gated_vs_{name}"] for r in res["per_seed"] if f"gated_vs_{name}" in r]
            if tests:
                res["summary"][f"gated_vs_{name}"] = {
                    "mean_diff": mean_of(tests, "mean_diff"),
                    "seeds_ci_excludes_0": sum(t["bootstrap_ci95_mean_diff"][0] > 0 or t["bootstrap_ci95_mean_diff"][1] < 0
                                               for t in tests)}
        out["matchings"][matching] = res
    return out


def class_level_attributes(attrs, cub_root, min_class_count=None):
    """The standard CUB CBM's (Koh et al. 2020) cleaned concept labels: per species, an attribute is present
    if at least half of that species' visible *train* annotations say so, and every image takes its species'
    value; then only Koh et al.'s 112 attributes are kept (or, with `min_class_count`, those present in at
    least that many species, recomputed here -- 109, 106 of them in the 112).
    Returns the (max_img_id + 1, n_kept) label matrix and the kept attribute indices."""
    def read(name):
        rows = np.loadtxt(os.path.join(cub_root, name), dtype=np.int64)
        return rows[:, 0], rows[:, 1]
    ids, cls = read("image_class_labels.txt")
    split_ids, is_train = read("train_test_split.txt")
    assert np.array_equal(ids, split_ids)
    n_classes = int(cls.max())
    cls_of = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    cls_of[ids] = cls
    train_img = np.zeros(int(ids.max()) + 1, dtype=bool)
    train_img[ids] = is_train == 1
    # As Koh et al.'s CUB preprocessing: votes from train images only; an "absent" annotation with
    # certainty 1 ("not visible") casts no vote; a tie counts as present.
    rows = np.loadtxt(os.path.join(cub_root, "attributes", "image_attribute_labels.txt"), usecols=(0, 1, 2, 3),
                      dtype=np.int64)
    img, attr, present, certainty = rows[:, 0], rows[:, 1] - 1, rows[:, 2], rows[:, 3]
    vote = train_img[img] & ~((present == 0) & (certainty == 1))
    count = np.zeros((n_classes, attrs.shape[1], 2))
    np.add.at(count, (cls_of[img[vote]] - 1, attr[vote], present[vote]), 1)
    class_attr = count[..., 1] >= count[..., 0]                              # (n_classes, 312), ties -> present
    kept = (np.array(KOH_112) if min_class_count is None
            else np.flatnonzero(class_attr.sum(0) >= min_class_count))
    labels = np.zeros((attrs.shape[0], len(kept)), dtype=np.int8)
    labels[ids] = class_attr[cls - 1][:, kept]
    return labels, kept


def main(args):
    attrs, attr_names = load_cub_attributes(args.cub_root)
    if args.labels == "class":
        attrs, kept = class_level_attributes(attrs, args.cub_root, args.min_class_count)
        attr_names = [attr_names[i] for i in kept]
        print(f"Class-level (Koh et al.) labels on {len(kept)} attributes")
    clip_model, _ = load_clip() if {"name", "term", "dict"} & set(args.matchings) else (None, None)
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
    p.add_argument("--matchings", nargs="+", default=["name", "data"], choices=["name", "term", "dict", "data"])
    p.add_argument("--dict-template", default="{}", help="text template around the attribute description for dict matching, "
                                                      "e.g. 'a photo of a bird with {}'")
    p.add_argument("--dict-null-percentile", type=float, default=100,
                   help="per-concept null for dict matching: percentile of the concept's similarity to Places-365 phrases")
    p.add_argument("--no-align-filter", dest="align_filter", action="store_false",
                   help="keep concepts whose name is not aligned with their dictionary vector (DN-CBM)")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "leakage"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "concept_accuracy"))
    p.add_argument("--cub-root", default=os.path.join(ROOT, "dataset", "CUB_200_2011"))
    p.add_argument("--match-percentile", type=float, default=95)
    p.add_argument("--n-shuffles", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--labels", default="image", choices=["image", "class"],
                   help="ground truth: CUB's per-image annotations, or the standard CBM's class-level majority-vote "
                        "labels restricted to attributes present in >= --min-class-count species (Koh et al. 2020)")
    p.add_argument("--min-class-count", type=int, default=None,
                   help="with --labels class: recompute the attribute subset (present in >= this many species) "
                        "instead of using Koh et al.'s published 112")
    p.add_argument("--controls", nargs="+", default=list(CONTROLS), choices=["baseline", *CONTROLS],
                   help="sets GCG is compared with: baseline = every concept the baseline head uses")
    p.add_argument("--n-examples", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
