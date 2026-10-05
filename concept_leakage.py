"""Concepts-task leakage (CTL) and interconcept leakage (ICL) for the CUB probes, after
Parisini et al., "Leakage and Interpretability in Concept-Based Models" (arXiv:2504.14094).

  CTL_i  = max(0, I(c_hat_i; y) / H(y) - I(c_i; y) / H(y))
  ICL_ij = max(0, I(c_hat_i; c_hat_j) / sqrt(H(c_hat_i) H(c_hat_j))
               - I(c_i; c_j)         / sqrt(H(c_i) H(c_j)))

where c_i is a ground-truth CUB attribute, c_hat_i the SAE concept matched to it (see
concept_accuracy.py for the matching), and y the species label. CTL asks how much *more*
the learned concept tells the head about the species than the attribute it is named after
does; ICL asks how much more it tells about the other attributes. Both are properties of
the concept layer alone (the gate scales activations *after* they are computed), so the
baseline and gated probes differ only through *which* concepts each one relies on. Scores
are therefore reported (a) over every matched concept, as the population reference, and
(b) averaged over the concept set each probe uses - the gate's open set vs. the baseline's
top-|open| by weight norm, the same pairing concept_accuracy.py evaluates.

Only concepts that matched a CUB attribute can be scored - unmatched ones have no c_i - so
every set also reports its coverage.

MI is estimated by plug-in on discretised variables with the Miller-Madow bias correction,
so predicted and ground-truth sides are estimated identically: y and c_i are already
discrete; c_hat_i is either binarised with the attribute-tuned thresholds from
concept_accuracy.py (`--encoding hard`, the paper's "hard CBM" reading) or bucketed into a
zero bin plus `--n-bins` quantile bins of its nonzero mass (`--encoding soft`; SAE
activations are mostly exactly zero, which k-NN estimators such as KSG handle badly).

Alongside the MI scores the paper's first design check is run: fit a linear head on the
*ground-truth* attributes matched by a concept set and compare its test accuracy with a
head refit on the *predicted* activations of the same set. The predicted-side surplus is
accuracy the head gets from information the annotated concepts do not carry.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src import config
from src.concepts import load_concept_names, load_vocab, save_vocab_embeddings, load_vocab_embeddings, name_concepts
from src.data import load_clip_features
from src.models import load_clip, load_autoencoder, LinearProbe, GatedProbe
from concept_accuracy import (
    load_cub_attribute_terms, load_cub_image_attributes, filter_and_merge_concepts, merge_activations,
    match_concepts_to_attributes, tune_concept_thresholds, to_group_indices, effective_weight,
    _latest_run, _probe_paths, _mean_std,
)


# ---------------------------------------------------------------------------------------------
# Discrete information estimates
# ---------------------------------------------------------------------------------------------

# Relabel a 1-d array's values to 0..K-1 so it can index a contingency table
def _codes(x):
    _, codes = np.unique(np.asarray(x), return_inverse=True)
    return codes.astype(np.int64)

# Plug-in entropy in bits with the Miller-Madow correction (K_nonempty - 1) / (2N), in bits
def entropy(codes):
    n = codes.shape[0]
    counts = np.bincount(codes).astype(np.float64)
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum() + (p.size - 1) / (2 * n * np.log(2)))

# Plug-in MI in bits, Miller-Madow corrected as H(x) + H(y) - H(x, y) so all three terms share the
# same correction (the joint's correction is (K_xy - 1) / 2N, so the net bias term is
# (K_x + K_y - K_xy - 1) / 2N and MI can come out slightly negative; clipped at 0)
def mutual_information(codes_a, codes_b):
    joint = codes_a * (codes_b.max() + 1) + codes_b
    return max(0.0, entropy(codes_a) + entropy(codes_b) - entropy(joint))

# Bucket a continuous concept activation into discrete codes: exact zeros get their own bin,
# the nonzero mass is split into `n_bins` quantile bins (fewer if there are too few distinct values)
def discretise_soft(acts_col, n_bins):
    x = np.asarray(acts_col, dtype=np.float64)
    codes = np.zeros(x.shape[0], dtype=np.int64)
    nonzero = x != 0
    if nonzero.sum() > 1:
        edges = np.unique(np.quantile(x[nonzero], np.linspace(0, 1, n_bins + 1)[1:-1]))
        codes[nonzero] = 1 + np.searchsorted(edges, x[nonzero], side='right')
    return codes


# ---------------------------------------------------------------------------------------------
# Leakage scores
# ---------------------------------------------------------------------------------------------

# Encode every matched concept both ways: predicted (soft bins or hard threshold) and ground truth
# (OR over its matched attribute ids). Concepts without a tuned threshold are skipped under hard
# encoding (never matched a positive/negative split), matching concept_accuracy's behaviour.
def encode_concepts(merged_acts, attrs, concept_idx_to_attr_ids, thresholds, encoding, n_bins):
    predicted, truth = {}, {}
    for concept_idx, attr_ids in concept_idx_to_attr_ids.items():
        if encoding == 'hard':
            if concept_idx not in thresholds:
                continue
            predicted[concept_idx] = (merged_acts[:, concept_idx] > thresholds[concept_idx]).numpy().astype(np.int64)
        else:
            predicted[concept_idx] = discretise_soft(merged_acts[:, concept_idx].numpy(), n_bins)
        truth[concept_idx] = attrs[:, [i - 1 for i in attr_ids]].any(dim=1).numpy().astype(np.int64)
    return predicted, truth

# Per-concept CTL (see module docstring). Also returns the two normalised MI terms so a score of 0
# can be told apart as "predicted carries less than truth" vs. "both carry nothing".
def concepts_task_leakage(predicted, truth, labels):
    y = _codes(labels)
    h_y = entropy(y)
    scores = {}
    for c in predicted:
        mi_pred = mutual_information(predicted[c], y) / h_y
        mi_true = mutual_information(truth[c], y) / h_y
        scores[c] = {'ctl': max(0.0, mi_pred - mi_true), 'nmi_pred_task': mi_pred, 'nmi_true_task': mi_true}
    return scores

# Pairwise ICL over `concepts` (upper triangle), geometric-mean entropy normalisation on each side.
# Pairs where either side has zero entropy (a constant variable) are undefined and skipped.
def interconcept_leakage(predicted, truth, concepts):
    h_pred = {c: entropy(predicted[c]) for c in concepts}
    h_true = {c: entropy(truth[c]) for c in concepts}
    scores = {}
    for a_i, a in enumerate(concepts):
        for b in concepts[a_i + 1:]:
            if min(h_pred[a], h_pred[b], h_true[a], h_true[b]) <= 0:
                continue
            nmi_pred = mutual_information(predicted[a], predicted[b]) / np.sqrt(h_pred[a] * h_pred[b])
            nmi_true = mutual_information(truth[a], truth[b]) / np.sqrt(h_true[a] * h_true[b])
            scores[(a, b)] = max(0.0, nmi_pred - nmi_true)
    return scores

# Summary of a concept set's leakage: mean CTL (unweighted, and weighted by `weights` when given -
# the probe's effective weight norm per concept, so concepts the head leans on count more), mean
# pairwise ICL, and coverage (how many of the set's concepts could be scored at all)
def summarise_set(concepts, ctl_scores, predicted, truth, n_used, weights=None):
    scored = [c for c in concepts if c in ctl_scores]
    out = {'n_used': n_used, 'n_matched': len(scored)}
    if not scored:
        return out
    ctl = np.array([ctl_scores[c]['ctl'] for c in scored])
    out['mean_ctl'] = float(ctl.mean())
    out['frac_ctl_positive'] = float((ctl > 0).mean())
    out['mean_nmi_pred_task'] = float(np.mean([ctl_scores[c]['nmi_pred_task'] for c in scored]))
    out['mean_nmi_true_task'] = float(np.mean([ctl_scores[c]['nmi_true_task'] for c in scored]))
    if weights is not None:
        w = np.array([weights[c] for c in scored])
        out['weighted_mean_ctl'] = float((ctl * w).sum() / w.sum()) if w.sum() > 0 else float('nan')
    icl = interconcept_leakage(predicted, truth, scored)
    out['n_icl_pairs'] = len(icl)
    out['mean_icl'] = float(np.mean(list(icl.values()))) if icl else float('nan')
    return out


# ---------------------------------------------------------------------------------------------
# Reliance check: accuracy from annotated concepts alone vs. from the predicted activations
# ---------------------------------------------------------------------------------------------

# Multinomial logistic regression fit on `x_train`, accuracy on `x_test`. Features are standardised
# from the train split so binary attributes and SAE activations get the same regulariser strength.
def linear_head_accuracy(x_train, y_train, x_test, y_test, C=1.0, seed=0):
    x_train, x_test = np.asarray(x_train, dtype=np.float64), np.asarray(x_test, dtype=np.float64)
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit((x_train - mu) / sd, y_train)
    return float((clf.predict((x_test - mu) / sd) == y_test).mean())

# For a concept set: (a) head on the ground-truth attributes its matched concepts point at
# (leakage-free by construction), (b) head refit on the same concepts' predicted activations.
# The (b) - (a) surplus is task accuracy that the concepts' annotated meaning does not explain.
def reliance_check(concepts, concept_idx_to_attr_ids, train_merged, test_merged, train_attrs, test_attrs,
                   y_train, y_test, C, seed):
    matched = [c for c in concepts if c in concept_idx_to_attr_ids]
    if not matched:
        return None
    attr_cols = sorted({i - 1 for c in matched for i in concept_idx_to_attr_ids[c]})
    acc_truth = linear_head_accuracy(train_attrs[:, attr_cols], y_train, test_attrs[:, attr_cols], y_test, C, seed)
    acc_pred = linear_head_accuracy(train_merged[:, matched], y_train, test_merged[:, matched], y_test, C, seed)
    return {'n_matched': len(matched), 'n_attributes': len(attr_cols),
            'acc_truth_attrs': acc_truth, 'acc_pred_acts': acc_pred, 'surplus': acc_pred - acc_truth}


# Probe weight norm per merged group (max |w| over each group's raw members, L2 over classes), as
# the importance the head assigns to a merged concept
def merged_weight_norm(probe, gated, group_indices):
    w = effective_weight(probe, gated).detach()
    return {g: w[:, idx].abs().max(dim=1).values.norm().item() for g, idx in enumerate(group_indices)}


def parse_args():
    parser = argparse.ArgumentParser(description='CTL / ICL leakage scores (Parisini et al. 2025) for CUB concepts, over the baseline and gated probes\' concept sets')
    parser.add_argument('--baseline-run', default=None, help='data/cub/model/<run>; defaults to the latest baseline_* run')
    parser.add_argument('--gated-run', default=None, help='data/cub/model/<run>; defaults to the latest gated_* run')
    parser.add_argument('--encoding', choices=['soft', 'hard'], default='soft', help='How the predicted concept is discretised for MI: quantile bins of the activation, or the attribute-tuned binary threshold')
    parser.add_argument('--n-bins', type=int, default=8, help='Nonzero quantile bins for --encoding soft (plus one bin for exact zeros)')
    parser.add_argument('--split', choices=['test', 'all'], default='test', help='Images the MI estimates are taken over')
    parser.add_argument('--align-percentile', type=float, default=95)
    parser.add_argument('--match-percentile', type=float, default=95)
    parser.add_argument('--val-frac', type=float, default=0.2)
    parser.add_argument('--gate-threshold', type=float, default=0.5)
    parser.add_argument('--head-C', type=float, default=1.0, help='Inverse L2 strength of the reliance-check logistic heads')
    parser.add_argument('--skip-reliance', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main(args):
    n_classes = 200
    SAVE      = os.path.join(config.DATA_PATH, 'cub')
    ACT_SAVE  = os.path.join(SAVE, 'activations')
    MODEL_DIR = os.path.join(SAVE, 'model')

    clip_model, _ = load_clip()
    autoencoder = load_autoencoder(config.sae_device)

    concept_names_path = os.path.join(SAVE, 'concept_names.csv')
    if os.path.exists(concept_names_path):
        concept_names = load_concept_names(concept_names_path)
    else:
        if not os.path.exists(config.EMB_PATH):
            save_vocab_embeddings(clip_model, load_vocab())
        concept_names = name_concepts(autoencoder, load_vocab_embeddings(), load_vocab(), csv_path=concept_names_path)

    train_acts = torch.load(os.path.join(ACT_SAVE, 'train_sae_acts.pt'), map_location='cpu')
    test_acts  = torch.load(os.path.join(ACT_SAVE, 'test_sae_acts.pt'),  map_location='cpu')
    train_attrs, test_attrs = load_cub_image_attributes(train=True), load_cub_image_attributes(train=False)
    _, train_labels = load_clip_features(ACT_SAVE, 'train')
    _, test_labels  = load_clip_features(ACT_SAVE, 'test')
    train_labels, test_labels = torch.as_tensor(train_labels), torch.as_tensor(test_labels)

    merged_names, group_indices, raw_idx_to_group_idx, _ = filter_and_merge_concepts(
        clip_model, autoencoder, concept_names, percentile=args.align_percentile)
    print(f'{len(merged_names)} merged concepts kept from {len(concept_names)} raw concepts')
    train_merged, test_merged = merge_activations(train_acts, group_indices), merge_activations(test_acts, group_indices)

    attribute_terms = load_cub_attribute_terms()
    attr_ids_of = {}
    for attr_id, _, term in attribute_terms:
        attr_ids_of.setdefault(term, []).append(attr_id)
    matches, match_threshold = match_concepts_to_attributes(clip_model, merged_names, attribute_terms, percentile=args.match_percentile)
    concept_idx_to_attr_ids = {c: attr_ids_of[term] for term, ms in matches.items() for c, _, _ in ms}
    print(f'{len(concept_idx_to_attr_ids)}/{len(merged_names)} merged concepts matched a CUB attribute above threshold={match_threshold:.4f}')

    thresholds = tune_concept_thresholds(train_merged, train_attrs, concept_idx_to_attr_ids, train_labels,
                                         val_frac=args.val_frac, seed=args.seed) if args.encoding == 'hard' else {}

    if args.split == 'all':
        eval_acts, eval_attrs, eval_labels = torch.cat([train_merged, test_merged]), torch.cat([train_attrs, test_attrs]), torch.cat([train_labels, test_labels])
    else:
        eval_acts, eval_attrs, eval_labels = test_merged, test_attrs, test_labels

    predicted, truth = encode_concepts(eval_acts, eval_attrs, concept_idx_to_attr_ids, thresholds, args.encoding, args.n_bins)
    ctl_scores = concepts_task_leakage(predicted, truth, eval_labels.numpy())
    all_matched = sorted(predicted.keys())
    population = summarise_set(all_matched, ctl_scores, predicted, truth, n_used=len(merged_names))
    print(f'\n[{args.encoding} encoding, {args.split} split, n={eval_acts.shape[0]}] all {population["n_matched"]} matched concepts: '
          f'mean CTL = {population["mean_ctl"]:.4f} ({population["frac_ctl_positive"] * 100:.0f}% > 0), '
          f'mean NMI(pred; y) = {population["mean_nmi_pred_task"]:.4f} vs NMI(true; y) = {population["mean_nmi_true_task"]:.4f}, '
          f'mean ICL = {population["mean_icl"]:.4f} over {population["n_icl_pairs"]} pairs')

    top = sorted(all_matched, key=lambda c: -ctl_scores[c]['ctl'])[:10]
    print('Highest-CTL concepts: ' + ', '.join(f'{merged_names[c]} ({ctl_scores[c]["ctl"]:.3f})' for c in top))

    y_train, y_test = train_labels.numpy(), test_labels.numpy()
    reliance_all = None
    if not args.skip_reliance:
        acc_all_attrs = linear_head_accuracy(train_attrs, y_train, test_attrs, y_test, args.head_C, args.seed)
        reliance_all = reliance_check(all_matched, concept_idx_to_attr_ids, train_merged, test_merged, train_attrs, test_attrs,
                                      y_train, y_test, args.head_C, args.seed)
        print(f'Completeness: linear head on all 312 ground-truth attributes = {acc_all_attrs:.4f}; on the {reliance_all["n_attributes"]} '
              f'attributes matched by any concept = {reliance_all["acc_truth_attrs"]:.4f}; refit on those concepts\' activations = {reliance_all["acc_pred_acts"]:.4f}')

    baseline_run = args.baseline_run or _latest_run(MODEL_DIR, 'baseline_')
    gated_run    = args.gated_run or _latest_run(MODEL_DIR, 'gated_')
    baseline_paths, gated_paths = _probe_paths(os.path.join(MODEL_DIR, baseline_run)), _probe_paths(os.path.join(MODEL_DIR, gated_run))
    n_seeds = min(len(baseline_paths), len(gated_paths))
    print(f'\nEvaluating baseline: {baseline_run} and gated: {gated_run} over {n_seeds} seed(s)')

    per_seed = {'baseline': [], 'gated': []}
    for seed_i, (b_path, g_path) in enumerate(zip(baseline_paths[:n_seeds], gated_paths[:n_seeds])):
        b_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        b_probe.load_state_dict(torch.load(b_path, map_location='cpu'))
        g_probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
        g_probe.load_state_dict(torch.load(g_path, map_location='cpu'))
        open_concepts = (g_probe.gate_probs() > args.gate_threshold).nonzero().squeeze(1).tolist()
        baseline_concepts = b_probe.linear.weight.norm(dim=0).topk(len(open_concepts)).indices.tolist()

        for label, probe, gated, raw_set, path in [('baseline', b_probe, False, baseline_concepts, b_path),
                                                   ('gated', g_probe, True, open_concepts, g_path)]:
            groups = to_group_indices(raw_set, raw_idx_to_group_idx)
            summary = summarise_set(groups, ctl_scores, predicted, truth, n_used=len(raw_set),
                                    weights=merged_weight_norm(probe, gated, group_indices))
            summary['probe_path'] = path
            summary['n_merged'] = len(groups)
            if not args.skip_reliance:
                summary['reliance'] = reliance_check(groups, concept_idx_to_attr_ids, train_merged, test_merged, train_attrs, test_attrs,
                                                     y_train, y_test, args.head_C, args.seed)
            per_seed[label].append(summary)
            if 'mean_ctl' not in summary:
                print(f'[seed {seed_i}] {label}: none of its {len(raw_set)} concepts ({len(groups)} merged) matched an attribute')
                continue
            line = (f'[seed {seed_i}] {label}: {len(raw_set)} raw / {len(groups)} merged / {summary["n_matched"]} matched, '
                    f'mean CTL = {summary["mean_ctl"]:.4f}, weight-weighted CTL = {summary["weighted_mean_ctl"]:.4f}, '
                    f'mean ICL = {summary["mean_icl"]:.4f}')
            if summary.get('reliance'):
                r = summary['reliance']
                line += f'; head acc on truth attrs = {r["acc_truth_attrs"]:.4f}, on pred acts = {r["acc_pred_acts"]:.4f} (surplus {r["surplus"]:+.4f})'
            print(line)

    results = {'encoding': args.encoding, 'n_bins': args.n_bins, 'split': args.split, 'n_eval_images': int(eval_acts.shape[0]),
               'align_percentile': args.align_percentile, 'match_percentile': args.match_percentile, 'match_threshold': match_threshold,
               'population': population, 'reliance_all_matched': reliance_all,
               'top_ctl_concepts': [{'concept': merged_names[c], **ctl_scores[c]} for c in top],
               'per_concept': {merged_names[c]: ctl_scores[c] for c in all_matched}}
    for label, run in [('baseline', baseline_run), ('gated', gated_run)]:
        seeds = per_seed[label]
        agg = {'run': run, 'n_seeds': n_seeds, 'per_seed': seeds}
        for key in ['mean_ctl', 'weighted_mean_ctl', 'mean_icl']:
            agg[f'{key}_mean'], agg[f'{key}_std'] = _mean_std([s.get(key) for s in seeds])
        if not args.skip_reliance:
            for key in ['acc_truth_attrs', 'acc_pred_acts', 'surplus']:
                agg[f'{key}_mean'], agg[f'{key}_std'] = _mean_std([s['reliance'][key] for s in seeds if s.get('reliance')])
        results[label] = agg
        if agg['mean_ctl_mean'] is not None:
            print(f'{label} ({run}) over {n_seeds} seed(s): mean CTL = {agg["mean_ctl_mean"]:.4f} +/- {agg["mean_ctl_std"]:.4f}, '
                  f'weighted CTL = {agg["weighted_mean_ctl_mean"]:.4f} +/- {agg["weighted_mean_ctl_std"]:.4f}, '
                  f'mean ICL = {agg["mean_icl_mean"]:.4f} +/- {agg["mean_icl_std"]:.4f}'
                  + (f', reliance surplus = {agg["surplus_mean"]:+.4f} +/- {agg["surplus_std"]:.4f}' if not args.skip_reliance else ''))

    out_path = os.path.join(SAVE, f"concept_leakage_{args.encoding}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main(parse_args())
