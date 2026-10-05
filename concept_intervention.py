import argparse
import json
import os
from datetime import datetime
import torch
from src import config
from src.concepts import load_vocab, save_vocab_embeddings, load_vocab_embeddings, name_concepts, load_concept_names
from src.data import load_clip_features
from src.models import load_clip, load_autoencoder, LinearProbe, GatedProbe
from src.visualise import plot_concept_intervention
from concept_accuracy import (
    load_cub_attribute_terms, load_cub_image_attributes, filter_and_merge_concepts,
    match_concepts_to_attributes, tune_concept_thresholds, merge_activations, _latest_run,
)
from concept_ablation import _probe_paths_by_seed

INTERVENE_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concept intervention curve for CUB (Koh et al.-style): replace increasing "
                     "fractions of SAE concepts matched to CUB ground-truth attributes with "
                     "ground-truth-informed activation values, and measure how test accuracy "
                     "responds. Plots baseline vs. gated probe over the same intervened concepts."
    )
    parser.add_argument('--baseline-run', default=None, help='data/cub/model/<run> to evaluate; defaults to the latest baseline_* run')
    parser.add_argument('--gated-run', default=None, help='data/cub/model/<run> to evaluate; defaults to the latest gated_* run')
    parser.add_argument('--align-percentile', type=float, default=95, help='Percentile of the mismatched-pairing null distribution used as the concept dictionary-vector/name alignment threshold')
    parser.add_argument('--match-percentile', type=float, default=95, help='Percentile of the unrelated-word null distribution used as the concept/attribute match threshold')
    parser.add_argument('--fractions', type=float, nargs='+', default=INTERVENE_FRACTIONS, help='Fractions of matched concepts to intervene')
    parser.add_argument('--n-orderings', type=int, default=5, help='Random concept orderings averaged per (seed, fraction), for the random-order curve')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Held-out fraction of train used to tune per-concept detection thresholds (needed for the uncertainty-order curve)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random concept-ordering draws and the threshold-tuning val split')
    return parser.parse_args()


# Ground-truth presence per image for each matched merged concept: attrs[:, attr_ids] OR'd
# together, same definition used throughout concept_accuracy.py (_prediction_matrices,
# tune_concept_thresholds).
def _ground_truth(attrs, concept_idx_to_attr_ids):
    return {
        c: attrs[:, [i - 1 for i in attr_ids]].any(dim=1)
        for c, attr_ids in concept_idx_to_attr_ids.items()
    }


# For each matched merged concept, the reference raw-activation value (per raw SAE dictionary
# index in its merged group) to clamp to when intervening: the training-set mean activation among
# images where the concept's linked attribute is ground-truth present, and separately among images
# where it's absent. A concept is dropped if train has no positive or no negative example, since
# neither reference value would be well-defined (same degenerate case tune_concept_thresholds
# guards against).
def positive_negative_refs(train_acts, train_truth, group_indices):
    pos_val, neg_val, valid_concepts = {}, {}, []
    for c, truth in train_truth.items():
        pos_mask, neg_mask = truth, ~truth
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        for raw_idx in group_indices[c]:
            pos_val[raw_idx] = train_acts[pos_mask, raw_idx].mean().item()
            neg_val[raw_idx] = train_acts[neg_mask, raw_idx].mean().item()
        valid_concepts.append(c)
    return pos_val, neg_val, valid_concepts


# Clamp every raw activation belonging to each concept in `concepts` to its ground-truth reference
# value (positive or negative, per image, per concept's test_truth) - i.e. "correct" those
# concepts to what the CUB attribute labels actually say about each test image, exactly like a
# classic concept-bottleneck intervention but expressed in this SAE's continuous activation space.
def intervene(acts, concepts, group_indices, test_truth, pos_val, neg_val):
    acts = acts.clone()
    for c in concepts:
        truth = test_truth[c]
        for raw_idx in group_indices[c]:
            acts[truth, raw_idx] = pos_val[raw_idx]
            acts[~truth, raw_idx] = neg_val[raw_idx]
    return acts


def evaluate(probe, acts, labels):
    with torch.no_grad():
        out = probe(acts)
        logits = out[0] if isinstance(out, tuple) else out
        return (logits.argmax(dim=1) == labels).float().mean().item()


# Mean/std test accuracy at each fraction of concepts intervened, averaged over n_orderings random
# permutations of `valid_concepts` (since which concepts get corrected first shouldn't bias the
# curve - Koh et al. average over random intervention orders for the same reason). Same global
# order is used for every test image - this is the "how much does correcting an arbitrary subset of
# concepts help, on average" curve, not the best-case one (see uncertainty_order_curve for that).
def random_order_curve(probe, test_acts, test_labels, group_indices, test_truth, pos_val, neg_val,
                        valid_concepts, fractions, n_orderings, seed):
    generator = torch.Generator().manual_seed(seed)
    fraction_accs = {f: [] for f in fractions}
    for _ in range(n_orderings):
        perm = torch.randperm(len(valid_concepts), generator=generator).tolist()
        order = [valid_concepts[i] for i in perm]
        for f in fractions:
            n = round(f * len(order))
            acts = intervene(test_acts, order[:n], group_indices, test_truth, pos_val, neg_val)
            fraction_accs[f].append(evaluate(probe, acts, test_labels))
    means = [sum(fraction_accs[f]) / len(fraction_accs[f]) for f in fractions]
    stds  = [torch.tensor(fraction_accs[f]).std().item() if len(fraction_accs[f]) > 1 else 0.0 for f in fractions]
    return means, stds


# Per-(image, concept) distance from that concept's own tuned detection threshold (on the merged,
# max-over-group activation) - a proxy for the concept detector's own confidence, since these
# thresholded SAE activations have no calibrated probability the way Koh et al.'s concept predictor
# does. Small distance = activation sits right on the decision boundary = the detector is least
# sure whether the concept fired, so that's the (image, concept) pair we intervene on first, per
# image, for the uncertainty-ranked curve below. Returns [n_test, len(valid_concepts)].
def boundary_distance(test_acts, group_indices, thresholds, valid_concepts):
    merged = torch.stack([test_acts[:, group_indices[c]].max(dim=1).values for c in valid_concepts], dim=1)
    thresh = torch.tensor([thresholds[c] for c in valid_concepts])
    return (merged - thresh.unsqueeze(0)).abs()


# Mean/std test accuracy at each fraction of concepts intervened, correcting - per test image - the
# concepts the detector is *least confident about first* (smallest boundary_distance), rather than a
# single order shared across all images. This is deterministic (no random orderings to average), and
# is the more informative "how much does fixing the model's actual mistakes help" curve: the
# detector's most-uncertain concepts are exactly where it's most likely to have gotten the
# presence/absence call wrong, so correcting those first should show a sharper rise than random order.
def uncertainty_order_curve(probe, test_acts, test_labels, group_indices, test_truth, pos_val, neg_val,
                             valid_concepts, boundary_dist, fractions):
    n_test, n_valid = boundary_dist.shape
    rank = boundary_dist.argsort(dim=1).argsort(dim=1)  # rank[i, j] = 0 -> most uncertain for image i

    means = []
    for f in fractions:
        k = round(f * n_valid)
        selected = rank < k  # [n_test, n_valid] bool: this image's k most-uncertain concepts
        acts = test_acts.clone()
        for j, c in enumerate(valid_concepts):
            sel_c = selected[:, j]
            if not sel_c.any():
                continue
            truth_c = test_truth[c]
            pos_mask = sel_c & truth_c
            neg_mask = sel_c & ~truth_c
            for raw_idx in group_indices[c]:
                acts[pos_mask, raw_idx] = pos_val[raw_idx]
                acts[neg_mask, raw_idx] = neg_val[raw_idx]
        means.append(evaluate(probe, acts, test_labels))
    return means


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
        text_emb = load_vocab_embeddings()
        concept_names = name_concepts(autoencoder, text_emb, load_vocab(), csv_path=concept_names_path)

    train_acts = torch.load(os.path.join(ACT_SAVE, 'train_sae_acts.pt'), map_location='cpu')
    test_acts  = torch.load(os.path.join(ACT_SAVE, 'test_sae_acts.pt'),  map_location='cpu')
    train_attrs = load_cub_image_attributes(train=True)
    test_attrs  = load_cub_image_attributes(train=False)
    _, train_labels = load_clip_features(ACT_SAVE, 'train')
    _, test_labels  = load_clip_features(ACT_SAVE, 'test')

    merged_names, group_indices, raw_idx_to_group_idx, align_threshold = filter_and_merge_concepts(
        clip_model, autoencoder, concept_names, percentile=args.align_percentile
    )
    print(f'{len(merged_names)} merged concepts kept from {len(concept_names)} raw concepts')

    attribute_terms = load_cub_attribute_terms()
    attr_ids_of = {}
    for attr_id, _, term in attribute_terms:
        attr_ids_of.setdefault(term, []).append(attr_id)

    matches, match_threshold = match_concepts_to_attributes(
        clip_model, merged_names, attribute_terms, percentile=args.match_percentile
    )
    concept_idx_to_attr_ids = {
        concept_idx: attr_ids_of[term]
        for term, concept_matches in matches.items()
        for concept_idx, _, _ in concept_matches
    }
    print(f'{len(concept_idx_to_attr_ids)}/{len(merged_names)} merged concepts matched a CUB attribute above threshold={match_threshold:.4f}')

    train_truth = _ground_truth(train_attrs, concept_idx_to_attr_ids)
    test_truth  = _ground_truth(test_attrs,  concept_idx_to_attr_ids)
    pos_val, neg_val, ref_concepts = positive_negative_refs(train_acts, train_truth, group_indices)

    # Per-concept detection threshold on the merged (max-over-group) train activation, used only to
    # rank (image, concept) pairs by the detector's own confidence for the uncertainty-order curve
    # below (tune_concept_thresholds drops the same degenerate all-pos/all-neg concepts as
    # positive_negative_refs, on its own internal val split - intersect so both curves share one
    # concept set and are directly comparable).
    train_merged_acts = merge_activations(train_acts, group_indices)
    thresholds = tune_concept_thresholds(
        train_merged_acts, train_attrs, concept_idx_to_attr_ids, train_labels, val_frac=args.val_frac, seed=args.seed
    )
    valid_concepts = sorted(set(ref_concepts) & set(thresholds.keys()))
    print(f'{len(valid_concepts)}/{len(concept_idx_to_attr_ids)} matched concepts have both a positive/negative '
          f'training example and a tuned threshold, and are usable for intervention')
    if not valid_concepts:
        raise RuntimeError('No matched concepts are usable for intervention; cannot build an intervention curve')

    boundary_dist = boundary_distance(test_acts, group_indices, thresholds, valid_concepts)

    results = {}
    for label, run_arg, prefix, probe_cls in [
        ('baseline', args.baseline_run, 'baseline_', LinearProbe),
        ('gated',    args.gated_run,    'gated_',    GatedProbe),
    ]:
        run = run_arg or _latest_run(MODEL_DIR, prefix)
        by_seed = _probe_paths_by_seed(os.path.join(MODEL_DIR, run))

        random_seed_curves, uncertainty_seed_curves = [], []
        for seed, path in sorted(by_seed.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
            probe = probe_cls(config.N_LEARNED_FEATURES, n_classes)
            probe.load_state_dict(torch.load(path, map_location='cpu'))
            probe.eval()
            seed_offset = seed if isinstance(seed, int) else 0

            random_means, _ = random_order_curve(
                probe, test_acts, test_labels, group_indices, test_truth, pos_val, neg_val,
                valid_concepts, args.fractions, args.n_orderings, args.seed + seed_offset,
            )
            uncertainty_means = uncertainty_order_curve(
                probe, test_acts, test_labels, group_indices, test_truth, pos_val, neg_val,
                valid_concepts, boundary_dist, args.fractions,
            )
            random_seed_curves.append(random_means)
            uncertainty_seed_curves.append(uncertainty_means)
            print(f'[{label} seed {seed}] random: 0%={random_means[0]:.4f} 100%={random_means[-1]:.4f}  '
                  f'uncertainty: 0%={uncertainty_means[0]:.4f} 100%={uncertainty_means[-1]:.4f}')

        results[label] = {'run': run, 'fractions': args.fractions}
        for order_label, seed_curves in [('random', random_seed_curves), ('uncertainty', uncertainty_seed_curves)]:
            seed_curves = torch.tensor(seed_curves)
            mean_curve = seed_curves.mean(0).tolist()
            std_curve  = seed_curves.std(0).tolist() if seed_curves.shape[0] > 1 else [0.0] * len(args.fractions)
            results[label][order_label] = {
                'n_seeds': seed_curves.shape[0], 'mean_accuracy': mean_curve, 'std_accuracy': std_curve,
            }
            for f, m, s in zip(args.fractions, mean_curve, std_curve):
                print(f'{label} ({order_label}) {f * 100:5.1f}% intervened: acc={m:.4f}+/-{s:.4f}')

    results['n_matched_concepts'] = len(concept_idx_to_attr_ids)
    results['n_valid_concepts'] = len(valid_concepts)
    results['align_percentile'] = args.align_percentile
    results['match_percentile'] = args.match_percentile
    results['n_orderings'] = args.n_orderings

    results_path = os.path.join(SAVE, f"concept_intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {results_path}')

    BASELINE_COLOR, GATED_COLOR = '#4C72B0', '#C44E52'
    curves = {
        'Baseline (random order)':      (results['baseline']['random']['mean_accuracy'],      results['baseline']['random']['std_accuracy'],      BASELINE_COLOR, '-'),
        'Baseline (uncertainty order)': (results['baseline']['uncertainty']['mean_accuracy'], results['baseline']['uncertainty']['std_accuracy'], BASELINE_COLOR, '--'),
        'Gated (random order)':         (results['gated']['random']['mean_accuracy'],          results['gated']['random']['std_accuracy'],          GATED_COLOR,    '-'),
        'Gated (uncertainty order)':    (results['gated']['uncertainty']['mean_accuracy'],     results['gated']['uncertainty']['std_accuracy'],    GATED_COLOR,    '--'),
    }
    plot_path = plot_concept_intervention(args.fractions, curves, 'cub')
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main(parse_args())
