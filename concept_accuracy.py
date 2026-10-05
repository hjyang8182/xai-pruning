import argparse
import glob
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from src import config
from src.concepts import embed_texts, load_concept_names, load_vocab, save_vocab_embeddings, load_vocab_embeddings, name_concepts
from src.data import load_cub, load_clip_features
from src.models import load_clip, load_autoencoder, LinearProbe, GatedProbe
from datetime import datetime

N_CUB_ATTRIBUTES = 312

# format per line: "<attribute_id> <category>::<descriptive_term> 
def load_cub_attribute_terms(data_root='./dataset'):
    path = os.path.join(os.path.expanduser(data_root), 'attributes.txt')
    terms = []
    with open(path) as f:
        for line in f:
            attr_id, full_name = line.strip().split(' ', 1)
            category, term = full_name.split('::', 1)
            terms.append((int(attr_id), category, term.replace('_', ' ')))
    return terms

# Null distribution of *best-match* similarity when words unrelated to any CUB attribute (sampled
# from the general CLIPDissect vocab) are matched against the attribute term bank. Taking a max over
# many candidate terms inflates similarity somewhat just by chance (more terms -> higher expected max),
# so a concept's real match should clearly beat what an arbitrary unrelated word achieves against the
# same term bank - this is a different null model than pick_alignment_threshold's shuffled 1:1 pairing,
# since concept<->attribute matching is many-to-many argmax, not a paired comparison.
def pick_match_threshold(clip_model, attr_emb, percentile=95, n_samples=500, seed=0):
    generator = torch.Generator().manual_seed(seed)
    vocab = load_vocab()
    sample_words = [vocab[i] for i in torch.randperm(len(vocab), generator=generator)[:n_samples].tolist()]
    sample_emb = F.normalize(embed_texts(clip_model, sample_words), dim=1)
    null_best = (sample_emb @ attr_emb.T).max(dim=1).values
    return torch.quantile(null_best, percentile / 100).item()

# Match each DNCBM concept (by index, since concept names can repeat across 8192 concepts drawn from
# a much smaller vocab) to its most similar CUB attribute term (by CLIP text-embedding cosine
# similarity), dropping matches below `threshold` and grouping concepts that share an attribute.
# If `threshold` is None, it's auto-picked via pick_match_threshold. Returns the matches and the
# threshold used.
def match_concepts_to_attributes(clip_model, concept_names, attribute_terms, percentile=95):
    terms = [term for _, _, term in attribute_terms]
    concept_emb = F.normalize(embed_texts(clip_model, concept_names), dim=1)
    attr_emb    = F.normalize(embed_texts(clip_model, terms), dim=1)
    sims = concept_emb @ attr_emb.T
    best_vals, best_idx = sims.max(dim=1)

    threshold = pick_match_threshold(clip_model, attr_emb, percentile=percentile)
    print(f'Auto-picked match threshold = {threshold:.4f} (null-distribution {percentile}th percentile)')

    matches = {}
    for concept_idx, (concept_name, idx, val) in enumerate(zip(concept_names, best_idx.tolist(), best_vals.tolist())):
        if val < threshold:
            continue
        matches.setdefault(terms[idx], []).append((concept_idx, concept_name, val))

    for term in matches:
        matches[term].sort(key=lambda triple: -triple[2])
    return matches, threshold

# Per-image binary CUB attribute matrix [n_images, 312], in the same image order CUBDataset builds
# for that split (images.txt order, filtered by train_test_split.txt) so it lines up with
# train_sae_acts.pt / test_sae_acts.pt row-for-row.
def load_cub_image_attributes(data_root='./dataset', train=True):
    base = os.path.join(os.path.expanduser(data_root), 'CUB_200_2011')

    img_id_to_split = {}
    with open(os.path.join(base, 'train_test_split.txt')) as f:
        for line in f:
            img_id, is_train = line.strip().split(' ')
            img_id_to_split[img_id] = int(is_train)

    want_split = 1 if train else 0
    split_img_ids = []
    with open(os.path.join(base, 'images.txt')) as f:
        for line in f:
            img_id = line.strip().split(' ', 1)[0]
            if img_id_to_split[img_id] == want_split:
                split_img_ids.append(img_id)
    row_of = {img_id: i for i, img_id in enumerate(split_img_ids)}

    attrs = torch.zeros(len(split_img_ids), N_CUB_ATTRIBUTES)
    with open(os.path.join(base, 'attributes', 'image_attribute_labels.txt')) as f:
        for line in f:
            img_id, attr_id, is_present = line.split()[:3]
            row = row_of.get(img_id)
            if row is not None:
                attrs[row, int(attr_id) - 1] = int(is_present)
    return attrs

# Per concept, pick the activation threshold (searched over a held-out slice of `acts`) that best
# separates images with vs. without the matched CUB attribute(s), using balanced accuracy so rare
# attributes don't just get thresholded away to "never present". `concept_idx_to_attr_ids` maps each
# concept to a *list* of attribute ids (several CUB attributes can share the same descriptive term,
# e.g. has_wing_color::blue and has_belly_color::blue both reduce to "blue") - the ground truth is
# their OR: "does this image have that descriptor anywhere". `labels` (species, row-aligned with
# `acts`/`attrs`) stratifies the val split so it's not dominated by a handful of species - CUB
# attributes are near-deterministic per species, so a plain random split can leave rare
# attributes' species entirely out of val and trip the pos==0/neg==0 skip below.
def tune_concept_thresholds(acts, attrs, concept_idx_to_attr_ids, labels, val_frac=0.2, seed=0):
    n = acts.shape[0]
    _, val_idx = train_test_split(range(n), test_size=val_frac, random_state=seed, stratify=labels)
    val_idx = torch.tensor(val_idx)

    thresholds = {}
    for concept_idx, attr_ids in concept_idx_to_attr_ids.items():
        concept_acts = acts[val_idx, concept_idx]
        labels = attrs[val_idx][:, [i - 1 for i in attr_ids]].any(dim=1).float()
        pos, neg = labels.sum().item(), (labels == 0).sum().item()
        if pos == 0 or neg == 0:
            continue

        best_thresh, best_score = concept_acts.min().item() - 1e-6, -1.0
        for t in concept_acts.unique().tolist():
            pred = concept_acts > t
            tpr = (pred & (labels == 1)).sum().item() / pos
            tnr = (~pred & (labels == 0)).sum().item() / neg
            score = 0.5 * (tpr + tnr)
            if score > best_score:
                best_score, best_thresh = score, t
        thresholds[concept_idx] = best_thresh
    return thresholds

# Apply per-concept thresholds to any split's activations to get a binary predicted concept set
def predict_concept_set(acts, thresholds):
    return {concept_idx: acts[:, concept_idx] > t for concept_idx, t in thresholds.items()}

# Null distribution of alignment scores from *mismatched* (concept, name) pairs - shuffle which
# name embedding goes with which dictionary vector and score those wrong pairings. A real concept
# should be more aligned with its actual name than a randomly mismatched one would be by chance, so
# the `percentile`-th value of this null distribution makes a data-driven threshold instead of a
# guessed constant, calibrated to this SAE's own dictionary/vocab rather than assumed CLIP norms.
def pick_alignment_threshold(dic_vec, name_emb, percentile=95, n_shuffles=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    null_scores = []
    for _ in range(n_shuffles):
        perm = torch.randperm(name_emb.shape[0], generator=generator)
        null_scores.append((dic_vec.T * name_emb[perm]).sum(dim=1))
    return torch.quantile(torch.cat(null_scores), percentile / 100).item()

# Filter out concepts whose dictionary vector isn't well-aligned with the text embedding of its
# assigned name (cosine similarity below `threshold`), then merge concepts that share the same
# assigned name into one group. If `threshold` is None, it's auto-picked via pick_alignment_threshold.
# Returns the surviving merged names, each group's raw SAE dictionary indices, a raw-index ->
# merged-group-index lookup (only covers indices that survived filtering), and the threshold used.
def filter_and_merge_concepts(clip_model, autoencoder, concept_names, percentile=95):
    dic_vec = F.normalize(autoencoder.decoder.weight.detach().cpu().squeeze(), dim=0)
    name_emb = F.normalize(embed_texts(clip_model, concept_names), dim=1)
    alignment = (dic_vec.T * name_emb).sum(dim=1)

    threshold = pick_alignment_threshold(dic_vec, name_emb, percentile=percentile)
    print(f'Auto-picked alignment threshold = {threshold:.4f} (null-distribution {percentile}th percentile)')

    groups = {}
    for concept_idx, (name, align) in enumerate(zip(concept_names, alignment.tolist())):
        if align < threshold:
            continue
        groups.setdefault(name, []).append(concept_idx)

    merged_names = list(groups.keys())
    group_indices = list(groups.values())
    raw_idx_to_group_idx = {c: m for m, indices in enumerate(group_indices) for c in indices}
    return merged_names, group_indices, raw_idx_to_group_idx, threshold

# Combine each merged group's raw SAE activation columns into one column via max (the continuous
# analogue of "did any concept in this group fire")
def merge_activations(acts, group_indices):
    return torch.stack([acts[:, indices].max(dim=1).values for indices in group_indices], dim=1)

# Map raw SAE concept indices (e.g. a probe's gated-open or top-weight concepts) to merged-group
# indices, dropping any that didn't survive the alignment filter
def to_group_indices(raw_indices, raw_idx_to_group_idx):
    return sorted({raw_idx_to_group_idx[c] for c in raw_indices if c in raw_idx_to_group_idx})

def _latest_run(model_dir, prefix):
    runs = sorted(d for d in os.listdir(model_dir) if d.startswith(prefix))
    if not runs:
        raise FileNotFoundError(f"No '{prefix}*' runs found in {model_dir}")
    return runs[-1]

# A run dir holds one probe_seed<seed>.pt per seed trained by train_cbm.py; older runs may only
# have a single probe.pt. Sorted so pairing baseline/gated paths by index lines up matching seeds
# when both runs were trained with the same --seeds list.
# The mask-and-refit deliverable of newer gated runs (probe_seed<seed>_refit.pt: {weight, bias,
# keep, tau}) is a plain linear head that is exactly zero outside `keep`. Wrap it in a GatedProbe
# with saturated gate logits (+30 / -30 -> sigmoid 1 / 0) so every downstream consumer
# (gate_logits > threshold for the open set, effective_weight, forward -> (logits, gates)) works
# unchanged and reproduces the refit head's predictions.
def _load_gated_probe(path, n_classes, refit=False):
    probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
    if refit:
        refit_path = path[:-3] + '_refit.pt'
        d = torch.load(refit_path, map_location='cpu')
        with torch.no_grad():
            probe.linear.weight.copy_(d['weight'])
            probe.linear.bias.copy_(d['bias'])
            probe.gate_logits.copy_(torch.where(d['keep'].bool(), torch.tensor(30.0), torch.tensor(-30.0)))
    else:
        probe.load_state_dict(torch.load(path, map_location='cpu'))
    probe.eval()
    return probe


def _probe_paths(run_dir):
    # Newer gated runs also save probe_seed<seed>_refit.pt (mask-and-refit linear heads);
    # those are not GatedProbe state dicts, so keep only the plain per-seed checkpoints.
    paths = sorted(p for p in glob.glob(os.path.join(run_dir, 'probe_seed*.pt')) if not p.endswith('_refit.pt'))
    if not paths:
        single = os.path.join(run_dir, 'probe.pt')
        paths = [single] if os.path.exists(single) else []
    if not paths:
        raise FileNotFoundError(f'No probe_seed*.pt or probe.pt found in {run_dir}')
    return paths

def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    t = torch.tensor(values)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)

# Stack predicted-presence and ground-truth-presence into two [n_test, n_valid_concepts] bool
# matrices for the given concept set, dropping concepts with no tuned threshold (never matched)
def _prediction_matrices(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs):
    # number of valid attributes the concept mapped to
    valid = [c for c in concept_indices if c in predicted]
    if not valid:
        return valid, None, None
    pred_matrix = torch.stack([predicted[c] for c in valid], dim=1)
    truth_matrix = torch.stack(
        [test_attrs[:, [i - 1 for i in concept_idx_to_attr_ids[c]]].any(dim=1) for c in valid],
        dim=1,
    )
    return valid, pred_matrix, truth_matrix

# Per-image Jaccard index between the predicted concept set and the ground-truth attribute set: for
# each image, how well the concepts that fired agree with the CUB attributes actually true of that
# bird. Unlike accuracy, agreeing on absence contributes nothing except in the fully-empty case, so
# it isn't inflated by how sparse the attributes are. Images where neither set has anything (nothing
# fired, no matched attribute present) count as perfect agreement (Jaccard = 1) rather than undefined.
def per_image_jaccard(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs):
    valid, pred_matrix, truth_matrix = _prediction_matrices(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs)
    if not valid:
        return None, 0
    intersection = (pred_matrix & truth_matrix).sum(dim=1).float()
    union = (pred_matrix | truth_matrix).sum(dim=1).float()
    jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return jaccard, len(valid)

# Paired image-level comparison of two per-image Jaccard vectors over the same test images. The
# across-seed std is 0 here (the gate's open set and the baseline's top-k set are seed-invariant),
# so the only real variance is across images; this tests whether the gated-minus-baseline
# difference is distinguishable from zero at that level. Reports a paired t-test, a Wilcoxon
# signed-rank test (ties/zeros dropped, as scipy does by default), a bootstrap CI on the mean
# difference, and Cohen's d_z for effect size.
def paired_jaccard_test(baseline_per_image, gated_per_image, n_boot=10000, seed=0):
    from scipy import stats
    b = baseline_per_image.double().numpy()
    g = gated_per_image.double().numpy()
    d = g - b
    t_stat, t_p = stats.ttest_rel(g, b)
    nz = d[d != 0]
    if len(nz) > 0:
        w_stat, w_p = stats.wilcoxon(g, b)
    else:
        w_stat, w_p = float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        'n_images': int(len(d)),
        'mean_baseline': float(b.mean()), 'mean_gated': float(g.mean()),
        'mean_diff': float(d.mean()), 'std_diff': float(d.std(ddof=1)),
        'bootstrap_ci95_mean_diff': [float(lo), float(hi)],
        'cohens_dz': float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan'),
        'frac_gated_better': float((d > 0).mean()), 'frac_baseline_better': float((d < 0).mean()),
        'frac_tied': float((d == 0).mean()),
        'paired_t': {'t': float(t_stat), 'p': float(t_p)},
        'wilcoxon': {'W': float(w_stat), 'p': float(w_p), 'n_nonzero_pairs': int(len(nz))},
    }

# Mean Jaccard index over the test set (see per_image_jaccard)
def concept_set_jaccard(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs):
    jaccard, n_valid = per_image_jaccard(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs)
    if jaccard is None:
        return None, 0
    return jaccard.mean().item(), jaccard.std().item(), n_valid

# Per-concept accuracy: for each matched concept, the fraction of test images where predicted
# presence matches ground-truth attribute presence, averaged across concepts (each weighted equally)
def concept_mean_accuracy(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs):
    valid, pred_matrix, truth_matrix = _prediction_matrices(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs)
    if not valid:
        return None, 0
    return (pred_matrix == truth_matrix).float().mean().item(), len(valid)

# The probe's own predicted class for this image, independent of which concepts get shown
# alongside it (that's now driven by the Jaccard-scored concept set, not this probe's per-image
# contributions - see jaccard_relevant_concepts).
def _predicted_class(probe, acts_row, gated):
    with torch.no_grad():
        output = probe(acts_row.unsqueeze(0))
        logits = output[0] if gated else output
        return logits.argmax(dim=1).item()

# The concepts from `concept_indices` (the fixed, image-independent set a probe is scored over -
# baseline_groups/gated_groups) that actually move this image's per_image_jaccard: matched to an
# attribute (present in `predicted`) and with predicted-presence or ground-truth-presence true.
# Concepts predicted-absent AND truth-absent are excluded since they contribute to neither the
# intersection nor the union, so they don't affect this image's score at all. This is exactly the
# concept set `_prediction_matrices` reduces to, so it directly explains the plotted Jaccard number.
def jaccard_relevant_concepts(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs, merged_names, image_idx):
    items = []
    for c in concept_indices:
        if c not in predicted:
            continue
        pred_present = predicted[c][image_idx].item()
        truth_present = test_attrs[image_idx, [i - 1 for i in concept_idx_to_attr_ids[c]]].any().item()
        if not pred_present and not truth_present:
            continue
        items.append((c, merged_names[c], pred_present, truth_present))
    return items

# Render a probe's Jaccard-relevant concepts as colored text lines: green if predicted-present and
# actually present (intersection), red if predicted-present but actually absent (false positive),
# orange if predicted-absent but actually present (false negative, missed). Caps display at top_k,
# summarizing anything past that by match/mismatch count rather than dropping it silently. Starts at
# height `y` in axes-fraction coordinates; returns the y after the last line.
def _draw_jaccard_concepts_list(ax, y, header, pred_class_name, items, top_k):
    ax.text(0, y, f'{header} (pred: {pred_class_name})', fontsize=8, fontweight='bold', va='top', ha='left', transform=ax.transAxes)
    y -= 0.09
    if not items:
        ax.text(0.03, y, '(no concepts affected this image\'s Jaccard)', fontsize=7, va='top', ha='left', transform=ax.transAxes)
        return y - 0.07
    # Matches (green, intersection) first, mismatches (FP/FN) last; stable sort keeps each
    # group in its original order otherwise.
    items = sorted(items, key=lambda item: item[2] != item[3])
    shown, rest = items[:top_k], items[top_k:]
    for _, name, pred_present, truth_present in shown:
        if pred_present and truth_present:
            mark, color = '✓', 'green'
        elif pred_present:
            mark, color = '✗ FP', 'red'
        else:
            mark, color = '✗ FN', 'orange'
        ax.text(0.03, y, f'{mark} {name}', fontsize=7, va='top', ha='left', color=color, transform=ax.transAxes)
        y -= 0.07
    if rest:
        n_match = sum(1 for _, _, p, t in rest if p == t)
        ax.text(0.03, y, f'... and {len(rest)} more ({n_match} match, {len(rest) - n_match} mismatch)',
                fontsize=7, va='top', ha='left', color='gray', style='italic', transform=ax.transAxes)
        y -= 0.07
    return y

# Render and save one annotated image per index in `indices`: the raw image plus, per probe, the
# concepts from its scored set (baseline_groups/gated_groups) that actually affect this image's
# per_image_jaccard - i.e. exactly what the plotted Jaccard number is computed from, capped at
# top_k_concepts. Shared by plot_gated_wins (win-selected indices) and plot_random_examples
# (uniformly sampled indices) so both use identical rendering.
def _plot_examples(indices, baseline_per_image, gated_per_image, test_acts, merged_names,
                    baseline_probe, gated_probe, baseline_groups, gated_groups,
                    concept_idx_to_attr_ids, predicted, test_attrs,
                    save_dir, top_k_concepts=6, data_root='./dataset'):
    _, _, test_raw = load_cub(data_root=data_root)
    os.makedirs(save_dir, exist_ok=True)

    for rank, idx in enumerate(indices):
        image, label = test_raw[idx]
        class_name = test_raw.classes[label].replace(' ', '_')
        acts_row = test_acts[idx]
        baseline_pred = _predicted_class(baseline_probe, acts_row, gated=False)
        gated_pred = _predicted_class(gated_probe, acts_row, gated=True)
        baseline_items = jaccard_relevant_concepts(baseline_groups, concept_idx_to_attr_ids, predicted, test_attrs, merged_names, idx)
        gated_items = jaccard_relevant_concepts(gated_groups, concept_idx_to_attr_ids, predicted, test_attrs, merged_names, idx)

        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={'width_ratios': [1, 1.3]})
        ax_img.imshow(image)
        ax_img.set_title(
            f'{test_raw.classes[label]}\nbase jaccard={baseline_per_image[idx]:.2f}  gated jaccard={gated_per_image[idx]:.2f}',
            fontsize=9,
        )
        ax_img.axis('off')

        ax_text.axis('off')
        y = _draw_jaccard_concepts_list(
            ax_text, 1.0, 'Baseline used', test_raw.classes[baseline_pred], baseline_items, top_k_concepts,
        )
        _draw_jaccard_concepts_list(
            ax_text, y - 0.05, 'Gated used', test_raw.classes[gated_pred], gated_items, top_k_concepts,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{rank:02d}_{idx}_{class_name}.png'), bbox_inches='tight')
        plt.close()

    print(f'Saved {len(indices)} images to {save_dir}')

# Plot the top_k_images test images where the gated probe's per-image Jaccard index beat the
# baseline's by the largest margin
def plot_gated_wins(baseline_per_image, gated_per_image, test_acts, merged_names, baseline_probe, gated_probe,
                     baseline_groups, gated_groups, concept_idx_to_attr_ids, predicted, test_attrs,
                     dataset='cub', top_k_images=8, top_k_concepts=6, data_root='./dataset'):
    diff = gated_per_image - baseline_per_image
    win_idx = (diff > 0).nonzero().squeeze(1)
    if win_idx.numel() == 0:
        print('No images where the gated probe outperformed the baseline in Jaccard index')
        return

    top_idx = win_idx[diff[win_idx].topk(min(top_k_images, win_idx.numel())).indices].tolist()
    save_dir = os.path.join(config.DATA_PATH, dataset, 'plot', f'gated_wins_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    _plot_examples(
        top_idx, baseline_per_image, gated_per_image, test_acts, merged_names,
        baseline_probe, gated_probe, baseline_groups, gated_groups, concept_idx_to_attr_ids, predicted, test_attrs,
        save_dir, top_k_concepts, data_root,
    )

# Plot a uniformly random sample of n test images (not selected by any win condition), for context
# on what "typical" predictions look like alongside the cherry-picked gated-wins examples
def plot_random_examples(baseline_per_image, gated_per_image, test_acts, merged_names, baseline_probe, gated_probe,
                          baseline_groups, gated_groups, concept_idx_to_attr_ids, predicted, test_attrs,
                          dataset='cub', n=8, top_k_concepts=6, data_root='./dataset', seed=0):
    generator = torch.Generator().manual_seed(seed)
    n_test = baseline_per_image.shape[0]
    idx = torch.randperm(n_test, generator=generator)[:min(n, n_test)].tolist()
    save_dir = os.path.join(config.DATA_PATH, dataset, 'plot', f'random_examples_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    _plot_examples(
        idx, baseline_per_image, gated_per_image, test_acts, merged_names,
        baseline_probe, gated_probe, baseline_groups, gated_groups, concept_idx_to_attr_ids, predicted, test_attrs,
        save_dir, top_k_concepts, data_root,
    )

# Aggregate a probe's raw per-concept weight into merged-group weight via max magnitude across each
# group's raw members (the weight analogue of merge_activations) - a merged concept's relevance to a
# class is however strongly its single most-relevant raw dictionary vector pulls that class's logit.
def merge_weight_matrix(weight, group_indices):
    return torch.stack([weight[:, indices].abs().max(dim=1).values for indices in group_indices], dim=1)

# Per-class boolean mask over merged concepts: True where a concept's merged weight is >= the
# `percentile`-th value of that class's own weight row (i.e. in that class's own top (100-percentile)%
# by magnitude). Row-wise rather than a single global cutoff, since weight magnitudes aren't
# comparable across classes.
def nonnegligible_weight_mask(weight_merged, percentile):
    cutoff = torch.quantile(weight_merged, percentile / 100, dim=1, keepdim=True)
    return weight_merged >= cutoff

# A probe's effective per-(class, raw-concept) weight: for the gated probe this is the linear weight
# scaled by that concept's learned gate, matching what forward() actually multiplies the activation by
# (x * sigmoid(gate_logits)) before the linear layer - so a concept the gate has closed contributes
# ~0 regardless of its raw linear weight. The baseline probe has no gate, so this is just its weight.
def effective_weight(probe, gated):
    if gated:
        return probe.linear.weight * torch.sigmoid(probe.gate_logits).unsqueeze(0)
    return probe.linear.weight

# For each test image, the merged concepts (restricted to `matched`, since only concepts with a tuned
# threshold/matched attribute can ever contribute to a Jaccard score) a probe "used" for its own
# predicted class: active (already firing per `active_matrix`, from predict_concept_set) AND
# non-negligibly weighted toward that image's predicted class (top `weight_percentile`% of that
# class's own weight row - see nonnegligible_weight_mask). Returns the boolean
# [n_test, len(matched)] selection matrix and the probe's predicted class per image.
def per_image_selected(probe, acts, group_indices, matched, active_matrix, weight_percentile, gated):
    with torch.no_grad():
        output = probe(acts)
        logits = output[0] if gated else output
    pred_class = logits.argmax(dim=1)

    weight_merged = merge_weight_matrix(effective_weight(probe, gated), group_indices)
    nonneg = nonnegligible_weight_mask(weight_merged, weight_percentile)  # [n_classes, n_merged]
    nonneg_matched = nonneg[:, matched]                                  # [n_classes, len(matched)]
    return active_matrix & nonneg_matched[pred_class], pred_class

# Null distribution of per-image Jaccard from choosing `k_i` concepts *uniformly at random* from
# `pool_matrix[i]` (baseline's active+non-negligible-weight candidates for that image/class) instead
# of whichever ones the gate actually selected, repeated `n_draws` times per image, all scored against
# the same fixed `truth_matrix` (every matched attribute, not just the drawn ones - so a true attribute
# the draw missed still counts against it, exactly like per_image_jaccard elsewhere in this file). This
# answers: at the same set size the gate actually used, how much of its Jaccard score is just from
# *how many* concepts it fires, versus firing the *right* ones? Images are skipped where the draw isn't
# well-defined: nothing selected (k_i == 0) or the baseline pool is smaller than k_i (can't sample that
# many without replacement).
def chance_jaccard_null(selected_matrix, pool_matrix, truth_matrix, n_draws=200, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n_test, n_matched = truth_matrix.shape
    k = selected_matrix.sum(dim=1)
    pool_size = pool_matrix.sum(dim=1)
    eligible = (k > 0) & (pool_size >= k)

    real_inter = (selected_matrix & truth_matrix).sum(dim=1).float()
    real_union = (selected_matrix | truth_matrix).sum(dim=1).float()
    real_jaccard = torch.where(real_union > 0, real_inter / real_union, torch.ones_like(real_union))

    null_mean = torch.full((n_test,), float('nan'))
    null_std = torch.full((n_test,), float('nan'))
    for i in eligible.nonzero().squeeze(1).tolist():
        pool_cols = pool_matrix[i].nonzero().squeeze(1)
        ki = k[i].item()
        keys = torch.rand(n_draws, pool_cols.numel(), generator=generator)
        draw_cols = pool_cols[keys.argsort(dim=1)[:, :ki]]  # [n_draws, ki], indices into n_matched
        pred = torch.zeros(n_draws, n_matched, dtype=torch.bool)
        pred.scatter_(1, draw_cols, True)
        truth_row = truth_matrix[i].unsqueeze(0).expand(n_draws, -1)
        inter = (pred & truth_row).sum(dim=1).float()
        union = (pred | truth_row).sum(dim=1).float()
        jacc = torch.where(union > 0, inter / union, torch.ones_like(union))
        null_mean[i], null_std[i] = jacc.mean(), jacc.std()

    return real_jaccard, null_mean, null_std, eligible

# Summarize the real-vs-chance comparison over eligible images: mean real (gate-selected) Jaccard,
# mean of the per-image chance means, and the fraction of images where the gate's real score beat its
# own per-image chance mean - a paired, per-image comparison rather than comparing the two overall
# means, since "chance" varies image-to-image with the size and quality of that image's baseline pool.
def summarize_chance_null(real_jaccard, null_mean, eligible):
    idx = eligible.nonzero().squeeze(1)
    if idx.numel() == 0:
        return None
    real, null = real_jaccard[idx], null_mean[idx]
    return {
        'n_eligible': idx.numel(),
        'mean_real_jaccard': real.mean().item(),
        'mean_chance_jaccard': null.mean().item(),
        'frac_images_beating_chance': (real > null).float().mean().item(),
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate semantic accuracy of CUB concepts against ground-truth attributes, for both the baseline and gated probes')
    parser.add_argument('--baseline-run', default=None, help='data/cub/model/<run> to evaluate; defaults to the latest baseline_* run')
    parser.add_argument('--gated-run', default=None, help='data/cub/model/<run> to evaluate; defaults to the latest gated_* run')
    parser.add_argument('--align-percentile', type=float, default=95, help='Percentile of the mismatched-pairing null distribution used as the concept dictionary-vector/name alignment threshold')
    parser.add_argument('--match-threshold', type=float, default=None, help='Min cosine similarity for a concept to count as matching a CUB attribute term; if unset, auto-picked from --match-percentile')
    parser.add_argument('--match-percentile', type=float, default=95, help='Percentile of the unrelated-word null distribution to use as --match-threshold when it is unset')
    parser.add_argument('--val-frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gate-threshold', type=float, default=0.5)
    parser.add_argument('--refit', action='store_true',
                        help='Score the gated run\'s mask-and-refit heads (probe_seed<N>_refit.pt) instead of the '
                             'gated probes: open set = the refit mask, weights = the refit linear head')
    parser.add_argument('--top-k-wins', type=int, default=8, help='Number of gated-outperforms-baseline example images to plot')
    parser.add_argument('--n-random', type=int, default=8, help='Number of uniformly random example images to plot, for context alongside the gated-wins examples')
    parser.add_argument('--top-k-concepts', type=int, default=6, help='Number of predicted concepts to show per probe on each plotted image')
    parser.add_argument('--weight-percentile', type=float, default=90, help='Per-class percentile cutoff on merged-group weight magnitude for a concept to count as non-negligibly weighted toward the predicted class (chance-null Jaccard test)')
    parser.add_argument('--n-null-draws', type=int, default=200, help='Random same-size draws from baseline\'s candidate pool per image, for the chance-null Jaccard test')
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
        text_emb = load_vocab_embeddings()
        concept_names = name_concepts(autoencoder, text_emb, load_vocab(), csv_path=concept_names_path)

    train_acts = torch.load(os.path.join(ACT_SAVE, 'train_sae_acts.pt'), map_location='cpu')
    test_acts  = torch.load(os.path.join(ACT_SAVE, 'test_sae_acts.pt'),  map_location='cpu')
    train_attrs = load_cub_image_attributes(train=True)
    test_attrs  = load_cub_image_attributes(train=False)
    _, train_labels = load_clip_features(ACT_SAVE, 'train')

    merged_names, group_indices, raw_idx_to_group_idx, align_threshold = filter_and_merge_concepts(
        clip_model, autoencoder, concept_names, percentile=args.align_percentile
    )
    print(f'{len(merged_names)} merged concepts kept from {len(concept_names)} raw concepts')
    train_merged_acts = merge_activations(train_acts, group_indices)
    test_merged_acts  = merge_activations(test_acts,  group_indices)

    attribute_terms = load_cub_attribute_terms()
    attr_ids_of = {}
    for attr_id, _, term in attribute_terms:
        # Find the attribute ids matching to an attribute term
        # ex. attr_ids_of["blue"] should have ids of attributes "has_wing_color::blue, has_belly_color::blue"
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

    thresholds = tune_concept_thresholds(train_merged_acts, train_attrs, concept_idx_to_attr_ids, train_labels, val_frac=args.val_frac, seed=args.seed)
    predicted  = predict_concept_set(test_merged_acts, thresholds)

    # Fixed truth universe for the chance-null Jaccard test: every matched concept, same as elsewhere
    # in this file, so real-vs-chance selections are scored against the same ground-truth space.
    matched = sorted(predicted.keys())
    _, active_matrix, truth_matrix_all = _prediction_matrices(matched, concept_idx_to_attr_ids, predicted, test_attrs)

    baseline_run = args.baseline_run or _latest_run(MODEL_DIR, 'baseline_')
    gated_run    = args.gated_run or _latest_run(MODEL_DIR, 'gated_')
    baseline_paths = _probe_paths(os.path.join(MODEL_DIR, baseline_run))
    gated_paths    = _probe_paths(os.path.join(MODEL_DIR, gated_run))

    n_seeds = min(len(baseline_paths), len(gated_paths))
    if len(baseline_paths) != len(gated_paths):
        print(f'Warning: {baseline_run} has {len(baseline_paths)} probe(s) but {gated_run} has '
              f'{len(gated_paths)}; pairing the first {n_seeds} of each by index')
    baseline_paths, gated_paths = baseline_paths[:n_seeds], gated_paths[:n_seeds]

    print(f'Evaluating concept accuracy of baseline: {baseline_run} and gated: {gated_run} over {n_seeds} seed(s)')

    # baseline_probe/gated_probe/baseline_groups/gated_groups are kept from the first seed pair
    # only, to drive the example plots below (one probe pair per plot, not one set per seed).
    baseline_probe = gated_probe = baseline_groups = gated_groups = None
    per_seed = {'baseline': [], 'gated': [], 'chance_null': []}

    for seed_i, (b_path, g_path) in enumerate(zip(baseline_paths, gated_paths)):
        b_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        b_probe.load_state_dict(torch.load(b_path, map_location='cpu'))
        b_probe.eval()

        g_probe = _load_gated_probe(g_path, n_classes, refit=args.refit)
        open_concepts = (torch.sigmoid(g_probe.gate_logits) > args.gate_threshold).nonzero().squeeze(1).tolist()

        # Compare both probes over equally-sized concept sets: baseline has no gating mechanism, so
        # its "used" concepts are taken as the top-|open_concepts| by learned weight magnitude
        weight_norm = b_probe.linear.weight.norm(dim=0)
        baseline_concepts = weight_norm.topk(len(open_concepts)).indices.tolist()

        # Both concept sets are raw SAE dictionary indices (probes were trained on all 8192 dims);
        # translate into merged-group indices before evaluating against concept_idx_to_attr_ids
        b_groups = to_group_indices(baseline_concepts, raw_idx_to_group_idx)
        g_groups = to_group_indices(open_concepts, raw_idx_to_group_idx)

        for label, concept_indices, path in [('baseline', b_groups, b_path), ('gated', g_groups, g_path)]:
            jaccard, jaccard_std, n_matched = concept_set_jaccard(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs)
            mean_acc, _ = concept_mean_accuracy(concept_indices, concept_idx_to_attr_ids, predicted, test_attrs)
            if jaccard is None:
                print(f'[seed {seed_i}] {label} ({os.path.basename(path)}): none of its {len(concept_indices)} concepts had a matched attribute')
            else:
                print(f'[seed {seed_i}] {label} ({os.path.basename(path)}): {len(concept_indices)} concepts, {n_matched} matched, '
                      f'mean concept accuracy = {mean_acc:.4f}, mean Jaccard index = {jaccard:.4f}')
            per_seed[label].append({
                'probe_path': path, 'n_concepts': len(concept_indices), 'n_matched': n_matched,
                'mean_accuracy': mean_acc, 'mean_jaccard': jaccard, 'std_jaccard': jaccard_std,
            })

        if active_matrix is not None:
            gated_selected, _    = per_image_selected(g_probe, test_acts, group_indices, matched, active_matrix, args.weight_percentile, gated=True)
            baseline_pool, _     = per_image_selected(b_probe, test_acts, group_indices, matched, active_matrix, args.weight_percentile, gated=False)
            real_jaccard, null_mean, _, eligible = chance_jaccard_null(
                gated_selected, baseline_pool, truth_matrix_all, n_draws=args.n_null_draws, seed=args.seed + seed_i,
            )
            null_summary = summarize_chance_null(real_jaccard, null_mean, eligible)
            if null_summary is None:
                print(f'[seed {seed_i}] chance-null: no eligible images (nothing selected, or baseline pool always smaller than the gate\'s selection)')
            else:
                print(f'[seed {seed_i}] chance-null: {null_summary["n_eligible"]} eligible images, '
                      f'mean gate-selected jaccard = {null_summary["mean_real_jaccard"]:.4f}, '
                      f'mean chance jaccard = {null_summary["mean_chance_jaccard"]:.4f}, '
                      f'gate beats chance on {null_summary["frac_images_beating_chance"] * 100:.1f}% of images')
            per_seed['chance_null'].append(null_summary)

        if seed_i == 0:
            baseline_probe, gated_probe = b_probe, g_probe
            baseline_groups, gated_groups = b_groups, g_groups

    results = {}
    for label, run in [('baseline', baseline_run), ('gated', gated_run)]:
        acc_mean, acc_std = _mean_std([s['mean_accuracy'] for s in per_seed[label]])
        jac_mean, jac_std = _mean_std([s['mean_jaccard'] for s in per_seed[label]])
        if acc_mean is None:
            print(f'{label} ({run}): no seed had any matched concept')
        else:
            print(f'{label} ({run}) over {n_seeds} seed(s): mean concept accuracy = {acc_mean:.4f} +/- {acc_std:.4f}, '
                  f'mean Jaccard index = {jac_mean:.4f} +/- {jac_std:.4f}')
        results[label] = {
            'run': run, 'refit': (args.refit if label == 'gated' else False),
            'align_percentile': args.align_percentile, 'match_percentile': args.match_percentile,
            'n_seeds': n_seeds, 'per_seed': per_seed[label],
            'mean_accuracy': acc_mean, 'std_accuracy_across_seeds': acc_std,
            'mean_jaccard': jac_mean, 'std_jaccard_across_seeds': jac_std,
        }

    chance_stats = [s for s in per_seed['chance_null'] if s]
    if chance_stats:
        real_mean, real_std = _mean_std([s['mean_real_jaccard'] for s in chance_stats])
        chance_mean, chance_std = _mean_std([s['mean_chance_jaccard'] for s in chance_stats])
        beat_mean, beat_std = _mean_std([s['frac_images_beating_chance'] for s in chance_stats])
        print(f'Chance-null over {len(chance_stats)} seed(s): mean gate-selected jaccard = {real_mean:.4f} +/- {real_std:.4f}, '
              f'mean chance jaccard = {chance_mean:.4f} +/- {chance_std:.4f}, '
              f'gate beats chance on {beat_mean * 100:.1f}% +/- {beat_std * 100:.1f}% of images')
        results['chance_null'] = {
            'weight_percentile': args.weight_percentile, 'n_null_draws': args.n_null_draws,
            'n_seeds': len(chance_stats), 'per_seed': chance_stats,
            'mean_real_jaccard': real_mean, 'std_real_jaccard_across_seeds': real_std,
            'mean_chance_jaccard': chance_mean, 'std_chance_jaccard_across_seeds': chance_std,
            'mean_frac_beating_chance': beat_mean, 'std_frac_beating_chance_across_seeds': beat_std,
        }

    baseline_per_image, _ = per_image_jaccard(baseline_groups, concept_idx_to_attr_ids, predicted, test_attrs)
    gated_per_image, _    = per_image_jaccard(gated_groups, concept_idx_to_attr_ids, predicted, test_attrs)
    if baseline_per_image is not None and gated_per_image is not None:
        results['paired_image_test'] = paired_jaccard_test(baseline_per_image, gated_per_image, seed=args.seed)
        pt = results['paired_image_test']
        print(f"Paired image-level test (seed-0 probes, n={pt['n_images']}): gated - baseline Jaccard = "
              f"{pt['mean_diff']:+.4f} (95% CI {pt['bootstrap_ci95_mean_diff'][0]:+.4f}..{pt['bootstrap_ci95_mean_diff'][1]:+.4f}), "
              f"paired t p={pt['paired_t']['p']:.2e}, Wilcoxon p={pt['wilcoxon']['p']:.2e}, d_z={pt['cohens_dz']:.3f}, "
              f"gated better on {pt['frac_gated_better'] * 100:.1f}% / worse on {pt['frac_baseline_better'] * 100:.1f}% of images")

    with open(os.path.join(SAVE, f"concept_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w') as f:
        json.dump(results, f, indent=2)

    if baseline_per_image is not None and gated_per_image is not None:
        plot_gated_wins(
            baseline_per_image, gated_per_image, test_acts, merged_names, baseline_probe, gated_probe,
            baseline_groups, gated_groups, concept_idx_to_attr_ids, predicted, test_attrs,
            top_k_images=args.top_k_wins, top_k_concepts=args.top_k_concepts,
        )
        plot_random_examples(
            baseline_per_image, gated_per_image, test_acts, merged_names, baseline_probe, gated_probe,
            baseline_groups, gated_groups, concept_idx_to_attr_ids, predicted, test_attrs,
            n=args.n_random, top_k_concepts=args.top_k_concepts, seed=args.seed,
        )

if __name__ == '__main__':
    main(parse_args())
