import os
import re
from collections import Counter
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator

from src.config import DATA_PATH, N_LEARNED_FEATURES


def _save_plot(dataset_name, label, dpi=150, save_dir=None):
    save_dir = save_dir or os.path.join(DATA_PATH, dataset_name, 'plot')
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_dir, f'{label}_{timestamp}.svg')
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close()
    return path


def _add_operating_point(ax, operating_point, color='#4C72B0'):
    """Vertical line marking the concept count actually used by a deployed gated probe."""
    if operating_point is None:
        return
    ax.axvline(
        x=operating_point, color=color, linestyle=':', linewidth=LW_REF, zorder=4,
        label='Reported model',
    )
    # Count annotated on the line rather than in the legend label: top of the line, tucked to its
    # left. Called after the curves are plotted, so widening the y-range here keeps the text off
    # whatever the topmost curve happens to be. x in data coords, y in axes fraction.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.text(
        operating_point, 0.97, f'{operating_point:.0f} ', transform=ax.get_xaxis_transform(),
        ha='right', va='top', fontsize=15, color=color, zorder=4,
    )


def _save_text(dataset_name, label, lines):
    save_dir = os.path.join(DATA_PATH, dataset_name, 'plot')
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_dir, f'{label}_{timestamp}.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


def explain_image(probe, acts, concept_names, y_true, class_names, top_k=5):
    device = next(probe.parameters()).device
    acts = acts.to(device)
    output = probe(acts.unsqueeze(0))
    if isinstance(output, tuple):
        logits, gates = output
        # GatedProbe.forward feeds x * gates into the linear layer, so a concept the gate has
        # shut off shouldn't show up as a large contribution just because its raw activation and
        # weight are large.
        effective_acts = acts * gates
    else:
        logits = output
        effective_acts = acts
    pred = logits.argmax(dim=1).item()
    contributions = effective_acts * probe.linear.weight[pred]
    top_idx = contributions.abs().topk(top_k).indices
    top = [(concept_names[i], contributions[i].item()) for i in top_idx]
    other_sum = contributions.sum().item() - sum(v for _, v in top)
    return pred, top, other_sum


_POS_COLOR, _NEG_COLOR, _OTHER_COLOR = '#55A868', '#C44E52', '#B0B0B0'


def n_available_concepts(probe, n_total):
    """How many concepts a probe can actually draw on: every one for a linear probe, the open
    gates (> 0.5) for a gated probe -- what the 'Other concepts (N)' bar should count over."""
    if hasattr(probe, 'gate_probs'):
        return int((probe.gate_probs() > 0.5).sum().item())
    if getattr(probe, 'keep', None) is not None:  # mask-and-refit head (explain_concepts.py --refit)
        return int(probe.keep.sum().item())
    return n_total


# ---------------------------------------------------------------------------------------- #
#  Paper figure style. One place to keep every curve plot legible *without zooming*: a small
#  canvas (so nothing is scaled down much when placed in a column), thick lines and big
#  markers, per-panel axis labels, and the shared legend below the panels.
# ---------------------------------------------------------------------------------------- #
GATED_LABEL = 'DN-CBM w/ GCG'          # how the gated probe is named in legends / titles
SPARSE_LABEL = '(Original) L1-Sparse Probe'  # the unmodified DN-CBM head, swept over lambda_sparse
PANEL_W, PANEL_H = 4.6, 3.7            # inches per grid panel (single-panel figures use ~5.4 x 4.1)
LW, MS, MEW = 3.5, 9, 1.4              # line width, marker size, marker edge width
LW_REF = 2.6                           # reference lines (baseline accuracy, deployed probe)
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 19, 18, 15, 17
ACC_AXIS_LABEL = 'Task Accuracy'


def _legend_below(fig, handles, labels, ncol=None, fontsize=FS_LEGEND):
    """Shared legend under the panels (constrained layout reserves the room)."""
    fig.legend(handles, labels, fontsize=fontsize, loc='outside lower center',
               ncol=ncol or len(labels), frameon=False, handlelength=2.6, columnspacing=1.6,
               borderpad=0.6)


def _style_axes(ax, title=None):
    ax.set_facecolor('#F7F7F9')
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FS_TICK, width=1.6, length=5)
    ax.grid(True, which='major', linestyle='-', linewidth=0.9, color='white', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


# How probe keys ('Baseline', 'Gated') are labelled in figures; the keys themselves stay as-is
# because the explanation JSONs index by them.
PROBE_DISPLAY = {'Gated': GATED_LABEL}

# Figure typography (explanation panels)
_FS_TITLE, _FS_TICK, _FS_VALUE, _FS_XLABEL = 19, 19, 16, 19


def _bar(ax, expl, other_sum, title, xlim=None, n_concepts=None):
    """Horizontal contribution bars for one probe. `xlim`, when the caller shares it across
    probes, is what makes bars comparable between panels; the aggregate 'other concepts' bar is
    greyed because it is a sum over the rest of the dictionary, not a single concept."""
    other_label = 'Other' + (f' ({n_concepts - len(expl)})' if n_concepts else '')
    names  = [n for n, _ in expl] + [other_label]
    vals   = [v for _, v in expl] + [other_sum]
    colors = [(_POS_COLOR if v >= 0 else _NEG_COLOR) for _, v in expl] + [_OTHER_COLOR]
    # barh stacks upward, so reverse to put the largest contributor on top.
    names, vals, colors = names[::-1], vals[::-1], colors[::-1]

    _style_panel(ax, len(names))
    # Numeric positions rather than the names themselves: the SAE dictionary can name two
    # features identically, and a categorical axis would merge those onto one row.
    pos = list(range(len(names)))
    bars = ax.barh(pos, vals, color=colors, edgecolor='white', linewidth=0.8, height=0.68, zorder=3)
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.2f}', xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if val >= 0 else -5, 0), textcoords='offset points',
                    va='center', ha='left' if val >= 0 else 'right', fontsize=_FS_VALUE, color='#2B2B2B')
    if xlim:
        ax.set_xlim(*xlim)
    if min(vals) < 0:
        ax.axvline(0, color='#555555', linewidth=1.0, zorder=4)
    # Long "<model>: <class> ✓" titles overrun a 5-inch panel; break after the model name.
    if len(title) > 24 and ': ' in title:
        title = title.replace(': ', ':\n', 1)
    ax.set_title(title, fontsize=_FS_TITLE, fontweight='bold')
    ax.set_xlabel('Contribution', fontsize=_FS_XLABEL)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    ax.tick_params(axis='x', labelsize=_FS_TICK - 2)


def plot_comparison(img_idx, baseline_probe, pruned_probe,
                    full_acts, pruned_acts, keep_idx,
                    concept_names, class_names, top_k=5,
                    pruned_concept_names=None, dataset=None, gated_probe=None,
                    dataset_name=None):
    image, y_true = dataset[img_idx][0], dataset[img_idx][1]
    pred_b, expl_b, other_b = explain_image(baseline_probe, full_acts[img_idx], concept_names, y_true, class_names, top_k)
    pruned_names = pruned_concept_names if pruned_concept_names is not None else [concept_names[i] for i in keep_idx]
    pred_p, expl_p, other_p = explain_image(pruned_probe, pruned_acts[img_idx], pruned_names, y_true, class_names, top_k)

    n_panels = 4 if gated_probe is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels - 1, 4))
    axes[0].imshow(image)
    axes[0].set_title(f'True: {class_names[y_true]}')
    axes[0].axis('off')
    _bar(axes[1], expl_b, other_b, f'Baseline: {class_names[pred_b]}')
    _bar(axes[2], expl_p, other_p, f'Pruned: {class_names[pred_p]}')
    if gated_probe is not None:
        pred_g, expl_g, other_g = explain_image(gated_probe, full_acts[img_idx], concept_names, y_true, class_names, top_k)
        _bar(axes[3], expl_g, other_g, f'{GATED_LABEL}: {class_names[pred_g]}')
    plt.tight_layout()
    _save_plot(dataset_name, 'gated' if gated_probe is not None else 'baseline')


def plot_gated_comparison(img_idx, baseline_probe, gated_probe, full_acts, concept_names, class_names, top_k=5, dataset=None, dataset_name=None):
    image, y_true = dataset[img_idx][0], dataset[img_idx][1]
    pred_b, expl_b, other_b = explain_image(baseline_probe, full_acts[img_idx], concept_names, y_true, class_names, top_k)
    pred_g, expl_g, other_g = explain_image(gated_probe, full_acts[img_idx], concept_names, y_true, class_names, top_k)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image)
    axes[0].set_title(f'True: {class_names[y_true]}')
    axes[0].axis('off')
    _bar(axes[1], expl_b, other_b, f'Baseline: {class_names[pred_b]}')
    _bar(axes[2], expl_g, other_g, f'{GATED_LABEL}: {class_names[pred_g]}')
    plt.tight_layout()
    _save_plot(dataset_name, 'gated')


# Like plot_gated_comparison, but takes an arbitrary {label: probe} mapping instead of a fixed
# baseline/gated pair, so a single image can be explained against just one probe or several.
def plot_concept_explanation(img_idx, probes, acts, concept_names, class_names, dataset, dataset_name, top_k=5, explain_dir=None):
    image, y_true = dataset[img_idx][0], dataset[img_idx][1]
    explanations = {label: explain_image(probe, acts, concept_names, y_true, class_names, top_k)
                    for label, probe in probes.items()}

    # One x-scale for every probe panel. They sit side by side and invite length comparison, so
    # letting each autoscale (baseline to 1.75, gated to 0.9) makes unequal bars look equal.
    values = [v for _, expl, _ in explanations.values() for _, v in expl]
    values += [other for _, _, other in explanations.values()]
    hi, lo = max(values), min(0.0, min(values))
    pad = 0.20 * (hi - lo)  # room for the value labels
    xlim = (lo - (1.6 * pad if lo < 0 else 0), hi + pad)

    n_panels = 1 + len(probes)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels - 1, 4.2))
    fig.patch.set_facecolor('white')
    axes[0].imshow(image)
    axes[0].set_title(f'True: {_format_class_name(class_names[y_true])}', fontsize=12.5, fontweight='bold')
    axes[0].axis('off')
    for ax, (label, (pred, expl, other)) in zip(axes[1:], explanations.items()):
        # Whether each probe got it right is the first thing you want to know when reading the
        # concepts it used, so it goes in the panel title rather than being left to the reader.
        mark = '✓' if pred == y_true else '✗'
        _bar(ax, expl, other, f'{label}: {_format_class_name(class_names[pred])} {mark}',
             xlim=xlim, n_concepts=n_available_concepts(probes[label], len(concept_names)))
    plt.tight_layout()
    return _save_plot(dataset_name, 'concept_explanation', save_dir=explain_dir)


def plot_concept_explanation_grid(img_indices, probes, test_acts, concept_names, class_names, dataset,
                                  dataset_name, top_k=5, explain_dir=None, plot_label='concept_explanation_grid'):
    """Several plot_concept_explanation figures stacked into one: a row per image, with the image
    and one contribution panel per probe. Each row shares an x-scale across its probe panels (as
    the single-image figure does) but rows are scaled independently, since logits differ a lot
    between images and one global scale would flatten the small ones."""
    n_panels = 1 + len(probes)
    fig, axes = plt.subplots(len(img_indices), n_panels, figsize=(5 * n_panels - 1, 4.0 * len(img_indices)),
                             squeeze=False)
    fig.patch.set_facecolor('white')
    n_avail = {label: n_available_concepts(probe, len(concept_names)) for label, probe in probes.items()}

    for row, img_idx in zip(axes, img_indices):
        image, y_true = dataset[img_idx][0], dataset[img_idx][1]
        acts = test_acts[img_idx]
        explanations = {label: explain_image(probe, acts, concept_names, y_true, class_names, top_k)
                        for label, probe in probes.items()}
        values = [v for _, expl, _ in explanations.values() for _, v in expl]
        values += [other for _, _, other in explanations.values()]
        hi, lo = max(values), min(0.0, min(values))
        pad = 0.20 * (hi - lo)
        # More room on the negative side: those value labels sit to the left of the bar,
        # right where the concept names are.
        xlim = (lo - (1.6 * pad if lo < 0 else 0), hi + pad)

        row[0].imshow(image)
        row[0].set_title(f'True: {_format_class_name(class_names[y_true])}', fontsize=_FS_TITLE, fontweight='bold')
        row[0].axis('off')
        for ax, (label, (pred, expl, other)) in zip(row[1:], explanations.items()):
            mark = '\u2713' if pred == y_true else '\u2717'
            _bar(ax, expl, other, f'{PROBE_DISPLAY.get(label, label)}: {_format_class_name(class_names[pred])} {mark}',
                 xlim=xlim, n_concepts=n_avail[label])

    plt.tight_layout(h_pad=2.0)
    return _save_plot(dataset_name, plot_label, save_dir=explain_dir)


def plot_explanation_examples(examples, probe_labels=('Baseline', 'Gated'), plot_label='explanation_examples'):
    """Cross-dataset explanation figure: one row per dataset with the image and one contribution
    panel per probe, drawn from the JSON explain_concepts.py writes rather than live probes.

    examples: list of (dataset_display_name, entry) where entry is one element of that JSON's
    'images' list (image_path, true_class, and per-probe predicted_class / correct /
    top_concepts / sum_of_other_features / n_concepts_available). Each row shares an x-scale
    across its probe panels; rows scale independently, as in plot_concept_explanation_grid."""
    n_panels = 1 + len(probe_labels)
    fig, axes = plt.subplots(len(examples), n_panels, figsize=(5 * n_panels - 1, 4.0 * len(examples)),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    for row, (dataset_name, entry) in zip(axes, examples):
        per_probe = {label: entry[label.lower()] for label in probe_labels}
        values = [c['contribution'] for e in per_probe.values() for c in e['top_concepts']]
        values += [e['sum_of_other_features'] for e in per_probe.values()]
        hi, lo = max(values), min(0.0, min(values))
        pad = 0.20 * (hi - lo)
        # More room on the negative side: those value labels sit to the left of the bar,
        # right where the concept names are.
        xlim = (lo - (1.6 * pad if lo < 0 else 0), hi + pad)

        row[0].imshow(plt.imread(entry['image_path']))
        row[0].set_title(f'{dataset_name}\nTrue: {_format_class_name(entry["true_class"])}',
                         fontsize=_FS_TITLE, fontweight='bold')
        row[0].axis('off')
        for ax, (label, e) in zip(row[1:], per_probe.items()):
            mark = '\u2713' if e['correct'] else '\u2717'
            expl = [(c['concept'], c['contribution']) for c in e['top_concepts']]
            _bar(ax, expl, e['sum_of_other_features'],
                 f'{PROBE_DISPLAY.get(label, label)}: {_format_class_name(e["predicted_class"])} {mark}',
                 xlim=xlim, n_concepts=e.get('n_concepts_available'))

    plt.tight_layout(h_pad=2.0)
    return _save_plot(None, plot_label, save_dir=os.path.join(DATA_PATH, 'plot'))


def plot_relevant_concepts(gated_probe, concept_names, dataset_name):
    gates = torch.sigmoid(gated_probe.gate_logits).detach()
    n_open = (gates > 0.5).sum().item()
    print(f'Open gates: {n_open} / {len(gates)}')
    print(f'Mean: {gates.mean():.3f}  Min: {gates.min():.3f}  Max: {gates.max():.3f}')

    open_idx = (gates > 0.5).nonzero().squeeze().tolist()
    sorted_pairs = sorted(zip([concept_names[i] for i in open_idx], gates[open_idx].tolist()), key=lambda x: -x[1])
    for name, val in sorted_pairs[:20]:
        print(f'  {name:25s} gate={val:.3f}')

    txt_path = _save_text(dataset_name, 'open_concepts', [f'{name}\tgate={val:.3f}' for name, val in sorted_pairs])
    print(f'Saved {len(sorted_pairs)} open concepts to {txt_path}')

    high_gates = [(n, v) for n, v in sorted_pairs if v > 0.8][:10]
    low_gates  = sorted(zip(concept_names, gates.tolist()), key=lambda x: x[1])[:10]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh([p[0] for p in high_gates][::-1], [p[1] for p in high_gates][::-1], color='steelblue')
    axes[0].set_xlabel('Gate Value')
    axes[0].set_title('Highest Gate Values')
    axes[1].barh([p[0] for p in low_gates][::-1],  [p[1] for p in low_gates][::-1],  color='tomato')
    axes[1].set_xlabel('Gate Value')
    axes[1].set_title('Lowest Gate Values')
    plt.suptitle(f'Concept Gates — {dataset_name}')
    plt.tight_layout()
    _save_plot(dataset_name, 'gated')


def plot_gate_distribution(gated_probe, dataset_name):
    gates = torch.sigmoid(gated_probe.gate_logits).detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(gates, bins=50, range=(0.0, 1.0))
    plt.axvline(x=0.5, color='red', linestyle='--', label='threshold')
    plt.xlabel('Gate value')
    plt.ylabel('Count')
    plt.title('Distribution of gate values')
    plt.legend()
    _save_plot(dataset_name, 'gated')


_SEED_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']


def plot_training_curves(histories_by_seed, dataset_name, label='baseline'):
    """Per-epoch train loss and test accuracy (and, for a gated probe, open-gate count) across the
    seeds of a train_probe/train_gated_probe run (see train_cbm.py's track_history=True), to check
    convergence before trusting a run's final accuracy. `histories_by_seed` is {seed: history},
    history a dict of equal-length per-epoch lists with keys 'train_loss', 'test_acc', and
    optionally 'open_gates' (gated probes only)."""
    plt.rcParams['font.family'] = 'sans-serif'
    has_gates = any('open_gates' in h for h in histories_by_seed.values())
    n_panels = 3 if has_gates else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    for ax in axes:
        ax.set_facecolor('#F7F7F9')
    fig.patch.set_facecolor('white')

    for (seed, history), color in zip(histories_by_seed.items(), _SEED_COLORS * (len(histories_by_seed) // len(_SEED_COLORS) + 1)):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], color=color, linewidth=1.8, label=f'seed {seed}')
        axes[1].plot(epochs, history['test_acc'], color=color, linewidth=1.8, label=f'seed {seed}')
        if has_gates:
            axes[2].plot(epochs, history['open_gates'], color=color, linewidth=1.8, label=f'seed {seed}')

    axes[0].set_ylabel('Train loss', fontsize=12)
    axes[1].set_ylabel('Test accuracy', fontsize=12)
    if has_gates:
        axes[2].set_ylabel('Open gates', fontsize=12)
    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='white', zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].legend(fontsize=9, loc='best', frameon=False)

    fig.suptitle(f'Training curves ({label}) — {dataset_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save_plot(dataset_name, f'{label}_training_curves')


def _plot_single_lambda_gate_sweep(lambda_gates, accs, n_open, dataset_name, split_label,
                                    accs_std=None, n_open_std=None,
                                    baseline_acc=None, baseline_acc_std=None, metric_name=ACC_AXIS_LABEL,
                                    overlays=None):
    """overlays: optional list of (mean, std, color, label) accuracy curves aligned to
    `lambda_gates`, drawn on the left (accuracy) axis alongside the gated curve -- used to
    lay the concept_ablation.py random-retrain sweep over the gated_sweep_test plot."""
    ACC_COLOR, GATE_COLOR = '#4C72B0', '#DD8452'
    plt.rcParams['font.family'] = 'sans-serif'
    metric_label = metric_name if metric_name.isupper() else metric_name.lower()

    fig, ax1 = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    if accs_std:
        lo = [a - s for a, s in zip(accs, accs_std)]
        hi = [a + s for a, s in zip(accs, accs_std)]
        ax1.fill_between(lambda_gates, lo, hi, color=ACC_COLOR, alpha=0.18, linewidth=0, zorder=2)
    ax1.plot(
        lambda_gates, accs, color=ACC_COLOR, marker='o', markersize=MS,
        markeredgecolor='white', markeredgewidth=MEW, linewidth=LW,
        label=GATED_LABEL, zorder=4,
    )
    OVERLAY_COLORS = ['#8FB3E0', '#2B4A7A']  # accuracy-axis blues (colour = axis), see the grid version
    for j, (mean, std, _, label) in enumerate(overlays or []):
        color = OVERLAY_COLORS[j % len(OVERLAY_COLORS)]
        if std and any(std):
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax1.fill_between(lambda_gates, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax1.plot(
            lambda_gates, mean, color=color, marker='s', linestyle='-.', markersize=MS,
            markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=4,
        )
    if baseline_acc is not None:
        if baseline_acc_std:
            ax1.axhspan(baseline_acc - baseline_acc_std, baseline_acc + baseline_acc_std,
                        color='#555555', alpha=0.1, zorder=1)
        ax1.axhline(y=baseline_acc, color='#555555', linestyle='--', linewidth=LW_REF,
                    label='Baseline', zorder=5)
    ax1.set_xscale('log')
    _style_axes(ax1)
    ax1.set_xlabel(r'$\lambda_{gate}$', fontsize=FS_LABEL, fontweight='bold')
    ax1.set_ylabel(metric_name, color=ACC_COLOR, fontsize=FS_LABEL, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=ACC_COLOR)
    # Paper figures render small, so a handful of large tick labels reads better than many small
    # ones - thin both y-axes down to ~5 ticks rather than matplotlib's denser default.
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax2 = ax1.twinx()
    if n_open_std:
        lo = [max(0, o - s) for o, s in zip(n_open, n_open_std)]
        hi = [o + s for o, s in zip(n_open, n_open_std)]
        ax2.fill_between(lambda_gates, lo, hi, color=GATE_COLOR, alpha=0.15, linewidth=0, zorder=2)
    ax2.plot(
        lambda_gates, n_open, color=GATE_COLOR, marker='D', markersize=MS - 1,
        markeredgecolor='white', markeredgewidth=MEW, linewidth=LW - 0.5, linestyle='--',
        alpha=0.9, label='Concepts', zorder=3,
    )
    ax2.set_ylabel('Used Concepts', color=GATE_COLOR, fontsize=FS_LABEL, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=GATE_COLOR, labelsize=FS_TICK, width=1.6, length=5)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    _legend_below(fig, lines1 + lines2, labels1 + labels2, ncol=min(3, len(labels1) + len(labels2)))

    return _save_plot(dataset_name, f'gated_sweep_{split_label}')


def plot_lambda_gate_sweep(lambda_gates, accs, n_open, dataset_name, accs_std=None, n_open_std=None,
                            baseline_acc=None, baseline_acc_std=None,
                            test_accs=None, test_accs_std=None,
                            baseline_test_acc=None, baseline_test_acc_std=None,
                            val_metric_name=ACC_AXIS_LABEL, test_overlays=None):
    """test_overlays: optional list of (mean, std, color, label) accuracy curves aligned to
    `lambda_gates`, overlaid only on the test-split plot (e.g. the random-retrain sweep)."""
    _plot_single_lambda_gate_sweep(
        lambda_gates, accs, n_open, dataset_name, 'val',
        accs_std=accs_std, n_open_std=n_open_std,
        baseline_acc=baseline_acc, baseline_acc_std=baseline_acc_std,
        metric_name=val_metric_name,
    )
    if test_accs is not None:
        _plot_single_lambda_gate_sweep(
            lambda_gates, test_accs, n_open, dataset_name, 'test',
            accs_std=test_accs_std, n_open_std=n_open_std,
            baseline_acc=baseline_test_acc, baseline_acc_std=baseline_test_acc_std,
            overlays=test_overlays,
        )


def plot_lambda_gate_sweep_grid(results_by_dataset, dataset_order=None, ncols=None, sharey=True,
                                 metric_name=ACC_AXIS_LABEL, n_open_as_pct=False, plot_label='gated_sweep_test_grid'):
    """One figure with one gated_sweep_test panel per dataset (see _plot_single_lambda_gate_sweep
    for the single-dataset version): accuracy curves on the left axis, used-concept count on a
    twin right axis, one legend shared across panels since the curves mean the same thing
    everywhere. Styled like plot_concept_ablation_grid.

    results_by_dataset: {dataset_name: {'lambda_gates', 'accs', 'accs_std', 'n_open',
    'n_open_std', 'baseline_acc', 'baseline_acc_std', 'overlays' (optional list of
    (mean, std, color, label))}} - the same fields _plot_single_lambda_gate_sweep takes.
    `sharey` shares the accuracy axis only; the concept-count axis is per-dataset because
    dictionaries differ in size -- unless `n_open_as_pct`, in which case 'n_open' is already a
    percentage of the dictionary and the right axes get a common 0-100% scale too.
    """
    ax2_max = None
    ACC_COLOR, GATE_COLOR = '#4C72B0', '#DD8452'
    plt.rcParams['font.family'] = 'sans-serif'

    order = dataset_order or list(results_by_dataset.keys())
    ncols = ncols or (2 if len(order) > 1 else 1)
    nrows = -(-len(order) // ncols)
    # Colour = axis: everything read on the accuracy axis is a shade of its blue, the concept
    # count is the orange of its axis. Overlays (e.g. a random subset of the same size) stay in
    # the accuracy family -- a lighter blue, dash-dot, square markers -- so they are told apart
    # from the gated curve without looking like they belong to some third axis.
    OVERLAY_COLORS = ['#8FB3E0', '#2B4A7A']
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_W * ncols, PANEL_H * nrows + 0.9), squeeze=False,
                             sharey=sharey, layout='constrained')
    axes_flat = axes.flatten()

    legend_lines, legend_labels, twin_axes = [], [], []
    for i, (ax1, dataset_name) in enumerate(zip(axes_flat, order)):
        d = results_by_dataset[dataset_name]
        lambda_gates = d['lambda_gates']

        curves = [(d['accs'], d.get('accs_std'), ACC_COLOR, '-', 'o', GATED_LABEL)]
        curves += [(mean, std, OVERLAY_COLORS[j % len(OVERLAY_COLORS)], '-.', 's', label)
                   for j, (mean, std, _, label) in enumerate(d.get('overlays') or [])]
        for mean, std, color, linestyle, marker, label in curves:
            if std and any(std):
                lo = [m - s for m, s in zip(mean, std)]
                hi = [m + s for m, s in zip(mean, std)]
                ax1.fill_between(lambda_gates, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
            ax1.plot(
                lambda_gates, mean, color=color, linestyle=linestyle, marker=marker, markersize=MS,
                markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=4,
            )
        if d.get('baseline_acc') is not None:
            if d.get('baseline_acc_std'):
                ax1.axhspan(d['baseline_acc'] - d['baseline_acc_std'], d['baseline_acc'] + d['baseline_acc_std'],
                            color='#555555', alpha=0.1, zorder=1)
            ax1.axhline(y=d['baseline_acc'], color='#555555', linestyle='--', linewidth=LW_REF,
                        label='Baseline', zorder=5)

        ax1.set_xscale('log')
        _style_axes(ax1, dataset_name)
        ax1.tick_params(axis='y', labelcolor=ACC_COLOR)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax1.set_xlabel(r'$\lambda_{gate}$', fontsize=FS_LABEL, fontweight='bold')
        if i % ncols == 0:
            ax1.set_ylabel(metric_name, color=ACC_COLOR, fontsize=FS_LABEL, fontweight='bold')

        ax2 = ax1.twinx()
        if d.get('n_open_std'):
            lo = [max(0, o - s) for o, s in zip(d['n_open'], d['n_open_std'])]
            hi = [o + s for o, s in zip(d['n_open'], d['n_open_std'])]
            ax2.fill_between(lambda_gates, lo, hi, color=GATE_COLOR, alpha=0.15, linewidth=0, zorder=2)
        ax2.plot(
            lambda_gates, d['n_open'], color=GATE_COLOR, marker='D', markersize=MS - 1,
            markeredgecolor='white', markeredgewidth=MEW, linewidth=LW - 0.5, linestyle='--',
            alpha=0.9, label='Concepts', zorder=3,
        )
        ax2.tick_params(axis='y', labelcolor=GATE_COLOR, labelsize=FS_TICK, width=1.6, length=5)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        if n_open_as_pct:
            ax2.yaxis.set_major_formatter(PercentFormatter())
        hi = max(o + (s or 0) for o, s in zip(d['n_open'], d.get('n_open_std') or [0] * len(d['n_open'])))
        ax2_max = max(ax2_max or 0, hi)
        twin_axes.append(ax2)
        # The concept axis is shared across panels (same scale, see below), so like the accuracy
        # axis it is titled and tick-labelled once per row: on the rightmost panel.
        if (i + 1) % ncols == 0 or i == len(order) - 1:
            ax2.set_ylabel('Used Concepts' + (' (%)' if n_open_as_pct else ''), color=GATE_COLOR,
                           fontsize=FS_LABEL, fontweight='bold')
        elif sharey:
            ax2.tick_params(axis='y', labelright=False)

        if not legend_lines:
            l1, lab1 = ax1.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            legend_lines, legend_labels = l1 + l2, lab1 + lab2

    for ax in axes_flat[len(order):]:
        ax.axis('off')
    for ax2 in twin_axes:  # one shared concept scale so panels are comparable at a glance
        ax2.set_ylim(0, min(100, ax2_max * 1.08) if n_open_as_pct else ax2_max * 1.08)

    fig.patch.set_facecolor('white')
    _legend_below(fig, legend_lines, legend_labels)

    return _save_plot(None, plot_label, save_dir=os.path.join(DATA_PATH, 'plot'))


def plot_lambda_sparse_sweep(lambda_sparses, accs, n_used, dataset_name,
                              accs_std=None, n_used_std=None):
    """Twin-axis plot of test accuracy and number of used concepts (count_used_concepts, see
    src/metrics.py) against lambda_sparse for the baseline (ungated) probe. Mirrors
    _plot_single_lambda_gate_sweep, but the x-axis is lambda_sparse (the L1 weight penalty) and the
    right-hand series is used-concept count rather than open-gate count, since the baseline probe
    has no gates."""
    ACC_COLOR, CONCEPT_COLOR = '#4C72B0', '#55A868'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.set_facecolor('#F7F7F9')
    fig.patch.set_facecolor('white')

    if accs_std:
        lo = [a - s for a, s in zip(accs, accs_std)]
        hi = [a + s for a, s in zip(accs, accs_std)]
        ax1.fill_between(lambda_sparses, lo, hi, color=ACC_COLOR, alpha=0.18, linewidth=0, zorder=2)
    ax1.plot(
        lambda_sparses, accs, color=ACC_COLOR, marker='o', markersize=7,
        markeredgecolor='white', markeredgewidth=1.2, linewidth=2.5,
        label='Test accuracy', zorder=4,
    )
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\lambda_{sparse}$', fontsize=13)
    ax1.set_ylabel(ACC_AXIS_LABEL, color=ACC_COLOR, fontsize=13, fontweight='medium')
    ax1.tick_params(axis='y', labelcolor=ACC_COLOR, labelsize=10.5)
    ax1.tick_params(axis='x', labelsize=10.5)
    ax1.grid(True, which='major', linestyle='-', linewidth=0.7, color='white', zorder=0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2 = ax1.twinx()
    if n_used_std:
        lo = [max(0, u - s) for u, s in zip(n_used, n_used_std)]
        hi = [u + s for u, s in zip(n_used, n_used_std)]
        ax2.fill_between(lambda_sparses, lo, hi, color=CONCEPT_COLOR, alpha=0.15, linewidth=0, zorder=2)
    ax2.plot(
        lambda_sparses, n_used, color=CONCEPT_COLOR, marker='D', markersize=6,
        markeredgecolor='white', markeredgewidth=1.1, linewidth=2, linestyle='--',
        alpha=0.9, label='Used concepts', zorder=3,
    )
    ax2.set_ylabel('Used concepts', color=CONCEPT_COLOR, fontsize=13, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=CONCEPT_COLOR, labelsize=10.5)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, fontsize=10.5, loc='upper center',
        bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False,
    )

    plt.title(f'Accuracy / Concept Count vs. $\\lambda_{{sparse}}$ — {dataset_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save_plot(dataset_name, 'lambda_sparse_sweep')


def plot_pruning_results(results, base_acc, keep_fractions, title, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    labels = ['Activation Frequency', 'Maximum Activation Value', 'Name Alignment', 'Activation Variance']

    for ax, (_, (fracs, accs)), color, label in zip(axes, results.items(), colors, labels):
        ax.plot(fracs, accs, color=color, linewidth=2, marker='o', markersize=5, label='Pruned probe')
        ax.axhline(y=base_acc, color='gray', linestyle='--', linewidth=1.5,  label=f'Baseline ({base_acc:.3f})')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('Fraction of Concepts Kept', fontsize=11)
        ax.set_ylabel(ACC_AXIS_LABEL, fontsize=11)
        ax.set_xlim(0.05, 1.05)
        ax.set_xticks(keep_fractions)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_plot(dataset_name, 'baseline')


def plot_concept_ablation(fractions, open_mean, closed_mean, open_std, closed_std, dataset_name,
                           random_mean=None, random_std=None, labels=None, xlabel=None,
                           plot_label='concept_ablation'):
    """`labels` renames the three curves for callers whose selections aren't the open/closed sets
    (e.g. the ranked-order mode, which walks one ranking over the whole dictionary)."""
    OPEN_COLOR, CLOSED_COLOR, RANDOM_COLOR = '#C44E52', '#4C72B0', '#8C8C8C'
    plt.rcParams['font.family'] = 'sans-serif'
    pct = [f * 100 for f in fractions]
    labels = labels or ('Open-first', 'Closed-first', 'Random')

    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    curves = [
        (open_mean,   open_std,   OPEN_COLOR,   'o', '-',  labels[0]),
        (closed_mean, closed_std, CLOSED_COLOR, 's', '--', labels[1]),
    ]
    if random_mean is not None:
        curves.append((random_mean, random_std, RANDOM_COLOR, 'D', '-.', labels[2]))

    for mean, std, color, marker, ls, label in curves:
        if std:
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(pct, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            pct, mean, color=color, marker=marker, linestyle=ls, markersize=MS,
            markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=3,
        )

    _style_axes(ax)
    ax.set_xlabel(xlabel or 'Concepts ablated (%)', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
    ax.xaxis.set_major_formatter(PercentFormatter())
    # Paper figures render small, so a handful of large tick labels reads better than many small
    # ones - thin both axes down to ~5 ticks rather than matplotlib's denser default.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, h, l)

    return _save_plot(dataset_name, plot_label)


def plot_concept_ablation_grid(results_by_dataset, dataset_order=None, ncols=None, sharey=False,
                                xlabel=None, plot_label='concept_ablation_grid'):
    """One figure with one ablation-curve panel per dataset, instead of a separate figure per
    dataset (see plot_concept_ablation). Panels share a single x-axis title ('Concepts ablated
    (%)') and y-axis title ('Task Accuracy') on every panel (each panel is titled with its
    dataset name), and one legend shared below all panels since the curves
    (Open-first/Closed-first/Random) mean the same thing everywhere.

    results_by_dataset: {dataset_name: {'fractions', 'open_acc_mean', 'open_acc_std',
    'closed_acc_mean', 'closed_acc_std', 'random_acc_mean' (optional), 'random_acc_std'
    (optional)}} - i.e. the same fields plot_concept_ablation takes, one entry per dataset.
    `ncols` defaults to a 2-wide grid; `sharey=True` gives every panel the same accuracy range
    (and drops the redundant tick labels on all but the leftmost panel of each row), so curves
    are comparable across datasets at a glance.
    """
    OPEN_COLOR, CLOSED_COLOR, RANDOM_COLOR = '#C44E52', '#4C72B0', '#8C8C8C'
    plt.rcParams['font.family'] = 'sans-serif'

    order = dataset_order or list(results_by_dataset.keys())
    ncols = ncols or (2 if len(order) > 1 else 1)
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_W * ncols, PANEL_H * nrows + 0.9), squeeze=False,
                             sharey=sharey, layout='constrained')
    axes_flat = axes.flatten()

    legend_lines, legend_labels = None, None
    for i, (ax, dataset_name) in enumerate(zip(axes_flat, order)):
        d = results_by_dataset[dataset_name]
        pct = [f * 100 for f in d['fractions']]

        curves = [
            (d['open_acc_mean'],   d.get('open_acc_std'),   OPEN_COLOR,   'o', '-',  'Open-first'),
            (d['closed_acc_mean'], d.get('closed_acc_std'), CLOSED_COLOR, 's', '--', 'Closed-first'),
        ]
        if d.get('random_acc_mean') is not None:
            curves.append((d['random_acc_mean'], d.get('random_acc_std'), RANDOM_COLOR, 'D', '-.', 'Random'))

        for mean, std, color, marker, ls, label in curves:
            if std:
                lo = [m - s for m, s in zip(mean, std)]
                hi = [m + s for m, s in zip(mean, std)]
                ax.fill_between(pct, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
            ax.plot(
                pct, mean, color=color, marker=marker, linestyle=ls, markersize=MS,
                markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=3,
            )

        _style_axes(ax, dataset_name)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel(xlabel or 'Concepts Zeroed (%)', fontsize=FS_LABEL, fontweight='bold')
        if i % ncols == 0:
            ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
        if legend_lines is None:
            legend_lines, legend_labels = ax.get_legend_handles_labels()

    for ax in axes_flat[len(order):]:
        ax.axis('off')

    fig.patch.set_facecolor('white')
    _legend_below(fig, legend_lines, legend_labels)

    return _save_plot(None, plot_label, save_dir=os.path.join(DATA_PATH, 'plot'))


def plot_explanation_fidelity(top_ks, gated_mean, gated_std, dataset_name,
                               baseline_mean=None, baseline_std=None):
    """Line plot of prediction-change rate vs top_k: the fraction of predictions that flip
    when each explanation is pruned down to only its top_k contributing concepts."""
    GATED_COLOR, BASELINE_COLOR = '#C44E52', '#4C72B0'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor('#F7F7F9')
    fig.patch.set_facecolor('white')

    curves = [(gated_mean, gated_std, GATED_COLOR, GATED_LABEL)]
    if baseline_mean is not None:
        curves.append((baseline_mean, baseline_std, BASELINE_COLOR, 'Baseline probe'))

    for mean, std, color, label in curves:
        if std:
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(top_ks, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            top_ks, mean, color=color, marker='o', markersize=6,
            markeredgecolor='white', markeredgewidth=1.1, linewidth=2.5, label=label, zorder=3,
        )

    ax.set_xscale('log')
    ax.set_xticks(top_ks)
    ax.set_xticklabels([str(k) for k in top_ks])
    ax.set_xlabel('Top-k concepts kept in explanation', fontsize=12)
    ax.set_ylabel('Prediction change rate', fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='white', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'Explanation fidelity — {dataset_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10.5, loc='upper right', frameon=False)

    fig.tight_layout()
    return _save_plot(dataset_name, 'explanation_fidelity')


_CONTRIB_DIST_COLORS = ['#4C72B0', '#C44E52', '#55A868', '#8172B2']


def plot_contribution_distribution(contribs_by_label, dataset_name, bins=100, save_dir=None,
                                    plot_label='contribution_distribution', title_suffix=''):
    """Overlaid histogram comparing |concept contribution| values across an arbitrary
    {label: contributions} mapping of probes - e.g. a gated probe vs a baseline one, either
    pooled over many examples or from a single image. Log y-axis since contributions cluster
    heavily near zero with a long tail of a few large ones."""
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor('#F7F7F9')
    fig.patch.set_facecolor('white')

    max_val = max(contribs.max() for contribs in contribs_by_label.values())
    bin_edges = np.linspace(0, max_val, bins + 1)

    for (label, contribs), color in zip(contribs_by_label.items(), _CONTRIB_DIST_COLORS):
        ax.hist(contribs, bins=bin_edges, color=color, alpha=0.55, density=True, label=label, zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel('|Concept contribution|', fontsize=12)
    ax.set_ylabel('Density (log scale)', fontsize=12)
    ax.set_title(f'Distribution of concept contributions — {dataset_name}{title_suffix}', fontsize=13, fontweight='bold')
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='white', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(fontsize=10.5, loc='upper right', frameon=False)

    fig.tight_layout()
    return _save_plot(dataset_name, plot_label, save_dir=save_dir)


PANEL_BG = '#F7F7F9'
_BLUE_SEQ = LinearSegmentedColormap.from_list('project_blue_seq', ['#EEF3FA', '#4C72B0', '#1B3B63'])


def _style_panel(ax, n_rows, x_axis_only=True):
    plt.rcParams['font.family'] = 'sans-serif'
    ax.figure.patch.set_facecolor('white')
    ax.set_facecolor(PANEL_BG)
    for i in range(0, n_rows, 2):
        ax.axhspan(i - 0.5, i + 0.5, color='white', alpha=0.55, zorder=0)
    ax.grid(True, axis='x' if x_axis_only else 'both', linestyle='-', linewidth=0.9, color='white', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _format_class_name(name):
    # Places365 classes are folder paths like '/s/swimming_pool/indoor': drop the letter bucket
    # and show the sub-scene in parentheses.
    name = re.sub(r'^/[a-z]/', '', name)
    if '/' in name:
        head, sub = name.split('/', 1)
        name = f'{head} ({sub.replace("/", ", ")})'
    return name.replace('_', ' ').title()


def plot_superclass_confusion_matrix(cm, class_names, dataset_name, model_name, normalize=True):
    """cm: (n_classes, n_classes) tensor/array of true (rows) vs predicted (cols) counts."""
    plt.rcParams['font.family'] = 'sans-serif'
    cm = cm.float() if torch.is_tensor(cm) else torch.tensor(cm, dtype=torch.float)
    if normalize:
        cm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
    cm_np = cm.numpy()
    class_names = [_format_class_name(name) for name in class_names]
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2.5))
    fig.patch.set_facecolor('white')
    im = ax.imshow(cm_np, cmap=_BLUE_SEQ, vmin=0, vmax=1 if normalize else cm_np.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticklabels(class_names, fontsize=8.5)
    ax.set_xlabel('Predicted superclass', fontsize=11.5, labelpad=8)
    ax.set_ylabel('True superclass', fontsize=11.5, labelpad=8)
    ax.set_title(f'Superclass confusion matrix — {model_name}\n{dataset_name}', fontsize=13, fontweight='bold', pad=14)

    # Thin white separators between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(which='major', length=0)

    thresh = 0.55 * (1 if normalize else cm_np.max())
    skip_below = 0.005 if normalize else 0
    for i in range(n):
        for j in range(n):
            val = cm_np[i, j]
            if val <= skip_below:
                continue
            text = f'{val:.2f}' if normalize else f'{int(val)}'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, text, ha='center', va='center', fontsize=7, fontweight=weight,
                    color='white' if val > thresh else '#2B2B2B')

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if normalize:
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    cbar.set_label('Share of true class' if normalize else 'Count', fontsize=10)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    return _save_plot(dataset_name, f'superclass_confusion_{model_name}')


def plot_superclass_accuracy(class_acc, dataset_name, model_name):
    """class_acc: dict mapping superclass name -> accuracy."""
    items = sorted(class_acc.items(), key=lambda kv: kv[1])
    names, accs = [_format_class_name(k) for k, _ in items], [v for _, v in items]
    mean_acc = sum(accs) / len(accs)

    norm = plt.Normalize(vmin=min(accs) - 0.08, vmax=max(accs) + 0.03)
    colors = [_BLUE_SEQ(norm(a)) for a in accs]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.6))
    _style_panel(ax, len(names))
    bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=0.8, height=0.68, zorder=3)
    ax.axvline(x=mean_acc, color='#555555', linestyle='--', linewidth=1.5, zorder=4,
               label=f'Mean = {mean_acc:.3f}')
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', va='center', fontsize=8.5, color='#2B2B2B')

    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_xlabel('Accuracy', fontsize=11.5)
    ax.set_title(f'Per-superclass accuracy — {model_name}\n{dataset_name}', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=9.5, frameon=False)
    plt.tight_layout()
    return _save_plot(dataset_name, f'superclass_accuracy_{model_name}')

def plot_superclass_delta(
    baseline_accs,
    gated_accs,
    dataset_name,
    baseline_name,
    gated_name,
    top_n=10,
):
    """
    Publication-style dumbbell plot comparing superclass accuracies.
    Shows the top_n superclasses ranked by absolute accuracy change.
    """

    deltas = {
        k: gated_accs[k] - baseline_accs[k]
        for k in baseline_accs
    }

    # Top N by absolute change
    ranked = sorted(
        deltas.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    # Largest change at the top
    items = ranked[::-1]

    names = [_format_class_name(k) for k, _ in items]
    baseline = [baseline_accs[k] for k, _ in items]
    gated = [gated_accs[k] for k, _ in items]

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    BASE = "#B8BCC2"      # light grey
    GATED = "#1F4E79"     # dark blue
    POS = "#2E8B57"       # improvement
    NEG = "#C44E52"       # degradation

    fig_h = max(4.0, 0.55 * len(names) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    ys = np.arange(len(names))

    # ------------------------------------------------------------------
    # Clean styling
    # ------------------------------------------------------------------
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)
    ax.grid(False, axis="y")

    max_delta = max(abs(g - b) for b, g in zip(baseline, gated))
    if max_delta == 0:
        max_delta = 1

    # ------------------------------------------------------------------
    # Dumbbell lines
    # ------------------------------------------------------------------
    for y, b, g in zip(ys, baseline, gated):

        delta = abs(g - b)

        # Emphasise larger improvements
        lw = 2 + 6 * np.sqrt(delta / max_delta)

        ax.plot(
            [b, g],
            [y, y],
            color=POS if g >= b else NEG,
            linewidth=lw,
            alpha=0.8,
            solid_capstyle="round",
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------
    ax.scatter(
        baseline,
        ys,
        s=70,
        color=BASE,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
        label="Baseline Probe",
    )

    ax.scatter(
        gated,
        ys,
        s=90,
        color=GATED,
        edgecolor="white",
        linewidth=1.1,
        zorder=4,
        label=GATED_LABEL,
    )

    # ------------------------------------------------------------------
    # Delta labels (percentage points)
    # ------------------------------------------------------------------
    lo = min(baseline + gated)
    hi = max(baseline + gated)

    offset = 0.015 * (hi - lo)

    for y, b, g in zip(ys, baseline, gated):

        delta_pp = (g - b) * 100

        ax.text(
            max(b, g) + offset,
            y,
            rf"$\Delta={delta_pp:+.1f}$ pp",
            fontsize=8.5,
            color=POS if delta_pp >= 0 else NEG,
            va="center",
            ha="left",
            fontweight="semibold",
        )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10)

    pad = max((hi - lo) * 0.05, 0.005)

    ax.set_xlim(lo - pad, hi + pad * 3)

    ax.set_xlabel("Classification Accuracy", fontsize=11)

    ax.set_title(
        f"Top {top_n} Superclass Accuracy Changes",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # ------------------------------------------------------------------
    # Remove clutter
    # ------------------------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="lower right",
    )

    plt.tight_layout()

    return _save_plot(
        dataset_name,
        f"superclass_delta_{gated_name}_vs_{baseline_name}",
    )

def plot_concept_intervention(fractions, curves, dataset_name):
    """Concept intervention curve: as increasing fractions of CUB-attribute-matched concepts are
    clamped to ground-truth-informed activation values (see concept_intervention.py), how does test
    accuracy respond. `curves` is {label: (mean_acc, std_acc, color, linestyle)}; a curve is skipped
    if its mean is None. Intended for one solid + one dashed line per probe (random-order vs.
    uncertainty-ranked intervention order, à la Koh et al. 2020), but works for any label set."""
    plt.rcParams['font.family'] = 'sans-serif'
    pct = [f * 100 for f in fractions]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor('#F7F7F9')
    fig.patch.set_facecolor('white')

    for label, (mean, std, color, linestyle) in curves.items():
        if mean is None:
            continue
        if std:
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(pct, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(
            pct, mean, color=color, marker='o', markersize=5,
            markeredgecolor='white', markeredgewidth=1.0, linewidth=2.2,
            linestyle=linestyle, label=label, zorder=3,
        )

    ax.set_xlabel('Percentage of concepts intervened', fontsize=12)
    ax.set_ylabel(ACC_AXIS_LABEL, fontsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_title(f'Concept intervention curve — {dataset_name}', fontsize=13, fontweight='bold')
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='white', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(fontsize=9.5, loc='lower right', frameon=False)

    fig.tight_layout()
    return _save_plot(dataset_name, 'concept_intervention')


def plot_random_concept_baseline(concept_counts, random_mean, random_std, dataset_name, operating_point=None):
    """Accuracy vs. concept count for a probe trained on top of an untrained, randomly initialized
    concept layer (see random_concept_baseline.py). Meant to be read alongside a real-concept
    probe's accuracy at the same concept counts (e.g. from train_cbm.py / prune_concepts.py),
    tracked separately - this curve tracking that one, especially at high concept counts, is
    evidence the real probe's accuracy comes from dimensionality alone, not from what the concepts
    encode. `operating_point`, if given, marks the concept count actually used by a deployed gated
    probe (e.g. mean of that run's probe_config.json open_gates_by_seed) with a vertical line."""
    RANDOM_COLOR = '#8C8C8C'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    if random_std:
        lo = [m - s for m, s in zip(random_mean, random_std)]
        hi = [m + s for m, s in zip(random_mean, random_std)]
        ax.fill_between(concept_counts, lo, hi, color=RANDOM_COLOR, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(
        concept_counts, random_mean, color=RANDOM_COLOR, marker='s', linestyle='-.', markersize=MS,
        markeredgecolor='white', markeredgewidth=MEW, linewidth=LW,
        label='Random (untrained Gaussian projection)', zorder=3,
    )
    _add_operating_point(ax, operating_point)

    _style_axes(ax, dataset_name)
    ax.set_xlabel('Concept count', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, h, l, ncol=1)

    return _save_plot(dataset_name, 'random_concept_baseline')


def plot_gated_vs_random(open_gates, gated_test_acc, gated_test_acc_std,
                          concept_counts, random_acc_mean, random_acc_std,
                          dataset_name, baseline_test_acc=None, baseline_test_acc_std=None,
                          operating_point=None):
    """Test accuracy vs. concept count: the gated probe's real accuracy at each lambda_gate (from
    sweep_lambda_gate.py's lambda_gate_sweep.json, x = mean open gates at that lambda_gate) against
    an untrained random-projection probe's accuracy at the same concept counts
    (random_concept_baseline.py). The random curve tracking the gated one is evidence accuracy comes
    from dimensionality alone; a persistent gap is evidence the gate's learned concept selection
    matters. `open_gates`/`gated_test_acc`(_std) and `concept_counts`/`random_acc_mean`(_std) need
    not share the same x-values - each is plotted on its own points along a shared log x-axis.
    `operating_point`, if given, marks the concept count actually used by a deployed gated probe
    (e.g. mean of that run's probe_config.json open_gates_by_seed) with a vertical line."""
    GATED_COLOR, RANDOM_COLOR = '#C44E52', '#8C8C8C'

    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    curves = [
        (open_gates,     gated_test_acc,  gated_test_acc_std, GATED_COLOR,  'o', '-',  GATED_LABEL),
        (concept_counts, random_acc_mean, random_acc_std,     RANDOM_COLOR, 's', '-.', 'Random projection'),
    ]
    for xs, mean, std, color, marker, ls, label in curves:
        if std:
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            xs, mean, color=color, marker=marker, linestyle=ls, markersize=MS,
            markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=3,
        )
    # The ungated probe over the full dictionary is a different model, not the lambda_gate -> 0
    # limit of the gated sweep (on Places365 it sits well below every gated probe), so it is a
    # reference line rather than a point joined onto the gated curve.
    if baseline_test_acc is not None:
        if baseline_test_acc_std:
            ax.axhspan(baseline_test_acc - baseline_test_acc_std, baseline_test_acc + baseline_test_acc_std,
                       color='#555555', alpha=0.1, zorder=1)
        ax.axhline(y=baseline_test_acc, color='#555555', linestyle='--', linewidth=LW_REF,
                   label='Baseline', zorder=4)
    _add_operating_point(ax, operating_point)

    ax.set_xscale('log')
    _style_axes(ax, dataset_name)
    ax.set_xlabel('Concept count', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, h, l, ncol=2)

    return _save_plot(dataset_name, 'gated_vs_random')


def _plot_acc_vs_count_grid(results_by_dataset, comparator_label, dataset_order=None, ncols=None,
                            sharey=True, counts_as_pct=True, frame_on_gated=False, plot_label='acc_vs_count_grid', legend_fontsize=FS_LEGEND):
    """Shared body of plot_gated_vs_random_grid / plot_gate_vs_sparse_grid: one panel per dataset
    of test accuracy vs. concept count for the gated sweep against one comparator curve, styled
    like the other grids (shared accuracy axis, one legend above, gated blue / comparator green /
    baseline dashed grey).

    results_by_dataset: {dataset_name: {'open_gates', 'gated_acc', 'gated_acc_std', 'cmp_x',
    'cmp_acc', 'cmp_acc_std', 'baseline_acc', 'baseline_acc_std', 'operating_point' (optional)}}
    - x-values in counts, or in % of the dictionary when `counts_as_pct` (the caller scales); the
    axis is logarithmic either way. `frame_on_gated` limits each panel's x-range to what the gated
    sweep covers (plus a margin) instead of the union of both curves -- see plot_gate_vs_sparse_sweep
    for why the L1 sweep needs that."""
    GATED_COLOR, CMP_COLOR = '#4C72B0', '#55A868'
    plt.rcParams['font.family'] = 'sans-serif'

    order = dataset_order or list(results_by_dataset.keys())
    ncols = ncols or (2 if len(order) > 1 else 1)
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_W * ncols, PANEL_H * nrows + 0.9), squeeze=False,
                             sharey=sharey, layout='constrained')
    axes_flat = axes.flatten()

    legend_lines, legend_labels = None, None
    for i, (ax, dataset_name) in enumerate(zip(axes_flat, order)):
        d = results_by_dataset[dataset_name]

        # Different marker per curve as well as colour, so they stay apart in greyscale too.
        curves = [
            (d['open_gates'], d['gated_acc'], d.get('gated_acc_std'), GATED_COLOR, 'o', '-',  GATED_LABEL),
            (d['cmp_x'],      d['cmp_acc'],   d.get('cmp_acc_std'),   CMP_COLOR,   's', '-.', comparator_label),
        ]
        for xs, mean, std, color, marker, ls, label in curves:
            if std:
                lo = [m - s for m, s in zip(mean, std)]
                hi = [m + s for m, s in zip(mean, std)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
            ax.plot(
                xs, mean, color=color, marker=marker, linestyle=ls, markersize=MS,
                markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=3,
            )
        if d.get('baseline_acc') is not None:
            if d.get('baseline_acc_std'):
                ax.axhspan(d['baseline_acc'] - d['baseline_acc_std'], d['baseline_acc'] + d['baseline_acc_std'],
                           color='#555555', alpha=0.1, zorder=1)
            ax.axhline(y=d['baseline_acc'], color='#555555', linestyle='--', linewidth=LW_REF,
                       label='Baseline', zorder=4)
        # Concept count actually used by the deployed gated probe for this dataset, annotated on
        # the line (as % when the axis is %) since the value differs per panel.
        if d.get('operating_point') is not None:
            op = d['operating_point']
            ax.axvline(x=op, color=GATED_COLOR, linestyle=':', linewidth=LW_REF, zorder=4, label='Reported model')
            ax.text(op, 0.97, (f'{op:.1f}% ' if counts_as_pct else f'{op:.0f} '),
                    transform=ax.get_xaxis_transform(), ha='right', va='top', fontsize=FS_TICK,
                    color=GATED_COLOR, zorder=5)

        ax.set_xscale('log')
        if frame_on_gated:
            ax.set_xlim(min(d['open_gates']) / 2.5, max(max(d['open_gates']), max(d['cmp_x'])) * 1.6)
        if counts_as_pct:
            ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        _style_axes(ax, dataset_name)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel('Concepts Used (%)' if counts_as_pct else 'Concept count', fontsize=FS_LABEL, fontweight='bold')
        if i % ncols == 0:
            ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
        if legend_lines is None:
            legend_lines, legend_labels = ax.get_legend_handles_labels()

    for ax in axes_flat[len(order):]:
        ax.axis('off')
    if any(results_by_dataset[n].get('operating_point') is not None for n in order):
        # headroom for the operating-point labels (once when the y-axis is shared, else per panel)
        for ax in (axes_flat[:1] if sharey else axes_flat[:len(order)]):
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.12 * (hi - lo))

    fig.patch.set_facecolor('white')
    _legend_below(fig, legend_lines, legend_labels, fontsize=legend_fontsize)

    return _save_plot(None, plot_label, save_dir=os.path.join(DATA_PATH, 'plot'))


def plot_gated_vs_random_grid(results_by_dataset, **kwargs):
    """Grid version of plot_gated_vs_random: gated sweep vs. an untrained random projection of the
    same size ('cmp_*' = random_concept_baseline.py's counts / accuracies). See _plot_acc_vs_count_grid."""
    return _plot_acc_vs_count_grid(results_by_dataset, 'Random projection',
                                   plot_label=kwargs.pop('plot_label', 'gated_vs_random_grid'), **kwargs)


def plot_gate_vs_sparse_grid(results_by_dataset, **kwargs):
    """Grid version of plot_gate_vs_sparse_sweep: gated sweep vs. the L1 sparse-probe sweep
    ('cmp_*' = lambda_sparse_sweep.json's used_concepts / accuracy). Framed on the gated sweep's
    x-range since L1 drives the count below 1 concept at chance accuracy. See _plot_acc_vs_count_grid."""
    kwargs.setdefault('legend_fontsize', FS_LEGEND + 3)  # only three entries, so there is room
    return _plot_acc_vs_count_grid(results_by_dataset, SPARSE_LABEL, frame_on_gated=True,
                                   plot_label=kwargs.pop('plot_label', 'gate_vs_sparse_grid'), **kwargs)


def plot_random_retrain_sweep(concept_counts, gate_mean, gate_std, rand_mean, rand_std,
                               baseline_acc, baseline_acc_std, dataset_name):
    """Test accuracy vs. concept count N for fresh probes retrained from scratch on N concepts
    drawn uniformly at random (concept_ablation.py --mode random-retrain), N taken from a
    sweep_lambda_gate.py run, against the all-concepts 'Baseline'. `gate_mean` is optional
    (pass None to omit); when given it is a second curve, e.g. a ranked top-N selection.
    A random curve that meets the baseline early means concept count alone drives accuracy."""
    GATE_COLOR, RANDOM_COLOR = '#C44E52', '#55A868'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    series = [(rand_mean, rand_std, RANDOM_COLOR, 's', '-.', 'Random subset')]
    if gate_mean is not None:
        series.insert(0, (gate_mean, gate_std, GATE_COLOR, 'o', '-', 'Gate-selected'))
    for mean, std, color, marker, ls, label in series:
        if std and any(std):
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(concept_counts, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            concept_counts, mean, color=color, marker=marker, linestyle=ls, markersize=MS,
            markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=3,
        )

    if baseline_acc is not None:
        if baseline_acc_std:
            ax.axhspan(baseline_acc - baseline_acc_std, baseline_acc + baseline_acc_std,
                       color='#555555', alpha=0.1, zorder=1)
        ax.axhline(y=baseline_acc, color='#555555', linestyle='--', linewidth=LW_REF,
                   label='Baseline', zorder=5)

    ax.set_xscale('log')
    _style_axes(ax, dataset_name)
    ax.set_xlabel('Concept count', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, h, l)

    return _save_plot(dataset_name, 'random_retrain_sweep')


def plot_gate_vs_sparse_sweep(open_gates, gated_test_acc, gated_test_acc_std,
                               used_concepts, sparse_test_acc, sparse_test_acc_std,
                               dataset_name, baseline_test_acc=None, baseline_test_acc_std=None):
    """Test accuracy vs. concept count for an existing sweep_lambda_gate.py run (x = mean open
    gates per lambda_gate) overlaid with an existing sweep_lambda_sparse.py run (x = mean used
    concepts per lambda_sparse, see count_used_concepts in src/metrics.py), to compare how the two
    sparsity mechanisms trade off concept count against accuracy on the same dataset."""
    GATE_COLOR, SPARSE_COLOR = '#C44E52', '#55A868'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    curves = [
        (open_gates,    gated_test_acc,  gated_test_acc_std,  GATE_COLOR,   'o', '-',  GATED_LABEL),
        (used_concepts, sparse_test_acc, sparse_test_acc_std, SPARSE_COLOR, 's', '-.', SPARSE_LABEL),
    ]
    for xs, mean, std, color, marker, ls, label in curves:
        if std:
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            xs, mean, color=color, marker=marker, linestyle=ls, markersize=MS,
            markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=3,
        )
    if baseline_test_acc is not None:
        if baseline_test_acc_std:
            ax.axhspan(baseline_test_acc - baseline_test_acc_std, baseline_test_acc + baseline_test_acc_std,
                        color='#555555', alpha=0.1, zorder=1)
        ax.axhline(y=baseline_test_acc, color='#555555', linestyle='--', linewidth=LW_REF,
                   label='Baseline (all concepts)', zorder=1)
    # L1 drives the concept count below 1 before it gives up, so on a log axis the sparse sweep
    # spans several decades in which it sits at chance -- squeezing both curves' interesting
    # range into the right-hand third. Frame on the range the gated sweep actually covers; any
    # sparse point left of that is off-scale rather than plotted.
    x_lo = min(open_gates) / 2.5
    x_hi = max(max(open_gates), max(used_concepts)) * 1.6
    ax.set_xlim(x_lo, x_hi)

    ax.set_xscale('log')
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    _style_axes(ax, dataset_name)
    ax.set_xlabel('Concept count', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel(ACC_AXIS_LABEL, fontsize=FS_LABEL, fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, which='minor', linestyle='-', linewidth=0.35, color='white', zorder=0)
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, h, l, ncol=2, fontsize=FS_LEGEND + 3)

    return _save_plot(dataset_name, 'gate_vs_sparse_sweep')


def plot_concept_stability(concept_frequency, n_seeds, mean_jaccard, dataset_name):
    counts = Counter(concept_frequency.values())
    xs = list(range(1, n_seeds + 1))
    ys = [counts.get(k, 0) for k in xs]

    plt.figure(figsize=(7, 4))
    plt.bar(xs, ys, color='steelblue')
    plt.xlabel('Number of seeds a concept is open in')
    plt.ylabel('Number of concepts')
    plt.xticks(xs)
    plt.title(f'Open-concept stability across {n_seeds} seeds — {dataset_name}\nMean pairwise Jaccard: {mean_jaccard:.3f}')
    plt.tight_layout()
    _save_plot(dataset_name, 'concept_stability')

def plot_gate_init_sweep(summary, dataset_name):
    """Gate-initialisation ablation (sweep_gate_init.py): task accuracy (gated probe and, when
    present, the mask-and-refit head) and open-gate count against the constant every gate logit
    starts at. Noisy starts (gate_init_std > 0) are drawn as separate markers at their mean."""
    ACC_COLOR, REFIT_COLOR, GATE_COLOR = '#4C72B0', '#55A868', '#DD8452'
    plt.rcParams['font.family'] = 'sans-serif'
    rows = summary['rows']
    const = [r for r in rows if r['gate_init_std'] == 0]
    noisy = [r for r in rows if r['gate_init_std'] > 0]
    x = [r['gate_init'] for r in const]

    fig, ax1 = plt.subplots(figsize=(5.4, 4.6), layout='constrained')
    fig.patch.set_facecolor('white')

    def _curve(ax, key, color, marker, ls, label):
        m = [r[f'{key}_mean'] for r in const]
        sd = [r[f'{key}_std'] for r in const]
        ax.fill_between(x, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, m, color=color, marker=marker, linestyle=ls, markersize=MS,
                markeredgecolor='white', markeredgewidth=MEW, linewidth=LW, label=label, zorder=4)
        for r in noisy:
            ax.errorbar(r['gate_init'], r[f'{key}_mean'], yerr=r[f'{key}_std'], color=color,
                        marker='*', markersize=MS + 6, markeredgecolor='white', linestyle='none',
                        capsize=4, zorder=5, label=f'{label}, noisy start' if r is noisy[0] else None)

    _curve(ax1, 'accuracy', ACC_COLOR, 'o', '-', GATED_LABEL)
    if all('refit_accuracy_mean' in r for r in rows):
        _curve(ax1, 'refit_accuracy', REFIT_COLOR, 's', '--', f'{GATED_LABEL} (refit)')
    _style_axes(ax1)
    ax1.set_xlabel('Initial gate logit  $\\mu_j(0)$', fontsize=FS_LABEL, fontweight='bold')
    ax1.set_ylabel(ACC_AXIS_LABEL, color=ACC_COLOR, fontsize=FS_LABEL, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=ACC_COLOR)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax2 = ax1.twinx()
    _curve(ax2, 'open_gates', GATE_COLOR, 'D', ':', 'Open gates')
    ax2.set_ylabel('Open gates', color=GATE_COLOR, fontsize=FS_LABEL, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=GATE_COLOR, labelsize=FS_TICK, width=1.6, length=5)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend_below(fig, h1 + h2, l1 + l2, ncol=2)
    return _save_plot(dataset_name, 'gate_init_sweep')
