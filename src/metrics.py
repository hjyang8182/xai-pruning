import numpy as np

# Backbone-agnostic concept-usage panel, shared with LF-CBM / UCBM / VLG-CBM. Re-exported
# here so DN-CBM code can `from src.metrics import concept_usage_report` alongside the
# DN-CBM-specific helpers below.
from concept_usage import concept_usage_report, mean_report  # noqa: F401


def compute_cea(num_classes, num_concepts, acc, beta=0.25):
    # Concept-efficient Accuracy as defined by Zhao et al. (PS-CBM)
    k = np.ceil(np.log2(num_classes))
    return acc / (np.log(num_concepts) / np.log(k)) ** beta


def compute_nec(weight, threshold=1e-3):
    # Number of Effective Concepts as defined by Zhao et al. (arXiv:2408.01432): the average,
    # across classes, of how many concepts have a non-negligible final-layer weight. `weight`
    # should already be restricted to whichever concepts are in play (e.g. the open set for a
    # gated probe). Weights here come from a soft L1 penalty rather than proximal sparsification,
    # so they're never exactly zero - `threshold` is what stands in for "unused".
    return (weight.abs() > threshold).float().sum(dim=1).mean(dim=0).item()


def count_used_concepts(weight, threshold=1e-3):
    # A concept counts as "used" if it has at least one non-negligible weight to some class,
    # rather than every concept in the dictionary counting as used just by existing (a concept
    # can have a weight to every class that never clears `threshold`, i.e. never actually drive
    # a prediction). `weight` is (n_classes, n_concepts); same threshold semantics as compute_nec.
    return (weight.abs() > threshold).any(dim=0).sum().item()
