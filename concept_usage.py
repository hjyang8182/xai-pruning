"""Backbone-agnostic concept-usage metrics.

How many of a CBM's concepts actually drive its predictions? Thresholding a single number
(non-zero final-layer weights, or gate > 0.5) is fragile: in the gated-probe case every gate
can fall below 0.5 while accuracy is unchanged, because the gate only rescales activations
that the linear layer is free to rescale back. This module reports a small panel instead,
all of it computed from two tensors any backbone can hand over:

    weight : (n_classes, n_concepts)   final linear-layer weight
    acts   : (n_examples, n_concepts)  concept activations as fed into that linear layer
                                       (optional; enables the activation-weighted and
                                       per-example metrics)

Everything operates on plain arrays/tensors, so it drops in for DN-CBM (src/), LF-CBM,
UCBM, VLG-CBM, or anything else that ends up with a concept-activation matrix and a linear
head -- no dependency on any backbone's config or data pipeline. Same spirit as gating.py.

Metrics (keys in the returned dict)
-----------------------------------
n_used_union            Concepts with |weight| > `weight_threshold` for at least one class
                        (a concept counts as used only if it can move some logit). Coarse
                        upper bound; threshold-sensitive; ignores a separate gate unless one
                        is passed / folded in.
nec / nec_median        Number of Effective Concepts (Zhao et al., arXiv:2408.01432): the
                        mean (and median) over classes of that per-class non-zero count.
n_effective_mass[p]     Concepts needed to account for fraction p of total concept
                        contribution, where contribution_c = ||weight[:, c]||_1 *
                        mean_x |acts[x, c]| -- the typical magnitude by which concept c moves
                        the logit vector. Sorted-cumulative, so it degrades gracefully when a
                        soft L1 penalty never drives weights to exactly zero.
participation_ratio     (sum_c s_c)^2 / sum_c s_c^2  with  s_c = contribution_c. A
                        threshold-free "effective number of concepts": ~n_concepts if all
                        contribute equally, ~1 if one dominates.
active_per_image_*      mean / std / median over examples of #{c : |acts[x, c]| >
                        `act_threshold`}. Only present when `acts` is given. Meaningful for a
                        sparse / non-negative code (post-ReLU SAE, UCBM similarities); for
                        dense standardised activations it approaches n_concepts and says
                        little -- `contribution_weighted_by_acts` in the dict flags whether
                        `acts` was supplied at all.

Gating
------
Pass `gate` (shape (n_concepts,)) only when the model applies a per-concept multiplier that
is *not already* reflected in `weight` (e.g. UCBM's gated classifier head, or a DN-CBM
GatedProbe whose `.linear.weight` is stored ungated). It is folded into both `weight` and
`acts` so every metric sees the gated model. Do NOT pass it when the weights already have
the gate baked in (LF-CBM arm C `W_g`, VLG-CBM arm C `W_g`) -- that would double-count it.
"""
import numpy as np


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def _fmt_level(p):
    # 0.9 -> "0.9", 0.95 -> "0.95"; stable, compact JSON keys
    return format(float(p), "g")


def concept_usage_report(weight, acts=None, gate=None, *,
                         weight_threshold=1e-5,
                         act_threshold=1e-8,
                         mass_levels=(0.90, 0.95, 0.99),
                         column_norm="l1"):
    """Return a dict of concept-usage metrics. See the module docstring for the definitions
    and for when to pass `gate`.

    Parameters
    ----------
    weight : array-like (n_classes, n_concepts)
        Final linear-layer weight matrix.
    acts : array-like (n_examples, n_concepts), optional
        Concept activations exactly as fed into the linear layer (post non-linearity/gate).
        Enables activation-weighted contribution and the per-image active-concept counts.
    gate : array-like (n_concepts,), optional
        Per-concept multiplier not already folded into `weight`/`acts`; folded into both.
    weight_threshold : float
        |weight| above this counts as a live connection (for n_used_union / nec).
    act_threshold : float
        |act| above this counts as an active concept for the per-image counts.
    mass_levels : iterable of float
        Coverage fractions for n_effective_mass.
    column_norm : {"l1", "l2"}
        Norm of each weight column used in the contribution score.
    """
    W = _to_numpy(weight)
    if W.ndim != 2:
        raise ValueError(f"weight must be 2D (n_classes, n_concepts), got shape {W.shape}")
    n_classes, n_concepts = W.shape

    A = _to_numpy(acts)
    if A is not None and (A.ndim != 2 or A.shape[1] != n_concepts):
        raise ValueError(
            f"acts must be (n_examples, {n_concepts}) to match weight, got {A.shape}")

    g = _to_numpy(gate)
    if g is not None:
        g = g.reshape(-1)
        if g.shape[0] != n_concepts:
            raise ValueError(f"gate must be ({n_concepts},), got {g.shape}")
        W = W * g[None, :]
        if A is not None:
            A = A * g[None, :]

    nonzero = np.abs(W) > weight_threshold        # (n_classes, n_concepts)
    per_class_count = nonzero.sum(axis=1)          # (n_classes,)
    used_union_mask = nonzero.any(axis=0)          # (n_concepts,)
    n_used_union = int(used_union_mask.sum())

    if column_norm == "l1":
        col = np.abs(W).sum(axis=0)
    elif column_norm == "l2":
        col = np.sqrt((W ** 2).sum(axis=0))
    else:
        raise ValueError("column_norm must be 'l1' or 'l2'")
    act_mag = np.abs(A).mean(axis=0) if A is not None else np.ones(n_concepts)
    s = col * act_mag                              # (n_concepts,) per-concept contribution

    total = float(s.sum())
    s_sorted = np.sort(s)[::-1]
    csum = np.cumsum(s_sorted)
    n_effective_mass = {}
    for p in mass_levels:
        if total <= 0:
            n_effective_mass[_fmt_level(p)] = 0
        else:
            idx = int(np.searchsorted(csum, p * total)) + 1
            n_effective_mass[_fmt_level(p)] = min(idx, n_concepts)

    sq = float((s ** 2).sum())
    participation_ratio = (total ** 2 / sq) if sq > 0 else 0.0

    report = {
        "n_concepts_total": int(n_concepts),
        "n_classes": int(n_classes),
        "weight_threshold": float(weight_threshold),
        "gated": g is not None,
        "n_used_union": n_used_union,
        "used_union_frac": (n_used_union / n_concepts) if n_concepts else 0.0,
        "nec": float(per_class_count.mean()),
        "nec_median": float(np.median(per_class_count)),
        "contribution_norm": column_norm,
        "contribution_weighted_by_acts": A is not None,
        "n_effective_mass": n_effective_mass,
        "participation_ratio": participation_ratio,
    }

    if A is not None:
        active = (np.abs(A) > act_threshold).sum(axis=1).astype(np.float64)
        report["act_threshold"] = float(act_threshold)
        report["active_per_image_mean"] = float(active.mean())
        report["active_per_image_std"] = float(active.std())
        report["active_per_image_median"] = float(np.median(active))

    return report


def mean_report(reports):
    """Elementwise mean of a list of `concept_usage_report` dicts, for averaging over seeds.
    Numeric scalars are averaged; the nested `n_effective_mass` is averaged per key; string
    and bool fields are copied from the first report."""
    reports = [r for r in reports if r]
    if not reports:
        return {}
    out = {}
    for k, v in reports[0].items():
        if isinstance(v, (bool, str)):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: float(np.mean([r[k][kk] for r in reports])) for kk in v}
        elif isinstance(v, (int, float)):
            out[k] = float(np.mean([r[k] for r in reports]))
        else:
            out[k] = v
    return out
