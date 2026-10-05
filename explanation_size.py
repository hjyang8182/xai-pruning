"""Local vs. global concept usage: how big is a single prediction's explanation?

Sparse CBMs (LF-CBM / VLG-CBM's GLM-SAGA head, UCBM's elastic-net head) control
*per-class* sparsity: NEC = mean #non-zero weights per class row. That says nothing
about the two numbers that decide whether the model as a whole is interpretable:

  * the global union  U = {j : some class has W[c,j] != 0}  -- every concept a user
    must understand to read the model at all; and
  * the *local* explanation size of a given image x -- the number of concepts that
    actually carry the predicted-class logit  z_yhat(x) = sum_j W[yhat,j] a_j(x) + b.

A low NEC with |U| ~ n_concepts means the concept set is only ever pruned locally
(per class / per image), never globally, and a hard image is free to recruit
whichever concepts happen to fire. This module measures exactly that, per image,
from the two tensors every substrate can hand over (same contract as
concept_usage.py / concept_retention.py):

    weight : (n_classes, n_concepts)   final linear-layer weight
    bias   : (n_classes,) or None
    acts   : (n_examples, n_concepts)  concept activations as fed into that layer
    labels : (n_examples,) optional    for the correct/incorrect split

Per-image sizes (all computed for the predicted class yhat)
-----------------------------------------------------------
n_nonzero(x)   #{j : |W[yhat,j]| > weight_threshold and |a_j(x)| > act_threshold}
               The full explanation, i.e. every bar in a LF-CBM-style contribution
               plot. For dense standardised activations (LF-CBM, VLG-CBM) it equals
               the class row's non-zero count; for a sparse code (UCBM, SAE) it is
               smaller.
k_suff(x)      Minimal sufficient prefix: rank concepts by their signed contribution
               W[yhat,j] a_j(x) (largest support first) and report the smallest k
               such that the top-k concepts alone (every other activation zeroed,
               bias kept) already predict yhat; k = 0 if the bias alone does.
               Answers "how many concepts must a reader be shown before the shown
               concepts alone justify the decision?". Greedy, so an upper bound on
               the true minimum-cardinality sufficient set.
k_mass[p](x)   Smallest k whose top-k |W[yhat,j] a_j(x)| sum to >= p of the total
               absolute contribution to z_yhat. Reader-facing "how many concepts
               explain p of the logit".

Set sizes
---------
global:  n_union (any class non-zero), n_intersection (every class non-zero), nec,
         per-class non-zero counts.
local:   union over the eval set of the k_suff / k_mass explanations
         (local_union_*), plus per-concept explanation frequencies so a "core"
         (appears in >= f of all explanations) can be reported.

Gating: pass `gate` only when it is NOT already reflected in `weight` / `acts`
(DN-CBM GatedProbe stores an ungated weight; UCBM when handing the pre-gate code).
With fold_gate=True it multiplies `weight`; with fold_gate=False it is assumed to be
baked in already and only used as metadata.
"""
import numpy as np

try:  # torch is available in every substrate's env; keep the import soft anyway.
    import torch
except Exception:  # pragma: no cover
    torch = None


def _to_numpy(x, dtype=np.float64):
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _fmt_level(p):
    return format(float(p), "g")


def _pct(v, qs):
    v = np.asarray(v, dtype=np.float64)
    return {f"p{int(q)}": float(np.percentile(v, q)) for q in qs}


def _size_stats(v, size_thresholds):
    v = np.asarray(v, dtype=np.float64)
    out = {"mean": float(v.mean()), "median": float(np.median(v)),
           "max": int(v.max()), "min": int(v.min())}
    out.update(_pct(v, (90, 95, 99)))
    out["frac_gt"] = {str(t): float((v > t).mean()) for t in size_thresholds}
    return out


def local_explanation_sizes(weight, bias, acts, labels=None, *, gate=None, fold_gate=True,
                            mass_levels=(0.9,), size_thresholds=(5, 10, 20, 50),
                            weight_threshold=1e-5, act_threshold=1e-8,
                            core_fracs=(0.5, 0.9), n_extreme=20, extreme_by="k_mass",
                            chunk_elems=40_000_000):
    """Per-image explanation sizes + global/local set sizes. See module docstring.

    Returns a JSON-able dict; the per-image arrays are under "per_example" (lists)
    so callers can dump / plot them, and the `n_extreme` largest examples (by
    `extreme_by`: "k_mass" = first mass level, ranked by |contribution|; or "k_suff",
    ranked by signed contribution) are listed under "extreme" with their
    explanation's concept indices.
    """
    W = _to_numpy(weight)
    if W.ndim != 2:
        raise ValueError(f"weight must be 2D (n_classes, n_concepts), got {W.shape}")
    n_classes, n_concepts = W.shape
    b = np.zeros(n_classes) if bias is None else _to_numpy(bias).reshape(-1)
    A = _to_numpy(acts)
    if A.ndim != 2 or A.shape[1] != n_concepts:
        raise ValueError(f"acts must be (n_examples, {n_concepts}), got {A.shape}")
    n_ex = A.shape[0]
    y = None if labels is None else _to_numpy(labels, dtype=np.int64).reshape(-1)

    g = _to_numpy(gate)
    if g is not None:
        g = g.reshape(-1)
        if g.shape[0] != n_concepts:
            raise ValueError(f"gate must be ({n_concepts},), got {g.shape}")
        if fold_gate:
            W = W * g[None, :]

    # ---- global (weight-only) set sizes -------------------------------------- #
    nonzero = np.abs(W) > weight_threshold                    # (C, J)
    per_class = nonzero.sum(axis=1)
    union_mask = nonzero.any(axis=0)
    inter_mask = nonzero.all(axis=0)

    # ---- per-image quantities ------------------------------------------------ #
    logits = A @ W.T + b[None, :]
    pred = logits.argmax(axis=1)
    correct = None if y is None else (pred == y)

    # contribution of each concept to the predicted-class logit
    contrib = A * W[pred, :]                                   # (N, J)
    nonzero_mask = (np.abs(W[pred, :]) > weight_threshold) & (np.abs(A) > act_threshold)
    n_nonzero = nonzero_mask.sum(axis=1)
    # union of the full (100%-mass) explanations: every concept that has a non-zero term in
    # *some* image's predicted-class logit. Equals the global union for dense activations.
    local_union_nonzero = int(nonzero_mask.any(axis=0).sum())
    del nonzero_mask

    # mass prefix on |contribution|
    abs_sorted = -np.sort(-np.abs(contrib), axis=1)
    csum = np.cumsum(abs_sorted, axis=1)
    total = csum[:, -1:]
    k_mass = {}
    for p in mass_levels:
        # first index where cumulative mass >= p * total; +1 for a count
        hit = csum >= p * total - 1e-12
        k = hit.argmax(axis=1) + 1
        k[total[:, 0] <= 0] = 0
        k_mass[_fmt_level(p)] = k

    # decision-sufficient prefix, ranked by signed support for yhat
    order = np.argsort(-contrib, axis=1)                        # (N, J)
    k_suff = np.empty(n_ex, dtype=np.int64)
    Wf = W.astype(np.float32)
    bsz = max(1, int(chunk_elems // (n_classes * n_concepts)))
    for s in range(0, n_ex, bsz):
        e = min(n_ex, s + bsz)
        o = order[s:e]                                          # (B, J)
        a_sorted = np.take_along_axis(A[s:e], o, axis=1).astype(np.float32)   # (B, J)
        # per-prefix logits: cum[b, c, k] = b_c + sum_{i<=k} W[c, o_i] a_{o_i}
        terms = np.transpose(Wf[:, o], (1, 0, 2)) * a_sorted[:, None, :]   # (B, C, J)
        cum = np.cumsum(terms, axis=2) + b.astype(np.float32)[None, :, None]
        pref_pred = cum.argmax(axis=1)                          # (B, J)
        match = pref_pred == pred[s:e, None]
        # prefix index i holds i+1 concepts -> first matching i gives k = i + 1; the
        # full prefix always matches, so argmax is well-defined. k = 0 if bias alone does.
        bias_only = b.argmax() == pred[s:e]
        k_suff[s:e] = np.where(bias_only, 0, match.argmax(axis=1) + 1)

    # ---- local unions / frequencies ----------------------------------------- #
    def _prefix_mask(k):
        # boolean (N, J) mask of each image's top-k (by signed contribution) concepts
        m = np.zeros((n_ex, n_concepts), dtype=bool)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(n_concepts)[None, :].repeat(n_ex, 0), axis=1)
        m[ranks < k[:, None]] = True
        return m

    suff_mask = _prefix_mask(k_suff)
    suff_freq = suff_mask.mean(axis=0)                          # (J,) frac of images using j
    local_union_suff = int((suff_freq > 0).sum())
    core_suff = {_fmt_level(f): int((suff_freq >= f).sum()) for f in core_fracs}

    # mass explanations are ranked by |contribution|, not signed
    order_abs = np.argsort(-np.abs(contrib), axis=1)
    ranks_abs = np.empty_like(order_abs)
    np.put_along_axis(ranks_abs, order_abs, np.arange(n_concepts)[None, :].repeat(n_ex, 0), axis=1)
    local_union_mass, core_mass, mass_freq = {}, {}, {}
    for key, k in k_mass.items():
        m = ranks_abs < k[:, None]
        f = m.mean(axis=0)
        mass_freq[key] = f
        local_union_mass[key] = int((f > 0).sum())
        core_mass[key] = {_fmt_level(c): int((f >= c).sum()) for c in core_fracs}

    # ---- extreme examples ---------------------------------------------------- #
    if extreme_by == "k_mass":
        first = _fmt_level(mass_levels[0])
        size_by, order_by = k_mass[first], order_abs
    elif extreme_by == "k_suff":
        size_by, order_by = k_suff, order
    else:
        raise ValueError("extreme_by must be 'k_mass' or 'k_suff'")
    top = np.argsort(-size_by, kind="stable")[:n_extreme]
    extreme = []
    for i in top:
        k = int(size_by[i])
        idx = order_by[i, :k]
        extreme.append({
            "index": int(i), "pred": int(pred[i]),
            "label": None if y is None else int(y[i]),
            "correct": None if correct is None else bool(correct[i]),
            "k_suff": int(k_suff[i]), "n_nonzero": int(n_nonzero[i]),
            "k_mass": {kk: int(v[i]) for kk, v in k_mass.items()},
            "ranked_by": extreme_by,
            "concepts": idx.tolist(),
            "contributions": contrib[i, idx].tolist(),
        })

    report = {
        "n_concepts_total": int(n_concepts),
        "n_classes": int(n_classes),
        "n_examples": int(n_ex),
        "acc": None if correct is None else float(correct.mean()),
        "gated": g is not None,
        "weight_threshold": float(weight_threshold),
        "act_threshold": float(act_threshold),
        "mass_levels": [float(p) for p in mass_levels],
        "extreme_by": extreme_by,
        "size_thresholds": [int(t) for t in size_thresholds],
        # global
        "n_union": int(union_mask.sum()),
        "union_frac": float(union_mask.mean()),
        "n_intersection": int(inter_mask.sum()),
        "nec": float(per_class.mean()),
        "nec_median": float(np.median(per_class)),
        "per_class_nonzero_max": int(per_class.max()),
        # local
        "n_nonzero": _size_stats(n_nonzero, size_thresholds),
        "k_suff": _size_stats(k_suff, size_thresholds),
        "k_mass": {k: _size_stats(v, size_thresholds) for k, v in k_mass.items()},
        "local_union_nonzero": local_union_nonzero,
        "local_union_nonzero_frac": local_union_nonzero / n_concepts,
        "local_union_suff": local_union_suff,
        "local_union_suff_frac": local_union_suff / n_concepts,
        "local_core_suff": core_suff,
        "local_union_mass": local_union_mass,
        "local_union_mass_frac": {k: v / n_concepts for k, v in local_union_mass.items()},
        "local_core_mass": core_mass,
        "extreme": extreme,
        "per_example": {
            "pred": pred.tolist(),
            "correct": None if correct is None else correct.tolist(),
            "n_nonzero": n_nonzero.tolist(),
            "k_suff": k_suff.tolist(),
            "k_mass": {k: v.tolist() for k, v in k_mass.items()},
        },
        "concept_freq_suff": suff_freq.tolist(),
        "concept_freq_mass": {k: v.tolist() for k, v in mass_freq.items()},
    }
    if correct is not None:
        for name, arr in (("n_nonzero", n_nonzero), ("k_suff", k_suff)):
            report[name]["correct"] = _size_stats(arr[correct], size_thresholds) if correct.any() else None
            report[name]["incorrect"] = _size_stats(arr[~correct], size_thresholds) if (~correct).any() else None
    return report


def mean_explanation_report(reports):
    """Average `local_explanation_sizes` dicts over seeds: scalar stats are averaged,
    the per-example arrays are concatenated (they are pooled histograms anyway), and
    `extreme` is the union re-sorted by k_suff (each entry tagged with its seed)."""
    reports = [r for r in reports if r]
    if not reports:
        return {}
    if len(reports) == 1:
        out = dict(reports[0])
        out["n_seeds"] = 1
        return out

    def _avg(key):
        return float(np.mean([r[key] for r in reports]))

    def _avg_stats(dicts):
        out = {}
        for k, v in dicts[0].items():
            if v is None:
                out[k] = None
            elif isinstance(v, dict):
                out[k] = _avg_stats([d[k] for d in dicts if d.get(k) is not None]) or None
            else:
                out[k] = float(np.mean([d[k] for d in dicts]))
        return out

    out = dict(reports[0])
    for key in ("acc", "n_union", "union_frac", "n_intersection", "nec", "nec_median",
                "per_class_nonzero_max", "local_union_suff", "local_union_suff_frac",
                "local_union_nonzero", "local_union_nonzero_frac"):
        if reports[0].get(key) is not None:
            out[key] = _avg(key)
    out["n_union_std"] = float(np.std([r["n_union"] for r in reports]))
    out["n_nonzero"] = _avg_stats([r["n_nonzero"] for r in reports])
    out["k_suff"] = _avg_stats([r["k_suff"] for r in reports])
    out["k_mass"] = {k: _avg_stats([r["k_mass"][k] for r in reports]) for k in reports[0]["k_mass"]}
    out["local_core_suff"] = _avg_stats([r["local_core_suff"] for r in reports])
    out["local_union_mass"] = {k: float(np.mean([r["local_union_mass"][k] for r in reports]))
                               for k in reports[0]["local_union_mass"]}
    out["local_union_mass_frac"] = {k: v / reports[0]["n_concepts_total"]
                                    for k, v in out["local_union_mass"].items()}
    out["local_core_mass"] = {k: _avg_stats([r["local_core_mass"][k] for r in reports])
                              for k in reports[0]["local_core_mass"]}
    pe = {"pred": [], "correct": [] if reports[0]["per_example"]["correct"] is not None else None,
          "n_nonzero": [], "k_suff": [], "k_mass": {k: [] for k in reports[0]["k_mass"]}}
    for r in reports:
        p = r["per_example"]
        pe["pred"] += p["pred"]
        if pe["correct"] is not None:
            pe["correct"] += p["correct"]
        pe["n_nonzero"] += p["n_nonzero"]
        pe["k_suff"] += p["k_suff"]
        for k in pe["k_mass"]:
            pe["k_mass"][k] += p["k_mass"][k]
    out["per_example"] = pe
    ext = []
    for si, r in enumerate(reports):
        for e in r["extreme"]:
            ext.append(dict(e, seed_index=si))
    n_extreme = len(reports[0]["extreme"])
    first = _fmt_level(reports[0]["mass_levels"][0])
    key = (lambda e: -e["k_suff"]) if reports[0]["extreme_by"] == "k_suff" else (lambda e: -e["k_mass"][first])
    out["extreme"] = sorted(ext, key=key)[:n_extreme]
    out["concept_freq_suff"] = np.mean([r["concept_freq_suff"] for r in reports], axis=0).tolist()
    out["concept_freq_mass"] = {k: np.mean([r["concept_freq_mass"][k] for r in reports], axis=0).tolist()
                                for k in reports[0]["concept_freq_mass"]}
    # keep the per-seed scalars so a seed-std can be reported next to the mean
    out["per_seed"] = {
        "acc": [r["acc"] for r in reports],
        "n_union": [r["n_union"] for r in reports],
        "nec": [r["nec"] for r in reports],
        "local_union_nonzero": [r["local_union_nonzero"] for r in reports],
        "n_nonzero_mean": [r["n_nonzero"]["mean"] for r in reports],
        "local_union_mass": {k: [r["local_union_mass"][k] for r in reports] for k in reports[0]["local_union_mass"]},
        "k_mass_mean": {k: [r["k_mass"][k]["mean"] for r in reports] for k in reports[0]["k_mass"]},
    }
    out["n_seeds"] = len(reports)
    return out


def format_extreme_examples(report, concept_names=None, class_names=None, max_concepts=None):
    """Human-readable dump of the largest explanations (for the paper's appendix /
    a quick look). `concept_names` / `class_names` are optional index->name lists."""
    def cname(j):
        return concept_names[j] if concept_names is not None and j < len(concept_names) else f"c{j}"

    def yname(c):
        if c is None:
            return "?"
        return class_names[c] if class_names is not None and c < len(class_names) else str(c)

    lines = []
    for e in report.get("extreme", []):
        head = (f"example {e['index']}  pred={yname(e['pred'])}  label={yname(e['label'])}  "
                f"correct={e['correct']}  k_suff={e['k_suff']}  n_nonzero={e['n_nonzero']}  "
                + "  ".join(f"k_mass@{k}={v}" for k, v in e["k_mass"].items()))
        if "seed_index" in e:
            head += f"  seed_index={e['seed_index']}"
        lines.append(head)
        pairs = list(zip(e["concepts"], e["contributions"]))
        if max_concepts is not None:
            pairs = pairs[:max_concepts]
        for j, c in pairs:
            lines.append(f"    {c:+.4f}  {cname(j)}")
        lines.append("")
    return "\n".join(lines)


def _int_bins(v, max_bins=60):
    """Integer-aligned histogram edges (width 1 unless the range needs coarsening)."""
    vmax = int(max(1, np.max(v)))
    w = max(1, int(np.ceil(vmax / max_bins)))
    return np.arange(-0.5, vmax + w + 0.5, w)


def plot_explanation_sizes(report, out_path, *, title=None, dataset_name=None, measure=None):
    """Histogram of per-image explanation size (`measure` in {"k_suff", "n_nonzero",
    "k_mass@<p>"}) with NEC, the global union and the local union marked, so the gap
    between "sparse per class" and "how many concepts a hard image actually uses"
    is visible at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pe = report["per_example"]
    if measure is None:
        measure = f"k_mass@{_fmt_level(report['mass_levels'][0])}"
    if measure.startswith("k_mass@"):
        v = np.asarray(pe["k_mass"][measure.split("@", 1)[1]], dtype=float)
        xlabel = f"# concepts covering {measure.split('@', 1)[1]} of the predicted-class logit"
    elif measure == "n_nonzero":
        v = np.asarray(pe["n_nonzero"], dtype=float)
        xlabel = "# non-zero concept terms in the predicted-class logit"
    else:
        v = np.asarray(pe["k_suff"], dtype=float)
        xlabel = "# concepts needed before the shown concepts alone give the prediction"
    n_concepts = report["n_concepts_total"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(v, bins=_int_bins(v), color="#00376d", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# test images")

    marks = [
        ("NEC (mean per class)", report["nec"], "#2e7d5b", "--"),
        ("median image", float(np.median(v)), "#2e7d5b", ":"),
        ("p99 image", float(np.percentile(v, 99)), "#c0654b", "--"),
        ("max image", float(v.max()), "#c0654b", "-"),
        (f"global union ({int(round(report['n_union']))}/{n_concepts})", report["n_union"], "#555555", "-."),
    ]
    for label, xv, col, ls in marks:
        if xv is None or xv <= 0:
            continue
        ax.axvline(xv, color=col, linestyle=ls, linewidth=1.2, label=f"{label}: {xv:.0f}")

    ttl = title or "Local explanation size"
    if dataset_name:
        ttl += f" -- {dataset_name}"
    ax.set_title(ttl)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summary_line(report, name=""):
    """One-line console/tabular summary."""
    ks, nz = report["k_suff"], report["n_nonzero"]
    km = next(iter(report["k_mass"].values())) if report["k_mass"] else None
    parts = [f"{name:>10s}" if name else "",
             f"J={report['n_concepts_total']}",
             f"acc={report['acc']:.3f}" if report.get("acc") is not None else "",
             f"NEC={report['nec']:.1f}",
             f"union={report['n_union']:.0f} ({100 * report['union_frac']:.0f}%)",
             f"inter={report['n_intersection']:.0f}",
             f"k_suff med/p95/p99/max={ks['median']:.0f}/{ks['p95']:.0f}/{ks['p99']:.0f}/{ks['max']:.0f}",
             f"n_nonzero med/max={nz['median']:.0f}/{nz['max']:.0f}",
             f"local_union(suff)={report['local_union_suff']:.0f}"]
    if km is not None:
        parts.append(f"k_mass med/p99/max={km['median']:.0f}/{km['p99']:.0f}/{km['max']:.0f}")
    return "  ".join(p for p in parts if p)
