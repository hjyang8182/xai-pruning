"""Backbone-agnostic concept-retention analysis.

A threshold-free, magnitude-aware answer to "how many of a CBM's concepts does it
actually need?". Instead of counting gates above 0.5 (fragile when the gate never
goes bimodal -- every gate can sit below 0.5 with accuracy unchanged, because the
linear layer is free to rescale), rank concepts by how much they move predictions
and measure the accuracy you keep as you drop the rest:

    c_j = colnorm(weight[:, j]) * actmag(acts[:, j])          (* gate_j, if separate)

    retain the top-k concepts (zero every other column of `acts`), evaluate, and
    report the smallest k whose accuracy is >= level * full-model accuracy, for
    level in {0.95, 0.99}.

That k is a knee on an ablation curve: threshold-free, and magnitude-aware because
`c_j` folds in both the weight column norm and the typical activation size, so a
mid-value gate on a high-weight concept still counts.

Everything operates on plain tensors any backbone can hand over -- DN-CBM (src/),
LF-CBM, UCBM, VLG-CBM -- exactly like `concept_usage.py`, with which the `c_j`
definition is deliberately shared (`column_norm`/`act_stat` default to the same
l2 * mean|a| contribution used there for `n_effective_mass` / `participation_ratio`).

    weight : (n_classes, n_concepts)   final linear-layer weight
    bias   : (n_classes,) or None      final linear-layer bias
    acts   : (n_examples, n_concepts)  concept activations as fed into that linear
                                       layer (post non-linearity, post any gate
                                       that is already reflected in `weight`)
    labels : (n_examples,)             integer class labels

Gating
------
Pass `gate` (shape (n_concepts,)) only when the model applies a per-concept
multiplier that is *not already* folded into `weight`/`acts` -- e.g. a DN-CBM
GatedProbe whose `.linear.weight` is stored ungated, or UCBM's gated head when you
pass the pre-gate signal. It is multiplied into both `weight` and `acts` up front
so every downstream number reflects the gated model. Do NOT pass it when the gate
is already baked into the weights (LF-CBM arm C `W_g`, VLG-CBM arm C `W_g`) or
when you already handed in the post-gate activations -- that double-counts it.
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


def contribution_scores(weight, acts, gate=None, *, column_norm="l2", act_stat="meanabs"):
    """Per-concept contribution score c_j (see module docstring).

    column_norm : {"l2", "l1"}   norm of each weight column.
    act_stat    : {"meanabs", "std", "rms", "none"}
        summary of |acts[:, j]| used as the activation magnitude. "none" -> 1
        (weight-only ranking, no activation matrix needed).
    """
    W = _to_numpy(weight)
    if W.ndim != 2:
        raise ValueError(f"weight must be 2D (n_classes, n_concepts), got {W.shape}")
    n_concepts = W.shape[1]

    g = _to_numpy(gate)
    if g is not None:
        g = g.reshape(-1)
        if g.shape[0] != n_concepts:
            raise ValueError(f"gate must be ({n_concepts},), got {g.shape}")
        W = W * g[None, :]

    if column_norm == "l2":
        col = np.sqrt((W ** 2).sum(axis=0))
    elif column_norm == "l1":
        col = np.abs(W).sum(axis=0)
    else:
        raise ValueError("column_norm must be 'l2' or 'l1'")

    if act_stat == "none" or acts is None:
        am = np.ones(n_concepts)
    else:
        A = _to_numpy(acts)
        if A.ndim != 2 or A.shape[1] != n_concepts:
            raise ValueError(f"acts must be (n_examples, {n_concepts}), got {A.shape}")
        if g is not None:
            A = A * g[None, :]
        if act_stat == "meanabs":
            am = np.abs(A).mean(axis=0)
        elif act_stat == "std":
            am = A.std(axis=0)
        elif act_stat == "rms":
            am = np.sqrt((A ** 2).mean(axis=0))
        else:
            raise ValueError("act_stat must be 'meanabs', 'std', 'rms' or 'none'")

    return col * am


def _default_grid(n):
    """k values for the display curve: every k up to 32, then ~8% geometric growth,
    always including n. Dense where the knee usually is, cheap in the tail."""
    ks = list(range(1, min(n, 32) + 1))
    k = ks[-1]
    while k < n:
        k = max(k + 1, int(round(k * 1.08)))
        if k < n:
            ks.append(k)
    ks.append(n)
    return sorted(set(ks))


def retention_curve(weight, bias, acts, labels, *, gate=None, fold_gate=True,
                    column_norm="l2", act_stat="meanabs",
                    retain_levels=(0.95, 0.99),
                    ranking="contribution",
                    n_random_trials=5, random_seed=0,
                    ks=None):
    """Rank concepts, sweep the number kept, and locate the retention knees.

    fold_gate : bool
        When a separate `gate` is passed, whether to multiply it into `weight`
        and `acts` before doing anything else (True, for a probe whose stored
        weights are ungated -- DN-CBM GatedProbe). Set False when the gate is
        already baked into the weights (LF-CBM / VLG-CBM arm C `W_g`) but you
        still want ranking="gate" to have the gate vector to sort by.

    ranking : {"contribution", "gate", "weight", "random"}
        "contribution" -> c_j (column_norm * act_stat, gate folded in)
        "gate"         -> gate_j (requires `gate`); the old open-set ordering
        "weight"       -> column_norm(weight[:, j]) only
        "random"       -> shuffled order (a single draw; use n_random_trials for
                          the averaged control curve, always computed regardless)

    Returns a JSON-serialisable dict:
        n_concepts_total, acc_full, ranking, column_norm, act_stat
        ks                    : list[int]           grid of "top-k kept"
        acc                   : list[float]         accuracy at each k
        random_acc_mean/std   : list[float]         random-subset control at each k
        retention_k           : {level: int}        smallest k with acc >= level*acc_full
        retention_frac        : {level: float}      that k / n_concepts_total
        scores                : list[float]         c_j in concept order (for reuse/plots)
    """
    W = _to_numpy(weight, dtype=np.float32)
    b = _to_numpy(bias, dtype=np.float32)
    A = _to_numpy(acts, dtype=np.float32)
    y = _to_numpy(labels, dtype=np.int64)
    if W.ndim != 2:
        raise ValueError(f"weight must be 2D, got {W.shape}")
    n_classes, n_concepts = W.shape
    if A.ndim != 2 or A.shape[1] != n_concepts:
        raise ValueError(f"acts must be (n_examples, {n_concepts}), got {A.shape}")
    if y.shape[0] != A.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and acts ({A.shape[0]}) disagree on n_examples")

    g = _to_numpy(gate)
    if g is not None:
        g = g.reshape(-1)
        if g.shape[0] != n_concepts:
            raise ValueError(f"gate must be ({n_concepts},), got {g.shape}")
        if fold_gate:
            W = W * g[None, :]
            A = A * g[None, :]

    scores = contribution_scores(W, A, None, column_norm=column_norm, act_stat=act_stat)

    rng = np.random.RandomState(random_seed)
    if ranking == "contribution":
        order = np.argsort(-scores, kind="stable")
    elif ranking == "weight":
        wcol = np.sqrt((W ** 2).sum(axis=0)) if column_norm == "l2" else np.abs(W).sum(axis=0)
        order = np.argsort(-wcol, kind="stable")
    elif ranking == "gate":
        if gate is None:
            raise ValueError("ranking='gate' needs `gate`")
        order = np.argsort(-_to_numpy(gate).reshape(-1), kind="stable")
    elif ranking == "random":
        order = rng.permutation(n_concepts)
    else:
        raise ValueError("ranking must be 'contribution', 'gate', 'weight' or 'random'")

    grid = sorted(set(int(k) for k in ks)) if ks is not None else _default_grid(n_concepts)
    if grid[-1] != n_concepts:
        grid.append(n_concepts)

    def _acc_from_logits(L):
        LL = L + b[None, :] if b is not None else L
        return float((LL.argmax(axis=1) == y).mean())

    def sweep_curve(perm, sample_ks):
        """acc at each k in ascending `sample_ks`, keeping perm[:k], by accumulating
        column contributions: logits(k) = logits(k_prev) + A[:, new] @ W[:, new].T.
        The whole curve costs one full A@W.T instead of one matmul per k."""
        L = np.zeros((A.shape[0], n_classes), dtype=A.dtype)
        out, prev = [], 0
        for k in sample_ks:
            if k > prev:
                cols = perm[prev:k]
                L = L + A[:, cols] @ W[:, cols].T
                prev = k
            out.append(_acc_from_logits(L))
        return out

    acc = sweep_curve(order, grid)
    acc_full = acc[-1]  # grid ends at n_concepts

    rand_curves = np.array([sweep_curve(rng.permutation(n_concepts), grid)
                            for _ in range(max(1, n_random_trials))])
    rand_mean, rand_std = [], []
    for j, k in enumerate(grid):
        if k >= n_concepts:
            rand_mean.append(acc_full); rand_std.append(0.0)
        else:
            rand_mean.append(float(rand_curves[:, j].mean()))
            rand_std.append(float(rand_curves[:, j].std()))

    # Exact retention_k: smallest grid point clearing the level, then walk integers
    # up from the previous grid point (acc(top-k) isn't guaranteed monotone, so take
    # the first crossing). The walk accumulates one column at a time -- cheap.
    retention_k, retention_frac = {}, {}
    for level in retain_levels:
        target = level * acc_full
        hit_i = next((i for i, a in enumerate(acc) if a >= target), None)
        if hit_i is None or hit_i == 0:
            retention_k[_fmt_level(level)] = int(grid[hit_i]) if hit_i == 0 else n_concepts
        else:
            lo, hi = grid[hit_i - 1], grid[hit_i]
            fine = sweep_curve(order, list(range(lo, hi + 1)))
            k_star = hi
            for off, a in enumerate(fine):
                if a >= target:
                    k_star = lo + off
                    break
            retention_k[_fmt_level(level)] = int(k_star)
        retention_frac[_fmt_level(level)] = float(
            retention_k[_fmt_level(level)] / n_concepts) if n_concepts else 0.0

    return {
        "n_concepts_total": int(n_concepts),
        "n_classes": int(n_classes),
        "n_examples": int(A.shape[0]),
        "ranking": ranking,
        "column_norm": column_norm,
        "act_stat": act_stat,
        "gate_folded": bool(g is not None and fold_gate),
        "acc_full": float(acc_full),
        "ks": [int(k) for k in grid],
        "acc": [float(a) for a in acc],
        "random_acc_mean": rand_mean,
        "random_acc_std": rand_std,
        "n_random_trials": int(max(1, n_random_trials)),
        "random_seed": int(random_seed),
        "retain_levels": [float(l) for l in retain_levels],
        "retention_k": retention_k,
        "retention_frac": retention_frac,
        "scores": [float(s) for s in scores],
    }


def mean_retention(reports):
    """Average a list of `retention_curve` dicts over seeds. Curves share the grid
    (same n_concepts), so `acc` / `random_acc_*` are averaged elementwise and
    `retention_k` is averaged per level (kept as a float -- it's a mean of ints)."""
    reports = [r for r in reports if r]
    if not reports:
        return {}
    if len({tuple(r["ks"]) for r in reports}) != 1:
        raise ValueError("reports have different k grids; can't average elementwise")
    out = dict(reports[0])
    for key in ("acc", "random_acc_mean"):
        out[key] = np.mean([r[key] for r in reports], axis=0).tolist()
    out["acc_std"] = np.std([r["acc"] for r in reports], axis=0).tolist()
    out["random_acc_std"] = np.mean([r["random_acc_std"] for r in reports], axis=0).tolist()
    out["acc_full"] = float(np.mean([r["acc_full"] for r in reports]))
    out["retention_k"] = {k: float(np.mean([r["retention_k"][k] for r in reports]))
                          for k in reports[0]["retention_k"]}
    out["retention_k_std"] = {k: float(np.std([r["retention_k"][k] for r in reports]))
                              for k in reports[0]["retention_k"]}
    out["retention_frac"] = {k: float(np.mean([r["retention_frac"][k] for r in reports]))
                             for k in reports[0]["retention_frac"]}
    out["scores"] = np.mean([r["scores"] for r in reports], axis=0).tolist()
    out["n_seeds"] = len(reports)
    return out


# --------------------------------------------------------------------------- #
#  Gate-threshold sensitivity                                                  #
# --------------------------------------------------------------------------- #

def _cea(n_classes, n_concepts, acc, beta=0.25):
    if n_concepts <= 1:
        return float(acc)
    k = np.ceil(np.log2(max(n_classes, 2)))
    return float(acc / (np.log(n_concepts) / np.log(k)) ** beta)


def threshold_sweep(weight, bias, acts, labels, gate, *, fold_gate=True,
                    taus=None, report_taus=(0.1, 0.3, 0.5, 0.7, 0.9),
                    plateau_tol=0.10, beta=0.25):
    """How much the open-concept count and the accuracy depend on where the
    gate open/closed threshold tau is put -- the diagnostic for a gate that
    never went bimodal, where `n_open @ 0.5` is an arbitrary number.

    At each tau: concepts with gate <= tau are hard-zeroed (their column of
    `acts` set to 0); n_open(tau) = #(gate > tau); acc(tau) is the resulting
    top-1 accuracy. Survivors keep whatever gate value the model learned, so
    this measures how well a hard cut at tau matches the trained soft model.

    gate       : (n_concepts,)  the per-concept gate values sigmoid(logits / T).
    fold_gate  : multiply `gate` into `weight`/`acts` first (True for a probe
                 whose stored weights are ungated; False when the gate is
                 already baked into the weights, LF-CBM / VLG-CBM arm C).
    taus       : thresholds to evaluate. Default: a 0..1 grid of ~200 points
                 unioned with the sorted unique gate values, so every step in
                 n_open(tau) is captured.

    Returns a JSON-serialisable dict:
        taus, n_open, acc, cea            per-tau curves
        acc_full, n_concepts_total
        n_open_at / acc_at / cea_at       dicts keyed by report_taus
        plateau_width                     width of the largest tau-interval
                                          around 0.5 where n_open stays within
                                          +/- plateau_tol of n_open(0.5)
        n_open_range_mid                  n_open(0.3) - n_open(0.7)  (spread of
                                          the count over a plausible tau band)
        acc_drop_mid                      acc(min tau) - acc(0.5)?  -> reported
                                          as acc(0.3) - acc(0.7)
        bimodal_mushy_frac               fraction of gates in (0.05, 0.95)
    """
    W = _to_numpy(weight, dtype=np.float32)
    b = _to_numpy(bias, dtype=np.float32)
    A = _to_numpy(acts, dtype=np.float32)
    y = _to_numpy(labels, dtype=np.int64)
    g = _to_numpy(gate).reshape(-1)
    n_classes, n_concepts = W.shape
    if g.shape[0] != n_concepts:
        raise ValueError(f"gate must be ({n_concepts},), got {g.shape}")
    if fold_gate:
        W = W * g[None, :]
        A = A * g[None, :]

    if taus is None:
        # a uniform 0..1 grid, plus gate-value quantiles so the steep parts of
        # n_open(tau) are still well sampled without one tau per concept.
        taus = np.unique(np.concatenate([
            np.linspace(0.0, 1.0, 201),
            np.clip(np.quantile(g, np.linspace(0.0, 1.0, 101)), 0.0, 1.0),
        ]))
    taus = np.asarray(taus, dtype=np.float64)

    # kept(tau) = {j : g_j > tau} is a prefix of the gate-descending order, and it
    # only grows as tau falls -- so accumulate column contributions along that order
    # and read the curve off at each tau's open count. One full A@W.T for the sweep.
    gorder = np.argsort(-g, kind="stable")
    n_open = np.array([int((g > t).sum()) for t in taus])

    def _acc_from_logits(L):
        LL = L + b[None, :] if b is not None else L
        return float((LL.argmax(axis=1) == y).mean())

    order_by_open = np.argsort(n_open, kind="stable")
    acc_arr = np.empty(len(taus), dtype=np.float64)
    L = np.zeros((A.shape[0], n_classes), dtype=A.dtype)
    prev = 0
    for i in order_by_open:
        k = n_open[i]
        if k > prev:
            cols = gorder[prev:k]
            L = L + A[:, cols] @ W[:, cols].T
            prev = k
        acc_arr[i] = _acc_from_logits(L)
    acc = acc_arr
    if prev < n_concepts:
        L = L + A[:, gorder[prev:]] @ W[:, gorder[prev:]].T
    acc_full = _acc_from_logits(L)
    cea = np.array([_cea(n_classes, max(no, 1), a, beta) for no, a in zip(n_open, acc)])

    def at(t):
        i = int(np.argmin(np.abs(taus - t)))
        return i

    n_open_at = {_fmt_level(t): int(n_open[at(t)]) for t in report_taus}
    acc_at = {_fmt_level(t): float(acc[at(t)]) for t in report_taus}
    cea_at = {_fmt_level(t): float(cea[at(t)]) for t in report_taus}

    # plateau: contiguous run of taus through 0.5 where n_open stays within tol
    i50 = at(0.5)
    base = max(n_open[i50], 1)
    lo = i50
    while lo > 0 and abs(n_open[lo - 1] - n_open[i50]) <= plateau_tol * base:
        lo -= 1
    hi = i50
    while hi < len(taus) - 1 and abs(n_open[hi + 1] - n_open[i50]) <= plateau_tol * base:
        hi += 1
    plateau_width = float(taus[hi] - taus[lo])

    mushy = float(np.mean((g > 0.05) & (g < 0.95)))

    return {
        "n_concepts_total": int(n_concepts),
        "n_classes": int(n_classes),
        "n_examples": int(A.shape[0]),
        "gate_folded": bool(fold_gate),
        "beta": float(beta),
        "acc_full": float(acc_full),
        "taus": [float(t) for t in taus],
        "n_open": [int(v) for v in n_open],
        "acc": [float(v) for v in acc],
        "cea": [float(v) for v in cea],
        "report_taus": [float(t) for t in report_taus],
        "n_open_at": n_open_at,
        "acc_at": acc_at,
        "cea_at": cea_at,
        "plateau_width": plateau_width,
        "plateau_tol": float(plateau_tol),
        "n_open_range_mid": int(n_open[at(0.3)] - n_open[at(0.7)]),
        "acc_drop_mid": float(acc[at(0.3)] - acc[at(0.7)]),
        "bimodal_mushy_frac": mushy,
    }


def mean_threshold_sweep(reports):
    """Average `threshold_sweep` dicts over seeds. They share the tau grid only
    if `taus` was passed explicitly; otherwise the grids differ (gate values
    vary by seed) and the curves are resampled onto the first report's taus."""
    reports = [r for r in reports if r]
    if not reports:
        return {}
    base_taus = np.asarray(reports[0]["taus"])
    out = dict(reports[0])

    def resamp(r, key):
        return np.interp(base_taus, np.asarray(r["taus"]), np.asarray(r[key]))

    for key in ("n_open", "acc", "cea"):
        stack = np.stack([resamp(r, key) for r in reports])
        out[key] = stack.mean(axis=0).tolist()
        out[key + "_std"] = stack.std(axis=0).tolist()
    out["acc_full"] = float(np.mean([r["acc_full"] for r in reports]))
    for k in ("plateau_width", "n_open_range_mid", "acc_drop_mid", "bimodal_mushy_frac"):
        vals = [r[k] for r in reports]
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    for k in ("n_open_at", "acc_at", "cea_at"):
        out[k] = {t: float(np.mean([r[k][t] for r in reports])) for t in reports[0][k]}
    out["n_seeds"] = len(reports)
    return out


def plot_threshold_sensitivity(report, out_path, *, title=None, dataset_name=None):
    """Twin-axis: n_open(tau) and acc(tau) vs the gate threshold tau, with tau=0.5
    marked. A long flat n_open plateau => the count is threshold-robust; a steep
    slope through 0.5 => it isn't. `report` is one `threshold_sweep` dict or a
    `mean_threshold_sweep` dict (uses *_std bands if present)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = np.array(report["taus"], dtype=float)
    n_open = np.array(report["n_open"], dtype=float)
    acc = np.array(report["acc"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(taus, acc, color="#00376d", linewidth=2.0, label="accuracy")
    if report.get("acc_std"):
        s = np.array(report["acc_std"], dtype=float)
        ax1.fill_between(taus, acc - s, acc + s, color="#00376d", alpha=0.15, linewidth=0)
    ax1.axhline(report["acc_full"], color="#555555", linestyle=":", linewidth=1.1, label="full model")
    ax1.set_xlabel(r"gate threshold $\tau$")
    ax1.set_ylabel("test accuracy", color="#00376d")
    ax1.tick_params(axis="y", labelcolor="#00376d")

    ax2 = ax1.twinx()
    ax2.plot(taus, n_open, color="#c0654b", linewidth=1.8, linestyle="--", label="open concepts")
    if report.get("n_open_std"):
        s = np.array(report["n_open_std"], dtype=float)
        ax2.fill_between(taus, n_open - s, n_open + s, color="#c0654b", alpha=0.12, linewidth=0)
    ax2.set_ylabel("open concepts (gate > τ)", color="#c0654b")
    ax2.tick_params(axis="y", labelcolor="#c0654b")
    ax2.grid(False)

    ax1.axvline(0.5, color="#2e7d5b", linewidth=1.0, alpha=0.7)
    ax1.annotate(f"plateau width {report['plateau_width']:.2f}\n"
                 f"mushy-gate frac {report['bimodal_mushy_frac']:.2f}",
                 (0.02, 0.02), xycoords="axes fraction", fontsize=9, va="bottom")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=10, frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, 0.12))
    ttl = title or "Gate threshold sensitivity"
    if dataset_name:
        ttl += f" -- {dataset_name}"
    ax1.set_title(ttl)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_retention_curve(report, out_path, *, title=None, dataset_name=None):
    """Accuracy vs. #concepts kept, with the random-subset control, the
    95%/99%-of-full lines, and the retention_k markers. `report` is one
    `retention_curve` dict or a `mean_retention` dict (uses acc_std if present)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = np.array(report["ks"], dtype=float)
    acc = np.array(report["acc"], dtype=float)
    acc_full = report["acc_full"]
    rmean = np.array(report["random_acc_mean"], dtype=float)
    rstd = np.array(report["random_acc_std"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if report.get("acc_std"):
        astd = np.array(report["acc_std"], dtype=float)
        ax.fill_between(ks, acc - astd, acc + astd, color="#00376d", alpha=0.15, linewidth=0)
    ax.plot(ks, acc, color="#00376d", marker="o", markersize=4, markeredgecolor="white",
            linewidth=2.0, label=f"top-k by {report['ranking']}")
    ax.fill_between(ks, rmean - rstd, rmean + rstd, color="#c0654b", alpha=0.12, linewidth=0)
    ax.plot(ks, rmean, color="#c0654b", marker="D", markersize=3, linewidth=1.5,
            linestyle="--", label="random k-subset")

    ax.axhline(acc_full, color="#555555", linestyle=":", linewidth=1.2, label="full model")
    for level in report["retain_levels"]:
        key = _fmt_level(level)
        yk = level * acc_full
        ax.axhline(yk, color="#888888", linestyle="--", linewidth=0.9)
        kk = report["retention_k"][key]
        ax.axvline(kk, color="#2e7d5b", linestyle="-", linewidth=1.0, alpha=0.7)
        ax.annotate(f"{int(round(kk))} @ {level:g}", (kk, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(3, 6), fontsize=9, color="#2e7d5b")

    ax.set_xscale("log")
    ax.set_xlabel("# concepts kept (ranked by contribution)")
    ax.set_ylabel("test accuracy")
    ttl = title or "Concept retention"
    if dataset_name:
        ttl += f" -- {dataset_name}"
    ax.set_title(ttl)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# tau as a hyperparameter: pick the open/closed threshold that maximises CEA
# (or the largest tau within a tolerance of the full model) on a held-out
# selection split, refitting the linear head on the survivors.
# ---------------------------------------------------------------------------

def _split_fit_select(n, select_frac, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_sel = max(1, int(round(select_frac * n)))
    return perm[n_sel:], perm[:n_sel]


def refit_head(A_fit, y_fit, keep, *, W_init, b_init, A_sel=None, y_sel=None,
               epochs=20, lr=1e-3, batch_size=4096, weight_decay=0.0,
               device=None, seed=0):
    """Re-train a linear head (CE loss, Adam) on the surviving concept columns
    `keep` only, warm-started from the columns of `W_init`/`b_init`. Columns
    outside `keep` stay exactly zero. If a selection split is given, the epoch
    with the best selection accuracy is returned (early stopping).

    A_fit / A_sel : (n, n_concepts) the exact input of the linear layer
    W_init        : (n_classes, n_concepts), gate already folded in if any.
    Returns (W, b, sel_acc_best) as float32 numpy, W full-width."""
    if torch is None:
        raise RuntimeError("refit_head needs torch")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    keep = np.asarray(keep)
    keep_idx = np.flatnonzero(keep) if keep.dtype == bool else keep
    n_classes, n_concepts = W_init.shape
    if keep_idx.size == 0:  # nothing left: bias-only model
        W = np.zeros((n_classes, n_concepts), dtype=np.float32)
        b = np.asarray(b_init, dtype=np.float32) if b_init is not None else np.zeros(n_classes, np.float32)
        acc = float((np.argmax(np.broadcast_to(b, (len(y_sel), n_classes)), 1) == y_sel).mean()) if A_sel is not None else None
        return W, b, acc

    torch.manual_seed(seed)
    kt = torch.as_tensor(keep_idx, dtype=torch.long)
    Af = torch.as_tensor(np.asarray(A_fit, dtype=np.float32))[:, kt]
    yf = torch.as_tensor(np.asarray(y_fit, dtype=np.int64))
    lin = torch.nn.Linear(len(keep_idx), n_classes).to(device)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(np.asarray(W_init, dtype=np.float32))[:, kt])
        lin.bias.copy_(torch.as_tensor(np.asarray(b_init, dtype=np.float32)) if b_init is not None
                       else torch.zeros(n_classes))
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss()

    if A_sel is not None:
        As = torch.as_tensor(np.asarray(A_sel, dtype=np.float32))[:, kt].to(device)
        ys = torch.as_tensor(np.asarray(y_sel, dtype=np.int64)).to(device)

        def sel_acc():
            with torch.no_grad():
                accs = []
                for i in range(0, As.shape[0], 65536):
                    accs.append((lin(As[i:i + 65536]).argmax(1) == ys[i:i + 65536]).float().sum())
                return float(torch.stack(accs).sum() / As.shape[0])
        best_acc, best_state = sel_acc(), {k: v.detach().clone() for k, v in lin.state_dict().items()}
    else:
        best_acc, best_state = None, None

    on_device = Af.numel() * 4 < 4e9  # keep the fit matrix resident on the GPU when it is small
    if on_device:
        Af, yf = Af.to(device), yf.to(device)
    n = Af.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=Af.device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Af[idx], yf[idx]
            if not on_device:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(lin(xb), yb).backward()
            opt.step()
        if A_sel is not None:
            a = sel_acc()
            if a > best_acc:
                best_acc, best_state = a, {k: v.detach().clone() for k, v in lin.state_dict().items()}
    if best_state is not None:
        lin.load_state_dict(best_state)
    W = np.zeros((n_classes, n_concepts), dtype=np.float32)
    W[:, keep_idx] = lin.weight.detach().cpu().numpy()
    b = lin.bias.detach().cpu().numpy().astype(np.float32)
    return W, b, best_acc


def _candidate_taus(g, n_tau, always=(0.5,)):
    """~n_tau thresholds whose open counts are spread log-uniformly between 1
    and the dictionary size (the interesting part of the curve is at the low
    counts), plus tau=0 (everything with g>0 open) and any `always` values."""
    fine = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001),
                                     np.clip(np.quantile(g, np.linspace(0, 1, 1001)), 0, 1)]))
    n_open = np.array([(g > t).sum() for t in fine])
    n_total = len(g)
    targets = np.unique(np.round(np.geomspace(1, max(n_total, 2), n_tau)).astype(int))
    picked = {0.0}
    for k in targets:
        i = int(np.argmin(np.abs(n_open - k)))
        picked.add(float(fine[i]))
    picked.update(float(t) for t in always)
    return np.array(sorted(picked))


def select_tau(weight, bias, gate, *, fit, select, evaluate=None, fold_gate=True,
               objective="cea", tol=0.01, beta=0.25, n_tau=25, refit=True,
               refit_kwargs=None, reference_taus=(0.5,), device=None):
    """Treat the gate threshold tau as a hyperparameter.

    For each candidate tau the concepts with gate <= tau are removed and the
    head is (optionally) refit on the survivors using `fit`; the objective is
    measured on `select`; the chosen tau* is then scored on `evaluate` (test).

    fit / select / evaluate : (acts, labels) tuples; acts are the exact input of
        the linear layer (see module docstring), labels int.
    objective : "cea"  -> tau* = argmax CEA(n_open, acc_select)
                "tol"  -> tau* = largest tau with acc_select >= (1 - tol) * acc_select(full)
    refit     : refit the head on the survivors (recommended). The "nofit"
                curves -- hard cut with the trained soft gate values kept -- are
                always reported too.

    Returns a JSON-serialisable dict with the candidate curves, the choice, and
    the evaluation of the chosen model (refit and nofit) next to the soft full
    model and each `reference_taus` (0.5 by default) for comparison."""
    W = _to_numpy(weight, dtype=np.float32)
    b = _to_numpy(bias, dtype=np.float32)
    g = _to_numpy(gate).reshape(-1)
    n_classes, n_concepts = W.shape
    if g.shape[0] != n_concepts:
        raise ValueError(f"gate must be ({n_concepts},), got {g.shape}")
    if fold_gate:
        W = W * g[None, :].astype(np.float32)

    def prep(split):
        if split is None:
            return None
        A, y = split
        A = _to_numpy(A, dtype=np.float32)
        if fold_gate:
            A = A * g[None, :].astype(np.float32)
        return A, _to_numpy(y, dtype=np.int64)

    fit, select, evaluate = prep(fit), prep(select), prep(evaluate)
    if b is None:
        b = np.zeros(n_classes, dtype=np.float32)

    def acc_of(Wm, bm, split, keep=None):
        A, y = split
        if keep is None:
            L = A @ Wm.T
        else:
            L = A[:, keep] @ Wm[:, keep].T
        return float(((L + bm[None, :]).argmax(1) == y).mean())

    rk = dict(epochs=20, lr=1e-3, batch_size=4096, weight_decay=0.0)
    rk.update(refit_kwargs or {})

    taus = _candidate_taus(g, n_tau, always=reference_taus)
    rows = []
    sel_full = acc_of(W, b, select)
    for t in taus:
        keep = g > t
        n_open = int(keep.sum())
        row = {"tau": float(t), "n_open": n_open,
               "sel_acc_nofit": acc_of(W, b, select, keep)}
        row["sel_cea_nofit"] = _cea(n_classes, max(n_open, 1), row["sel_acc_nofit"], beta)
        if evaluate is not None:
            row["eval_acc_nofit"] = acc_of(W, b, evaluate, keep)
            row["eval_cea_nofit"] = _cea(n_classes, max(n_open, 1), row["eval_acc_nofit"], beta)
        if refit:
            Wr, br, sel_acc = refit_head(fit[0], fit[1], keep, W_init=W, b_init=b,
                                         A_sel=select[0], y_sel=select[1], device=device, **rk)
            row["sel_acc_refit"] = float(sel_acc)
            row["sel_cea_refit"] = _cea(n_classes, max(n_open, 1), sel_acc, beta)
            if evaluate is not None:
                row["eval_acc_refit"] = acc_of(Wr, br, evaluate, keep)
                row["eval_cea_refit"] = _cea(n_classes, max(n_open, 1), row["eval_acc_refit"], beta)
        rows.append(row)

    mode = "refit" if refit else "nofit"
    sel_acc_key, sel_cea_key = f"sel_acc_{mode}", f"sel_cea_{mode}"
    if objective == "cea":
        i_star = int(np.argmax([r[sel_cea_key] for r in rows]))
    elif objective == "tol":
        ok = [i for i, r in enumerate(rows) if r[sel_acc_key] >= (1.0 - tol) * sel_full]
        i_star = max(ok, key=lambda i: rows[i]["tau"]) if ok else 0
    else:
        raise ValueError(f"unknown objective {objective}")
    chosen = dict(rows[i_star])

    refs = {}
    for t in reference_taus:
        i = int(np.argmin(np.abs(taus - t)))
        refs[_fmt_level(t)] = dict(rows[i])

    out = {
        "n_concepts_total": int(n_concepts), "n_classes": int(n_classes),
        "n_fit": int(fit[0].shape[0]), "n_select": int(select[0].shape[0]),
        "n_eval": int(evaluate[0].shape[0]) if evaluate is not None else None,
        "gate_folded": bool(fold_gate), "objective": objective, "tol": float(tol),
        "beta": float(beta), "refit": bool(refit), "refit_kwargs": rk,
        "sel_acc_full": sel_full,
        "eval_acc_full": acc_of(W, b, evaluate) if evaluate is not None else None,
        "candidates": rows,
        "tau_star": chosen["tau"], "chosen": chosen,
        "reference": refs,
        "bimodal_mushy_frac": float(np.mean((g > 0.05) & (g < 0.95))),
    }
    if evaluate is not None:
        out["eval_cea_full"] = _cea(n_classes, max(int((g > 0).sum()), 1), out["eval_acc_full"], beta)
    return out


def select_tau_crossfit(weight, bias, gate, *, fit, evaluate, n_folds=2, seed=0, **kw):
    """`select_tau` with the selection split carved out of `evaluate` rather than
    out of the training data: the soft head has (over)fit its training set, so a
    train-held-out selection split rewards keeping everything. Splits `evaluate`
    into `n_folds` folds, selects tau on one fold and scores on the rest, for
    each fold in turn, and averages (every example is scored exactly n_folds-1
    times, never by a tau chosen on itself). The head refit still uses `fit`.
    Returns the averaged dict (see `mean_select_tau`) with `per_fold` attached."""
    A, y = evaluate
    n = len(_to_numpy(y))
    perm = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(perm, n_folds)
    reports = []
    for f, sel_idx in enumerate(folds):
        ev_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        A_np = _to_numpy(A, dtype=np.float32)
        y_np = _to_numpy(y, dtype=np.int64)
        rep = select_tau(weight, bias, gate, fit=fit,
                         select=(A_np[sel_idx], y_np[sel_idx]),
                         evaluate=(A_np[ev_idx], y_np[ev_idx]), **kw)
        rep["fold"] = f
        reports.append(rep)
    out = mean_select_tau(reports)
    out["per_fold"] = [{k: r[k] for k in ("fold", "tau_star", "chosen", "sel_acc_full", "eval_acc_full")}
                       for r in reports]
    out["n_folds"] = n_folds
    return out


def mean_select_tau(reports):
    """Average `select_tau` dicts over seeds (the chosen point and the
    references; candidate curves are kept from the first report only)."""
    reports = [r for r in reports if r]
    if not reports:
        return {}
    out = dict(reports[0])
    scalar_keys = [k for k, v in reports[0]["chosen"].items() if isinstance(v, (int, float))]
    out["chosen"] = {k: float(np.mean([r["chosen"][k] for r in reports])) for k in scalar_keys}
    out["chosen_std"] = {k: float(np.std([r["chosen"][k] for r in reports])) for k in scalar_keys}
    out["tau_star"] = out["chosen"]["tau"]
    out["reference"] = {t: {k: float(np.mean([r["reference"][t][k] for r in reports])) for k in scalar_keys}
                        for t in reports[0]["reference"]}
    for k in ("sel_acc_full", "eval_acc_full", "eval_cea_full", "bimodal_mushy_frac"):
        if reports[0].get(k) is not None:
            out[k] = float(np.mean([r[k] for r in reports]))
    out["per_seed_chosen"] = [r["chosen"] for r in reports]
    out["n_seeds"] = len(reports)
    return out


def plot_tau_selection(report, out_path, *, title=None, dataset_name=None):
    """Left: selection-split accuracy vs open concepts, refit and hard-cut-only,
    with the soft full model and the chosen point. Right: the selection
    objective (CEA) vs tau with tau* marked. One y-axis per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["candidates"]
    n_open = np.array([r["n_open"] for r in rows], dtype=float)
    taus = np.array([r["tau"] for r in rows], dtype=float)
    refit = report["refit"]
    c_refit, c_nofit, ink2 = "#2a78d6", "#eb6834", "#52514e"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    order = np.argsort(n_open)
    ax.plot(n_open[order], [rows[i]["sel_acc_nofit"] for i in order], color=c_nofit, marker="o",
            markersize=4, linewidth=1.6, label="hard cut, soft head kept")
    if refit:
        ax.plot(n_open[order], [rows[i]["sel_acc_refit"] for i in order], color=c_refit, marker="o",
                markersize=4, linewidth=2.0, label="hard cut + refit head")
    ax.axhline(report["sel_acc_full"], color=ink2, linestyle=":", linewidth=1.0, label="soft full model")
    ch = report["chosen"]
    ax.scatter([ch["n_open"]], [ch["sel_acc_refit" if refit else "sel_acc_nofit"]],
               s=90, facecolors="none", edgecolors="#0b0b0b", linewidths=1.6, zorder=5)
    ax.annotate(f"τ*={ch['tau']:.2f}: {int(round(ch['n_open']))} open",
                (ch["n_open"], ch["sel_acc_refit" if refit else "sel_acc_nofit"]),
                xytext=(8, -14), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("open concepts (gate > τ)")
    ax.set_ylabel("selection-split accuracy")
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    key = "sel_cea_refit" if refit else "sel_cea_nofit"
    ax2.plot(taus, [r[key] for r in rows], color=c_refit if refit else c_nofit, marker="o",
             markersize=4, linewidth=2.0)
    ax2.axvline(ch["tau"], color="#0b0b0b", linewidth=1.0, linestyle="--")
    ax2.axvline(0.5, color=ink2, linewidth=1.0, linestyle=":")
    ax2.text(0.5, ax2.get_ylim()[0], " τ=0.5", color=ink2, fontsize=8, va="bottom")
    ax2.set_xlabel("gate threshold τ")
    ax2.set_ylabel(f"selection-split CEA (β={report['beta']:g})" + (" after refit" if refit else ""))
    ax2.set_xlim(0, 1)
    for a in (ax, ax2):
        a.grid(True, color="#e6e5e0", linewidth=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ttl = title or "Threshold selection"
    if dataset_name:
        ttl += f" -- {dataset_name}"
    fig.suptitle(ttl, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_tau_summary_line(rep, label=""):
    ch, ref = rep["chosen"], rep["reference"].get("0.5")
    mode = "refit" if rep["refit"] else "nofit"
    s = (f"[{label}] tau*={ch['tau']:.3f} ({rep['objective']}) -> {ch['n_open']:.0f}/{rep['n_concepts_total']} open, "
         f"sel acc {ch[f'sel_acc_{mode}']:.4f}")
    if rep.get("eval_acc_full") is not None:
        s += (f" | eval: full(soft) {rep['eval_acc_full']:.4f}, tau* {mode} {ch[f'eval_acc_{mode}']:.4f}"
              f", tau* nofit {ch['eval_acc_nofit']:.4f}")
        if ref is not None:
            s += f" | @0.5: {ref['n_open']:.0f} open, nofit {ref['eval_acc_nofit']:.4f}"
            if rep["refit"]:
                s += f", refit {ref['eval_acc_refit']:.4f}"
    return s
