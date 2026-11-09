"""Standard (supervised) concept bottleneck models on CUB with the GCG gate on the bottleneck: does the gate
reduce leakage, and what does it do to interventions?

Concepts are CUB's 312 attributes with CUB's own per-image annotations (default), or with --concepts koh112 /
--labels class Koh et al.'s 112 with their species majority-vote labels
(concept_accuracy_pipelines.class_level_attributes). Every bottleneck unit *is* an attribute, so CTL/ICL need
no concept-attribute matching. The input is the frozen CLIP RN50 image embedding (DN-CBM's backbone,
data/cub/activations/*_clip_features.pt), standardised on train; the concept predictor is an MLP on it.

Models (one per --modes entry, each at every --lambda-gates value and --seeds seed):
  sequential  x -> c trained alone (BCE, then frozen); c_hat -> y head trained on the predicted probabilities
  independent x -> c as sequential, but the head (and gate) are trained on the *true* concepts (Koh et al.)
  joint       x -> c -> y end to end, CE + alpha * sum_j BCE_j (Koh et al.'s joint CBM), one run per --alphas
The bottleneck is c_hat = sigmoid(concept logits). With lambda_gate > 0 the head sees c_hat * m, m the hard
straight-through gate of gating.GatedProbe (1[sigmoid(g) > 0.5] forward, sigmoid gradient), and the loss adds
lambda_gate * sum_j sigmoid(g_j); lambda_gate = 0 is the ungated CBM (no gate parameter). In the joint model the
gate is trained with everything else, so it can change what the bottleneck encodes, not only which units the
head reads. After training the head is refit on the open concepts (concept predictor frozen) -- GCG's
mask-and-refit -- and both heads are reported.

Per run, on the test images:
  task accuracy (gated head as trained, and refit head)
  concept balanced accuracy (per concept, mean of TPR/TNR at 0.5), over all 112 and over the open set
  CTL per concept and ICL per pair, hard (c_hat > 0.5) and soft (--n-bins quantile bins of c_hat) encodings
  intervention curve: accuracy when k concepts (random order, --n-orders orders) are set to their true 0/1 value
Summary per (mode, alpha, lambda_gate, seed > baseline), with K = the open set:
  gated model on K   vs   the same seed's ungated model on K (same concepts: does the gate change leakage?)
                     vs   the ungated model on all 112 (the model you would otherwise deploy)
                     vs   the ungated model on random |K| subsets (selection alone)

    python cub_cbm_gate.py                                    # defaults: sequential + joint, 3 seeds
    python cub_cbm_gate.py --modes joint --alphas 0.01 --lambda-gates 0 0.01 --seeds 0
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from concept_accuracy_pipelines import KOH_112, class_level_attributes
from concept_leakage import _codes, discretise_soft, entropy, mutual_information
from concept_leakage_pipelines import load_cub_attributes
from gating import hard_gate_ste

ROOT = os.path.dirname(os.path.abspath(__file__))


def split_ids(cub_root, train):
    """CUB image ids of a split in images.txt order -- the row order of the cached CLIP features."""
    def read(name):
        with open(os.path.join(cub_root, name)) as f:
            return dict(line.split()[:2] for line in f)
    split = read("train_test_split.txt")
    with open(os.path.join(cub_root, "images.txt")) as f:
        return np.array([int(line.split()[0]) for line in f if split[line.split()[0]] == ("1" if train else "0")])


def load_data(args):
    attrs, names = load_cub_attributes(args.cub_root)
    if args.labels == "class":   # species majority vote (Koh et al.); --concepts all keeps every attribute
        labels, kept = class_level_attributes(attrs, args.cub_root, None if args.concepts == "koh112" else 0)
    else:                        # CUB's own per-image annotations
        kept = np.array(KOH_112) if args.concepts == "koh112" else np.arange(attrs.shape[1])
        labels = attrs[:, kept]
    out = {"concept_names": [names[i] for i in kept]}
    for split in ("train", "test"):
        d = torch.load(os.path.join(args.features_dir, f"{split}_clip_features.pt"), map_location="cpu")
        ids = split_ids(args.cub_root, split == "train")
        assert len(ids) == len(d["features"]), (split, len(ids), len(d["features"]))
        out[split] = {"x": d["features"].float(), "y": torch.as_tensor(d["labels"]).long(),
                      "c": torch.from_numpy(labels[ids]).float()}
    if args.val_frac > 0:  # species-stratified hold-out from train for choosing lambda (fixed across seeds)
        rng = np.random.default_rng(0)
        y = out["train"]["y"].numpy()
        val = np.zeros(len(y), dtype=bool)
        for k in np.unique(y):
            idx = np.flatnonzero(y == k)
            val[rng.choice(idx, max(1, int(round(args.val_frac * len(idx)))), replace=False)] = True
        full = out["train"]
        out["val"] = {k: v[torch.from_numpy(val)] for k, v in full.items()}
        out["train"] = {k: v[torch.from_numpy(~val)] for k, v in full.items()}
    mu, sd = out["train"]["x"].mean(0), out["train"]["x"].std(0).clamp_min(1e-6)
    for split in [s for s in ("train", "val", "test") if s in out]:
        out[split]["x"] = (out[split]["x"] - mu) / sd
    return out


class CBM(nn.Module):
    def __init__(self, d_in, n_concepts, n_classes, hidden, gated, gate_init=2.0, bottleneck="soft"):
        super().__init__()
        self.bottleneck = bottleneck
        self.concept = (nn.Linear(d_in, n_concepts) if hidden == 0 else
                        nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_concepts)))
        self.head = nn.Linear(n_concepts, n_classes)
        self.gated = gated
        if gated:
            self.gate_logits = nn.Parameter(torch.full((n_concepts,), float(gate_init)))

    def mask(self):
        if not self.gated:
            return None
        return hard_gate_ste(torch.sigmoid(self.gate_logits))

    def open_set(self):
        n = self.head.in_features
        return torch.ones(n, dtype=torch.bool) if not self.gated else (torch.sigmoid(self.gate_logits) > 0.5).detach()

    def classify(self, c):
        c = squash(c, self.bottleneck)
        m = self.mask()
        return self.head(c if m is None else c * m)

    def forward(self, x):
        c = torch.sigmoid(self.concept(x))
        return c, self.classify(c)


def squash(c, bottleneck):
    """What the head reads: the concept probabilities (soft), or 1[c > 0.5] with a straight-through gradient
    (hard) -- a hard bottleneck has no sub-threshold channel for the head to read species information from."""
    if bottleneck == "soft":
        return c
    return (c > 0.5).to(c.dtype) + c - c.detach()


def batches(n, batch_size, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def gate_penalty(model):
    return torch.sigmoid(model.gate_logits).sum() if model.gated else torch.zeros(())


def optimiser(params, gate_params, args):
    groups = [{"params": params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if gate_params:
        groups.append({"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0})
    return torch.optim.Adam(groups)


def train_concepts(model, data, args, gen):
    opt = optimiser(model.concept.parameters(), [], args)
    x, c = data["x"], data["c"]
    for _ in range(args.concept_epochs):
        model.train()
        for idx in batches(len(x), args.batch_size, gen):
            loss = F.binary_cross_entropy_with_logits(model.concept(x[idx]), c[idx], reduction="none").sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def train_head(model, c_hat, y, args, gen, lam, epochs):
    gate = [model.gate_logits] if model.gated else []
    opt = optimiser(model.head.parameters(), gate, args)
    for _ in range(epochs):
        for idx in batches(len(y), args.batch_size, gen):
            loss = F.cross_entropy(model.classify(c_hat[idx]), y[idx]) + lam * gate_penalty(model)
            opt.zero_grad(); loss.backward(); opt.step()


def train_joint(model, data, args, gen, lam, alpha):
    gate = [model.gate_logits] if model.gated else []
    opt = optimiser([*model.concept.parameters(), *model.head.parameters()], gate, args)
    x, c, y = data["x"], data["c"], data["y"]
    for _ in range(args.joint_epochs):
        model.train()
        for idx in batches(len(x), args.batch_size, gen):
            logits_c = model.concept(x[idx])
            logits_y = model.classify(torch.sigmoid(logits_c))
            loss = (F.cross_entropy(logits_y, y[idx])
                    + alpha * F.binary_cross_entropy_with_logits(logits_c, c[idx], reduction="none").sum(1).mean()
                    + lam * gate_penalty(model))
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def predict_concepts(model, x):
    model.eval()
    return torch.sigmoid(model.concept(x))


def refit_head(c_hat, y, keep, n_classes, args, gen, bottleneck="soft"):
    """GCG's mask-and-refit: a fresh linear head on the open concepts only (concept predictor frozen)."""
    head = nn.Linear(int(keep.sum()), n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.head_epochs):
        for idx in batches(len(y), args.batch_size, gen):
            loss = F.cross_entropy(head(squash(c_hat[idx], bottleneck)[:, keep]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    W = torch.zeros(n_classes, len(keep))
    W[:, keep] = head.weight.detach()
    return W, head.bias.detach()


def task_leakage(c_tr, c_te, keep, y_tr, y_te, args, seed):
    """Set-level leakage (Havasi et al. 2022): how much more a head refit on the *soft* predicted concepts of
    `keep` gets than one refit on their thresholded 0/1 values -- the species information the head can read out
    of the probabilities beyond the concepts themselves. Both heads are trained on predicted train concepts."""
    n_y = int(y_tr.max()) + 1
    out = {}
    for bn in ("soft", "hard"):
        W, b = refit_head(c_tr, y_tr, keep, n_y, args, torch.Generator().manual_seed(seed), bn)
        logits = squash(c_te, bn) @ W.T + b
        out[f"acc_{bn}"] = float((logits.argmax(1) == y_te).float().mean())
        out[f"nll_{bn}"] = float(F.cross_entropy(logits, y_te))
    out["acc_gap"] = out["acc_soft"] - out["acc_hard"]
    out["nll_gap"] = out["nll_hard"] - out["nll_soft"]  # nats of species information carried only by the soft values
    return out


def balanced_accuracy(c_hat, c):
    pred, truth = c_hat > 0.5, c > 0.5
    pos, neg = truth.sum(0).float(), (~truth).sum(0).float()
    tpr = (pred & truth).sum(0) / pos.clamp_min(1)
    tnr = (~pred & ~truth).sum(0) / neg.clamp_min(1)
    ok = (pos > 0) & (neg > 0)
    return torch.where(ok, 0.5 * (tpr + tnr), torch.full_like(tpr, float("nan")))


def leakage(c_hat, c, y, n_bins):
    """Per-concept CTL and the pairwise ICL matrix (NaN where undefined), hard and soft encodings."""
    y = _codes(y.numpy())
    h_y = entropy(y)
    truth = [c[:, j].numpy().astype(np.int64) for j in range(c.shape[1])]
    h_true = np.array([entropy(t) for t in truth])
    out = {}
    for enc in ("hard", "soft"):
        pred = [(c_hat[:, j] > 0.5).numpy().astype(np.int64) if enc == "hard"
                else discretise_soft(c_hat[:, j].numpy(), n_bins) for j in range(c.shape[1])]
        ctl = np.array([max(0.0, (mutual_information(p, y) - mutual_information(t, y)) / h_y)
                        for p, t in zip(pred, truth)])
        h_pred = np.array([entropy(p) for p in pred])
        n = len(pred)
        icl = np.full((n, n), np.nan)
        for a in range(n):
            for b in range(a + 1, n):
                if min(h_pred[a], h_pred[b], h_true[a], h_true[b]) <= 0:
                    continue
                v = (mutual_information(pred[a], pred[b]) / np.sqrt(h_pred[a] * h_pred[b])
                     - mutual_information(truth[a], truth[b]) / np.sqrt(h_true[a] * h_true[b]))
                icl[a, b] = icl[b, a] = max(0.0, v)
        out[enc] = {"ctl": ctl, "icl": icl}
    return out


def set_leakage(leak, members):
    members = np.asarray(members, dtype=np.int64)
    res = {}
    for enc, d in leak.items():
        if len(members) == 0:  # the gate closed everything
            res[enc] = {"ctl": None, "icl": None}
            continue
        sub = d["icl"][np.ix_(members, members)]
        iu = np.triu_indices(len(members), 1)
        res[enc] = {"ctl": float(np.mean(d["ctl"][members])) if len(members) else None,
                    "icl": float(np.nanmean(sub[iu])) if len(members) > 1 and np.isfinite(sub[iu]).any() else None}
    return res


@torch.no_grad()
def intervention_curve(classify, c_hat, c, y, ks, n_orders, seed):
    rng = np.random.default_rng(seed)
    n = c.shape[1]
    orders = [rng.permutation(n) for _ in range(n_orders)]
    curve = []
    for k in ks:
        accs = []
        for order in orders:
            ci = c_hat.clone()
            ci[:, order[:k]] = c[:, order[:k]]
            accs.append(float((classify(ci).argmax(1) == y).float().mean()))
        curve.append(float(np.mean(accs)))
    return curve


def fmt(v):
    return "nan" if v is None else f"{v:.4f}"


def run(mode, alpha, lam, seed, data, args):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr, te = data["train"], data["test"]
    n_c, n_y = tr["c"].shape[1], int(tr["y"].max()) + 1
    model = CBM(tr["x"].shape[1], n_c, n_y, args.hidden, gated=lam > 0, bottleneck=args.bottleneck)
    if mode in ("sequential", "independent"):
        train_concepts(model, tr, args, gen)
        # independent (Koh et al.): head and gate see the *true* training concepts, so neither can be rewarded
        # for species information that leaks into the predicted ones; sequential: they see the predictions
        head_in = tr["c"] if mode == "independent" else predict_concepts(model, tr["x"])
        train_head(model, head_in, tr["y"], args, gen, lam, args.head_epochs)
    else:
        train_joint(model, tr, args, gen, lam, alpha)
    model.eval()
    c_tr, c_te = predict_concepts(model, tr["x"]), predict_concepts(model, te["x"])
    keep = model.open_set()
    with torch.no_grad():
        acc_gated = float((model.classify(c_te).argmax(1) == te["y"]).float().mean())
    W, b = refit_head(tr["c"] if mode == "independent" else c_tr, tr["y"], keep, n_y, args, gen, args.bottleneck)
    refit = lambda c: squash(c, args.bottleneck) @ W.T + b  # noqa: E731
    acc_refit = float((refit(c_te).argmax(1) == te["y"]).float().mean())
    val_gated = val_refit = None
    if "val" in data:  # for choosing lambda without touching test
        c_va = predict_concepts(model, data["val"]["x"])
        with torch.no_grad():
            val_gated = float((model.classify(c_va).argmax(1) == data["val"]["y"]).float().mean())
        val_refit = float((refit(c_va).argmax(1) == data["val"]["y"]).float().mean())
    bal = balanced_accuracy(c_te, te["c"])
    ks = sorted({int(round(f * n_c)) for f in np.linspace(0, 1, args.n_points)})
    res = {"mode": mode, "alpha": alpha if mode == "joint" else None, "lambda_gate": lam, "seed": seed,
           "bottleneck": args.bottleneck, "labels": args.labels, "concepts": args.concepts,
           "n_open": int(keep.sum()), "open": np.flatnonzero(keep.numpy()).tolist(),
           "task_acc_gated": acc_gated, "task_acc_refit": acc_refit,
           "val_acc_gated": val_gated, "val_acc_refit": val_refit,
           "concept_bal_acc_all": float(np.nanmean(bal.numpy())),
           "concept_bal_acc_open": float(np.nanmean(bal[keep].numpy())),
           "concept_bal_acc_per": [None if not np.isfinite(v) else float(v) for v in bal.numpy()],
           "intervention_ks": ks,
           "intervention_gated": intervention_curve(model.classify, c_te, te["c"], te["y"], ks, args.n_orders, seed),
           "intervention_refit": intervention_curve(refit, c_te, te["c"], te["y"], ks, args.n_orders, seed)}
    leak = leakage(c_te, te["c"], te["y"], args.n_bins)
    res["leakage_open"] = set_leakage(leak, res["open"])
    res["leakage_all"] = set_leakage(leak, np.arange(n_c))
    res["task_leakage_open"] = task_leakage(c_tr, c_te, keep, tr["y"], te["y"], args, seed) if keep.any() else None
    print(f"[{mode}{'' if mode == 'sequential' else f' a={alpha}'} lam={lam} seed={seed}] open {res['n_open']}/{n_c}  "
          f"acc gated {acc_gated:.4f} refit {acc_refit:.4f} (val {fmt(val_refit)})  concept bal acc {res['concept_bal_acc_all']:.4f}  "
          f"CTL/ICL hard open {fmt(res['leakage_open']['hard']['ctl'])}/{fmt(res['leakage_open']['hard']['icl'])}  "
          f"full intervention {res['intervention_refit'][-1]:.4f}", flush=True)
    ctx = {"c_tr": tr["c"] if mode == "independent" else c_tr, "c_tr_pred": c_tr, "c_te": c_te,
           "head_W": model.head.weight.detach()}
    return res, leak, ctx


def species_informativeness(c, y):
    """NMI(true attribute; species) on the train images, per attribute -- what a task-driven selection favours."""
    y = _codes(y.numpy())
    h = entropy(y)
    return np.array([mutual_information(c[:, j].numpy().astype(np.int64), y) / h for j in range(c.shape[1])])


def control_sets(K, base_ctx, data, rng):
    """Size-|K| concept sets chosen without the gate, all read off the ungated model / the data."""
    k = len(K)
    n_c = base_ctx["head_W"].shape[1]
    if "informativeness" not in data:
        data["informativeness"] = species_informativeness(data["train"]["c"], data["train"]["y"])
    return {"magnitude": np.argsort(-base_ctx["head_W"].norm(dim=0).numpy())[:k],   # top-|K| by baseline weight
            "informative": np.argsort(-data["informativeness"])[:k],              # top-|K| by species info of the attribute
            "random": rng.choice(n_c, k, replace=False)}


def evaluate_set(members, base_ctx, data, args, seed):
    """Refit a head on `members` of the ungated model's concepts; its accuracy and intervention curve."""
    tr, te = data["train"], data["test"]
    keep = torch.zeros(base_ctx["head_W"].shape[1], dtype=torch.bool)
    keep[torch.as_tensor(members, dtype=torch.long)] = True
    gen = torch.Generator().manual_seed(seed)
    W, b = refit_head(base_ctx["c_tr"], tr["y"], keep, base_ctx["head_W"].shape[0], args, gen, args.bottleneck)
    refit = lambda c: squash(c, args.bottleneck) @ W.T + b  # noqa: E731
    ks = sorted({int(round(f * len(keep))) for f in np.linspace(0, 1, args.n_points)})
    return {"task_acc_refit": float((refit(base_ctx["c_te"]).argmax(1) == te["y"]).float().mean()),
            "intervention_refit": intervention_curve(refit, base_ctx["c_te"], te["c"], te["y"], ks, args.n_orders, seed),
            "task_leakage": task_leakage(base_ctx["c_tr_pred"], base_ctx["c_te"], keep, tr["y"], te["y"], args, seed)}


def compare(gated, base, base_leak, args, base_ctx=None, data=None):
    """The gated run's open set K against the same seed's ungated model: same concepts, all, random |K|, and
    (with base_ctx) size-|K| sets chosen without the gate -- each refit and scored on accuracy, interventions
    and leakage, so the gate's own contribution is what separates them."""
    K = gated["open"]
    rng = np.random.default_rng(gated["seed"])
    n_c = len(base_leak["hard"]["ctl"])
    rand = [set_leakage(base_leak, rng.choice(n_c, len(K), replace=False)) for _ in range(args.n_random)]
    out = {}
    for enc in ("hard", "soft"):
        out[enc] = {}
        for m in ("ctl", "icl"):
            vals = [r[enc][m] for r in rand if r[enc][m] is not None]
            g = gated["leakage_open"][enc][m]
            out[enc][m] = {"gated_on_K": g,
                           "baseline_on_K": set_leakage(base_leak, K)[enc][m],
                           "baseline_all": base["leakage_all"][enc][m],
                           "baseline_random_K": float(np.mean(vals)) if vals else None,
                           "gated_pct_vs_baseline_random": float(np.mean(np.array(vals) < g) * 100) if vals and g is not None else None}
    if base_ctx is not None and len(K):
        out["controls"] = {"gated": {"task_acc_refit": gated["task_acc_refit"], "intervention_refit": gated["intervention_refit"],
                                     "leakage": gated["leakage_open"], "task_leakage": gated["task_leakage_open"]}}
        # same concepts K, but read off the ungated model: isolates what gated *training* did to the concepts
        sets = {"baseline_K": np.asarray(K), **control_sets(K, base_ctx, data, rng)}
        for name, members in sets.items():
            out["controls"][name] = {**evaluate_set(members, base_ctx, data, args, gated["seed"]),
                                     "leakage": set_leakage(base_leak, members)}
        c = out["controls"]
        print("    vs controls at |K|=%d: " % len(K) + "  ".join(
            f"{n} acc {v['task_acc_refit']:.4f} int {v['intervention_refit'][-1]:.4f} CTL {fmt(v['leakage']['hard']['ctl'])} "
            f"softgap {v['task_leakage']['acc_gap']:.4f}/{v['task_leakage']['nll_gap']:.3f}"
            for n, v in c.items()), flush=True)
    return out


def main(args):
    data = load_data(args)
    print(f"CUB: {len(data['train']['y'])} train / {len(data['val']['y']) if 'val' in data else 0} val / "
          f"{len(data['test']['y'])} test images, "
          f"{data['train']['c'].shape[1]} concepts ({args.concepts}, {args.labels}-level labels)")
    configs = ([("sequential", None)] * ("sequential" in args.modes) + [("independent", None)] * ("independent" in args.modes)
               + [("joint", a) for a in args.alphas] * ("joint" in args.modes))
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"cub_cbm_gate_{args.tag + '_' if args.tag else ''}{stamp}"
    partial = os.path.join(args.out_dir, f"{tag}.runs.jsonl")  # one line per finished run, survives a crash
    lams = sorted(set([0.0, *args.lambda_gates]))
    runs, comparisons = [], []
    for mode, alpha in configs:
        for seed in args.seeds:
            base, base_leak = None, None
            for lam in lams:
                res, leak, ctx = run(mode, alpha, lam, seed, data, args)
                runs.append(res)
                with open(partial, "a") as f:
                    f.write(json.dumps(res) + "\n")
                if lam == 0:
                    base, base_leak, base_ctx = res, leak, ctx
                else:
                    comparisons.append({"mode": mode, "alpha": alpha, "lambda_gate": lam, "seed": seed,
                                        "bottleneck": args.bottleneck, "labels": args.labels, "concepts": args.concepts,
                                        "n_open": res["n_open"], **compare(res, base, base_leak, args, base_ctx, data)})
                    with open(partial.replace(".runs.jsonl", ".comparisons.jsonl"), "a") as f:
                        f.write(json.dumps(comparisons[-1]) + "\n")
    summary = {}
    for mode, alpha in configs:
        for lam in lams:
            rs = [r for r in runs if r["mode"] == mode and r["alpha"] == alpha and r["lambda_gate"] == lam]
            cs = [c for c in comparisons if c["mode"] == mode and c["alpha"] == alpha and c["lambda_gate"] == lam]
            key = f"{mode}{'' if alpha is None else f'_a{alpha}'}_{args.bottleneck}_lam{lam}"
            mean = lambda xs: (float(np.mean(xs)), float(np.std(xs))) if xs else None  # noqa: E731
            entry = {k: mean([r[k] for r in rs if r[k] is not None and np.isfinite(r[k])]) for k in ("n_open", "task_acc_gated", "task_acc_refit",
                                                           "concept_bal_acc_all", "concept_bal_acc_open")}
            entry["intervention_ks"] = rs[0]["intervention_ks"]
            entry["intervention_refit"] = np.mean([r["intervention_refit"] for r in rs], 0).tolist()
            entry["intervention_gated"] = np.mean([r["intervention_gated"] for r in rs], 0).tolist()
            entry["leakage_open"] = {enc: {m: mean([r["leakage_open"][enc][m] for r in rs if r["leakage_open"][enc][m] is not None])
                                           for m in ("ctl", "icl")} for enc in ("hard", "soft")}
            if cs:
                entry["vs_baseline"] = {enc: {m: {k: mean([c[enc][m][k] for c in cs if c[enc][m][k] is not None])
                                                  for k in cs[0][enc][m]} for m in ("ctl", "icl")} for enc in ("hard", "soft")}
            summary[key] = entry
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{tag}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "concept_names": data["concept_names"], "summary": summary,
                   "comparisons": comparisons, "runs": runs}, f, indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["sequential", "joint"], choices=["sequential", "independent", "joint"])
    p.add_argument("--bottleneck", default="soft", choices=["soft", "hard"],
                   help="head reads concept probabilities (soft) or 1[c > 0.5] with a straight-through gradient (hard)")
    p.add_argument("--tag", default="", help="added to the output file names")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1], help="joint CBM concept-loss weight")
    p.add_argument("--lambda-gates", nargs="+", type=float, default=[0.0005, 0.001, 0.002, 0.003, 0.005],
                   help="gate penalties; 0 (the ungated CBM) is always added")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 10, 20])
    p.add_argument("--concepts", default="all", choices=["all", "koh112"],
                   help="all 312 CUB attributes, or Koh et al.'s 112")
    p.add_argument("--labels", default="image", choices=["image", "class"],
                   help="CUB's per-image attribute annotations, or the species majority vote (Koh et al.)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="species-stratified fraction of train held out to choose lambda by validation accuracy")
    p.add_argument("--hidden", type=int, default=512, help="concept predictor hidden width (0 = linear)")
    p.add_argument("--concept-epochs", type=int, default=60)
    p.add_argument("--head-epochs", type=int, default=60)
    p.add_argument("--joint-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-bins", type=int, default=8, help="soft-encoding quantile bins for CTL/ICL")
    p.add_argument("--n-random", type=int, default=200)
    p.add_argument("--n-orders", type=int, default=10, help="random intervention orders per curve")
    p.add_argument("--n-points", type=int, default=9, help="points on the intervention curve")
    p.add_argument("--cub-root", default=os.path.join(ROOT, "dataset", "CUB_200_2011"))
    p.add_argument("--features-dir", default=os.path.join(ROOT, "data", "cub", "activations"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "cub_cbm_gate"))
    main(p.parse_args())
