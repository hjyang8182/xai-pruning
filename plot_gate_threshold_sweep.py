"""One figure + one table for "how much does the 0.5 gate threshold matter?".

Takes the gate_threshold_sensitivity_*.json files written by each substrate's
`concept_retention.py --analysis threshold` (LF-CBM / VLG-CBM / UCBM, any dataset)
and draws, per model, accuracy vs the open/closed threshold tau (top row) and the
number of concepts that survive at that tau (bottom row). Concepts with gate <= tau
are hard-zeroed and the rest keep their trained gate value -- no retraining -- so
the curves say how well a hard cut at tau reproduces the soft model the gate
actually trained. The band where accuracy stays within `--tol` of the full model
is shaded; its right edge is the fewest concepts a threshold can keep at that cost.
Also prints a markdown table of the same numbers.

Usage:
    python plot_gate_threshold_sweep.py --out img/gate_threshold_cifar100.png \\
        "LF-CBM (CIFAR-100)=prelim/Label-free-CBM/seed_sweep_results/cifar100_gate_threshold_sensitivity_*.json" \\
        "VLG-CBM (CIFAR-100)=prelim/VLG-CBM/saved_models/cifar100/.../gate_threshold_sensitivity_*.json" \\
        "UCBM (CIFAR-100)=prelim/ucbm/save/RESULTS/.../cifar100_gate_threshold_sensitivity_*.json"
Each positional arg is  <panel label>=<json path or glob>  (latest match is used).
"""
import argparse
import glob
import json

import numpy as np

ACC_COLOR = "#2a78d6"    # categorical slot 1
OPEN_COLOR = "#eb6834"   # categorical slot 2
BAND_COLOR = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _load(spec):
    label, pattern = spec.split("=", 1)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"nothing matched {pattern}")
    with open(paths[-1]) as f:
        return label, json.load(f), paths[-1]


def _at(r, t):
    taus = np.asarray(r["taus"])
    return int(np.argmin(np.abs(taus - t)))


def tolerance_band(r, tol):
    """Widest contiguous tau interval containing 0.5 (or, if 0.5 is outside it,
    the widest interval anywhere) where acc >= (1 - tol) * acc_full. Returns
    (tau_lo, tau_hi, n_open_lo, n_open_hi); n_open_hi is the fewest concepts a
    hard threshold keeps at <= tol relative accuracy loss."""
    taus = np.asarray(r["taus"])
    acc = np.asarray(r["acc"])
    n_open = np.asarray(r["n_open"])
    ok = acc >= (1.0 - tol) * r["acc_full"]
    if not ok.any():
        return None
    # contiguous runs of ok
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(ok) - 1))
    i50 = _at(r, 0.5)
    inside = [rn for rn in runs if rn[0] <= i50 <= rn[1]]
    lo, hi = (inside[0] if inside else max(runs, key=lambda rn: taus[rn[1]] - taus[rn[0]]))
    return float(taus[lo]), float(taus[hi]), int(n_open[lo]), int(n_open[hi])


def table_rows(entries, tol, report_taus):
    rows = []
    for label, r, _ in entries:
        band = tolerance_band(r, tol)
        rows.append({
            "model": label, "J": r["n_concepts_total"], "acc_full": r["acc_full"],
            "n_seeds": r.get("n_seeds", 1),
            "at": {t: (int(round(r["n_open"][_at(r, t)])), float(r["acc"][_at(r, t)])) for t in report_taus},
            "band": band, "mushy": r["bimodal_mushy_frac"],
            "n_open_range_mid": r["n_open_range_mid"], "acc_drop_mid": r["acc_drop_mid"],
        })
    return rows


def markdown_table(rows, tol, report_taus):
    tcols = " | ".join(f"open / acc @τ={t:g}" for t in report_taus)
    hdr = (f"| model | #concepts | acc (full) | {tcols} | τ band within {100 * tol:.0f}% of full "
           f"(open at edges) | open(0.3)−open(0.7) | mushy gates |")
    lines = [hdr, "|" + "---|" * (6 + len(report_taus))]
    for r in rows:
        tcells = " | ".join(f"{r['at'][t][0]} / {r['at'][t][1]:.3f}" for t in report_taus)
        if r["band"]:
            lo, hi, nlo, nhi = r["band"]
            band = f"[{lo:.2f}, {hi:.2f}] ({nlo} → {nhi})"
        else:
            band = "—"
        seeds = f" (×{r['n_seeds']} seeds)" if r["n_seeds"] > 1 else ""
        lines.append(
            f"| {r['model']}{seeds} | {r['J']} | {r['acc_full']:.3f} | {tcells} | {band} | "
            f"{r['n_open_range_mid']:.0f} | {r['mushy']:.2f} |")
    return "\n".join(lines)


def plot(entries, out_path, tol, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), sharex=True, squeeze=False)
    for j, (label, r, _) in enumerate(entries):
        taus = np.asarray(r["taus"])
        acc = np.asarray(r["acc"])
        n_open = np.asarray(r["n_open"])
        band = tolerance_band(r, tol)
        i50 = _at(r, 0.5)

        ax = axes[0, j]
        ax.plot(taus, acc, color=ACC_COLOR, linewidth=2.0)
        if r.get("acc_std"):
            s = np.asarray(r["acc_std"])
            ax.fill_between(taus, acc - s, acc + s, color=ACC_COLOR, alpha=0.15, linewidth=0)
        ax.axhline(r["acc_full"], color=INK2, linestyle=":", linewidth=1.0)
        ax.text(0.99, r["acc_full"], "full model", color=INK2, fontsize=8, ha="right", va="bottom")
        ax.set_title(label, fontsize=11, color=INK)
        ax.set_ylim(0, 1.0 if acc.max() > 0.85 else max(0.05, acc.max() * 1.15))
        if j == 0:
            ax.set_ylabel("top-1 accuracy after hard cut at τ", color=INK2)

        ax2 = axes[1, j]
        ax2.plot(taus, n_open, color=OPEN_COLOR, linewidth=2.0)
        if r.get("n_open_std"):
            s = np.asarray(r["n_open_std"])
            ax2.fill_between(taus, n_open - s, n_open + s, color=OPEN_COLOR, alpha=0.15, linewidth=0)
        ax2.set_ylim(0, r["n_concepts_total"] * 1.05)
        ax2.axhline(r["n_concepts_total"], color=INK2, linestyle=":", linewidth=1.0)
        ax2.set_xlabel("gate threshold τ  (concepts with gate ≤ τ zeroed)", color=INK2)
        if j == 0:
            ax2.set_ylabel("open concepts (gate > τ)", color=INK2)

        for a in (ax, ax2):
            if band:
                a.axvspan(band[0], band[1], color=BAND_COLOR, alpha=0.08, linewidth=0)
            a.axvline(0.5, color=INK2, linewidth=1.0, linestyle="--", alpha=0.8)
            a.set_xlim(0, 1)
            a.grid(True, color=GRID, linewidth=0.8)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(colors=INK2)

        ax.annotate(f"τ=0.5: {int(round(n_open[i50]))} open, acc {acc[i50]:.3f}",
                    (0.5, acc[i50]), xytext=(6, -14), textcoords="offset points",
                    fontsize=8, color=INK, ha="left")
        if band:
            lo, hi, nlo, nhi = band
            ax2.annotate(f"within {100 * tol:.0f}%: τ∈[{lo:.2f},{hi:.2f}]\n{nlo}→{nhi} open",
                         (0.02, 0.96), xycoords="axes fraction", fontsize=8, color=INK, va="top")
    fig.suptitle(title or "Gate threshold sensitivity: hard cut at τ, no retraining",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="<label>=<json path or glob>")
    p.add_argument("--out", required=True, help="Output figure path (.png)")
    p.add_argument("--tol", type=float, default=0.01,
                   help="Relative accuracy loss defining the acceptable tau band (default 1%%)")
    p.add_argument("--report-taus", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--title", default=None)
    p.add_argument("--table-out", default=None, help="Also write the markdown table here")
    return p.parse_args()


def main(args):
    entries = [_load(s) for s in args.specs]
    for label, _, path in entries:
        print(f"{label}: {path}")
    rows = table_rows(entries, args.tol, args.report_taus)
    md = markdown_table(rows, args.tol, args.report_taus)
    print()
    print(md)
    if args.table_out:
        with open(args.table_out, "w") as f:
            f.write(md + "\n")
    plot(entries, args.out, args.tol, title=args.title)
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main(parse_args())
