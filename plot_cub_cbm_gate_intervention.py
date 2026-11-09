"""Intervention curve for the GCG gate on the joint CUB CBM (a=0.01, class labels, lambda 0.005):
gated model vs the ungated model refit on the same kept concepts. Mean +- std over seeds."""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pat, lam = (sys.argv[1] if len(sys.argv) > 1 else "ca_cls_joint001_s*"), float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
out = sys.argv[3] if len(sys.argv) > 3 else "figures/cub_cbm_gate_intervention_joint001.png"
curves = {"gated": [], "baseline_K": []}
K = []
for f in sorted(glob.glob(f"data/cub_cbm_gate/cub_cbm_gate_{pat}.comparisons.jsonl")):
    for line in open(f):
        d = json.loads(line)
        if d["mode"] == "joint" and d["alpha"] == 0.01 and d["lambda_gate"] == lam:
            K.append(d["n_open"])
            for s in curves:
                curves[s].append(d["controls"][s]["intervention_refit"])
n_pts = len(curves["gated"][0])
x = np.linspace(0, 100, n_pts)  # % of the kept concepts set to their true values
fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=200)
for s, label, col in [("baseline_K", "Ungated, same K", "#2a78d6"), ("gated", "Gated", "#eb6834")]:
    c = np.array(curves[s]); m, sd = c.mean(0), c.std(0)
    ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.2, lw=0)
    ax.plot(x, m, color=col, lw=2, marker="o", ms=4, label=label)
ax.set_xlabel(f"Kept concepts intervened on (%, K = {np.mean(K):.0f})")
ax.set_ylabel("Test accuracy")
ax.set_title(f"Joint CBM (α = .01), λ = {lam:g}", fontsize=10)
ax.grid(alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(out); print(out, "n seeds", len(K))
for s in curves:
    print(s, " ".join("%.3f±%.3f" % (a, b) for a, b in zip(np.mean(curves[s], 0), np.std(curves[s], 0))))
