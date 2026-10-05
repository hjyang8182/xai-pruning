"""One gate-value histogram per CBM substrate, all in the same style.

Each gated run saves its own gate histogram from inside its own pipeline
(DN-CBM: src/visualise.plot_gate_distribution; LF-CBM / VLG-CBM: train_cbm.py;
UCBM: plotter/plotter.py), and the four came out looking different -- different
sizes, fonts, colours, titles and annotations. This reads the *gate values* back
out of a finished run and redraws them identically for all four: gate value on x,
count on y, the 0.5 threshold, nothing else.

Gate values are sigmoid(gate_logits / T) with T the temperature the run converged
at -- read from the run's own metadata, since two of the pipelines anneal it and a
plain sigmoid would then show the wrong distribution.

Usage:
    python plot_gate_distributions.py                      # the four runs in RUNS
    python plot_gate_distributions.py --outdir img --width 2.6

    # or point it at any gated run directory:
    python plot_gate_distributions.py "LF-CBM=prelim/Label-free-CBM/saved_models/<run>"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from panel_style import wrap_to_width

BAR = "#2a78d6"        # same blue as the other CIFAR-100 panels
THRESH = "#c0654b"

# The four CIFAR-100 gated runs the figure is built from, <label>: <run dir>.
RUNS = {
    "LF-CBM":  "prelim/Label-free-CBM/saved_models/cifar100_cbm_armC_gated_2026_08_22_01_11",
    "UCBM":    "prelim/ucbm/save/RESULTS/cifar100-resnet50_v2/classifier/concepts_1000_64/"
               "gated_2026_08_20_-_11_23_11",
    "VLG-CBM": "prelim/VLG-CBM/saved_models/cifar100/cifar100_cbm_gated_2026_08_24_23_57_43",
    "DN-CBM":  "data/cifar100/model/gated_20260921_210058",
}


def load_gates(run_dir, seed=None):
    """(gate values, temperature) for a finished gated run, whichever pipeline wrote it.

    The pipelines store the gate differently -- LF-CBM and VLG-CBM save raw
    `gate_logits.pt` next to an args/metrics file, UCBM pickles the whole classifier,
    DN-CBM saves one GatedProbe state dict per seed -- so dispatch on what is there.
    """
    logits_path = os.path.join(run_dir, "gate_logits.pt")
    if os.path.exists(logits_path):                      # LF-CBM / VLG-CBM
        logits = torch.load(logits_path, map_location="cpu").float()
        temp = 1.0
        metrics = os.path.join(run_dir, "metrics.txt")
        if os.path.exists(metrics):                      # LF-CBM records the converged T
            with open(metrics) as f:
                m = json.load(f)
            temp = float(m.get("gate_temperature_used", temp))
        else:                                            # VLG-CBM: T from its args
            with open(os.path.join(run_dir, "args.txt")) as f:
                a = json.load(f)
            temp = float(a.get("gate_temperature_final") or a.get("gate_temperature") or 1.0)
        return torch.sigmoid(logits / temp).numpy(), temp

    clf = os.path.join(run_dir, "classifier.pth")
    if os.path.exists(clf):                              # UCBM
        ckpt = torch.load(clf, map_location="cpu", weights_only=False)
        logits = ckpt["model_state_dict"]["gate_logits"].float()
        temp = float(ckpt.get("gate_temperature", 1.0) or 1.0)
        return torch.sigmoid(logits / temp).numpy(), temp

    probes = sorted(p for p in os.listdir(run_dir)
                    if p.startswith("probe_seed") and p.endswith(".pt") and "refit" not in p)
    if probes:                                           # DN-CBM
        pick = probes[0] if seed is None else f"probe_seed{seed}.pt"
        sd = torch.load(os.path.join(run_dir, pick), map_location="cpu")
        return torch.sigmoid(sd["gate_logits"].float()).numpy(), 1.0

    raise FileNotFoundError(f"no gate values found in {run_dir}")


def plot(gates, out_path, width=4.0, height=None, font_scale=1.0, bins=50, log_y=False):
    # Typography matches the other narrow CIFAR-100 panels: large type on a small figure,
    # so the panel survives the shrink to column width.
    FS_AXIS = 16 * font_scale
    FS_TICK = 14 * font_scale
    FS_LEG = 15 * font_scale
    plt.rcParams.update({"font.family": "sans-serif", "font.size": FS_TICK})

    fig, ax = plt.subplots(figsize=(width, height or width * 0.72))

    ax.hist(gates, bins=bins, range=(0.0, 1.0), color=BAR, zorder=2)
    ax.axvline(0.5, color=THRESH, linewidth=1.6, linestyle=(0, (3, 2)), zorder=3,
               label="threshold")

    if log_y:
        ax.set_yscale("log")

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.tick_params(axis="both", length=0, labelsize=FS_TICK, colors="black")

    ax.grid(axis="y", color="#eeede9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")

    ax.legend(fontsize=FS_LEG, frameon=False, loc="upper center",
              handlelength=1.4, handletextpad=0.5, borderaxespad=0.1,
              bbox_to_anchor=(0.5, 1.22))

    # Axis labels are centred on the *figure*: on this narrow an axes they would need
    # wrapping, and text that overhangs would widen the saved panel past `width`.
    fig.supxlabel(wrap_to_width(fig, "Gate value", FS_AXIS, width - 0.10),
                  fontsize=FS_AXIS, color="black", y=0.005)
    fig.supylabel("Count", fontsize=FS_AXIS, color="black", x=0.005)

    # The left margin has to hold the y tick labels *and* the "Count" label beside them,
    # and how wide the tick labels are depends on the counts (160 vs 6,000), so measure
    # them rather than guess -- otherwise the label lands on top of the numbers.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_w = max((t.get_window_extent(renderer).width / fig.dpi
                  for t in ax.get_yticklabels()), default=0.0)
    legend_h = FS_LEG * 1.9 / 72
    fig.subplots_adjust(
        left=(0.10 + FS_AXIS * 1.35 / 72 + tick_w) / width,
        right=0.99,
        top=1 - legend_h / fig.get_figheight(),
        bottom=(0.30 + FS_AXIS * 1.35 / 72) / fig.get_figheight(),
    )

    root, _ = os.path.splitext(out_path)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 250}), (".svg", {})):
        fig.savefig(root + ext, bbox_inches="tight", **kw)
    plt.close(fig)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="*", help="<label>=<run dir> (default: the four runs in RUNS)")
    p.add_argument("--outdir", default="img")
    p.add_argument("--prefix", default="gate_distribution_cifar100")
    p.add_argument("--width", type=float, default=4.0, help="figure width in inches")
    p.add_argument("--height", type=float, default=None, help="figure height (default 0.72 * width)")
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="multiplies every font size")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--log-y", action="store_true",
                   help="log count axis -- the closed-gate spike otherwise flattens the rest")
    p.add_argument("--seed", type=int, default=None, help="DN-CBM: which seed's probe to read")
    args = p.parse_args()

    runs = dict(s.split("=", 1) for s in args.specs) if args.specs else RUNS
    os.makedirs(args.outdir, exist_ok=True)
    for label, run_dir in runs.items():
        gates, temp = load_gates(run_dir, seed=args.seed)
        n_open = int((gates > 0.5).sum())
        out = os.path.join(args.outdir, f"{args.prefix}_{label.lower().replace('-', '')}.pdf")
        root = plot(gates, out, width=args.width, height=args.height,
                    font_scale=args.font_scale, bins=args.bins, log_y=args.log_y)
        print(f"{label:8s} {n_open:>5,}/{len(gates):,} open (T={temp:g})  <- {run_dir}\n"
              f"         saved {root}.pdf / {root}.png / {root}.svg")


if __name__ == "__main__":
    main()
