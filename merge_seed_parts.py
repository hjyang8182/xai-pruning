"""Merge gate-sweep JSONs that each re-ran some arms for a subset of seeds (refit_from_gates.py /
rerun_random_control.py with --seeds, one seed per GPU) into one sweep JSON.

Every point is matched by lambda_gate. For each --arms arm, the per-seed lists of all parts (every
"<x>_per_seed" key, aligned by the arm's "seeds") are concatenated in seed order and "<x>_mean" /
"<x>_std" recomputed (population s.d., as the sweeps report). Everything else, including arms not
named, comes from the first part.

    python merge_seed_parts.py --arms magnitude_refit group_lasso_refit --out merged.json part0.json part10.json ...
"""
import argparse
import json
import statistics

import numpy as np


def merge_arm(parts):
    rows = {}
    for arm in parts:
        seeds = arm["seeds"]
        for j, seed in enumerate(seeds):
            rows[seed] = {k[:-len("_per_seed")]: v[j] for k, v in arm.items()
                          if k.endswith("_per_seed") and isinstance(v, list) and len(v) == len(seeds)}
    seeds = sorted(rows)
    out = {"seeds": seeds}
    for key in rows[seeds[0]]:
        vals = [rows[s][key] for s in seeds]
        out[f"{key}_per_seed"] = vals
        out[f"{key}_mean"] = statistics.fmean(vals)
        out[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return out


def main(args):
    sweeps = []
    for path in args.parts:
        with open(path) as f:
            sweeps.append(json.load(f))
    out = {**sweeps[0], "merged_from": args.parts}
    points = []
    for point in sweeps[0]["points"]:
        lam = point["lambda_gate"]
        others = [next(p for p in s["points"] if np.isclose(p["lambda_gate"], lam)) for s in sweeps]
        new = {**point}
        for arm in args.arms:
            new[arm] = merge_arm([p[arm] for p in others if arm in p])
        points.append(new)
    out["points"] = points
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for p in points:
        print(f"lam {p['lambda_gate']:g}: " + "  ".join(
            f"{a} {p[a]['acc_mean']:.4f} (seeds {p[a]['seeds']})" for a in args.arms))
    print(f"Saved merged sweep to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parts", nargs="+")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
