"""Paired prior-vs-planner statistics for stage-13 CSVs (design doc 14).

    python scripts/sim2sim/paired_stats.py results/stage13/eval_trust0_server.csv results/stage13/eval_night_server.csv

For every (model_prefix, condition) with both <prefix>_prior and <prefix>_planner rows the script pairs runs
on (speed, seed) and reports falls per 1k steps, reward/step, forward progress for both arms, the paired
difference (planner - prior) and a 95 % paired bootstrap CI over the matched runs.  Extra pairs can be
given as --pair NAME:MODEL_A:MODEL_B (difference = B - A).
"""
import argparse
import csv
import re
from collections import defaultdict

import numpy as np

STEPS = 5000.0


def load(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    by = defaultdict(dict)
    for r in rows:
        key = (float(r["speed"]), int(r["seed"]))
        by[(r["model"], r["condition"])][key] = r
    return by


def metrics(r):
    return dict(falls=float(r["num_falls"]) / STEPS * 1000.0,
                reward=float(r["run_reward_per_step"]),
                progress=float(r["forward_progress"]))


def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def compare(by, a, b, cond, label):
    ra, rb = by.get((a, cond)), by.get((b, cond))
    if not ra or not rb:
        return None
    keys = sorted(set(ra) & set(rb))
    if len(keys) < 4:
        return None
    out = {"label": label, "cond": cond, "n": len(keys)}
    for m in ("falls", "reward", "progress"):
        va = np.array([metrics(ra[k])[m] for k in keys])
        vb = np.array([metrics(rb[k])[m] for k in keys])
        lo, hi = boot_ci(vb - va)
        out[m] = (va.mean(), vb.mean(), (vb - va).mean(), lo, hi)
    hi_keys = [k for k in keys if k[0] >= 0.9]
    out["falls09"] = (np.mean([metrics(ra[k])["falls"] for k in hi_keys]) if hi_keys else float("nan"),
                      np.mean([metrics(rb[k])["falls"] for k in hi_keys]) if hi_keys else float("nan"))
    return out


def fmt(o):
    def cell(t, nd):
        a, b, d, lo, hi = t
        star = "*" if (lo > 0 or hi < 0) else ""
        return f"{a:.{nd}f} → {b:.{nd}f} ({d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]){star}"
    return (f"| {o['label']} | {o['cond']} | {o['n']} | {cell(o['falls'], 2)} | "
            f"{o['falls09'][0]:.2f} → {o['falls09'][1]:.2f} | {cell(o['reward'], 4)} | {cell(o['progress'], 1)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--pair", action="append", default=[], help="NAME:MODEL_A:MODEL_B (diff = B - A)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    by = load(args.csvs)
    models = sorted({m for m, _ in by})
    conds = sorted({c for _, c in by})
    lines = ["| pair | condition | n | falls/1k A → B (B−A [95% CI]) | falls/1k @0.9 A → B | reward/step A → B | progress m A → B |",
             "|---|---|---|---|---|---|---|"]
    for m in models:
        mm = re.match(r"^(.*)_prior(_switch)?$", m)
        if not mm:
            continue
        planner = f"{mm.group(1)}_planner{mm.group(2) or ''}"
        if planner not in models:
            continue
        for c in conds:
            o = compare(by, m, planner, c, f"{mm.group(1)}{mm.group(2) or ''}: prior → planner")
            if o:
                lines.append(fmt(o))
    for spec in args.pair:
        name, a, b = spec.split(":")
        for c in conds:
            o = compare(by, a, b, c, name)
            if o:
                lines.append(fmt(o))
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
