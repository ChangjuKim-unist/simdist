"""Design 18 section 9.7: F_cl - A0 versus the replanning horizon L (branch_h<L>_<cell>.csv, candidates A0 and Fcl).
Pairs within each run (clone means per branch point), restricted to the branch points present in every run.

    python scripts/sim2sim/summarize_horizon.py results/stage18/branch_h25_papermotor_friction0.15.csv results/stage18/branch_h125_... results/stage18/branch_h500_...
"""

import csv
import re
import sys

import numpy as np


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, np.float64)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(path):
    pts = {}
    for r in csv.DictReader(open(path)):
        k = (round(float(r["speed"]), 3), int(r["seed"]), int(r["step"]))
        pts.setdefault(k, {}).setdefault(r["group"], []).append((float(r["ret"]), int(r["fell"])))
    return {k: v for k, v in pts.items() if "A0" in v and "Fcl" in v}


def main():
    runs = {}
    for p in sys.argv[1:]:
        L = int(re.search(r"branch_h(\d+)_", p).group(1)); runs[L] = load(p)
    common = set.intersection(*[set(v) for v in runs.values()])
    keys = sorted(common)
    print(f"common branch points: {len(keys)} (per run: " + ", ".join(f"L={L}: {len(v)}" for L, v in sorted(runs.items())) + ")")
    print("| 재계획 구간 L (스텝) | 수익 구간 | 분기점 | A0 평균 | F_cl − A0 [95 % CI] | A0 대비 % | 넘어짐 A0 / F_cl |")
    print("|---|---|---|---|---|---|---|")
    for L, v in sorted(runs.items()):
        a0 = np.array([np.mean([x[0] for x in v[k]["A0"]]) for k in keys]); f = np.array([np.mean([x[0] for x in v[k]["Fcl"]]) for k in keys])
        fa = np.mean([np.mean([x[1] for x in v[k]["A0"]]) for k in keys]); ff = np.mean([np.mean([x[1] for x in v[k]["Fcl"]]) for k in keys])
        d = f - a0; ci = boot_ci(d)
        pct = f"{100 * d.mean() / a0.mean():+.1f} %" if a0.mean() > 0.5 else "—"
        print(f"| {L} | {L + 100} | {len(keys)} | {a0.mean():.2f} | {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {pct} | {fa:.2f} / {ff:.2f} |")
    # per speed
    print("\n| L | 속도 | 분기점 | A0 평균 | F_cl − A0 | 넘어짐 A0 / F_cl |\n|---|---|---|---|---|---|")
    for L, v in sorted(runs.items()):
        for sp in sorted(set(k[0] for k in keys)):
            ks = [k for k in keys if k[0] == sp]
            a0 = np.array([np.mean([x[0] for x in v[k]["A0"]]) for k in ks]); f = np.array([np.mean([x[0] for x in v[k]["Fcl"]]) for k in ks])
            fa = np.mean([np.mean([x[1] for x in v[k]["A0"]]) for k in ks]); ff = np.mean([np.mean([x[1] for x in v[k]["Fcl"]]) for k in ks])
            print(f"| {L} | {sp} | {len(ks)} | {a0.mean():.2f} | {(f - a0).mean():+.3f} | {fa:.2f} / {ff:.2f} |")


if __name__ == "__main__":
    main()
