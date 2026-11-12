"""Summarize a sim-to-sim adaptation experiment directory into a markdown table.

    python scripts/sim2sim/summarize.py results/sim2sim/low_friction
"""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(result_dir: str):
    rows = list(csv.DictReader(open(os.path.join(result_dir, "eval.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    conditions = []
    for r in rows:
        if r["condition"] not in conditions:
            conditions.append(r["condition"])

    forgetting = {}
    fpath = os.path.join(result_dir, "forgetting.json")
    if os.path.exists(fpath):
        forgetting = json.load(open(fpath))

    def stat(vals):
        vals = np.array([float(v) for v in vals if v not in ("", "nan")])
        if len(vals) == 0:
            return "n/a"
        return f"{vals.mean():.1f} ± {vals.std():.1f}" if len(vals) > 1 else f"{vals[0]:.1f}"

    lines = ["| model | " + " | ".join(f"{c}: reward / progress [m] / no-fall" for c in conditions)
             + " | value RMSE vs base | corr vs sim labels |",
             "|---|" + "---|" * (len(conditions) + 2)]
    for m in models:
        cells = []
        for c in conditions:
            rs = by.get((m, c), [])
            if not rs:
                cells.append("n/a")
                continue
            nofall = sum(1 for r in rs if r["episode_length"] not in ("", "nan") and float(r["episode_length"]) >= 1000)
            cells.append(f"{stat(r['total_reward'] for r in rs)} / {stat(r['forward_progress'] for r in rs)} / {nofall}/{len(rs)}")
        f = forgetting.get(m, {})
        cells.append(f"{f['rmse_vs_base']:.4f}" if "rmse_vs_base" in f else "n/a")
        cells.append(f"{f['corr_vs_label']:.3f}" if "corr_vs_label" in f else "n/a")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(result_dir, "summary.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sim2sim/low_friction")
