"""Design 18 gate 0 summary (docs/design_branch_headroom_ko.md section 3).

    python scripts/sim2sim/summarize_branch.py results/stage18/branch_*.csv [--fidelity] [--by-speed] [--out summary.md]

Per (model, condition) [x speed]:
  (a) noise floor: SD of the A0 return across the J clones (mean over branch points) and the restore error
      (A0 clone 0 return - the anchor's own realised return): mean and mean absolute value
  (b) per group: share of candidates (clone-mean) below A0 (clone-mean), fall share (rows)
  (c) optimistic bound: max over candidates of the clone-mean return - A0 clone-mean, per group and pooled (all groups but A0)
  (d) cross-validated bound: pick by the mean of the other clones, score with the held-out clone minus A0's held-out clone,
      averaged over the J held-out splits; mean and paired bootstrap 95% CI over branch points
  (e) F - A0 (clone-mean), paired bootstrap 95% CI
Gains in return units (sum of the evaluator's reward over 125 steps) and in % of the A0 mean return.
"""

import argparse
import sys

import numpy as np
import pandas as pd

KEY = ["speed", "seed", "step"]
GROUPS = ["A1", "B", "C", "D", "E", "F"]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell_stats(d, groups, combos=None):
    """d: rows of one (model, condition[, speed]) cell. Returns a dict of statistics (numpy, pandas-version independent).
    combos: optional {name: [groups]} unions of candidate groups scored like a group (design 18 section 8.1)."""
    J = int(d.clone.max()) + 1
    d = d.copy()
    d["bp"] = pd.factorize(list(zip(d.speed.round(3), d.seed, d.step)))[0]
    cands = sorted(set(zip(d.group, d.cand)), key=lambda t: (t[0], t[1]))
    cidx = {c: i for i, c in enumerate(cands)}
    d["ci"] = [cidx[c] for c in zip(d.group, d.cand)]
    nb, nc = d.bp.max() + 1, len(cands)
    R = np.full((nb, nc, J), np.nan); F = np.zeros((nb, nc, J), bool)
    R[d.bp.values, d.ci.values, d.clone.values] = d.ret.values; F[d.bp.values, d.ci.values, d.clone.values] = d.fell.values.astype(bool)
    i0 = cidx[("A0", 0)]
    keep = ~np.isnan(R).any(axis=(1, 2)) if nc > 1 else ~np.isnan(R[:, i0]).any(axis=1)
    R, F = R[keep], F[keep]
    a0 = R[:, i0]  # bp x clone
    out = {"n_bp": int(keep.sum()), "a0_mean": float(a0.mean())}
    out["a0_clone_sd"] = float(a0.std(axis=1, ddof=1).mean()) if J > 1 else np.nan
    anc = d[(d.group == "A0") & (d.clone == 0)].drop_duplicates("bp").set_index("bp").sort_index()
    anc = anc.loc[np.where(keep)[0]]
    err = anc.ret.values - anc.anchor_ret.values
    out["restore_err_mean"], out["restore_err_abs"] = float(err.mean()), float(np.abs(err).mean())
    out["a0_fell"] = float(F[:, i0].mean()); out["anchor_fell"] = float(anc.anchor_fell.mean())
    cm = R.mean(axis=2); a0m = cm[:, i0]  # clone means
    gidx = {g: [cidx[c] for c in cands if c[0] == g] for g in groups if any(c[0] == g for c in cands)}

    def cross(ids):
        """(d): choose among candidate ids by the mean of the other clones, score with the held-out clone minus A0."""
        vals = []
        for h in range(J):
            others = [c for c in range(J) if c != h]
            if not others:
                return None
            sel = R[:, ids][:, :, others].mean(axis=2)  # bp x cand
            pick = np.array(ids)[sel.argmax(axis=1)]
            vals.append(R[np.arange(len(R)), pick, h] - a0[:, h])
        return np.mean(vals, axis=0)

    per_group = {}
    for g, ids in gidx.items():
        gain = cm[:, ids] - a0m[:, None]
        st = {"n_cand": len(ids), "worse": float((gain < 0).mean()), "fell": float(F[:, ids].mean()), "mean_gain": float(gain.mean())}
        best = gain.max(axis=1); st["opt"] = float(best.mean()); st["opt_ci"] = boot_ci(best)
        cv = cross(ids)
        if cv is not None:
            st["cv"] = float(cv.mean()); st["cv_ci"] = boot_ci(cv)
        per_group[g] = st
    out["groups"] = per_group
    for name, gs in (combos or {}).items():
        ids = [i for g in gs if g in gidx for i in gidx[g]]
        if not ids:
            continue
        gain = cm[:, ids] - a0m[:, None]
        st = {"n_cand": len(ids), "worse": float((gain < 0).mean()), "fell": float(F[:, ids].mean()), "mean_gain": float(gain.mean())}
        best = gain.max(axis=1); st["opt"] = float(best.mean()); st["opt_ci"] = boot_ci(best)
        cv = cross(ids)
        if cv is not None:
            st["cv"] = float(cv.mean()); st["cv_ci"] = boot_ci(cv)
        per_group[name] = st
    all_ids = [i for ids in gidx.values() for i in ids]
    if all_ids:
        best = (cm[:, all_ids] - a0m[:, None]).max(axis=1); out["opt_all"] = float(best.mean()); out["opt_all_ci"] = boot_ci(best)
        cv = cross(all_ids)
        if cv is not None:
            out["cv_all"] = float(cv.mean()); out["cv_all_ci"] = boot_ci(cv)
    dep_ids = [i for g, ids in gidx.items() if g != "E" for i in ids]  # deployable candidates (E = surrogate upper bound)
    if dep_ids:
        best = (cm[:, dep_ids] - a0m[:, None]).max(axis=1); out["opt_dep"] = float(best.mean()); out["opt_dep_ci"] = boot_ci(best)
        cv = cross(dep_ids)
        if cv is not None:
            out["cv_dep"] = float(cv.mean()); out["cv_dep_ci"] = boot_ci(cv)
    if "F" in gidx:
        f = cm[:, gidx["F"][0]] - a0m; out["F_gain"] = float(f.mean()); out["F_ci"] = boot_ci(f)
    return out


def fmt(v, a0):
    if not a0 > 0.5:  # a negative or tiny A0 mean (cells where the base controller falls) makes a % meaningless
        return f"{v:+.3f}"
    return f"{v:+.3f} ({100 * v / a0:+.1f} %)"


def fmt_ci(ci, a0):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+"); ap.add_argument("--by-speed", action="store_true"); ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--groups", default=",".join(GROUPS))
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in a.csv], ignore_index=True)
    groups = a.groups.split(",")
    lines = []
    cells = ["model", "condition"] + (["speed"] if a.by_speed else [])
    if a.fidelity:
        lines.append("| model | condition | speed | n bp | A0 mean return | unperturbed clone SD (A0 / A1) | restore error mean / mean abs | A0 fell | anchor fell |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for k, x in d.groupby(["model", "condition", "speed"]):
            s = cell_stats(x, ["A1"])
            sd1 = x[x.group == "A1"].groupby(["speed", "seed", "step"])["ret"].std().mean()
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {s['n_bp']} | {s['a0_mean']:.3f} | {s['a0_clone_sd']:.4f} / {sd1:.4f} | {s['restore_err_mean']:+.4f} / {s['restore_err_abs']:.4f} | {s['a0_fell']:.3f} | {s['anchor_fell']:.3f} |")
    else:
        lines.append("| model | condition | speed | n bp | A0 mean | (a) clone SD / restore err | (c) opt all | (d) cv all [95% CI] | (d) cv without E [95% CI] | (e) F − A0 [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        detail = ["| model | condition | speed | group | n cand | (b) worse / fell | (c) opt | (d) cv [95% CI] |", "|---|---|---|---|---|---|---|---|"]
        for k, x in d.groupby(cells):
            s = cell_stats(x, groups); a0 = s["a0_mean"]
            sp = k[2] if a.by_speed else "all"
            fg = f"{fmt(s['F_gain'], a0)} {fmt_ci(s['F_ci'], a0)}" if "F_gain" in s else "-"
            lines.append(f"| {k[0]} | {k[1]} | {sp} | {s['n_bp']} | {a0:.3f} | {s['a0_clone_sd']:.3f} / {s['restore_err_abs']:.3f} | {fmt(s.get('opt_all', np.nan), a0)} | {fmt(s.get('cv_all', np.nan), a0)} {fmt_ci(s.get('cv_all_ci', (np.nan, np.nan)), a0)} | {fmt(s.get('cv_dep', np.nan), a0)} {fmt_ci(s.get('cv_dep_ci', (np.nan, np.nan)), a0)} | {fg} |")
            for g, st in s["groups"].items():
                cvs = f"{fmt(st['cv'], a0)} {fmt_ci(st['cv_ci'], a0)}" if "cv" in st else "-"
                detail.append(f"| {k[0]} | {k[1]} | {sp} | {g} | {st['n_cand']} | {st['worse']:.2f} / {st['fell']:.3f} | {fmt(st['opt'], a0)} | {cvs} |")
        lines += ["", "Per group:", ""] + detail
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
