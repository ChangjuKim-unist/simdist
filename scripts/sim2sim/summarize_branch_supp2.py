"""Design 18 section 8.2 / 8.3 (checks 2 and 3): tables from the supplement branch rows.

    python scripts/sim2sim/summarize_branch_supp2.py --check2 results/stage18/branch2_*.csv --check3 results/stage18/branch3_*.csv --out results/stage18/summary_supp2.md

Check 2 (per cell, paired over branch points, clone means): A1 - A0 (loss of feedback alone), F - A0 (shadow plan open loop),
F_cl - A0 (planner replanning every step), F_cl - F, C_cl - A0 (generator mean closed loop); fall shares; per speed;
reward-term decomposition of the candidate window (25 steps) for D best-of-6, F_cl and C_cl: tracking terms vs penalties.
Check 3 (generator-mean base controller): noise floor, cv headroom with / without E, per group (D, E, C open loop, G closed loop),
feedback-keeping-only headroom (D+E, D).  Returns = sum of the evaluator's reward over 125 steps, higher is better.
"""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from summarize_branch import boot_ci, cell_stats, fmt, fmt_ci  # noqa: E402
from summarize_branch_supp import NAMES, CELL_ORDER  # noqa: E402

KEY = ["speed", "seed", "step"]
TRACK = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "foot_clearance", "gait", "feet_height_gait"]


def cells_of(d):
    cells = [c for c in CELL_ORDER if ((d.model == c[0]) & (d.condition == c[1])).any()]
    return cells + sorted({(m, c) for m, c in zip(d.model, d.condition)} - set(cells))


def paired(d, g, ref="A0", col="ret"):
    """clone-mean of group g minus clone-mean of ref per branch point -> (mean, ci, fall share of g, n)."""
    a = d[d.group == ref].groupby(KEY)[col].mean()
    x = d[d.group == g].groupby(KEY).agg(r=(col, "mean"), f=("fell", "mean")).join(a.rename("a"), how="inner")
    diff = (x.r - x.a).values
    return float(diff.mean()), boot_ci(diff), float(x.f.mean()), len(x)


def term_cols(d, prefix):
    return [c for c in d.columns if c.startswith(prefix)]


def term_split(d, g, ref="A0"):
    """Candidate-window (25 steps) reward-term difference g - ref, split into tracking-type terms and penalties."""
    cc = term_cols(d, "c_")
    if not cc:
        return None
    a = d[d.group == ref].groupby(KEY)[cc].mean()
    x = d[d.group == g].groupby(KEY)[cc].mean().join(a, how="inner", rsuffix="_ref")
    out = {}
    for c in cc:
        out[c[2:]] = float((x[c] - x[c + "_ref"]).mean())
    tr = sum(v for k, v in out.items() if k in TRACK); pen = sum(v for k, v in out.items() if k not in TRACK)
    top = sorted(out.items(), key=lambda kv: -abs(kv[1]))[:3]
    return tr, pen, top


def best_of_D(d):
    """Rows of the optimistic best-of-6 D variant per branch point (clone-mean), as a pseudo group 'Dbest'."""
    dd = d[d.group == "D"]
    if dd.empty:
        return d
    m = dd.groupby(KEY + ["cand"]).ret.mean().reset_index()
    pick = m.loc[m.groupby(KEY).ret.idxmax(), KEY + ["cand"]]
    rows = dd.merge(pick, on=KEY + ["cand"]).copy(); rows["group"] = "Dbest"; rows["cand"] = 0
    return pd.concat([d, rows], ignore_index=True)


def check2(files, L):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = best_of_D(d)
    L += ["### 확인 2. 공정한 비교 (같은 런 안에서 짝지은 복제본 평균 차이, 125스텝 수익, 클수록 좋음; [ ] = 분기점 부트스트랩 95 % CI)", "",
          "| 모델 · 조건 | 분기점 | A0 평균 | A1 − A0 (피드백 상실만) | F − A0 (계획 개루프) | F_cl − A0 (매 스텝 재계획) | F_cl − F | C_cl − A0 (생성기 평균 폐루프) | 넘어짐 A0 / A1 / F / F_cl / C_cl |",
          "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for m, c in cells_of(d):
        x = d[(d.model == m) & (d.condition == c)]
        a0 = x[x.group == "A0"].groupby(KEY).ret.mean(); a0m = float(a0.mean()); n = len(a0)
        r = {g: paired(x, g) for g in ("A1", "F", "Fcl", "Ccl") if (x.group == g).any()}
        ff = paired(x, "Fcl", ref="F") if "F" in r and "Fcl" in r else None
        f0 = float(x[x.group == "A0"].groupby(KEY).fell.mean().mean())
        fs = " / ".join([f"{f0:.2f}"] + [f"{r[g][2]:.2f}" if g in r else "-" for g in ("A1", "F", "Fcl", "Ccl")])
        cell = lambda g: f"{fmt(r[g][0], a0m)} {fmt_ci(r[g][1], a0m)}" if g in r else "-"  # noqa: E731
        L.append(f"| {NAMES.get(m, m)} {c} | {n} | {a0m:.2f} | {cell('A1')} | {cell('F')} | {cell('Fcl')} | "
                 f"{(fmt(ff[0], a0m) + ' ' + fmt_ci(ff[1], a0m)) if ff else '-'} | {cell('Ccl')} | {fs} |")
        res[(m, c)] = {"a0": a0m, "r": r, "ff": ff}
    # per speed
    L += ["", "#### 확인 2 속도별 (A0 대비 차이)", "", "| 모델 · 조건 | 속도 | 분기점 | A0 평균 | A1 − A0 | F − A0 | F_cl − A0 | C_cl − A0 |", "|---|---|---|---|---|---|---|---|"]
    for m, c in cells_of(d):
        for v, x in d[(d.model == m) & (d.condition == c)].groupby("speed"):
            a0 = x[x.group == "A0"].groupby(KEY).ret.mean(); a0m = float(a0.mean())
            r = {g: paired(x, g) for g in ("A1", "F", "Fcl", "Ccl") if (x.group == g).any()}
            cell = lambda g: fmt(r[g][0], a0m) if g in r else "-"  # noqa: E731
            L.append(f"| {NAMES.get(m, m)} {c} | {v} | {len(a0)} | {a0m:.2f} | {cell('A1')} | {cell('F')} | {cell('Fcl')} | {cell('Ccl')} |")
    # term decomposition
    L += ["", "#### 확인 2 보상 항 분해 (후보 25스텝 구간, 후보 − A0; 추종형 = 속도·각속도 추종·발 체공·발 높이·보행 항, 벌점 = 나머지 9항; 상위 3항)", "",
          "| 모델 · 조건 | 후보 | 추종형 합 | 벌점 합 | 상위 3항 (후보 − A0) |", "|---|---|---|---|---|"]
    for m, c in cells_of(d):
        x = d[(d.model == m) & (d.condition == c)]
        for g, name in (("Dbest", "D 최선(낙관)"), ("D", "D 평균"), ("Fcl", "F_cl"), ("F", "F"), ("Ccl", "C_cl"), ("A1", "A1")):
            if not (x.group == g).any():
                continue
            ts = term_split(x, g)
            if ts is None:
                continue
            tr, pen, top = ts
            L.append(f"| {NAMES.get(m, m)} {c} | {name} | {tr:+.3f} | {pen:+.3f} | " + ", ".join(f"{k} {v:+.3f}" for k, v in top) + " |")
    return res


def check3(files, L):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    groups = ["D", "E", "C", "G"]; combos = {"D+E": ["D", "E"], "D only": ["D"], "C+G": ["C", "G"], "all but E": ["D", "C", "G"]}
    L += ["", "### 확인 3. 생성기 평균 정책을 기본 제어기로 (A0 = 생성기 폐루프; 교차 상한 cv = 다른 복제본으로 고르고 남은 복제본으로 채점, A0 대비)", "",
          "| 조건 | 분기점 | A0 평균 | 잡음 바닥 (복제본 SD) | 복원 오차 | A0 넘어짐 | cv E 제외 [95 % CI] | cv 전체 | cv D | cv E | cv C (개루프) | cv G (폐루프) | 넘어짐 D / E / C / G |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["nominal", "motor0.4", "friction0.15", "mass3", "motor0.35", "friction0.1"]
    conds = [c for c in order if (d.condition == c).any()] + sorted(set(d.condition) - set(order))
    res = {}
    for c in conds:
        x = d[d.condition == c]
        s = cell_stats(x, groups, combos); a0 = s["a0_mean"]; G = s["groups"]
        f = lambda k: fmt(G[k]["cv"], a0) if k in G and "cv" in G[k] else "-"  # noqa: E731
        fl = " / ".join(f"{G[k]['fell']:.2f}" if k in G else "-" for k in ("D", "E", "C", "G"))
        L.append(f"| {c} | {s['n_bp']} | {a0:.2f} | {s['a0_clone_sd']:.3f} | {s['restore_err_abs']:.3f} | {s['a0_fell']:.3f} | "
                 f"{fmt(s.get('cv_dep', np.nan), a0)} {fmt_ci(s.get('cv_dep_ci', (np.nan, np.nan)), a0)} | {fmt(s.get('cv_all', np.nan), a0)} | {f('D')} | {f('E')} | {f('C')} | {f('G')} | {fl} |")
        res[c] = s
    L += ["", "#### 확인 3 속도별 (cv E 제외 / cv D+E / cv D)", "", "| 조건 | 속도 | 분기점 | A0 평균 | cv E 제외 | cv D+E | cv D | cv C+G |", "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        for v, x in d[d.condition == c].groupby("speed"):
            s = cell_stats(x, groups, combos); a0 = s["a0_mean"]; G = s["groups"]
            f = lambda k: fmt(G[k]["cv"], a0) if k in G and "cv" in G[k] else "-"  # noqa: E731
            L.append(f"| {c} | {v} | {s['n_bp']} | {a0:.2f} | {fmt(s.get('cv_dep', np.nan), a0)} | {f('D+E')} | {f('D only')} | {f('C+G')} |")
    L += ["", "#### 확인 3 피드백 유지 후보만의 여지", "", "| 조건 | A0 평균 | 잡음 바닥 | D+E [95 % CI] | D만 [95 % CI] | D+E 넘어짐 | D 넘어짐 |", "|---|---|---|---|---|---|---|"]
    for c in conds:
        s = res[c]; a0 = s["a0_mean"]; G = s["groups"]; de, dd = G.get("D+E", {}), G.get("D only", {})
        L.append(f"| {c} | {a0:.2f} | {s['a0_clone_sd']:.3f} | {fmt(de.get('cv', np.nan), a0)} {fmt_ci(de.get('cv_ci', (np.nan, np.nan)), a0)} | "
                 f"{fmt(dd.get('cv', np.nan), a0)} {fmt_ci(dd.get('cv_ci', (np.nan, np.nan)), a0)} | {de.get('fell', np.nan):.3f} | {dd.get('fell', np.nan):.3f} |")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check2", nargs="*", default=[]); ap.add_argument("--check3", nargs="*", default=[]); ap.add_argument("--out")
    a = ap.parse_args(); L = []
    if a.check2:
        check2(a.check2, L)
    if a.check3:
        check3(a.check3, L)
    text = "\n".join(L); print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
