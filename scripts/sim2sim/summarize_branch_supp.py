"""Design 18 section 8.1 (check 1): decomposition of the gate-0 headroom by candidate group, on the gate-0 rows.

    python scripts/sim2sim/summarize_branch_supp.py results/stage18/branch_*.csv --out results/stage18/summary_supp.md

Tables (all returns = sum of the evaluator's reward over the 125 steps after the branch point, higher is better;
gains = minus A0, the base controller closed loop; cv = cross-validated bound of summarize_branch.py):
  1. per cell x group (all 11 cells): cv headroom, share of candidates worse than A0, fall share; unions
     feedback-keeping (D, E) / feedback-free (A1, B, C, F) / D alone / D+E
  2. command variants D per label: mean gain, fall share, and the gain split by the A0 outcome at the branch point
     (A0 fell within 125 steps -> "fall avoidance" share; A0 stayed up -> "tracking/posture" share)
  3. per speed command: cv of all deployable (no E), of D+E, of D
  4. the remaining headroom with feedback-keeping candidates only (D+E and D), per cell
"""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from summarize_branch import cell_stats, fmt, fmt_ci  # noqa: E402

COMBOS = {"D+E (feedback kept)": ["D", "E"], "A1+B+C+F (open loop)": ["A1", "B", "C", "F"], "D only": ["D"], "all but E": ["A1", "B", "C", "D", "F"]}
CELL_ORDER = [("broadwm", "nominal"), ("broadwm", "mass3"), ("broadwm", "friction0.15"), ("broadwm", "motor0.4"),
              ("paper", "nominal"), ("paper", "friction0.15"), ("paper", "motor0.4"),
              ("paperfr", "friction0.15"), ("paperfr", "motor0.4"), ("papermotor", "friction0.15"), ("papermotor", "motor0.4")]
NAMES = {"broadwm": "넓은 모델", "paper": "논문 모델", "paperfr": "마찰 적응 모델", "papermotor": "모터 적응 모델", "gen": "생성기 정책"}


def label_stats(d):
    """Per D label: clone-mean gain vs A0, fall share, split by the A0 outcome (any A0 clone fell)."""
    J = int(d.clone.max()) + 1
    key = ["speed", "seed", "step"]
    a0 = d[d.group == "A0"].groupby(key).agg(a0=("ret", "mean"), a0_fell=("fell", "max"), n=("ret", "size"))
    a0 = a0[a0.n == J]
    rows = []
    dd = d[d.group == "D"]
    per_lab = {lab: x.groupby(key).agg(r=("ret", "mean"), fell=("fell", "mean")) for lab, x in dd.groupby("label")}
    # optimistic best-of-D per branch point (clone-mean max over the six variants)
    best = pd.concat(per_lab.values(), keys=per_lab.keys(), names=["label"]).reset_index().groupby(key).apply(
        lambda g: g.loc[g.r.idxmax(), ["r", "fell"]]).rename(columns={"r": "r", "fell": "fell"})
    per_lab["D best-of-6 (낙관)"] = best
    for lab, g in per_lab.items():
        g = g.join(a0, how="inner")
        gain = g.r - g.a0; fell_a0 = g.a0_fell.astype(bool)
        share = fell_a0.mean()
        rows.append({"label": lab, "n": len(g), "gain": gain.mean(), "fell": g.fell.mean(), "worse": (gain < 0).mean(),
                     "a0_fell_share": share, "gain_when_a0_fell": gain[fell_a0].mean() if fell_a0.any() else np.nan,
                     "gain_when_a0_up": gain[~fell_a0].mean() if (~fell_a0).any() else np.nan,
                     "contrib_fall_avoid": (gain[fell_a0].sum() / len(g)) if fell_a0.any() else 0.0,
                     "contrib_tracking": (gain[~fell_a0].sum() / len(g)) if (~fell_a0).any() else 0.0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in a.csv], ignore_index=True)
    groups = ["A1", "B", "C", "D", "E", "F", "G", "Fcl", "Ccl"]
    cells = [c for c in CELL_ORDER if ((d.model == c[0]) & (d.condition == c[1])).any()]
    cells += sorted({(m, c) for m, c in zip(d.model, d.condition)} - set(cells))
    L = []
    # ---- table 1: per cell x group -------------------------------------------------------------------------
    L += ["### 표 1. 칸 × 후보군 (cv = 교차 상한, 수익 단위와 A0 대비 %; worse = A0보다 나쁜 후보 비율; fell = 넘어진 행 비율)", "",
          "| 모델 · 조건 | A0 평균 | 잡음 바닥 | 후보군 | cv [95 % CI] | worse | fell |", "|---|---|---|---|---|---|---|"]
    stats = {}
    for m, c in cells:
        x = d[(d.model == m) & (d.condition == c)]
        s = cell_stats(x, groups, COMBOS); stats[(m, c)] = s; a0 = s["a0_mean"]
        first = True
        for g, st in s["groups"].items():
            cv = f"{fmt(st['cv'], a0)} {fmt_ci(st['cv_ci'], a0)}" if "cv" in st else "-"
            L.append(f"| {NAMES.get(m, m)} {c} | {a0:.2f} | {s['a0_clone_sd']:.3f} | {g} | {cv} | {st['worse']:.2f} | {st['fell']:.3f} |" if first
                     else f"| | | | {g} | {cv} | {st['worse']:.2f} | {st['fell']:.3f} |")
            first = False
    # ---- table 2: D per label --------------------------------------------------------------------------------
    L += ["", "### 표 2. 명령 변형 D 변형별 (gain = 복제본 평균 수익 − A0; A0가 넘어진 분기점의 몫 = 넘어짐 회피, A0가 선 분기점의 몫 = 추종·자세)", "",
          "| 모델 · 조건 | 변형 | 분기점 | gain | worse | fell | A0 넘어진 분기점 비율 | gain (A0 넘어짐) | gain (A0 섬) | 회피 몫 | 추종 몫 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, c in cells:
        x = d[(d.model == m) & (d.condition == c)]
        if not (x.group == "D").any():
            continue
        t = label_stats(x); a0 = stats[(m, c)]["a0_mean"]
        for i, r in t.iterrows():
            L.append(f"| {NAMES.get(m, m)} {c} | {r.label} | {r.n} | {fmt(r.gain, a0)} | {r.worse:.2f} | {r.fell:.3f} | {r.a0_fell_share:.3f} | "
                     f"{r.gain_when_a0_fell:+.2f} | {r.gain_when_a0_up:+.3f} | {r.contrib_fall_avoid:+.3f} | {r.contrib_tracking:+.3f} |")
    # ---- table 3: per speed ----------------------------------------------------------------------------------
    L += ["", "### 표 3. 속도 명령별 교차 상한 (A0 대비)", "", "| 모델 · 조건 | 속도 | 분기점 | A0 평균 | cv E 제외 | cv D+E | cv D | cv 개루프(A1+B+C+F) |", "|---|---|---|---|---|---|---|---|"]
    for m, c in cells:
        for v, x in d[(d.model == m) & (d.condition == c)].groupby("speed"):
            s = cell_stats(x, groups, COMBOS); a0 = s["a0_mean"]; G = s["groups"]
            f = lambda k: fmt(G[k]["cv"], a0) if k in G and "cv" in G[k] else "-"  # noqa: E731
            L.append(f"| {NAMES.get(m, m)} {c} | {v} | {s['n_bp']} | {a0:.2f} | {fmt(s.get('cv_dep', np.nan), a0)} | {f('D+E (feedback kept)')} | {f('D only')} | {f('A1+B+C+F (open loop)')} |")
    # ---- table 4: remaining headroom with feedback-keeping candidates -----------------------------------------
    L += ["", "### 표 4. 피드백 유지 후보만으로 남는 여지 (교차 상한, A0 대비; 관문 1 설계의 입력)", "",
          "| 모델 · 조건 | A0 평균 | 잡음 바닥 | 전체 (E 제외) | D+E [95 % CI] | D만 [95 % CI] | D+E 넘어짐 | D 넘어짐 | D+E 몫 (E 포함 전체 대비) | D 몫 (E 제외 전체 대비) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, c in cells:
        s = stats[(m, c)]; a0 = s["a0_mean"]; G = s["groups"]
        de, dd = G.get("D+E (feedback kept)", {}), G.get("D only", {})
        share = de.get("cv", np.nan) / s["cv_all"] if s.get("cv_all") else np.nan
        share_d = dd.get("cv", np.nan) / s["cv_dep"] if s.get("cv_dep") else np.nan
        L.append(f"| {NAMES.get(m, m)} {c} | {a0:.2f} | {s['a0_clone_sd']:.3f} | {fmt(s.get('cv_dep', np.nan), a0)} | {fmt(de.get('cv', np.nan), a0)} {fmt_ci(de.get('cv_ci', (np.nan, np.nan)), a0)} | "
                 f"{fmt(dd.get('cv', np.nan), a0)} {fmt_ci(dd.get('cv_ci', (np.nan, np.nan)), a0)} | {de.get('fell', np.nan):.3f} | {dd.get('fell', np.nan):.3f} | {share:.2f} | {share_d:.2f} |")
    text = "\n".join(L); print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
