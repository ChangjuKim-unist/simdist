"""Design doc 36 check C3: does the planner's score rank its own candidates like their realised return?

Input: branch.csv files of scripts/sim2sim/branch_eval.py mode=branch with group M (anchor drive=planner, branch.m_record=true),
one per condition. Per branch point there are 62 sampled candidates with one run each (gauss / policy of the first and last MPPI
iteration, top = best-scored of the last iteration) and the executed plan (mean) and the policy head's chunk with 3 clones each.
Every candidate ran open loop for 25 steps, then the policy head to 125 steps.

  R1  Spearman rank correlation per branch point between model score (m_score) and realised 125-step return (ret), over the
      62 sampled candidates and over the last-iteration ones; also against the 25-step return (ret_chunk)
  R2  the same within the score-stratified Gaussian candidates
  R3  top 10 % by model score per point: share of Gaussian samples, share whose return is below the head chunk (mean of clones)
  R4  paired realised-return differences across points: executed mean - head, best-scored candidate - head (% of mean |head|);
      clone noise SD of mean and head
  R5  fall rate per candidate kind
Means over branch points with a bootstrap 95 % CI over points (5000 resamples, seed 36). numpy only.

    python scripts/sim2sim/stage36_rank.py tow20=/x/c3_tow20_branch.csv nominal=/x/c3_nominal_branch.csv \
        --md /x/stage36_rank.md --json /x/stage36_rank.json
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

NB, SEED = 5000, 36
SAMPLED = ("gauss", "policy", "top")
KINDS = ("gauss_first", "policy_first", "gauss_last", "policy_last", "top_last", "mean", "head")


def rankdata(x):
    """Ranks 1..n with average ranks for ties."""
    x = np.asarray(x, np.float64); o = np.argsort(x, kind="mergesort"); xs = x[o]
    _, start, cnt = np.unique(xs, return_index=True, return_counts=True)
    r = np.empty(len(x)); r[o] = np.repeat(start + (cnt - 1) / 2.0, cnt) + 1.0
    return r


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def boot(num, den=None):
    """Mean over points (or 100 x mean(num) / mean(|den|)) with a percentile bootstrap 95 % CI over points."""
    num = np.asarray(num, np.float64)
    ok = np.isfinite(num) if den is None else np.isfinite(num) & np.isfinite(den)
    num = num[ok]; n = len(num)
    if n == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    idx = np.random.default_rng(SEED).integers(0, n, (NB, n))
    if den is None:
        est, bs = num.mean(), num[idx].mean(1)
    else:
        den = np.abs(np.asarray(den, np.float64)[ok])
        est, bs = 100.0 * num.mean() / den.mean(), 100.0 * num[idx].mean(1) / den[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(est), "lo": float(lo), "hi": float(hi), "n": int(n)}


def load_points(path):
    pts = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["group"] != "M":
                continue
            pts[(r["model"], r["condition"], float(r["speed"]), int(r["seed"]), int(r["step"]))].append(r)
    return pts


def point_stats(rows):
    """Per-point numbers (None if the point is incomplete)."""
    f = lambda rs, k: np.array([float(r[k]) for r in rs])  # noqa: E731
    smp = [r for r in rows if r["m_kind"] in SAMPLED]
    mean = [r for r in rows if r["m_kind"] == "mean"]; head = [r for r in rows if r["m_kind"] == "head"]
    if len(smp) < 3 or not mean or not head:
        return None
    s, ret, r25 = f(smp, "m_score"), f(smp, "ret"), f(smp, "ret_chunk")
    last = np.array([r["m_iter"] == "last" for r in smp]); gs = np.array([r["m_kind"] == "gauss" for r in smp])
    isg = np.array([int(r["m_gauss"]) == 1 for r in smp])
    head_ret = f(head, "ret").mean(); mean_ret = f(mean, "ret").mean()
    ntop = max(1, int(round(0.1 * len(smp)))); top = np.argsort(-s, kind="stable")[:ntop]
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan  # noqa: E731
    out = {"rho_all": spearman(s, ret), "rho_last": spearman(s[last], ret[last]), "rho_all25": spearman(s, r25),
           "rho_gauss": spearman(s[gs], ret[gs]), "rho_gauss_last": spearman(s[gs & last], ret[gs & last]),
           "top_gauss": float(isg[top].mean()), "top_below_head": float((ret[top] < head_ret).mean()),
           "d_mean_head": mean_ret - head_ret, "d_best_head": float(ret[int(np.argmax(s))]) - head_ret, "head": head_ret,
           "var_mean": sd(f(mean, "ret")) ** 2, "var_head": sd(f(head, "ret")) ** 2}
    for k in KINDS:
        rs = [r for r in rows if (r["m_kind"] + (f"_{r['m_iter']}" if r["m_iter"] else "")) == k]
        out[f"fall_{k}"] = float(np.mean([int(r["fell"]) for r in rs])) if rs else np.nan
    return out


def summarize(pts):
    P = [p for p in (point_stats(rows) for rows in pts.values()) if p is not None]
    col = lambda k: np.array([p[k] for p in P], np.float64)  # noqa: E731
    res = {"points": len(P), "points_skipped": len(pts) - len(P)}
    for k in ("rho_all", "rho_last", "rho_all25", "rho_gauss", "rho_gauss_last", "top_gauss", "top_below_head"):
        v = col(k); res[k] = boot(v)
        if k.startswith("rho"):  # share of points with a positive correlation (points with an undefined one left out)
            v = v[np.isfinite(v)]; res[k + "_pos"] = float((v > 0).mean()) if len(v) else np.nan
    res["d_mean_head_pct"] = boot(col("d_mean_head"), col("head")); res["d_best_head_pct"] = boot(col("d_best_head"), col("head"))
    res["head_abs_mean"] = float(np.mean(np.abs(col("head")))) if len(P) else np.nan
    for k in ("mean", "head"):  # pooled clone noise SD = sqrt(mean per-point variance), also in % of mean |head|
        v = col(f"var_{k}"); ok = np.isfinite(v); n = int(ok.sum())
        if n:
            idx = np.random.default_rng(SEED).integers(0, n, (NB, n)); bs = np.sqrt(v[ok][idx].mean(1))
            res[f"sd_{k}"] = {"mean": float(np.sqrt(v[ok].mean())), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": n}
        else:
            res[f"sd_{k}"] = {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    for k in KINDS:
        res[f"fall_{k}"] = boot(col(f"fall_{k}"))
    return res


def fmt(b, d=3, pct=False):
    u = " %" if pct else ""
    return f"{b['mean']:.{d}f}{u}", f"[{b['lo']:.{d}f}, {b['hi']:.{d}f}]{u}"


def report(R):
    names = list(R); L = ["# 설계 36 C3: planner 점수가 자기 후보의 실제 성적 순서를 맞히는가", ""]
    L += ["각 분기점(branch point)은 적응된 planner가 실제로 걸으며 지나간 상태 하나다. 그 제어 단계에서 planner가 점수를 매긴 후보 "
          "64개(첫/마지막 반복의 Gaussian 표본을 점수 순위로 고르게 뽑은 것, 정책 혼합 표본, 마지막 반복의 최고 점수 후보, "
          "실제로 실행한 평균 계획 mean, 정책 헤드 chunk head)를 시뮬레이션에서 25 step open loop로 실행한 뒤 정책 헤드로 125 step까지 "
          "이어 걸었다. '모델 점수' = 25 step 예측 보상의 할인 합 + 끝 value (planner가 후보를 고를 때 쓰는 값). "
          "'실제 return' = 시뮬레이션에서 받은 125 step 보상 합. 괄호는 분기점에 대한 bootstrap 95 % 신뢰구간(5000회, seed 36).", ""]
    L += ["| 조건 | 분기점 수 | 빠진 분기점 |", "|---|---|---|"] + [f"| {c} | {R[c]['points']} | {R[c]['points_skipped']} |" for c in names] + [""]

    def rho_table(title, what, k):
        t = [f"## {title}", "", what, "", "| 조건 | 평균 Spearman | 95 % CI | 양(+)인 분기점 비율 |", "|---|---|---|---|"]
        for c in names:
            m, ci = fmt(R[c][k]); t.append(f"| {c} | {m} | {ci} | {R[c][k + '_pos']:.2f} |")
        return t + [""]

    def mean_table(title, what, k, head="평균", d=3, pct=False):
        t = [f"## {title}", "", what, "", f"| 조건 | {head} | 95 % CI |", "|---|---|---|"]
        for c in names:
            m, ci = fmt(R[c][k], d, pct); t.append(f"| {c} | {m} | {ci} |")
        return t + [""]

    L += rho_table("R1a. 표본 후보 62개: 모델 점수 순위 vs 실제 125 step return 순위",
                   "분기점마다 62개 표본 후보(두 반복 합침)의 모델 점수 순위와 실제 return 순위의 Spearman 상관을 구해 분기점 평균. "
                   "1 = 순서를 완전히 맞힘, 0 = 무관.", "rho_all")
    L += rho_table("R1b. 마지막 반복 후보만: 모델 점수 vs 실제 125 step return",
                   "planner가 실제로 고르는 단계(마지막 반복)의 후보 34개(Gaussian 24, 정책 2, 최고 점수 8)만으로 같은 상관.", "rho_last")
    L += rho_table("R1c. 표본 후보 62개: 모델 점수 vs 실제 25 step return",
                   "R1a와 같은 후보, 실제 return을 후보 자신이 실행된 첫 25 step으로만 잰 것(모델 점수가 예측하는 보상 구간).", "rho_all25")
    L += rho_table("R2a. Gaussian 후보만(두 반복 48개): 모델 점수 vs 실제 125 step return",
                   "점수 순위로 고르게 뽑은 Gaussian 표본만의 상관(점수 범위 전체를 덮는 후보 집합).", "rho_gauss")
    L += rho_table("R2b. 마지막 반복 Gaussian 후보만(24개): 모델 점수 vs 실제 125 step return",
                   "R2a 중 마지막 반복만.", "rho_gauss_last")
    L += mean_table("R3a. 모델 점수 상위 10 % 후보 중 Gaussian 표본의 비율",
                    "분기점마다 62개 표본 후보 중 모델 점수 상위 10 %(6개)가 Gaussian 표본(정책 혼합 표본이 아닌 것)인 비율의 평균.",
                    "top_gauss", "Gaussian 비율")
    L += mean_table("R3b. 모델 점수 상위 10 % 후보 중 실제 return이 정책 헤드보다 낮은 비율",
                    "상위 10 % 후보 각각의 실제 125 step return이 같은 분기점의 정책 헤드 chunk(clone 3개 평균)보다 낮은 비율의 평균. "
                    "높을수록 점수 꼬리가 '실제로는 헤드보다 나쁜' 후보로 채워져 있다는 뜻.", "top_below_head", "헤드보다 낮은 비율")
    L += mean_table("R4a. 실제로 실행한 계획(mean) - 정책 헤드 chunk: 실제 125 step return 차이",
                    "분기점마다 (mean clone 3개 평균 return) - (head clone 3개 평균 return)을 구해 평균하고, 헤드 return 절댓값 평균에 대한 % 로 표시. "
                    "양수 = planner가 고른 계획이 헤드보다 실제로 낫다.", "d_mean_head_pct", "차이 (헤드 절댓값 평균 대비 %)", 2, True)
    L += mean_table("R4b. 모델 점수 1위 표본 후보 - 정책 헤드 chunk: 실제 125 step return 차이",
                    "62개 중 모델 점수가 가장 높은 후보 하나의 실제 return - 헤드 clone 평균, 헤드 절댓값 평균에 대한 %.",
                    "d_best_head_pct", "차이 (헤드 절댓값 평균 대비 %)", 2, True)
    t = ["## R4c. clone 잡음: 같은 계획을 미세하게 흔든 상태에서 다시 실행했을 때 return의 표준편차", "",
         "mean과 head 각각 clone 3개(1, 2번은 작은 상태 섭동)의 분기점별 분산을 평균해 제곱근(합동 SD). R4a/R4b 차이를 읽을 때의 잡음 크기.", "",
         "| 조건 | 후보 | 합동 SD | 95 % CI | 헤드 절댓값 평균 대비 % |", "|---|---|---|---|---|"]
    for c in names:
        for k in ("mean", "head"):
            m, ci = fmt(R[c][f"sd_{k}"]); t.append(f"| {c} | {k} | {m} | {ci} | {100 * R[c][f'sd_{k}']['mean'] / R[c]['head_abs_mean']:.2f} |")
    L += t + [""]
    t = ["## R5. 후보 종류별 넘어짐 비율", "", "125 step 안에 넘어진 실행의 비율(분기점별 비율의 평균, 괄호는 95 % CI). 열 = 조건.", "",
         "| 후보 종류 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in KINDS:
        t.append(f"| {k} | " + " | ".join("{} {}".format(*fmt(R[c][f'fall_{k}'])) for c in names) + " |")
    L += t + [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="name=path/to/branch.csv (one per condition)")
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    R = {}
    for item in a.csvs:
        name, path = item.split("=", 1)
        R[name] = summarize(load_points(path))
    md = report(R)
    with open(a.md, "w") as f:
        f.write(md + "\n")
    with open(a.json, "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=float)
    print(md)


if __name__ == "__main__":
    main()
