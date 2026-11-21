"""Design doc 36 check C2 (docs/design36_sampling_checks_ko.md): planner vs policy alone on the design-34 logs (no new runs).

(a) task score S (design doc 35) of policy alone (POL), pretrained planner (PRE), adapted planner (ADP) per condition
(b) inside PRE / ADP runs, 50-step mid-walk windows: promised gain dJ = model score of the executed plan - of the policy
    head's plan (R + V, as recorded), policy deviation d = |a - a_pol|, realised S per step; within-robot Spearman(dJ, S)
    and the S gap between the top and bottom quarter of windows by d

    python3 scripts/sim2sim/stage36_logs.py --d34 results/stage34 --out results/stage36/c2.md
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage35_score as s35  # noqa: E402

CONDS = ["legR50", "delay2", "slope5", "tow20", "mot50"]
KO = {"nominal": "평소", "legR50": "뒷다리 하나 50 %", "delay2": "지연 40 ms", "slope5": "마찰 0.3 + 오르막 5°", "tow20": "끄는 짐 20 N",
      "mot50": "모터 전체 50 %"}
WIN = 50; NB = 5000


def per_step(path):
    z = np.load(path, allow_pickle=True)
    names = list(z["term_names"]); terms = z["terms"].astype(np.float64)
    s = z["since"].astype(np.int64); ok = (~z["resetting"].astype(bool)) & (s >= s35.MID)
    v = z["lin_vel"].astype(np.float64); w = z["ang_vel"][..., 2].astype(np.float64); sp = z["speeds"].astype(np.float64)
    S = s35.DT * (1.5 * np.exp(-(((v[..., 0] - sp[None]) ** 2) + v[..., 1] ** 2) / 0.25) + 0.75 * np.exp(-(w ** 2) / 0.25))
    S = S + sum(terms[..., names.index(k)] for k in s35.PEN if k in names)
    dJ = (z["R_exec"] + z["V_exec"] - z["R_pol"] - z["V_pol"]).astype(np.float64)
    d = np.linalg.norm(z["act"].astype(np.float64) - z["act_pol"].astype(np.float64), axis=-1)
    return S, dJ, d, z["reward"].astype(np.float64), ok


def ranks(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); sx = x[o]
    i = 0
    while i < len(x):  # average ties
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0; i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def windows(S, dJ, d, rew, ok):
    """per robot: list of (mean dJ, mean d, mean S, mean reward) over non-overlapping 50-step windows fully mid-walk"""
    out = []
    for r in range(S.shape[1]):
        rows = []
        for t0 in range(0, S.shape[0] - WIN + 1, WIN):
            sl = slice(t0, t0 + WIN)
            if ok[sl, r].all() and np.isfinite(dJ[sl, r]).all():
                rows.append((dJ[sl, r].mean(), d[sl, r].mean(), S[sl, r].mean(), rew[sl, r].mean()))
        out.append(np.array(rows) if rows else np.zeros((0, 4)))
    return out


def robot_stats(W):
    """per robot: Spearman(dJ, S), Spearman(dJ, reward), S(top quarter by d) - S(bottom quarter by d)"""
    res = []
    for w in W:
        if len(w) < 8:
            continue
        q = len(w) // 4; o = np.argsort(w[:, 1])
        res.append((spearman(w[:, 0], w[:, 2]), spearman(w[:, 0], w[:, 3]), w[o[-q:], 2].mean() - w[o[:q], 2].mean()))
    return np.array(res)


def boot(x, seed=36):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); b = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d34", default="results/stage34"); ap.add_argument("--out", default="results/stage36/c2.md")
    a = ap.parse_args(); L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    J = {"a": {}, "b": {}}
    f = lambda c, arm: os.path.join(a.d34, f"diag_{c}_{arm}.npz")  # noqa: E731
    P("## 설계 36 C2: planner와 정책만 (설계 34 로그, 새 실행 없음)"); P()
    P("S = 과제 점수(설계 35 정의), 걷는 중 스텝 평균. 차이는 정책만 S 대비 %, [95 % 구간] = 로봇 18대 짝 bootstrap."); P()
    P("### (a) 정책만 대비 planner의 S"); P()
    P("| 조건 | 정책만 S | 적응 전 planner − 정책만 | 적응 뒤 planner − 정책만 | 판정 (적응 뒤 ≥ +2 %, 하한 > 0) |"); P("|---|---|---|---|---|")
    for c in ["nominal"] + CONDS:
        pol, pre, adp = (s35.load(f(c, arm)) for arm in ("POL", "PRE", "ADP"))
        if pol is None or pre is None:
            continue
        ref = s35.pooled(pol, "S"); dp = s35.diff(pre, pol, "S", ref); da = s35.diff(adp, pol, "S", ref) if adp is not None else None
        ok = da is not None and da[0] >= 2.0 and da[1] > 0
        fm = lambda t: f"{t[0]:+.1f} % [{t[1]:+.1f}, {t[2]:+.1f}]" if t is not None else "-"  # noqa: E731
        P(f"| {KO[c]} | {ref:.4f} | {fm(dp)} | {fm(da)} | {'낫다' if ok else ('-' if da is None else '아님')} |")
        J["a"][c] = {"pol_S": ref, "pre_minus_pol": dp, "adp_minus_pol": da, "better": bool(ok)}
    P()
    P(f"### (b) planner 런 안의 {WIN}스텝 창: 모델이 약속한 이득과 실제 결과"); P()
    P("ΔĴ = 실행 계획의 모델 점수 − 같은 입력의 정책 head 계획 점수(예측 보상 합 + value). d = ‖실행 행동 − 정책 행동‖. "
      "상관은 로봇마다 창끼리 Spearman을 내고 로봇 평균. d 상하위 차 = 로봇 안에서 d 상위 1/4 창의 S − 하위 1/4 창의 S (정책만 S 대비 %)."); P()
    P("| 조건 | 팔 | 창 수 | ΔĴ 평균 | Spearman(ΔĴ, 실제 S) | Spearman(ΔĴ, 실제 보상) | d 상위 − 하위 1/4 S | 정보 없음 (< 0.1) | 벗어날수록 손해 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for c in ["nominal"] + CONDS:
        ref = s35.pooled(s35.load(f(c, "POL")), "S") if os.path.exists(f(c, "POL")) else np.nan
        for arm in ("PRE", "ADP"):
            if not os.path.exists(f(c, arm)):
                continue
            S, dJ, d, rew, ok = per_step(f(c, arm)); W = windows(S, dJ, d, rew, ok); R = robot_stats(W)
            if len(R) == 0:
                continue
            nwin = sum(len(w) for w in W); mdj = float(np.mean(np.concatenate([w[:, 0] for w in W if len(w)])))
            sS, sR, gap = boot(R[:, 0]), boot(R[:, 1]), boot(100.0 * R[:, 2] / ref)
            noinfo = sS[0] < 0.1; worse = gap[2] < 0
            P(f"| {KO[c]} | {'적응 전' if arm == 'PRE' else '적응 뒤'} | {nwin} | {mdj:+.3f} | {sS[0]:+.2f} [{sS[1]:+.2f}, {sS[2]:+.2f}] | "
              f"{sR[0]:+.2f} [{sR[1]:+.2f}, {sR[2]:+.2f}] | {gap[0]:+.1f} % [{gap[1]:+.1f}, {gap[2]:+.1f}] | {'예' if noinfo else '아니오'} | {'예' if worse else '아니오'} |")
            J["b"][f"{c}_{arm}"] = {"n_windows": nwin, "dJ_mean": mdj, "rho_S": sS, "rho_reward": sR, "d_gap_pct": gap,
                                   "no_info": bool(noinfo), "worse_when_far": bool(worse)}
    P()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n"); json.dump(J, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
