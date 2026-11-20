"""Design doc 34 (docs/design34_condition_screen_ko.md): screen table, chosen levels, and the SimDist adaptation result.

    python3 scripts/sim2sim/stage34_report.py --d results/stage34   ->  results/stage34/report.md
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage34_select  # noqa: E402
from stage34_select import metrics, agg, FAMILY  # noqa: E402

RNG = np.random.default_rng(34); NB = 5000
KO = {"nominal": "평소", "legF70": "앞다리 하나 70 %", "legF50": "앞다리 하나 50 %", "legR50": "뒷다리 하나 50 %", "delay1": "지연 20 ms", "delay2": "지연 40 ms",
      "slope3": "마찰 0.3 + 오르막 3°", "slope5": "마찰 0.3 + 오르막 5°", "tow10": "끄는 짐 10 N", "tow20": "끄는 짐 20 N", "mot50": "모터 전체 50 %"}
ARM = {"POL": "정책만", "PRE": "적응 전 planner", "ADP": "적응 뒤 planner"}


def boot(x, y, key):
    n = len(x["n_mid"]); idx = RNG.integers(0, n, (NB, n))
    b = np.array([agg(y, i)[key] - agg(x, i)[key] for i in idx])
    return agg(y)[key] - agg(x)[key], np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="results/stage34"); ap.add_argument("--out", default=None)
    ap.add_argument("--mid", type=int, default=None, help="testing only: first mid-walk step (registered value 300)")
    a = ap.parse_args()
    if a.mid is not None:
        stage34_select.MID = a.mid
    out = a.out or os.path.join(a.d, "report.md"); L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    nom = {arm: metrics(a.d, f"nominal_{arm}") for arm in ("POL", "PRE")}
    P("## 설계 34: 중간 난이도 조건 선별과 SimDist 적응 (논문 모델, 로봇 18대 × 5000스텝)"); P()
    P("걷는 중 = 리셋 뒤 300스텝 이후. 넘어짐은 1000스텝당, 보상은 스텝당, 속도 달성 = 걷는 중 평균 전진 속도 / 목표 속도."); P()
    P("### 선별: 적응 없이"); P()
    P("| 조건 | 팔 | 전체 넘어짐 | 걷는 중 넘어짐 | 걷는 중 시간 비율 | 걷는 중 보상 | 평소 대비 보상 | 속도 달성 | 옆 방향 속도 [m/s] |"); P("|---|---|---|---|---|---|---|---|---|")
    for c in ["nominal"] + list(FAMILY):
        for arm in ("POL", "PRE"):
            x = metrics(a.d, f"{c}_{arm}")
            if x is None:
                continue
            g = agg(x); ref = agg(nom[arm])["rew"] if nom[arm] is not None else np.nan
            P(f"| {KO[c]} | {ARM[arm]} | {g['falls']:.2f} | {g['haz']:.2f} | {g['share']:.2f} | {g['rew']:.4f} | {(g['rew'] / ref if ref else float('nan')):.2f} | {g['vratio']:.2f} | {g['vy']:.3f} |")
    P()
    ch = os.path.join(a.d, "chosen.txt")
    if not os.path.exists(ch):
        open(out, "w").write("\n".join(L) + "\n"); return
    chosen = [ln.split() for ln in open(ch) if ln.strip()]
    P("### 고른 수준 (등록한 규칙): " + ", ".join(f"{KO[c]}{'' if w == 'walks' else ' (걷는 수준 없음)'}" for _, c, w in chosen)); P()
    P("### 적응: 고른 조건에서 dynamics fine-tuning (시드 42) 전후, planner"); P()
    P("| 조건 | 팔 | 전체 넘어짐 | 출발 구간 넘어짐 | 걷는 중 넘어짐 | 걷는 중 보상 | 속도 달성 |"); P("|---|---|---|---|---|---|---|")
    rows = []
    for _, c, _ in chosen:
        for arm in ("POL", "PRE", "ADP"):
            x = metrics(a.d, f"{c}_{arm}")
            if x is None:
                continue
            g = agg(x); P(f"| {KO[c]} | {ARM[arm]} | {g['falls']:.2f} | {g['start']:.2f} | {g['haz']:.2f} | {g['rew']:.4f} | {g['vratio']:.2f} |")
        pre, adp = metrics(a.d, f"{c}_PRE"), metrics(a.d, f"{c}_ADP")
        if pre is not None and adp is not None:
            rows.append((c, boot(pre, adp, "rew"), boot(pre, adp, "falls"), boot(pre, adp, "haz"), agg(pre)["rew"]))
    P()
    if rows:
        P("적응 뒤 − 적응 전 (같은 로봇끼리 짝지은 bootstrap, [95 % CI])"); P()
        P("| 조건 | 걷는 중 보상 차이 | 보상 변화율 | 전체 넘어짐 차이 | 걷는 중 넘어짐 차이 |"); P("|---|---|---|---|---|")
        f = lambda t, k="{:+.4f} [{:.4f}, {:.4f}]": k.format(*t)  # noqa: E731
        for c, r, fa, hz, base in rows:
            P(f"| {KO[c]} | {f(r)} | {(100 * r[0] / base if base else float('nan')):+.0f} % | {f(fa, '{:+.2f} [{:.2f}, {:.2f}]')} | {f(hz, '{:+.2f} [{:.2f}, {:.2f}]')} |")
        P()
    P("### 보존: 적응한 모델을 평소 조건에 놓았을 때 (planner)"); P()
    P("| 적응한 조건 | 전체 넘어짐 | 걷는 중 보상 | 적응 전 대비 보상 |"); P("|---|---|---|---|")
    if nom["PRE"] is not None:
        g0 = agg(nom["PRE"]); P(f"| (적응 전) | {g0['falls']:.2f} | {g0['rew']:.4f} | 1.00 |")
        for _, c, _ in chosen:
            x = metrics(a.d, f"{c}_ADPnom")
            if x is not None:
                g = agg(x); P(f"| {KO[c]} | {g['falls']:.2f} | {g['rew']:.4f} | {(g['rew'] / g0['rew'] if g0['rew'] else float('nan')):.2f} |")
    P()
    open(out, "w").write("\n".join(L) + "\n"); print("written", out)


if __name__ == "__main__":
    main()
