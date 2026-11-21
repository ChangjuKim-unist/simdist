"""Design doc 36 check C1 (docs/design36_sampling_checks_ko.md): current warm start (init_mode=shift) vs the fixed one
(shift_fwd) on five planner cells. Paired over the 18 robots (same speed / seed), bootstrap 5000.

    python3 scripts/sim2sim/stage36_c1.py --d results/stage36 --out results/stage36/c1.md
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage35_score as s35  # noqa: E402
import stage27_metrics as s27  # noqa: E402

CELLS = [("nominal_PRE", "논문 모델, 평소"), ("tow20_PRE", "논문 모델, 끄는 짐 20 N"), ("mot50_PRE", "논문 모델, 모터 50 %"),
         ("tow20_ADP", "적응 모델, 끄는 짐 20 N"), ("mot50_ADP", "적응 모델, 모터 50 %")]
NB = 5000


def extra(path):
    """per robot over mid-walk steps: jitter (action 2nd difference), lateral speed, yaw rate, policy deviation, ESS / policy share"""
    a = s27.load(path); m = s27.phase(a, "mid", s35.MID); rm = s27.robot_metrics(a, m)
    z = np.load(path, allow_pickle=True)
    dev = np.linalg.norm(z["act"].astype(np.float64) - z["act_pol"].astype(np.float64), axis=-1)
    mm = lambda x: np.array([np.nanmean(np.where(m[:, i], x[:, i], np.nan)) for i in range(x.shape[1])])  # noqa: E731
    out = {"jerk": rm["act_jerk"], "lat": rm["lat_speed"], "yaw": rm["yaw_rate"], "dev": mm(dev)}
    for k in ("ess_last", "pol_last", "ess_first", "pol_first"):
        out[k] = mm(z[k].astype(np.float64)) if k in z.files else np.full(a["N"], np.nan)
    return out


def ratio(x, y, seed=36):
    """(mean y / mean x - 1) in %, paired bootstrap over robots"""
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (NB, len(x)))
    b = 100.0 * (y[idx].mean(1) / x[idx].mean(1) - 1.0)
    return 100.0 * (y.mean() / x.mean() - 1.0), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="results/stage36"); ap.add_argument("--out", default="results/stage36/c1.md")
    a = ap.parse_args(); L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    f = lambda cell, mode: os.path.join(a.d, f"diag_c1_{cell}_{mode}.npz")  # noqa: E731
    J = {}; fm = lambda t: f"{t[0]:+.1f} % [{t[1]:+.1f}, {t[2]:+.1f}]"  # noqa: E731
    P("## 설계 36 C1: warm start 고친 판(shift_fwd) − 지금 코드(shift)"); P()
    P("걷는 중(리셋 뒤 300스텝 이후), 로봇 18대 짝 bootstrap. S 차이는 지금 코드 S 대비 %, 떨림·이탈은 비율 변화 %."); P()
    P("| 칸 | 지금 S | S 차이 | 넘어짐 (지금 → 고친 판) | 떨림 변화 | 정책 이탈 변화 | 옆 속도 변화 | 중요 (S ≥ 2 % 또는 떨림 ≥ 20 %, 구간이 0 제외) |")
    P("|---|---|---|---|---|---|---|---|")
    for cell, ko in CELLS:
        r0, r1 = s35.load(f(cell, "shift")), s35.load(f(cell, "fwd"))
        if r0 is None or r1 is None:
            P(f"| {ko} | - | - | - | - | - | - | - |"); continue
        ref = s35.pooled(r0, "S"); dS = s35.diff(r1, r0, "S", ref)
        e0, e1 = extra(f(cell, "shift")), extra(f(cell, "fwd"))
        jk, dv, lt = ratio(e0["jerk"], e1["jerk"]), ratio(e0["dev"], e1["dev"]), ratio(e0["lat"], e1["lat"])
        imp = (abs(dS[0]) >= 2.0 and (dS[1] > 0 or dS[2] < 0)) or (abs(jk[0]) >= 20.0 and (jk[1] > 0 or jk[2] < 0))
        P(f"| {ko} | {ref:.4f} | {fm(dS)} | {s35.pooled(r0, 'falls'):.2f} → {s35.pooled(r1, 'falls'):.2f} | {fm(jk)} | {fm(dv)} | {fm(lt)} | {'예' if imp else '아니오'} |")
        J[cell] = {"S_ref": ref, "dS_pct": dS, "falls": [s35.pooled(r0, "falls"), s35.pooled(r1, "falls")], "jerk_pct": jk, "dev_pct": dv,
                   "lat_pct": lt, "important": bool(imp),
                   "ess_last": [float(np.nanmean(e0["ess_last"])), float(np.nanmean(e1["ess_last"]))],
                   "pol_last": [float(np.nanmean(e0["pol_last"])), float(np.nanmean(e1["pol_last"]))],
                   "ess_first": [float(np.nanmean(e0["ess_first"])), float(np.nanmean(e1["ess_first"]))],
                   "pol_first": [float(np.nanmean(e0["pol_first"])), float(np.nanmean(e1["pol_first"]))]}
    P()
    P("### 진단 지표: elite ESS 비율(1/Σw²/64)과 정책 표본의 가중 몫, 걷는 중 평균 (지금 코드 / 고친 판)"); P()
    P("| 칸 | ESS 첫 반복 | ESS 마지막 반복 | 정책 몫 첫 반복 | 정책 몫 마지막 반복 |"); P("|---|---|---|---|---|")
    for cell, ko in CELLS:
        if cell in J:
            j = J[cell]; g = lambda k: f"{j[k][0]:.3f} / {j[k][1]:.3f}"  # noqa: E731
            P(f"| {ko} | {g('ess_first')} | {g('ess_last')} | {g('pol_first')} | {g('pol_last')} |")
    P()
    if all(c in J for c in ("nominal_PRE", "tow20_PRE", "mot50_PRE")):
        n = J["nominal_PRE"]
        for k in ("pol_last", "ess_last"):
            for c in ("tow20_PRE", "mot50_PRE"):
                r = J[c][k][0] / n[k][0] if n[k][0] else np.nan
                P(f"- 진단 지표 채택 판정 ({k}, 논문 모델 지금 코드, {c} / 평소): {r:.2f}배 → {'2배 이상 차이' if (r >= 2 or r <= 0.5) else '2배 미만'}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n"); json.dump(J, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
