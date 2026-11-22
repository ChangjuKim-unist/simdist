"""Design 40 E0: tables and registered verdicts.

Files in --d:
  tune_free_sq{sq}_lam{lam}.{json,npz}   (nominal, seeds 100-107)
  tune_clock_w{w}.{json,npz}             (nominal, seeds 100-107)
  e0_{arm}_{cond}.{json,npz}             arm in ppo/free/clock, seeds 0-15
  python scripts/toy/stage40_e0_report.py tune --d results/stage40     -> picks tuned values
  python scripts/toy/stage40_e0_report.py report --d results/stage40 --md results/stage40/e0.md
"""
import argparse, glob, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from toy_common import CONDITIONS, EVAL_FROM, boot_ci  # noqa: E402

BIAS = np.array([0.0, 0.5, 0.5, 0.0])
PERIOD, R_GAIT, DT = 0.5, 0.5, 0.02


def clock_match(npz):
    con = npz["contact"].astype(bool)                    # (E, T, 4)
    fallen = np.maximum.accumulate(npz["fallen"].astype(bool), 1)
    T = con.shape[1]
    t = (np.arange(T) + 1) * DT
    des = np.mod(t[:, None] / PERIOD + BIAS[None], 1.0) < R_GAIT
    ok = (con == des[None]).mean(-1)                     # (E, T)
    alive = ~fallen
    m = np.where(alive[:, EVAL_FROM:], ok[:, EVAL_FROM:], np.nan)
    return np.nanmean(m, 1)                              # per episode (nan if fell before 2 s)


def load(d, name):
    js = json.load(open(os.path.join(d, name + ".json")))
    npz = np.load(os.path.join(d, name + ".npz"))
    js["match"] = clock_match(npz).tolist()
    return js


def tune(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.d, "tune_free_*.json"))):
        n = os.path.basename(f)[:-5]
        js = load(a.d, n)
        sq, lam = re.findall(r"sq([\d.]+)_lam([\d.]+)", n)[0]
        rows.append((float(sq), float(lam), float(np.mean(js["J"])), int(np.sum(js["fell"]))))
    ok = [r for r in rows if r[3] == 0]
    best = max(ok, key=lambda r: r[2]) if ok else None
    print("M-free tuning (sigma_q, lam, J, falls):")
    for r in rows:
        print("  ", r)
    print("pick:", best)
    crow = []
    for f in sorted(glob.glob(os.path.join(a.d, "tune_clock_w*.json"))):
        n = os.path.basename(f)[:-5]
        js = load(a.d, n)
        w = float(re.findall(r"w([\d.]+)", n)[0])
        crow.append((w, float(np.mean(js["J"])), int(np.sum(js["fell"])), float(np.nanmean(js["match"]))))
    crow.sort()
    pick = None
    if best:
        for w, J, fl, m in crow:
            if m >= 0.80 and fl == 0 and J >= 0.9 * best[2]:
                pick = (w, "rule")
                break
        if pick is None:
            nf = [c for c in crow if c[2] == 0]
            if nf:
                pick = (max(nf, key=lambda c: c[3])[0], "fallback: highest match without falls")
    print("M-clock tuning (w, J, falls, match):")
    for c in crow:
        print("  ", c)
    print("pick:", pick)
    out = {"free": rows, "free_pick": best, "clock": crow, "clock_pick": pick}
    json.dump(out, open(os.path.join(a.d, "tune.json"), "w"), indent=1)


def fmt(m, lo, hi, p=3):
    return f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def report(a):
    R = {}
    for arm in ("ppo", "free", "clock"):
        for c in CONDITIONS:
            n = f"e0_{arm}_{c}"
            if os.path.exists(os.path.join(a.d, n + ".json")):
                R[(arm, c)] = load(a.d, n)
    L = ["# 설계 40 E0 결과 (자동 생성)", ""]
    L += ["## 1. 조건별 J (0–1, 1 = 완벽한 속도 추종, 넘어진 뒤 0) 와 넘어짐 (16개 중)", "",
          "| 조건 | PPO | M-free | M-clock |", "|---|---|---|---|"]
    for c in CONDITIONS:
        cells = []
        for arm in ("ppo", "free", "clock"):
            if (arm, c) in R:
                js = R[(arm, c)]
                cells.append(f"{np.mean(js['J']):.3f} ({int(np.sum(js['fell']))})")
            else:
                cells.append("–")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 구현 확인", ""]
    verdict = {}
    for arm, k in (("ppo", "K1"), ("free", "K2")):
        if (arm, "nominal") in R:
            js = R[(arm, "nominal")]
            if k == "K1":
                ok = np.mean(js["J"]) >= 0.7 and np.sum(js["fell"]) == 0
            else:   # K2 (revised before registration): no falls, mean forward speed >= 0.4 m/s
                ok = np.sum(js["fell"]) == 0 and np.nanmean(js["vx_mean"]) >= 0.4
            verdict[k] = bool(ok)
            L.append(f"- {k} ({arm} nominal): J {np.mean(js['J']):.3f}, 넘어짐 {int(np.sum(js['fell']))}, "
                     f"전진 {np.nanmean(js['vx_mean']):.2f} m/s → {'통과' if ok else '실패'}")
    L += ["", "## 3. 시계 효과 Δ_clock = J(M-free) − J(M-clock), 같은 시드 짝, 95 % 구간", "",
          "| 조건 | Δ_clock | 접촉 일치율 free / clock |", "|---|---|---|"]
    D = {}
    for c in CONDITIONS:
        if ("free", c) in R and ("clock", c) in R:
            d = np.array(R[("free", c)]["J"]) - np.array(R[("clock", c)]["J"])
            D[c] = boot_ci(d)
            mf = np.nanmean(R[("free", c)]["match"])
            mc = np.nanmean(R[("clock", c)]["match"])
            L.append(f"| {c} | {fmt(*D[c])} | {mf:.2f} / {mc:.2f} |")
    if all(c in D for c in ("nominal", "kneeRR0", "kneeFR0")):
        nom_ok = abs(D["nominal"][0]) <= 0.05
        block = [c for c in ("kneeRR0", "kneeFR0") if D[c][0] >= 0.10 and D[c][1] > 0]
        harmless = all(D[c][0] < 0.05 or D[c][1] <= 0 for c in ("kneeRR0", "kneeFR0"))
        if block and nom_ok:
            v = f"막는다 ({', '.join(block)})"
        elif harmless:
            v = "무해"
        else:
            v = "중간"
        verdict["clock"] = v
        L += ["", f"**판정: {v}** (nominal |Δ| ≤ 0.05: {nom_ok})"]
    L += ["", "## 4. 참고: M-free − PPO (J, 같은 시드 짝)", "", "| 조건 | J 차이 | 넘어짐 PPO / M-free |", "|---|---|---|"]
    for c in CONDITIONS:
        if ("free", c) in R and ("ppo", c) in R:
            d = np.array(R[("free", c)]["J"]) - np.array(R[("ppo", c)]["J"])
            L.append(f"| {c} | {fmt(*boot_ci(d))} | {int(np.sum(R[('ppo', c)]['fell']))} / {int(np.sum(R[('free', c)]['fell']))} |")
    L += ["", "## 5. 동작 품질 (판정에 안 씀; 넘어지기 전 2 s 이후 평균)", "",
          "| 조건 | 방법 | 전진 m/s | 방향 틀어짐 ° | roll SD ° | pitch SD ° | 토크² | 목표 jerk | 미끄러짐 m/s | duty FR FL RR RL |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        for arm in ("ppo", "free", "clock"):
            if (arm, c) in R:
                js = R[(arm, c)]
                duty = np.nanmean(np.array(js["duty"], float), 0)
                L.append(f"| {c} | {arm} | {np.nanmean(js['vx_mean']):.2f} | {np.nanmean(js['yaw_drift_deg']):.1f} | "
                         f"{np.nanmean(js['roll_sd_deg']):.1f} | {np.nanmean(js['pitch_sd_deg']):.1f} | {np.nanmean(js['tau2']):.1f} | "
                         f"{np.nanmean(js['ctrl_jerk']):.4f} | {np.nanmean(js['slip']):.3f} | {' '.join(f'{x:.2f}' for x in duty)} |")
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"verdict": verdict, "delta_clock": D}, open(a.md[:-3] + ".json", "w"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tune", "report"))
    ap.add_argument("--d", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    tune(a) if a.mode == "tune" else report(a)


if __name__ == "__main__":
    main()
