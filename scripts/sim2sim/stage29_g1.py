"""Design doc 29 gate G1 analysis (docs/design29_gate_physcode_ko.md §3): robust-until-confident scoring over
{adapted, pretrained} vs the adapted model alone.

    python3 scripts/sim2sim/stage29_g1.py --d results/stage29 [--out results/stage29/g1]

Per robot (18 = speeds 0.3/0.6/0.9 x seeds 11-16, 5000 steps): falls per 1000 steps (all, startup = steps since reset
< 300, mid-walk), reward per step, forward progress; from diag npz (if present) the planner's deviation from the policy
head in the first 2 s after release (steps since reset 25-124) and mid-walk motion metrics (stage27_metrics).
Ratios / gain retention: robots resampled jointly (paired bootstrap, 5000).
Scenario verdicts (registered): F forgetting  (X+R)/X <= 0.7 with CI upper < 1 and X+R <= 1.15 x PRE on the other
condition; G gain kept (PRE - X+R)/(PRE - X) >= 0.7 on the adapted condition (if the adaptation itself shows no gain,
PRE - X <= 0, G is 'n/a' and the scenario cannot pass); N nominal reward (X+R)/X >= 0.95 and falls increase <= 0.05.
"""
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(29)
NB = 5000
STEPS = 5000
ARMS = ["PRE", "ICE", "ICER", "MOT", "MOTR"]
ARM_KO = {"PRE": "사전학습", "ICE": "빙판 적응", "ICER": "빙판 적응 + 강건 출발", "MOT": "모터 적응", "MOTR": "모터 적응 + 강건 출발"}
CONDS = ["nom", "ice", "mot"]
COND_KO = {"nom": "평소", "ice": "빙판", "mot": "모터 40 %"}
MODEL_KO = {"P": "논문 모델", "B": "넓은 DR 모델"}
SCEN = [("P", "MOT", "ice", "mot"), ("P", "ICE", "mot", "ice"), ("B", "MOT", "ice", "mot"), ("B", "ICE", "mot", "ice")]  # model, arm, other, own


def load_cell(d, name):
    f = os.path.join(d, f"eval_{name}.csv")
    if not os.path.exists(f):
        return None
    rows = sorted(csv.DictReader(open(f)), key=lambda r: (float(r["speed"]), int(r["seed"])))
    key = [(float(r["speed"]), int(r["seed"])) for r in rows]
    out = {"key": key, "falls_1k": np.array([int(r["num_falls"]) for r in rows]) / STEPS * 1000.0,
           "reward": np.array([float(r["run_reward_per_step"]) for r in rows]), "fwd": np.array([float(r["forward_progress"]) for r in rows])}
    st = {k: 0 for k in key}; mid = {k: 0 for k in key}
    ff = os.path.join(d, f"falls_{name}.csv")
    if os.path.exists(ff):
        for r in csv.DictReader(open(ff)):
            k = (float(r["speed"]), int(r["seed"]))
            (st if int(r["steps_since_reset"]) < 300 else mid)[k] += 1
    out["start_1k"] = np.array([st[k] for k in key]) / STEPS * 1000.0
    out["mid_1k"] = np.array([mid[k] for k in key]) / STEPS * 1000.0
    dg = os.path.join(d, f"diag_{name}.npz")
    if os.path.exists(dg):
        z = np.load(dg, allow_pickle=True)
        order = np.lexsort((z["seeds"], z["speeds"]))
        since = z["since"][:, order].astype(np.int64); res = z["resetting"][:, order].astype(bool)
        dev = np.linalg.norm(z["act"][:, order].astype(np.float32) - z["act_pol"][:, order].astype(np.float32), axis=-1)
        m0 = (~res) & (since >= 25) & (since < 125); m1 = (~res) & (since >= 300)
        out["dev_start"] = np.array([dev[m0[:, i], i].mean() if m0[:, i].any() else np.nan for i in range(dev.shape[1])])
        out["dev_mid"] = np.array([dev[m1[:, i], i].mean() if m1[:, i].any() else np.nan for i in range(dev.shape[1])])
        try:
            import stage28_analyze as A28, stage27_metrics as M27
            dd = {k: (v[:, order] if (isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == since.shape[0] and k != "term_names") else v) for k, v in z.items()}
            for k in ("speeds", "seeds", "robot_mass"):
                if k in dd:
                    dd[k] = z[k][order]
            a = A28.m27_view(dd, name)
            met = M27.robot_metrics(a, a["valid"] & (a["since"] >= 300))
            for k in ("act_jerk", "stride_cv", "short_cps", "slip", "tilt", "torque_sq_d", "power"):
                out[k] = np.asarray(met[k], float)
        except Exception as e:  # motion metrics are optional
            print(f"motion metrics skipped for {name}: {e}")
    return out


def boot(fn, *arrs):
    n = len(arrs[0]); idx = RNG.integers(0, n, (NB, n))
    b = np.array([fn(*[a[i] for a in arrs]) for i in idx], float)
    return fn(*arrs), np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def ratio(x, y):  # mean(y) / mean(x)
    return y.mean() / x.mean() if x.mean() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="results/stage29"); ap.add_argument("--out", default=None)
    args = ap.parse_args(); out = args.out or os.path.join(args.d, "g1")
    C = {}
    for m in ("P", "B"):
        for a in ARMS:
            for c in CONDS:
                C[(m, a, c)] = load_cell(args.d, f"{m}_{a}_{c}")
        for a, c in (("PRE", "ice"), ("MOT", "ice"), ("PRE", "mot"), ("ICE", "mot")):
            C[(m, a, c + "_ramp")] = load_cell(args.d, f"{m}_{a}_{c}_ramp")
    keys0 = next(v["key"] for v in C.values() if v is not None)
    for k, v in C.items():
        if v is not None:
            assert v["key"] == keys0, f"robot order differs in {k}"
    L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    R = {"cells": {}, "scenarios": {}, "ramp": {}}
    P("## 설계 29 G1: 확신 전 강건 계획 (출발 2초 동안 {적응 모델, 사전학습 모델} 최저 점수 혼합)"); P()
    P("1000스텝당 넘어짐, 18대(속도 0.3/0.6/0.9 × 시드 11~16) × 5000스텝, 낮을수록 좋음. 출발 = 리셋 후 300스텝 미만."); P()
    for m in ("P", "B"):
        for metric, label in (("falls_1k", "넘어짐 전체"), ("start_1k", "출발 중 넘어짐"), ("mid_1k", "걷는 중 넘어짐"), ("reward", "실제 보상/스텝 (높을수록 좋음)")):
            P(f"### {MODEL_KO[m]}: {label}"); P()
            P("| 팔 | " + " | ".join(COND_KO[c] for c in CONDS) + " |"); P("|---|" + "---|" * len(CONDS))
            for a in ARMS:
                vals = []
                for c in CONDS:
                    v = C[(m, a, c)]
                    vals.append("—" if v is None else (f"{v[metric].mean():.4f}" if metric == "reward" else f"{v[metric].mean():.2f}"))
                    if v is not None:
                        R["cells"][f"{m}_{a}_{c}"] = {k: float(np.nanmean(v[k])) for k in v if k != "key"}
                P(f"| {ARM_KO[a]} | " + " | ".join(vals) + " |")
            P()
        if all(C[(m, a, "ice")] is not None and "dev_start" in C[(m, a, "ice")] for a in ARMS):
            P(f"### {MODEL_KO[m]}: 계획기가 정책에서 벗어난 크기 (출발 2초 / 걷는 중)"); P()
            P("| 팔 | " + " | ".join(COND_KO[c] for c in CONDS) + " |"); P("|---|" + "---|" * len(CONDS))
            for a in ARMS:
                P(f"| {ARM_KO[a]} | " + " | ".join("—" if C[(m, a, c)] is None or "dev_start" not in C[(m, a, c)] else
                                                    f"{np.nanmean(C[(m, a, c)]['dev_start']):.2f} / {np.nanmean(C[(m, a, c)]['dev_mid']):.2f}" for c in CONDS) + " |")
            P()
    P("### 시나리오 판정 (등록 기준 §3)"); P()
    P("| 시나리오 | F 망각비 (X+R)/X [CI] | X+R / 사전학습 | G 이득 보존율 [CI] | N 평소 보상비, 넘어짐 증가 | 판정 |"); P("|---|---|---|---|---|---|")
    any_pass = False
    for m, a, other, own in SCEN:
        X, XR, PRE = (lambda c: C[(m, a, c)]), (lambda c: C[(m, a + "R", c)]), (lambda c: C[(m, "PRE", c)])
        if any(v is None for v in (X(other), XR(other), PRE(other), X(own), XR(own), PRE(own), X("nom"), XR("nom"))):
            P(f"| {MODEL_KO[m]} {ARM_KO[a]} | 미완 | | | | |"); continue
        f_r = boot(ratio, X(other)["falls_1k"], XR(other)["falls_1k"])
        f_pre = XR(other)["falls_1k"].mean() / max(PRE(other)["falls_1k"].mean(), 1e-9)
        F = bool(f_r[0] <= 0.7 and f_r[2] < 1.0 and f_pre <= 1.15)
        gain = PRE(own)["falls_1k"].mean() - X(own)["falls_1k"].mean()
        if gain <= 0:
            g_r = (np.nan, np.nan, np.nan); G = None
        else:
            g_r = boot(lambda p, x, xr: (p.mean() - xr.mean()) / (p.mean() - x.mean()) if p.mean() - x.mean() > 0 else np.nan,
                       PRE(own)["falls_1k"], X(own)["falls_1k"], XR(own)["falls_1k"])
            G = bool(g_r[0] >= 0.7)
        n_rew = XR("nom")["reward"].mean() / X("nom")["reward"].mean(); n_fall = XR("nom")["falls_1k"].mean() - X("nom")["falls_1k"].mean()
        N = bool(n_rew >= 0.95 and n_fall <= 0.05)
        ok = bool(F and G is True and N); any_pass |= ok
        R["scenarios"][f"{m}_{a}"] = {"F_ratio": f_r, "F_vs_pre": f_pre, "F": F, "gain_adapt": gain, "G_ret": g_r, "G": G, "N_reward": n_rew, "N_fall": n_fall, "N": N, "pass": ok}
        gtxt = "적응 이득 없음 (n/a)" if G is None else f"{g_r[0]:.2f} [{g_r[1]:.2f}, {g_r[2]:.2f}]"
        P(f"| {MODEL_KO[m]} {ARM_KO[a]} (다른 조건 {COND_KO[other]}) | {f_r[0]:.2f} [{f_r[1]:.2f}, {f_r[2]:.2f}] | {f_pre:.2f} | {gtxt} | {n_rew:.3f}, {n_fall:+.2f} | "
          f"F {'O' if F else 'X'} · G {'n/a' if G is None else ('O' if G else 'X')} · N {'O' if N else 'X'} → {'통과' if ok else '불통과'} |")
    R["G1_pass"] = any_pass
    P(); P(f"G1 통과(네 시나리오 중 하나 이상 F·G·N): **{any_pass}**"); P()
    P("### 보조 Q1r: 1초 램프 출발에서도 망각이 남나 (적응 모델 / 사전학습, 다른 조건 넘어짐)"); P()
    P("| 모델 | 비교 | 램프 없음 | 1초 램프 |"); P("|---|---|---|---|")
    for m in ("P", "B"):
        for a, c in (("MOT", "ice"), ("ICE", "mot")):
            r0 = (C[(m, a, c)], C[(m, "PRE", c)]); r1 = (C[(m, a, c + "_ramp")], C[(m, "PRE", c + "_ramp")])
            t0 = "—" if None in r0 else "{:.2f} [{:.2f}, {:.2f}]".format(*boot(ratio, r0[1]["falls_1k"], r0[0]["falls_1k"]))
            t1 = "—" if None in r1 else "{:.2f} [{:.2f}, {:.2f}]".format(*boot(ratio, r1[1]["falls_1k"], r1[0]["falls_1k"]))
            if None not in r1:
                R["ramp"][f"{m}_{a}_{c}"] = {"pre": float(r1[1]["falls_1k"].mean()), "adapted": float(r1[0]["falls_1k"].mean())}
            P(f"| {MODEL_KO[m]} | {ARM_KO[a]} / 사전학습, {COND_KO[c]} | {t0} | {t1} |")
    P()
    mot_keys = ("act_jerk", "stride_cv", "short_cps", "slip", "tilt", "torque_sq_d", "power")
    if C.get(("P", "MOT", "nom")) is not None and "act_jerk" in C[("P", "MOT", "nom")]:
        P("### 움직임 품질 (걷는 중, X+R / X 비, 1에 가까우면 강건 출발이 걷는 중 움직임을 안 바꿈)"); P()
        P("| 셀 | 떨림 | 보폭 변동계수 | 짧은 접지 | 미끄럼 | 기울기 | 토크² | 일률 |"); P("|---|---|---|---|---|---|---|---|")
        for m in ("P", "B"):
            for a in ("ICE", "MOT"):
                for c in CONDS:
                    x, xr = C[(m, a, c)], C[(m, a + "R", c)]
                    if x is None or xr is None or "act_jerk" not in x or "act_jerk" not in xr:
                        continue
                    P(f"| {MODEL_KO[m]} {ARM_KO[a]} {COND_KO[c]} | " + " | ".join(f"{np.nanmean(xr[k]) / np.nanmean(x[k]):.2f}" for k in mot_keys) + " |")
        P()

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o
    json.dump(clean(R), open(out + ".json", "w"), indent=1)
    open(out + ".md", "w").write("\n".join(L) + "\n"); print("written", out + ".md")


if __name__ == "__main__":
    main()
