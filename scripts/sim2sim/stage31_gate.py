"""Design doc 31 closed-loop gates G1 / G2 (docs/design31_token_gate_ko.md §3.1-3.2): the condition-token model with the
adapted token (tA) or the simulation token (tS), against the design-29 reference runs of the same protocol
(PRE = pretrained planner, ICE / MOT = dynamics-only fine-tuned planner; 18 robots = speeds 0.3/0.6/0.9 x seeds 11-16,
5000 steps).

    python3 scripts/sim2sim/stage31_gate.py --d results/stage31 --ref results/stage29 [--out results/stage31/gate]

G1 (gain kept under weight sharing), own condition:  (PRE - tA) / (PRE - DYN) >= 0.70 with CI lower > 0.40; cells where the
   dynamics-only adaptation itself has no gain (PRE - DYN <= 0) are 'n/a'.  Gate: >= 2 of the cells with a gain.
G2 (the simulation token keeps the pretrained behaviour), other out-of-coverage condition:
   tS - PRE <= max(0.3 x PRE, 0.15) falls per 1000 steps.  Gate: >= 3 of 4 cells.
Robots are resampled jointly (paired bootstrap, 5000).  Fall timing is normalised by time at risk: startup = share of
episodes that fall within their first 125 steps, mid-walk = falls per 1000 steps with steps-since-reset >= 300.
"""
import argparse, csv, json, os
import numpy as np

RNG = np.random.default_rng(31); NB = 5000; STEPS = 5000
MODEL_KO = {"P": "논문 모델", "B": "넓은 DR 모델"}; AD_KO = {"ICE": "빙판 적응", "MOT": "모터 40 % 적응"}
COND_KO = {"ice": "빙판", "mot": "모터 40 %", "nom": "평소"}
CELLS = [(m, a) for m in "PB" for a in ("ICE", "MOT")]
own = lambda a: "ice" if a == "ICE" else "mot"  # noqa: E731
oth = lambda a: "mot" if a == "ICE" else "ice"  # noqa: E731


def load(d, name):
    f = os.path.join(d, f"eval_{name}.csv")
    if not os.path.exists(f):
        return None
    rows = sorted(csv.DictReader(open(f)), key=lambda r: (float(r["speed"]), int(r["seed"])))
    out = {"falls": np.array([int(r["num_falls"]) for r in rows]) / STEPS * 1000.0,
           "reward": np.array([float(r["run_reward_per_step"]) for r in rows]), "n": len(rows)}
    dg = os.path.join(d, f"diag_{name}.npz")
    if os.path.exists(dg):
        z = np.load(dg, allow_pickle=True); s = z["since"].astype(np.int64); f_ = z["fall"].astype(bool); ok = ~z["resetting"].astype(bool)
        n_ep = ((s == 124) | (f_ & (s < 125))).sum()
        out["p_start"] = float((f_ & (s < 125)).sum() / max(1, n_ep)); out["n_ep"] = int(n_ep)
        mid = ok & (s >= 300); out["mid_haz"] = float(1000.0 * (f_ & (s >= 300)).sum() / max(1, mid.sum()))
        dev = np.linalg.norm(z["act"].astype(np.float32) - z["act_pol"].astype(np.float32), axis=-1)
        out["dev_mid"] = float(dev[mid].mean()) if mid.any() else float("nan")
    return out


def boot(fn, *arrs):
    n = len(arrs[0]); idx = RNG.integers(0, n, (NB, n))
    b = np.array([fn(*[x[i] for x in arrs]) for i in idx], float)
    return float(fn(*arrs)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="results/stage31"); ap.add_argument("--ref", default="results/stage29"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.d, "gate")
    L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    R = {"G1": {}, "G2": {}, "arms": {}}
    f3 = lambda t: "{:.2f} [{:.2f}, {:.2f}]".format(*t)  # noqa: E731
    m2 = lambda c: "없음" if c is None else f"{c['falls'].mean():.2f}"  # noqa: E731

    P("## 설계 31 폐루프 관문 (1000스텝당 넘어짐, 로봇 18대 × 5000스텝)"); P()
    P("### G1: 적응 토큰으로 자기 조건에서 이득이 유지되나"); P()
    P("| 칸 | 사전학습 | 동역학 fine-tuning | 토큰 모델 (적응 토큰) | 유지된 이득 [95 % CI] | 판정 |"); P("|---|---|---|---|---|---|")
    g1_pass = g1_n = 0
    for m, ad in CELLS:
        c = own(ad); pre = load(a.ref, f"{m}_PRE_{c}"); dyn = load(a.ref, f"{m}_{ad}_{c}"); ta = load(a.d, f"{m}_{ad}tA_{c}")
        key = f"{m}_{ad}"
        if pre is None or dyn is None or ta is None:
            P(f"| {MODEL_KO[m]}, {AD_KO[ad]} | {m2(pre)} | {m2(dyn)} | {m2(ta)} | — | 미실행 |"); continue
        gain = pre["falls"].mean() - dyn["falls"].mean()
        if gain <= 0:
            R["G1"][key] = {"pre": pre["falls"].mean(), "dyn": dyn["falls"].mean(), "tA": ta["falls"].mean(), "verdict": "n/a"}
            P(f"| {MODEL_KO[m]}, {AD_KO[ad]} | {m2(pre)} | {m2(dyn)} | {m2(ta)} | — | 해당 없음 (적응 이득 없음) |"); continue
        k = boot(lambda p, d, t: (p.mean() - t.mean()) / (p.mean() - d.mean()) if p.mean() - d.mean() > 0 else np.nan, pre["falls"], dyn["falls"], ta["falls"])
        ok = k[0] >= 0.70 and k[1] > 0.40; g1_n += 1; g1_pass += ok
        R["G1"][key] = {"pre": pre["falls"].mean(), "dyn": dyn["falls"].mean(), "tA": ta["falls"].mean(), "kept": k, "verdict": bool(ok)}
        P(f"| {MODEL_KO[m]}, {AD_KO[ad]} | {m2(pre)} | {m2(dyn)} | {m2(ta)} | {f3(k)} | {'통과' if ok else '불통과'} |")
    R["G1"]["pass"] = bool(g1_n > 0 and g1_pass >= 2); R["G1"]["cells"] = [int(g1_pass), int(g1_n)]
    P(); P(f"G1: 이득이 있는 {g1_n}칸 중 {g1_pass}칸 통과 → {'통과' if R['G1']['pass'] else '불통과'} (기준 2칸 이상)"); P()

    P("### G2: 시뮬 토큰으로 다른 범위 밖 조건에서 사전학습 수준이 유지되나"); P()
    P("| 칸 (평가 조건) | 사전학습 | 동역학 fine-tuning | 토큰 모델 (시뮬 토큰) | 토큰 − 사전학습 [95 % CI] | 허용 폭 | 판정 |"); P("|---|---|---|---|---|---|---|")
    g2_pass = g2_n = 0
    for m, ad in CELLS:
        c = oth(ad); pre = load(a.ref, f"{m}_PRE_{c}"); dyn = load(a.ref, f"{m}_{ad}_{c}"); ts = load(a.d, f"{m}_{ad}tS_{c}")
        key = f"{m}_{ad}"
        if pre is None or ts is None:
            P(f"| {MODEL_KO[m]}, {AD_KO[ad]} ({COND_KO[c]}) | {m2(pre)} | {m2(dyn)} | {m2(ts)} | — | — | 미실행 |"); continue
        dlt = boot(lambda p, t: t.mean() - p.mean(), pre["falls"], ts["falls"]); margin = max(0.3 * pre["falls"].mean(), 0.15)
        ok = dlt[0] <= margin; g2_n += 1; g2_pass += ok
        R["G2"][key] = {"pre": pre["falls"].mean(), "dyn": None if dyn is None else dyn["falls"].mean(), "tS": ts["falls"].mean(), "diff": dlt, "margin": margin, "verdict": bool(ok)}
        P(f"| {MODEL_KO[m]}, {AD_KO[ad]} ({COND_KO[c]}) | {m2(pre)} | {m2(dyn)} | {m2(ts)} | {f3(dlt)} | {margin:.2f} | {'통과' if ok else '불통과'} |")
    R["G2"]["pass"] = bool(g2_pass >= 3); R["G2"]["cells"] = [int(g2_pass), int(g2_n)]
    P(); P(f"G2: {g2_n}칸 중 {g2_pass}칸 통과 → {'통과' if R['G2']['pass'] else '불통과'} (기준 3칸 이상)"); P()

    P("### 모든 팔: 넘어짐, 보상, 넘어지는 시점 (출발 = 에피소드가 첫 125스텝 안에 넘어질 확률, 걷는 중 = 걷는 중 1000스텝당)"); P()
    P("| 칸 | 평가 조건 | 팔 | 넘어짐 | 스텝당 보상 | 출발 | 걷는 중 | 걷는 중 정책 이탈 |"); P("|---|---|---|---|---|---|---|---|")
    for m, ad in CELLS:
        for c in (own(ad), oth(ad), "nom"):
            for arm, d, name in (("사전학습", a.ref, f"{m}_PRE_{c}"), ("동역학 fine-tuning", a.ref, f"{m}_{ad}_{c}"),
                                 ("토큰 모델, 적응 토큰", a.d, f"{m}_{ad}tA_{c}"), ("토큰 모델, 시뮬 토큰", a.d, f"{m}_{ad}tS_{c}")):
                x = load(d, name)
                if x is None:
                    continue
                R["arms"][name] = {k: (float(v.mean()) if isinstance(v, np.ndarray) else v) for k, v in x.items()}
                g = lambda k, fmt="{:.2f}": fmt.format(x[k]) if k in x else "—"  # noqa: E731
                P(f"| {MODEL_KO[m]}, {AD_KO[ad]} | {COND_KO[c]} | {arm} | {x['falls'].mean():.2f} | {x['reward'].mean():.4f} | {g('p_start')} | {g('mid_haz')} | {g('dev_mid')} |")
    P()
    json.dump(R, open(out + ".json", "w"), indent=1)
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("written", out + ".md")


if __name__ == "__main__":
    main()
