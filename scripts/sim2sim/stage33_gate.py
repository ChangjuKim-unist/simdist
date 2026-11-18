"""Design doc 33 (docs/design33_two_model_ko.md): two deployed systems, three training seeds each, on the design-29 protocol
(18 robots x 5000 steps, falls per 1000 steps, paired bootstrap over robots).

    python3 scripts/sim2sim/stage33_gate.py --d results/stage33 --ref results/stage29 [--d32 results/stage32]

Systems: T = adapted model A (from S) + policy start + online one-step error switch to S;  R = single replay model + policy start.
References: PRE = pretrained planner (design 29), DYN = dynamics-only fine-tune, seeds 42 (design 29) / 43 / 44 (arm d43, d44),
POL = policy head alone, PREps = pretrained planner + policy start.
Cell criteria (thresholds unchanged from designs 31 / 32):
  gain      own condition:   (PRE - arm) / (PRE - mean of the DYN seeds) >= 0.70 with CI lower > 0.40  (3 cells with a gain)
  retention other condition: arm - PRE <= max(0.3 x PRE, 0.15)                                         (4 cells)
A system passes G if every seed passes >= 2 of 3 gain cells, K if every seed passes >= 3 of 4 retention cells.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage31_gate import load, boot, MODEL_KO, AD_KO, COND_KO, CELLS, own, oth  # noqa: E402

GAIN_CELLS = [("P", "ICE"), ("P", "MOT"), ("B", "ICE")]; SEEDS = (42, 43, 44)
SYS_KO = {"T": "두 벌 + 정책 출발 + 오차 전환", "R": "replay 한 모델 + 정책 출발"}
CELL_KO = lambda m, a: f"{MODEL_KO[m].replace(' 모델', '')}·{AD_KO[a].replace(' 적응', '')}"  # noqa: E731


def modes(d, name):
    """share of the planner-or-fallback steps (after the policy start) that ran on the fallback model"""
    f = os.path.join(d, f"diag_{name}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    if "mode" not in z.files:
        return None
    md = z["mode"]; ok = ~z["resetting"].astype(bool)
    return {"policy": float(((md == 1) & ok).sum() / max(1, ok.sum())), "fallback": float(((md >= 2) & ok).sum() / max(1, ((md != 1) & ok).sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="results/stage33"); ap.add_argument("--ref", default="results/stage29")
    ap.add_argument("--d32", default="results/stage32"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.d, "gate")
    L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    R = {"gain": {}, "ret": {}, "verdict": {}, "arms": {}, "modes": {}}
    f3 = lambda t: "{:.2f} [{:.2f}, {:.2f}]".format(*t)  # noqa: E731
    mm = lambda x: "—" if x is None else f"{x['falls'].mean():.2f}"  # noqa: E731

    def dyn_runs(m, ad, c):
        return [x for x in [load(a.ref, f"{m}_{ad}_{c}")] + [load(a.d, f"{m}_{ad}d{s}_{c}") for s in (43, 44)] if x is not None]

    def gain(name, m, ad):
        c = own(ad); pre = load(a.ref, f"{m}_PRE_{c}"); dyn = dyn_runs(m, ad, c); x = load(a.d, name)
        if x is None or pre is None or not dyn:
            return None
        dbar = np.mean([q["falls"] for q in dyn], axis=0)  # per robot, mean over the available DYN seeds
        k = boot(lambda p, q, t: (p.mean() - t.mean()) / (p.mean() - q.mean()) if p.mean() - q.mean() > 0 else np.nan, pre["falls"], dbar, x["falls"])
        return {"pre": float(pre["falls"].mean()), "dyn": float(dbar.mean()), "n_dyn": len(dyn), "arm": float(x["falls"].mean()), "kept": k, "ok": bool(k[0] >= 0.70 and k[1] > 0.40)}

    def ret(name, m, ad):
        c = oth(ad); pre = load(a.ref, f"{m}_PRE_{c}"); x = load(a.d, name)
        if x is None or pre is None:
            return None
        dlt = boot(lambda p, t: t.mean() - p.mean(), pre["falls"], x["falls"]); mg = max(0.3 * pre["falls"].mean(), 0.15)
        return {"pre": float(pre["falls"].mean()), "arm": float(x["falls"].mean()), "diff": dlt, "margin": float(mg), "ok": bool(dlt[0] <= mg)}

    P("## 설계 33: 두 벌 + 정책 출발 + 오차 전환, replay 한 모델 + 정책 출발 (1000스텝당 넘어짐, 로봇 18대 × 5000스텝)"); P()
    P("### 비교 대상 (넘어짐)"); P()
    P("| 모델 | 조건 | 사전학습 | 사전학습 + 정책 출발 | 정책만 | 동역학 fine-tuning 시드 42 / 43 / 44 (그 조건에 적응) |"); P("|---|---|---|---|---|---|")
    for m in "PB":
        for c in ("ice", "mot"):
            ad = "ICE" if c == "ice" else "MOT"
            P(f"| {MODEL_KO[m]} | {COND_KO[c]} | {mm(load(a.ref, f'{m}_PRE_{c}'))} | {mm(load(a.d, f'{m}_PREps_{c}'))} | {mm(load(a.d, f'{m}_POL_{c}'))} | "
              + " / ".join([mm(load(a.ref, f"{m}_{ad}_{c}"))] + [mm(load(a.d, f"{m}_{ad}d{s}_{c}")) for s in (43, 44)]) + " |")
    P()
    for kind, title, cells in (("gain", "이득: 자기 조건 (칸 값 = 넘어짐 → 유지된 이득 [95 % CI], 굵게 = 칸 통과)", GAIN_CELLS),
                               ("ret", "보존: 다른 범위 밖 조건 (칸 값 = 넘어짐 → 사전학습과의 차이 [95 % CI], 굵게 = 허용 폭 안)", CELLS)):
        P(f"### {title}"); P()
        P("| 시스템 | 학습 시드 | " + " | ".join(CELL_KO(m, ad) + (f" ({COND_KO[oth(ad)]})" if kind == "ret" else "") for m, ad in cells) + " | 통과 칸 |")
        P("|---|---|" + "---|" * (len(cells) + 1))
        for sy in "TR":
            for s in SEEDS:
                cs = []; n_ok = n = 0
                for m, ad in cells:
                    name = f"{m}_{ad}{sy}{s}_{own(ad) if kind == 'gain' else oth(ad)}"
                    r = (gain if kind == "gain" else ret)(name, m, ad); R[kind][name] = r
                    if r is None:
                        cs.append("미실행"); continue
                    n += 1; n_ok += r["ok"]
                    v = f"{r['arm']:.2f} → " + (f3(r["kept"]) if kind == "gain" else f"{r['diff'][0]:+.2f} [{r['diff'][1]:.2f}, {r['diff'][2]:.2f}] (폭 {r['margin']:.2f})")
                    cs.append(("**" + v + "**") if r["ok"] else v)
                R["verdict"][f"{kind}:{sy}{s}"] = [int(n_ok), int(n)]
                P(f"| {SYS_KO[sy]} | {s} | " + " | ".join(cs) + f" | {n_ok}/{n} |")
        P()
    V = R["verdict"]; ok = lambda k, need: V.get(k, [0, 0])[1] > 0 and V[k][0] >= need  # noqa: E731
    R["system"] = {sy: {"G": all(ok(f"gain:{sy}{s}", 2) for s in SEEDS), "K": all(ok(f"ret:{sy}{s}", 3) for s in SEEDS)} for sy in "TR"}
    P("판정 (세 시드가 각각 넘어야 함. G = 이득 3칸 중 2칸 이상, K = 보존 4칸 중 3칸 이상): "
      + "; ".join(f"{SYS_KO[sy]}: G {'통과' if v['G'] else '불통과'}, K {'통과' if v['K'] else '불통과'}" for sy, v in R["system"].items())); P()

    P("### 전환이 물러난 비율 (두 벌 시스템, 정책 출발 뒤의 스텝 중 S 모델로 계획한 비율; 시드 42 / 43 / 44)"); P()
    P("| 칸 | 자기 조건 (잘못 물러남) | 다른 범위 밖 조건 (물러나야 함) | 평소 (시드 42) |"); P("|---|---|---|---|")
    for m, ad in CELLS:
        row = []
        for c, ss in ((own(ad), SEEDS), (oth(ad), SEEDS), ("nom", (42,))):
            v = [modes(a.d, f"{m}_{ad}T{s}_{c}") for s in ss]
            R["modes"].update({f"{m}_{ad}T{s}_{c}": x for s, x in zip(ss, v)})
            row.append(" / ".join("—" if x is None else f"{x['fallback']:.2f}" for x in v))
        P(f"| {CELL_KO(m, ad)} | " + " | ".join(row) + " |")
    P()
    P("### 해 없음 칸과 평소 조건 (기준 없음): 넓은 DR·모터 40 %의 자기 조건, 평소 조건(시드 42)"); P()
    P("| 시스템 | 넓은 DR·모터 자기 조건, 시드 42 / 43 / 44 (사전학습 " + mm(load(a.ref, "B_PRE_mot")) + ") | 평소 조건 4칸, 시드 42 (넘어짐) | 평소 조건 4칸 (스텝당 보상) |"); P("|---|---|---|---|")
    for sy in "TR":
        nom = [load(a.d, f"{m}_{ad}{sy}42_nom") for m, ad in CELLS]
        P(f"| {SYS_KO[sy]} | " + " / ".join(mm(load(a.d, f"B_MOT{sy}{s}_mot")) for s in SEEDS) + " | " + " / ".join(mm(x) for x in nom) + " | "
          + " / ".join("—" if x is None else f"{x['reward'].mean():.4f}" for x in nom) + " |")
    P()
    P("### 모든 팔: 넘어짐, 보상, 넘어지는 시점 (출발 = 에피소드가 첫 125스텝 안에 넘어질 확률, 걷는 중 = 걷는 중 1000스텝당)"); P()
    P("| 칸 | 평가 조건 | 팔 | 넘어짐 | 스텝당 보상 | 출발 | 걷는 중 | 걷는 중 정책 이탈 |"); P("|---|---|---|---|---|---|---|---|")
    for m, ad in CELLS:
        for c in (own(ad), oth(ad)):
            rows = [("사전학습", a.ref, f"{m}_PRE_{c}"), ("사전학습 + 정책 출발", a.d, f"{m}_PREps_{c}"), ("정책만", a.d, f"{m}_POL_{c}"),
                    ("동역학 fine-tuning 42", a.ref, f"{m}_{ad}_{c}")] + [(f"동역학 fine-tuning {s}", a.d, f"{m}_{ad}d{s}_{c}") for s in (43, 44)]
            rows += [("replay 한 모델 42, 정책 출발 없음 (설계 32)", a.d32, f"{m}_{ad}n1_{c}")]
            rows += [(f"{SYS_KO[sy]} {s}", a.d, f"{m}_{ad}{sy}{s}_{c}") for sy in "TR" for s in SEEDS]
            for arm, d, name in rows:
                x = load(d, name)
                if x is None:
                    continue
                R["arms"][name] = {k: (float(v.mean()) if isinstance(v, np.ndarray) else v) for k, v in x.items()}
                g = lambda k, fmt="{:.2f}": fmt.format(x[k]) if k in x else "—"  # noqa: E731
                P(f"| {CELL_KO(m, ad)} | {COND_KO[c]} | {arm} | {x['falls'].mean():.2f} | {x['reward'].mean():.4f} | {g('p_start')} | {g('mid_haz')} | {g('dev_mid')} |")
    P()
    json.dump(R, open(out + ".json", "w"), indent=1, default=float)
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("written", out + ".md")


if __name__ == "__main__":
    main()
