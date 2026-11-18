"""Design doc 32 part A (docs/design32_recipe_ko.md): replication of the design-31 gates with new training seeds and the
ingredient ablation of the tok recipe.  Same protocol and references as design 31 (PRE = pretrained planner, DYN =
dynamics-only fine-tuned planner from results/stage29; 18 robots x 5000 steps; paired bootstrap over robots).

    python3 scripts/sim2sim/stage32_gate.py --d results/stage32 --d31 results/stage31 --ref results/stage29

Cell criteria (unchanged from design 31):
  gain      own condition:   (PRE - arm) / (PRE - DYN) >= 0.70 with CI lower > 0.40   (3 cells with a gain; broad motor n/a)
  retention other condition: arm - PRE <= max(0.3 x PRE, 0.15) falls per 1000 steps  (4 cells)
R1 = gain with the adapted token, R2 = retention with the simulation token, R3 = retention with the adapted token
(no switch); each for training seeds 43 and 44 (seed 42 = design 31, shown for reference).
Ablation arms (training seed 42, deployed as one model without a switch): n2 = no token + both replay sets,
t1 = token + pretraining windows only (adapted token), n1 = no token + pretraining windows only.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage31_gate import load, boot, MODEL_KO, AD_KO, COND_KO, CELLS, own, oth  # noqa: E402

GAIN_CELLS = [("P", "ICE"), ("P", "MOT"), ("B", "ICE")]
CELL_KO = lambda m, a: f"{MODEL_KO[m].replace(' 모델', '')}·{AD_KO[a].replace(' 적응', '')}"  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="results/stage32"); ap.add_argument("--d31", default="results/stage31")
    ap.add_argument("--ref", default="results/stage29"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.d, "gate")
    L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    R = {"gain": {}, "ret": {}, "verdict": {}, "arms": {}}
    # arm key -> (label, directory, infix of the adapted-token / single-model run, infix of the simulation-token run)
    ARMS = {"tok42": ("토큰 모델 시드 42 (설계 31)", a.d31, "tA", "tS"), "tok43": ("토큰 모델 시드 43", a.d, "t43A", "t43S"),
            "tok44": ("토큰 모델 시드 44", a.d, "t44A", "t44S"),
            "n2": ("토큰 없음 + replay 둘 다", a.d, "n2", None), "t1": ("토큰 + 사전학습 창만", a.d, "t1A", None),
            "n1": ("토큰 없음 + 사전학습 창만", a.d, "n1", None)}

    def gain(d, infix, m, ad):
        c = own(ad); pre = load(a.ref, f"{m}_PRE_{c}"); dyn = load(a.ref, f"{m}_{ad}_{c}"); x = load(d, f"{m}_{ad}{infix}_{c}")
        if x is None or pre is None or dyn is None:
            return None
        k = boot(lambda p, q, t: (p.mean() - t.mean()) / (p.mean() - q.mean()) if p.mean() - q.mean() > 0 else np.nan, pre["falls"], dyn["falls"], x["falls"])
        return {"pre": float(pre["falls"].mean()), "dyn": float(dyn["falls"].mean()), "arm": float(x["falls"].mean()), "kept": k,
                "ok": bool(k[0] >= 0.70 and k[1] > 0.40)}

    def ret(d, infix, m, ad):
        c = oth(ad); pre = load(a.ref, f"{m}_PRE_{c}"); dyn = load(a.ref, f"{m}_{ad}_{c}"); x = load(d, f"{m}_{ad}{infix}_{c}")
        if x is None or pre is None:
            return None
        dlt = boot(lambda p, t: t.mean() - p.mean(), pre["falls"], x["falls"]); mg = max(0.3 * pre["falls"].mean(), 0.15)
        return {"pre": float(pre["falls"].mean()), "dyn": None if dyn is None else float(dyn["falls"].mean()), "arm": float(x["falls"].mean()),
                "diff": dlt, "margin": float(mg), "ok": bool(dlt[0] <= mg)}

    f3 = lambda t: "{:.2f} [{:.2f}, {:.2f}]".format(*t)  # noqa: E731

    def table(title, kind, rows, cells):
        """rows: (key, label, dir, infix). One comparison per table: cell = falls -> kept gain, or falls -> (arm - PRE)."""
        P(title); P()
        P("| 팔 | " + " | ".join(CELL_KO(m, ad) + (f" ({COND_KO[oth(ad)]})" if kind == "ret" else "") for m, ad in cells) + " | 통과 칸 |")
        P("|---|" + "---|" * (len(cells) + 1))
        for key, label, d, infix in rows:
            cs = []; n_ok = n = 0
            for m, ad in cells:
                r = (gain if kind == "gain" else ret)(d, infix, m, ad)
                R[kind].setdefault(key, {})[f"{m}_{ad}"] = r
                if r is None:
                    cs.append("미실행"); continue
                n += 1; n_ok += r["ok"]
                v = f"{r['arm']:.2f} → " + (f3(r["kept"]) if kind == "gain" else f"{r['diff'][0]:+.2f} [{r['diff'][1]:.2f}, {r['diff'][2]:.2f}] (폭 {r['margin']:.2f})")
                cs.append(("**" + v + "**") if r["ok"] else v)
            R["verdict"][f"{kind}:{key}"] = [int(n_ok), int(n)]
            P(f"| {label} | " + " | ".join(cs) + f" | {n_ok}/{n} |")
        P()

    def refs(c_of):
        return ", ".join(f"{CELL_KO(m, ad)} {load(a.ref, f'{m}_PRE_{c_of(ad)}')['falls'].mean():.2f} / {load(a.ref, f'{m}_{ad}_{c_of(ad)}')['falls'].mean():.2f}" for m, ad in CELLS)

    P("## 설계 32-A: 재현과 성분 (1000스텝당 넘어짐, 로봇 18대 × 5000스텝)"); P()
    P(f"기준 (사전학습 / 동역학 fine-tuning). 자기 조건: {refs(own)}. 다른 범위 밖 조건: {refs(oth)}."); P()
    seeds = ("tok42", "tok43", "tok44")
    table("### R1: 적응 토큰, 자기 조건 (칸 값 = 넘어짐 → 유지된 이득 [95 % CI], 굵게 = 칸 통과)", "gain", [(k, ARMS[k][0], ARMS[k][1], ARMS[k][2]) for k in seeds], GAIN_CELLS)
    table("### R2: 시뮬 토큰, 다른 범위 밖 조건 (칸 값 = 넘어짐 → 사전학습과의 차이 [95 % CI], 굵게 = 허용 폭 안)", "ret", [(k + "S", ARMS[k][0], ARMS[k][1], ARMS[k][3]) for k in seeds], CELLS)
    table("### R3: 적응 토큰 그대로(전환 없음), 다른 범위 밖 조건", "ret", [(k + "A", ARMS[k][0], ARMS[k][1], ARMS[k][2]) for k in seeds], CELLS)
    V = R["verdict"]; ok = lambda k, need: V.get(k, [0, 0])[1] > 0 and V[k][0] >= need  # noqa: E731
    rep = {"R1": all(ok(f"gain:tok{s}", 2) for s in (43, 44)), "R2": all(ok(f"ret:tok{s}S", 3) for s in (43, 44)), "R3": all(ok(f"ret:tok{s}A", 3) for s in (43, 44))}
    R["replication"] = rep
    P("재현 판정 (새 시드 43, 44가 각각 기준을 넘어야 함): " + ", ".join(f"{k} {'재현' if v else '재현 안 됨'}" for k, v in rep.items())); P()

    abl = ("tok42", "n2", "t1", "n1")
    table("### 성분, 이득: 자기 조건 (칸 값 = 넘어짐 → 유지된 이득 [95 % CI])", "gain", [(k + "_abl", ARMS[k][0], ARMS[k][1], ARMS[k][2]) for k in abl], GAIN_CELLS)
    table("### 성분, 보존: 한 모델을 그대로 쓸 때 다른 범위 밖 조건 (칸 값 = 넘어짐 → 사전학습과의 차이 [95 % CI])", "ret", [(k + "_abl", ARMS[k][0], ARMS[k][1], ARMS[k][2]) for k in abl], CELLS)
    R["recipe"] = {k: bool(ok(f"gain:{k}_abl", 2) and ok(f"ret:{k}_abl", 3)) for k in abl}
    P("레시피 판정 (이득 2칸 이상 + 보존 3칸 이상): " + ", ".join(f"{ARMS[k][0]} {'통과' if v else '불통과'}" for k, v in R["recipe"].items())); P()

    P("### 학습 시드 사이 폭 (토큰 모델 시드 42 / 43 / 44의 넘어짐, 적응 토큰)과 성분 팔"); P()
    P("| 칸 | 평가 조건 | 토큰 모델 3시드 | 토큰 없음 + replay 둘 다 | 토큰 + 사전학습 창만 | 토큰 없음 + 사전학습 창만 |"); P("|---|---|---|---|---|---|")
    mm = lambda d, n: (lambda x: "—" if x is None else f"{x['falls'].mean():.2f}")(load(d, n))  # noqa: E731
    for m, ad in CELLS:
        for c in (own(ad), oth(ad)):
            P(f"| {CELL_KO(m, ad)} | {COND_KO[c]} | " + " / ".join(mm(ARMS[k][1], f"{m}_{ad}{ARMS[k][2]}_{c}") for k in seeds) + " | "
              + " | ".join(mm(a.d, f"{m}_{ad}{i}_{c}") for i in ("n2", "t1A", "n1")) + " |")
    P()
    P("### 모든 팔: 넘어짐, 보상, 넘어지는 시점 (출발 = 에피소드가 첫 125스텝 안에 넘어질 확률, 걷는 중 = 걷는 중 1000스텝당)"); P()
    P("| 칸 | 평가 조건 | 팔 | 넘어짐 | 스텝당 보상 | 출발 | 걷는 중 | 걷는 중 정책 이탈 |"); P("|---|---|---|---|---|---|---|---|")
    for m, ad in CELLS:
        for c in (own(ad), oth(ad), "nom"):
            rows = [("사전학습", a.ref, f"{m}_PRE_{c}"), ("동역학 fine-tuning", a.ref, f"{m}_{ad}_{c}"), ("토큰 42, 적응 토큰", a.d31, f"{m}_{ad}tA_{c}"), ("토큰 42, 시뮬 토큰", a.d31, f"{m}_{ad}tS_{c}")]
            rows += [(f"토큰 {s}, {'적응' if t == 'A' else '시뮬'} 토큰", a.d, f"{m}_{ad}t{s}{t}_{c}") for s in (43, 44) for t in "AS"]
            rows += [(ARMS[k][0], a.d, f"{m}_{ad}{ARMS[k][2]}_{c}") for k in ("n2", "t1", "n1")]
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
