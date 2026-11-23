"""Design 42: ladder decision (--decide) and full report with the registered verdict (docs/design42_dose_ladder_ko.md §5)."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from toy_common import boot_ci  # noqa: E402

ARMS = ["E", "M"]
SEEDS = [0, 1, 2]
CONDS = ["kneeRR0", "kneeFR0"]
CELLS = ["A1", "A2", "A3"]                  # increasing dose; A4 = diagnostic only
DOSE = {"A1": 8000, "A2": 50000, "A3": 50000}


def load(d, name):
    p = os.path.join(d, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def arr(js, k):
    return None if js is None else np.array(js[k], float)


def oracle(d41):
    return {c: float(np.mean(load(d41, f"oracle_{c}")["J"])) for c in ["nominal"] + CONDS}


def ladder(d, d41):
    """Per cell / arm / condition: seed-mean J, seed-mean falls, pass flag; and the chosen procedure P(c)."""
    U = oracle(d41)
    res, P = {}, {}
    for c in CONDS:
        thr = 0.5 * U[c]
        for cell in CELLS:
            for A in ARMS:
                js = [load(d, f"cl_{A}{s}_{c}_{cell}") for s in SEEDS]
                if any(j is None for j in js):
                    res[(cell, A, c)] = None
                    continue
                J = [float(np.mean(j["J"])) for j in js]
                f = [float(np.sum(j["fell"])) for j in js]
                ok = np.mean(J) >= thr and np.mean(f) <= 8
                res[(cell, A, c)] = dict(J=J, falls=f, Jm=float(np.mean(J)), fm=float(np.mean(f)), ok=bool(ok))
        for cell in CELLS:
            r = [res[(cell, A, c)] for A in ARMS]
            if all(x is not None and x["ok"] for x in r):
                P[c] = cell
                break
    return U, res, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d"); ap.add_argument("--d41"); ap.add_argument("--md", default="")
    ap.add_argument("--decide", action="store_true")
    a = ap.parse_args()
    d, d41 = a.d, a.d41
    U, res, P = ladder(d, d41)
    if a.decide:
        out = {c: dict(cell=P[c], dose=DOSE[P[c]]) for c in P}
        json.dump(out, open(os.path.join(d, "ladder.json"), "w"), indent=1)
        print(json.dumps(out))
        return

    L = ["# 설계 42 결과 (자동 생성)", ""]
    L += ["## 1. 특권 데이터 용량 사다리: 적응 후 J (시드 0/1/2), 넘어짐 /16 평균",
          "", f"통과 = 시드 평균 J ≥ 상한의 절반 (kneeRR0 {0.5 * U['kneeRR0']:.3f}, kneeFR0 {0.5 * U['kneeFR0']:.3f}) 그리고 넘어짐 평균 ≤ 8/16", ""]
    for c in CONDS:
        L += [f"### {c} (상한 {U[c]:.3f})", "", "| 칸 | 팔 | J 시드별 | 평균 J | 넘어짐 평균 | 통과 | nominal J 하락 |", "|---|---|---|---|---|---|---|"]
        for cell in CELLS:
            for A in ARMS:
                r = res[(cell, A, c)]
                if r is None:
                    L.append(f"| {cell} | {A} | (없음) | | | | |"); continue
                drop = []
                for s in SEEDS:
                    b, x = load(d41, f"cl_{A}{s}_nominal_before"), load(d, f"cl_{A}{s}_{c}_{cell}nom")
                    if b and x:
                        drop.append(np.mean(b["J"]) - np.mean(x["J"]))
                L.append(f"| {cell} | {A} | {' / '.join(f'{v:.3f}' for v in r['J'])} | {r['Jm']:.3f} | {r['fm']:.1f} | "
                         f"{'예' if r['ok'] else '아니오'} | {np.mean(drop) if drop else float('nan'):+.3f} |")
        j4 = load(d, f"cl_A4_{c}")
        if j4:
            L.append(f"| A4 (진단, 고장 데이터만으로 처음부터) | – | {np.mean(j4['J']):.3f} | | {int(np.sum(j4['fell']))} | | |")
        L.append("")
    L += ["선택된 절차 P(c): " + (", ".join(f"{c} → {P[c]} ({DOSE[P[c]]} transition)" for c in P) if P else "없음"), ""]

    if not P:
        verdict = "장난감 경로 닫음 (어떤 용량에서도 world model MPPI가 고장 조건을 걷지 못함)"
        L += [f"**판정: {verdict}**"]
        rep = dict(verdict=verdict, P=P)
    else:
        # ---- stage B
        L += ["## 2. 자기 수집 (넘어지면 다시 세움): usable transition / 로봇 스텝 / 에피소드 / 넘어짐", "",
              "| 팔 시드 | 조건 | usable | 로봇 스텝 | 에피소드 | 넘어진 에피소드 |", "|---|---|---|---|---|---|"]
        for A in ARMS:
            for s in SEEDS:
                for c in P:
                    p = os.path.join(d, f"col_{A}{s}_{c}.txt")
                    if os.path.exists(p):
                        u, r, e, f = open(p).read().split()[:4]
                        L.append(f"| {A}{s} | {c} | {u} | {r} | {e} | {f} |")
        Jb = {(A, s, c): arr(load(d41, f"cl_{A}{s}_{c}_before"), "J") for A in ARMS for s in SEEDS for c in ["nominal"] + CONDS}
        Ja = {(A, s, c): arr(load(d, f"cl_{A}{s}_{c}_B"), "J") for A in ARMS for s in SEEDS for c in P}
        Jn = {(A, s, c): arr(load(d, f"cl_{A}{s}_{c}_Bnom"), "J") for A in ARMS for s in SEEDS for c in P}
        Fb = {(A, s, c): load(d41, f"cl_{A}{s}_{c}_before") for A in ARMS for s in SEEDS for c in P}
        Fa = {(A, s, c): load(d, f"cl_{A}{s}_{c}_B") for A in ARMS for s in SEEDS for c in P}

        def G(A, s, c):
            b, x = Jb[(A, s, c)], Ja[(A, s, c)]
            return None if b is None or x is None else float(x.mean() - b.mean())

        def Lr(A, s):
            b = Jb[(A, s, "nominal")]
            v = [b.mean() - Jn[(A, s, c)].mean() for c in P if Jn[(A, s, c)] is not None]
            return float(np.mean(v)) if v else None

        L += ["", "## 3. 적응 이득 G = J(적응 후) − J(적응 전), 시드 평균 [시드별], 넘어짐 전→후 (/48)", "",
              "| 팔 | " + " | ".join(P) + " |", "|---|" + "---|" * len(P)]
        for A in ARMS:
            row = []
            for c in P:
                g = [G(A, s, c) for s in SEEDS]
                if any(x is None for x in g):
                    row.append("(없음)"); continue
                fb = sum(int(np.sum(Fb[(A, s, c)]["fell"])) for s in SEEDS)
                fa = sum(int(np.sum(Fa[(A, s, c)]["fell"])) for s in SEEDS)
                row.append(f"{np.mean(g):+.3f} [{', '.join(f'{x:+.2f}' for x in g)}] ({fb}→{fa})")
            L.append(f"| {A} | " + " | ".join(row) + " |")
        Lm = {A: [Lr(A, s) for s in SEEDS] for A in ARMS}
        L += ["", "## 4. 보존 손실 L = J(적응 전, nominal) − J(적응 후, nominal), 포함 조건 평균", "", "| 팔 | 시드별 | 평균 |", "|---|---|---|"]
        for A in ARMS:
            if all(x is not None for x in Lm[A]):
                L.append(f"| {A} | {' / '.join(f'{x:+.3f}' for x in Lm[A])} | {np.mean(Lm[A]):+.3f} |")

        # ---- eligibility: room >= 0.10 (ladder pass replaces positive control)
        elig = []
        L += ["", "## 5. 조건 자격 (여지 ≥ 0.10, 사다리 통과)", ""]
        for c in P:
            room = U[c] - max(np.mean([Jb[(A, s, c)].mean() for s in SEEDS]) for A in ARMS)
            ok = room >= 0.10
            L.append(f"- {c}: 여지 {room:+.3f}, 사다리 {P[c]} → {'자격' if ok else '제외'}")
            if ok:
                elig.append(c)

        # ---- main comparison
        Dres = {}
        L += ["", "## 6. 주 비교 D(c) = G(M, c) − G(E, c) [시드별], 95 % 구간 (짝 에피소드 48개)", "",
              "| 조건 | D | 시드별 | 구간 |", "|---|---|---|---|"]
        for c in P:
            ds, pairs = [], []
            for s in SEEDS:
                gm, ge = G("M", s, c), G("E", s, c)
                if gm is None or ge is None:
                    continue
                ds.append(gm - ge)
                pairs.append((Ja[("M", s, c)] - Jb[("M", s, c)]) - (Ja[("E", s, c)] - Jb[("E", s, c)]))
            if len(ds) == 3:
                ci = boot_ci(np.concatenate(pairs))
                Dres[c] = (float(np.mean(ds)), ds)
                L.append(f"| {c} | {np.mean(ds):+.3f} | {', '.join(f'{x:+.2f}' for x in ds)} | [{ci[1]:+.3f}, {ci[2]:+.3f}] |")

        def supports(sign):
            if not elig or any(c not in Dres for c in elig):
                return False
            good = [c for c in elig if sign * Dres[c][0] >= 0.10 and all(sign * x > 0 for x in Dres[c][1])]
            need = 1 if len(elig) == 1 else 2
            LM, LE = np.mean(Lm["M"]), np.mean(Lm["E"])
            lok = (LM - LE <= 0.05) if sign > 0 else (LE - LM <= 0.05)
            return len(good) >= need and lok

        if not elig:
            verdict = "측정 불가 (자격 조건 없음)"
        elif supports(+1):
            verdict = "가설 지지"
        elif supports(-1):
            verdict = "가설 반대"
        elif all(c in Dres and abs(Dres[c][0]) < 0.05 for c in elig):
            verdict = "차이 없음"
        else:
            verdict = "중간"
        L += ["", f"**판정: {verdict}** (자격 조건: {', '.join(elig) if elig else '없음'})"]

        # ---- motion quality (report only)
        L += ["", "## 7. 참고: 동작 품질 (시드 평균)", "", "| 칸 | 전진 m/s | roll SD ° | pitch SD ° | 토크² | 미끄러짐 m/s |", "|---|---|---|---|---|---|"]
        for A in ARMS:
            for c in P:
                for lab, dd, nm in (("적응 전", d41, "before"), ("적응 후", d, "B")):
                    js = [load(dd, f"cl_{A}{s}_{c}_{nm}") for s in SEEDS]
                    if any(j is None for j in js):
                        continue
                    q = lambda k: np.nanmean([np.nanmean(j[k]) for j in js])
                    L.append(f"| {A} {c} {lab} | {q('vx_mean'):.2f} | {q('roll_sd_deg'):.1f} | {q('pitch_sd_deg'):.1f} | {q('tau2'):.1f} | {q('slip'):.3f} |")
        rep = dict(verdict=verdict, P=P, eligible=elig, D={c: Dres[c][0] for c in Dres},
                   L={A: float(np.mean(Lm[A])) for A in ARMS if all(x is not None for x in Lm[A])})
    rep["ladder"] = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in res.items()}
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(rep, open(a.md.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
