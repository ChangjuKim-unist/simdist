"""Design 41: tables and the registered verdict (docs/design41_coverage_matched_ko.md §5)."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from toy_common import boot_ci  # noqa: E402

ARMS = ["E", "EN", "M"]
SEEDS = [0, 1, 2]
CONDS = ["kneeRR0", "kneeFR0", "tow40"]


def load(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def Jv(d, name):
    js = load(d, name)
    return None if js is None else np.array(js["J"], float)


def falls(d, name):
    js = load(d, name)
    return None if js is None else int(np.sum(js["fell"]))


def mean_or_nan(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d"); ap.add_argument("--md"); a = ap.parse_args()
    d = a.d
    L = ["# 설계 41 결과 (자동 생성)", ""]

    # ---- oracle ceilings
    U = {}
    L += ["## 1. 상한 U(c): 같은 planner + 진짜 조건 모델 (16개)", "", "| 조건 | J | 넘어짐 /16 | 전진 m/s |", "|---|---|---|---|"]
    for c in ["nominal"] + CONDS:
        js = load(d, f"oracle_{c}")
        if js:
            U[c] = float(np.mean(js["J"]))
            L.append(f"| {c} | {U[c]:.3f} | {int(np.sum(js['fell']))} | {np.nanmean(js['vx_mean']):.2f} |")

    # ---- per arm / seed quantities
    def n(A, s):
        return f"{A}{s}"

    Jb = {(A, s, c): Jv(d, f"cl_{n(A, s)}_{c}_before") for A in ARMS for s in SEEDS for c in ["nominal"] + CONDS}
    Ja = {(A, s, c): Jv(d, f"cl_{n(A, s)}_{c}_after") for A in ARMS for s in SEEDS for c in CONDS}
    Jan = {(A, s, c): Jv(d, f"cl_{n(A, s)}_{c}_afternom") for A in ARMS for s in SEEDS for c in CONDS}
    Jp = {(A, s, c): Jv(d, f"cl_{n(A, s)}_{c}_pafter") for A in ARMS for s in SEEDS for c in CONDS}
    Jpn = {(A, s, c): Jv(d, f"cl_{n(A, s)}_{c}_pafternom") for A in ARMS for s in SEEDS for c in CONDS}

    def G(A, s, c, after=Ja):
        b, x = Jb[(A, s, c)], after[(A, s, c)]
        return None if b is None or x is None else float(x.mean() - b.mean())

    def Lret(A, s, afternom=Jan):
        b = Jb[(A, s, "nominal")]
        if b is None:
            return None
        v = [b.mean() - afternom[(A, s, c)].mean() for c in CONDS if afternom[(A, s, c)] is not None]
        return float(np.mean(v)) if v else None

    # ---- walk gate
    G0 = {}
    L += ["", "## 2. 걷기 관문 G0: 적응 전 nominal J (시드 0 / 1 / 2, 평균 ≥ 0.40 통과), 넘어짐 /16", "",
          "| 팔 | J 시드별 | 평균 | 넘어짐 시드별 | 판정 |", "|---|---|---|---|---|"]
    for A in ARMS:
        js = [Jb[(A, s, "nominal")] for s in SEEDS]
        if any(j is None for j in js):
            continue
        m = float(np.mean([j.mean() for j in js]))
        G0[A] = m >= 0.40
        fl = [falls(d, f"cl_{n(A, s)}_nominal_before") for s in SEEDS]
        L.append(f"| {A} | {' / '.join(f'{j.mean():.3f}' for j in js)} | {m:.3f} | {' / '.join(map(str, fl))} | "
                 f"{'통과' if G0[A] else '실패'} |")

    # ---- gains, positive control, retention
    L += ["", "## 3. 적응 이득 G = J(적응 후) − J(적응 전), 시드 평균 [시드별], 넘어짐 전→후 (시드 합, /48)", "",
          "| 팔 | " + " | ".join(CONDS) + " |", "|---|---|---|---|"]
    for A in ARMS:
        cells = []
        for c in CONDS:
            g = [G(A, s, c) for s in SEEDS]
            if any(x is None for x in g):
                cells.append("–"); continue
            fb = sum(falls(d, f"cl_{n(A, s)}_{c}_before") for s in SEEDS)
            fa = sum(falls(d, f"cl_{n(A, s)}_{c}_after") for s in SEEDS)
            cells.append(f"{np.mean(g):+.3f} [{', '.join(f'{x:+.2f}' for x in g)}] ({fb}→{fa})")
        L.append(f"| {A} | " + " | ".join(cells) + " |")
    L += ["", "## 4. 적응 전 J (시드 평균)와 여지 U(c) − 적응 전", "", "| 팔 | " + " | ".join(CONDS) + " |", "|---|---|---|---|"]
    for A in ARMS:
        cells = []
        for c in CONDS:
            b = mean_or_nan([Jb[(A, s, c)].mean() if Jb[(A, s, c)] is not None else None for s in SEEDS])
            cells.append(f"{b:.3f} (여지 {U.get(c, float('nan')) - b:+.3f})")
        L.append(f"| {A} | " + " | ".join(cells) + " |")
    L += ["", "## 5. 양성 대조 G⁺ (특권 데이터로 같은 미세조정), 시드 평균 [시드별]", "", "| 팔 | " + " | ".join(CONDS) + " | 보존 손실 L⁺ |",
          "|---|---|---|---|---|"]
    Gp = {}
    for A in ARMS:
        cells = []
        for c in CONDS:
            g = [G(A, s, c, Jp) for s in SEEDS]
            if any(x is None for x in g):
                cells.append("–"); continue
            Gp[(A, c)] = float(np.mean(g))
            cells.append(f"{np.mean(g):+.3f} [{', '.join(f'{x:+.2f}' for x in g)}]")
        lp = [Lret(A, s, Jpn) for s in SEEDS]
        L.append(f"| {A} | " + " | ".join(cells) + f" | {mean_or_nan(lp):+.3f} |")
    Lr = {}
    L += ["", "## 6. 보존 손실 L = J(적응 전, nominal) − J(적응 후, nominal), 세 조건 평균", "", "| 팔 | 시드별 | 평균 |", "|---|---|---|"]
    for A in ARMS:
        l = [Lret(A, s) for s in SEEDS]
        if any(x is None for x in l):
            continue
        Lr[A] = float(np.mean(l))
        L.append(f"| {A} | {' / '.join(f'{x:+.3f}' for x in l)} | {Lr[A]:+.3f} |")

    # ---- condition eligibility
    elig = []
    L += ["", "## 7. 조건 자격 (여지 ≥ 0.10 그리고 max G⁺(E, M) ≥ 0.05)", ""]
    for c in CONDS:
        bE = mean_or_nan([Jb[("E", s, c)].mean() if Jb[("E", s, c)] is not None else None for s in SEEDS])
        bM = mean_or_nan([Jb[("M", s, c)].mean() if Jb[("M", s, c)] is not None else None for s in SEEDS])
        room = U.get(c, float("nan")) - max(bE, bM)
        absorb = max(Gp.get(("E", c), -9), Gp.get(("M", c), -9))
        ok = room >= 0.10 and absorb >= 0.05
        if ok:
            elig.append(c)
        L.append(f"- {c}: 여지 {room:+.3f}, max G⁺ {absorb:+.3f} → {'자격' if ok else '제외'}")

    # ---- D tables
    def Dtab(X, Y):
        out = {}
        for c in CONDS:
            ds, pairs = [], []
            for s in SEEDS:
                need = [Jb[(X, s, c)], Ja[(X, s, c)], Jb[(Y, s, c)], Ja[(Y, s, c)]]
                if any(v is None for v in need):
                    break
                ds.append(G(X, s, c) - G(Y, s, c))
                pairs.append((Ja[(X, s, c)] - Jb[(X, s, c)]) - (Ja[(Y, s, c)] - Jb[(Y, s, c)]))
            if len(ds) == len(SEEDS):
                out[c] = (float(np.mean(ds)), ds, boot_ci(np.concatenate(pairs)))
        return out

    L += ["", "## 8. 주 비교 D(c) = G(M, c) − G(E, c) [시드별], 95 % 구간 (짝 에피소드 48개)", "", "| 조건 | D | 시드별 | 구간 |", "|---|---|---|---|"]
    DM = Dtab("M", "E")
    for c, (m, ds, ci) in DM.items():
        L.append(f"| {c} | {m:+.3f} | {', '.join(f'{x:+.2f}' for x in ds)} | [{ci[1]:+.3f}, {ci[2]:+.3f}] |")
    L += ["", "참고: G(M, c) − G(EN, c)", "", "| 조건 | D | 시드별 | 구간 |", "|---|---|---|---|"]
    for c, (m, ds, ci) in Dtab("M", "EN").items():
        L.append(f"| {c} | {m:+.3f} | {', '.join(f'{x:+.2f}' for x in ds)} | [{ci[1]:+.3f}, {ci[2]:+.3f}] |")

    # ---- verdict
    verdict = None
    if "E" in G0 and "M" in G0 and all(c in DM for c in CONDS) and "E" in Lr and "M" in Lr:
        if not G0["M"] and not G0["E"]:
            verdict = "멈춤 (E, M 모두 걷기 관문 실패)"
        elif G0["M"] and not G0["E"]:
            verdict = "planner 단계 의존 (E만 걷기 관문 실패)"
        elif not G0["M"]:
            verdict = "멈춤 (M 걷기 관문 실패)"
        elif not elig:
            verdict = "측정 불가 (자격 조건 없음)"
        else:
            need = 2 if len(elig) >= 2 else 1
            sup = sum(DM[c][0] >= 0.10 and all(x > 0 for x in DM[c][1]) for c in elig) >= need and Lr["M"] - Lr["E"] <= 0.05
            ag = sum(-DM[c][0] >= 0.10 and all(x < 0 for x in DM[c][1]) for c in elig) >= need and Lr["E"] - Lr["M"] <= 0.05
            none = all(abs(DM[c][0]) < 0.05 for c in elig)
            verdict = "가설 지지" if sup else "가설 반대" if ag else "차이 없음" if none else "중간"
        L += ["", f"**판정: {verdict}** (자격 조건: {', '.join(elig) if elig else '없음'})"]

    # ---- reference: ranking and motion quality
    L += ["", "## 9. 참고: 후보 순위 Spearman (시드 평균)", "", "| 팔 | nominal | kneeRR0 | tow40 |", "|---|---|---|---|"]
    for A in ARMS:
        cells = []
        for c in ("mot100", "kneeRR0", "tow40"):
            v = [load(d, f"rank_{n(A, s)}_{c}") for s in SEEDS]
            v = [x for x in v if x]
            cells.append(f"{np.mean([x['rho_mean'] for x in v]):.2f}" if v else "–")
        L.append(f"| {A} | " + " | ".join(cells) + " |")
    L += ["", "## 10. 참고: 동작 품질 (넘어지기 전 2 s 이후, 시드 평균)", "",
          "| 칸 | 전진 m/s | roll SD ° | pitch SD ° | 토크² | 미끄러짐 m/s |", "|---|---|---|---|---|---|"]
    rows = [(f"상한 {c}", [f"oracle_{c}"]) for c in ["nominal"] + CONDS]
    for A in ARMS:
        rows += [(f"{A} nominal 적응 전", [f"cl_{n(A, s)}_nominal_before" for s in SEEDS])]
        for c in CONDS:
            rows += [(f"{A} {c} 적응 전", [f"cl_{n(A, s)}_{c}_before" for s in SEEDS]),
                     (f"{A} {c} 적응 후", [f"cl_{n(A, s)}_{c}_after" for s in SEEDS])]
    for label, names in rows:
        js = [load(d, x) for x in names]
        js = [x for x in js if x]
        if not js:
            continue
        def m(k):
            return np.nanmean([np.nanmean(np.array(x[k], float)) for x in js])
        L.append(f"| {label} | {m('vx_mean'):.2f} | {m('roll_sd_deg'):.1f} | {m('pitch_sd_deg'):.1f} | {m('tau2'):.1f} | {m('slip'):.3f} |")

    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"verdict": verdict, "eligible": elig, "U": U, "G0": G0, "L": Lr,
               "D_M_E": {c: v[0] for c, v in DM.items()}, "Gplus": {f"{k[0]}|{k[1]}": v for k, v in Gp.items()}},
              open(a.md[:-3] + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
