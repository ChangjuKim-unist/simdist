"""Design doc 38 (docs/design38_privileged_mask_ko.md): how much of the gap between the deployed policy and the privileged
expert is explained by the expert's privileged inputs.  L(X, c) = 1 - S(X, c) / S(X, nominal); G(c) = L(deployed, c) - L(expert, c);
share(m, c) = [L(expert masked m, c) - L(expert, c)] / G(c).  Robots paired (same speed / seed), one bootstrap index set for all runs.

    python3 scripts/sim2sim/stage38_mask.py --d results/stage38 --d34 results/stage34 --md results/stage38/report.md
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage35_score as s35  # noqa: E402

LEVELS = [("legR50", "뒷다리 하나 50 %"), ("slope5", "마찰 0.3 + 오르막 5°"), ("tow20", "끄는 짐 20 N"), ("mot50", "모터 전체 50 %")]
MASKS = [("F", "발 접촉력"), ("H", "발 높이"), ("P", "물리값"), ("A", "특권 입력 전부")]
NB = 5000


def S_of(r, idx):
    """robot-pooled S for bootstrap index rows idx (NB, N) -> (NB,)"""
    return (r["S"][idx] * r["n"][idx]).sum(1) / r["n"][idx].sum(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="results/stage38"); ap.add_argument("--d34", default="results/stage34")
    ap.add_argument("--md", default="results/stage38/report.md"); a = ap.parse_args()
    ld = lambda name: s35.load(os.path.join(a.d, f"diag_{name}.npz"))  # noqa: E731
    pol = {c: s35.load(os.path.join(a.d34, f"diag_{c}_POL.npz")) for c in ["nominal"] + [l for l, _ in LEVELS]}
    ex = {(m, c): ld(f"{c}_{m}") for m in ["none"] + [k for k, _ in MASKS] for c in ["nominal"] + [l for l, _ in LEVELS]}
    N = len(ex[("none", "nominal")]["n"]); rng = np.random.default_rng(38); idx = rng.integers(0, N, (NB, N))
    full = np.tile(np.arange(N), (1, 1))
    L = lambda r_c, r_n, ix: 1.0 - S_of(r_c, ix) / S_of(r_n, ix)  # noqa: E731
    out, J = [], {}
    P = lambda s="": (print(s), out.append(s))  # noqa: E731
    ci = lambda b: (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))  # noqa: E731
    P("## 설계 38: 특권 입력 가리기"); P()
    P("조건 손실 L = 1 − S(조건)/S(평소), 각자 자기 평소 대비. 배치 정책 = 논문 모델 정책 head(설계 34). 95 % 구간은 로봇 bootstrap."); P()
    P("### 조건 손실 (%)"); P()
    P("| 조건 | 배치 정책 | expert | " + " | ".join(f"expert, {ko} 가림" for _, ko in MASKS) + " |"); P("|---" * (3 + len(MASKS)) + "|")
    for c, ko in LEVELS:
        cells = []
        for r_c, r_n in [(pol[c], pol["nominal"])] + [(ex[(m, c)], ex[(m, "nominal")]) for m in ["none"] + [k for k, _ in MASKS]]:
            if r_c is None or r_n is None:
                cells.append("-"); continue
            p = 100 * L(r_c, r_n, full)[0]; b = 100 * L(r_c, r_n, idx); lo, hi = ci(b)
            cells.append(f"{p:.1f} [{lo:.1f}, {hi:.1f}]")
        P(f"| {ko} | " + " | ".join(cells) + " |")
    P()
    P("### 가림이 설명하는 격차의 몫 share = [L(가림) − L(expert)] / [L(배치 정책) − L(expert)]"); P()
    P("| 조건 | 격차 G (%) | " + " | ".join(f"{ko} ({k})" for k, ko in MASKS) + " |"); P("|---" * (2 + len(MASKS)) + "|")
    for c, ko in LEVELS:
        if any(x is None for x in (pol[c], pol["nominal"], ex[("none", c)], ex[("none", "nominal")])):
            P(f"| {ko} | - |" + " - |" * len(MASKS)); continue
        G0 = L(pol[c], pol["nominal"], full)[0] - L(ex[("none", c)], ex[("none", "nominal")], full)[0]
        Gb = L(pol[c], pol["nominal"], idx) - L(ex[("none", c)], ex[("none", "nominal")], idx)
        row = [f"{100 * G0:.1f}"]; J[c] = {"G": float(G0)}
        for k, _ in MASKS:
            if ex[(k, c)] is None or ex[(k, "nominal")] is None:
                row.append("-"); continue
            e0 = L(ex[(k, c)], ex[(k, "nominal")], full)[0] - L(ex[("none", c)], ex[("none", "nominal")], full)[0]
            eb = L(ex[(k, c)], ex[(k, "nominal")], idx) - L(ex[("none", c)], ex[("none", "nominal")], idx)
            sh, shb = e0 / G0, eb / Gb; lo, hi = ci(shb)
            row.append(f"{sh:.2f} [{lo:.2f}, {hi:.2f}]"); J[c][k] = {"share": float(sh), "ci": [lo, hi]}
        P(f"| {ko} | " + " | ".join(row) + " |")
    P()
    # registered verdicts
    shA = {c: J.get(c, {}).get("A") for c, _ in LEVELS}
    n_expl = sum(1 for v in shA.values() if v and v["share"] >= 0.5 and v["ci"][0] > 0)
    n_none = sum(1 for v in shA.values() if v and v["share"] < 0.2)
    verdict = "정보가 격차를 설명" if n_expl >= 3 else ("정보로는 설명 안 됨" if n_none >= 3 else "부분")
    P(f"- 판정 (share(A) ≥ 0.5이고 하한 > 0인 조건 {n_expl}/4, share(A) < 0.2인 조건 {n_none}/4): **{verdict}**")
    for c, ko in LEVELS:
        a_ = (J.get(c, {}).get("A") or {}).get("share")
        if a_ is None:
            continue
        main = [k for k, _ in MASKS[:3] if (J[c].get(k) or {}).get("share", -9) >= 0.5 * a_] if a_ > 0 else []
        P(f"- 주 신호, {ko}: {', '.join(main) if main else '없음'}")
    nomA = ex[("A", "nominal")]; nom0 = ex[("none", "nominal")]
    if nomA is not None and nom0 is not None:
        d = s35.diff(nomA, nom0, "S", s35.pooled(nom0, "S"))
        P(f"- 평소 조건에서 A 가림의 S 변화: {d[0]:+.1f} % [{d[1]:+.1f}, {d[2]:+.1f}] (R4 기준: −5 % 안)")
    P()
    P("### 움직임 (로봇 합동): S, 넘어짐 /1000스텝, 속도 비, 옆 속도 |vy|, 회전 속도 SD"); P()
    P("| 런 | S | 넘어짐 | 속도 비 | |vy| | 회전 SD |"); P("|---|---|---|---|---|---|")
    for c in ["nominal"] + [l for l, _ in LEVELS]:
        for m in ["none"] + [k for k, _ in MASKS]:
            r = ex[(m, c)]
            if r is not None:
                P(f"| {c}_{m} | {s35.pooled(r, 'S'):.4f} | {s35.pooled(r, 'falls'):.2f} | {s35.pooled(r, 'vratio'):.3f} | "
                  f"{s35.pooled(r, 'vy'):.3f} | {s35.pooled(r, 'wz_sd'):.3f} |")
        if pol[c] is not None:
            r = pol[c]
            P(f"| {c}_배치정책 | {s35.pooled(r, 'S'):.4f} | {s35.pooled(r, 'falls'):.2f} | {s35.pooled(r, 'vratio'):.3f} | "
              f"{s35.pooled(r, 'vy'):.3f} | {s35.pooled(r, 'wz_sd'):.3f} |")
    os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
    open(a.md, "w").write("\n".join(out) + "\n"); json.dump({"share": J, "verdict": verdict}, open(a.md.replace(".md", ".json"), "w"), indent=1)
    print("written", a.md)


if __name__ == "__main__":
    main()
