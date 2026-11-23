"""Design 43: tables and the registered verdict (docs/design43_own_state_coverage_ko.md §5)."""
import argparse, json, os

import numpy as np

CONDS = ["kneeRR0", "kneeFR0"]
SEEDS = [0, 1, 2]
ROUNDS = [0, 1, 2, 3, 4]


def load(d, name):
    p = os.path.join(d, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d"); ap.add_argument("--d41"); ap.add_argument("--md")
    a = ap.parse_args()
    d = a.d
    U = {c: float(np.mean(load(a.d41, f"oracle_{c}")["J"])) for c in CONDS}

    def cell(arm, c, r):
        nm = "DG" if r == 0 else arm                    # round 0 is shared (base data only)
        js = [load(d, f"cl_{nm}_{c}_s{s}_r{r}") for s in SEEDS]
        if any(j is None for j in js):
            return None
        J = [float(np.mean(j["J"])) for j in js]
        f = [float(np.sum(j["fell"])) for j in js]
        vx = float(np.nanmean([np.nanmean(j["vx_mean"]) for j in js]))
        ok = np.mean(J) >= 0.5 * U[c] and np.mean(f) <= 8
        return dict(J=J, f=f, Jm=float(np.mean(J)), fm=float(np.mean(f)), vx=vx, ok=bool(ok))

    L = ["# 설계 43 결과 (자동 생성)", "",
         "통과 = 시드 평균 J ≥ 상한의 절반 그리고 넘어짐 평균 ≤ 8/16. 판정은 4라운드 값으로 한다.", ""]
    res = {}
    for c in CONDS:
        L += [f"## {c} (상한 {U[c]:.3f}, 통과선 {0.5 * U[c]:.3f})", "",
              "| 라운드 | 데이터 (기본 + 추가) | DG: J 시드별 / 평균 (넘어짐) | CT: J 시드별 / 평균 (넘어짐) |", "|---|---|---|---|"]
        for r in ROUNDS:
            row = [str(r), f"50k + {25 * r}k"]
            for arm in ["DG", "CT"]:
                x = cell(arm, c, r)
                res[(arm, c, r)] = x
                row.append("(없음)" if x is None else
                           f"{' / '.join(f'{v:.3f}' for v in x['J'])} / **{x['Jm']:.3f}** ({x['fm']:.1f}){' 통과' if x['ok'] else ''}")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        L += ["전진 m/s (시드 평균): " + ", ".join(
            f"{arm} r{r} {res[(arm, c, r)]['vx']:.2f}" for arm in ["DG", "CT"] for r in ROUNDS if res.get((arm, c, r))), ""]

    def ok(arm, c):
        x = res.get((arm, c, 4))
        return x is not None and x["ok"]

    dg = [c for c in CONDS if ok("DG", c)]
    ct = [c for c in CONDS if ok("CT", c)]
    if any(c not in ct for c in dg):
        verdict = "자기 상태 덮기가 열쇠 (DG 통과, 같은 양의 CT 실패인 조건 있음)"
    elif dg:
        verdict = "양으로 충분 (DG와 CT 모두 통과)"
    elif ct:
        verdict = "중간 (CT만 통과)"
    else:
        verdict = "실패 (4라운드에서 DG가 두 조건 모두 통과 못 함)"
    L += [f"**판정: {verdict}** (DG 통과: {', '.join(dg) or '없음'}; CT 통과: {', '.join(ct) or '없음'})"]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(dict(verdict=verdict, dg_pass=dg, ct_pass=ct, U=U,
                   cells={f"{k[0]}_{k[1]}_r{k[2]}": v for k, v in res.items()}),
              open(a.md.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
