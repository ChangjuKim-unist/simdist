"""Design 40 §10: E1 / E2 tables and the registered verdict."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from toy_common import boot_ci  # noqa: E402

SRC = ["P", "PN", "R", "ME", "MC"]
E2C = ["kneeRR0", "kneeFR0", "tow40"]


def J(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        return None
    js = json.load(open(p))
    return np.array(js["J"]), int(np.sum(js["fell"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d"); ap.add_argument("--md"); a = ap.parse_args()
    L = ["# 설계 40 E1·E2 결과 (자동 생성)", "", "## E1 후보 순위 (Spearman 평균 / 예측 1위 regret 상대값; 앵커당 후보 16개)", "",
         "| 원천 | nominal | kneeRR0 | tow40 | 열린 루프 오차 (nominal) |", "|---|---|---|---|---|"]
    for s in SRC:
        cells, mse = [], "–"
        for c in ("mot100", "kneeRR0", "tow40"):
            p = os.path.join(a.d, f"rank_{s}_{c}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                cells.append(f"{r['rho_mean']:.2f} / {r['regret_rel']:.2f}")
                if c == "mot100":
                    mse = f"{r['open_loop_mse']:.2f}"
            else:
                cells.append("–")
        L.append(f"| {s} | " + " | ".join(cells) + f" | {mse} |")
    L += ["", "## E1 폐루프 (world model MPPI, nominal, 16개): J / 넘어짐", ""]
    for s in SRC:
        r = J(a.d, f"cl_{s}_nominal_before")
        L.append(f"- {s}: " + (f"J {r[0].mean():.3f}, 넘어짐 {r[1]}" if r else "–"))
    L += ["", "## E2 이득 G = J(적응 후) − J(적응 전), 조건별 (95 % 구간), 넘어짐 전→후", "",
          "| 원천 | " + " | ".join(E2C) + " | 보존 손실 L (nominal 평균) |", "|---|---|---|---|---|"]
    G, Lr = {}, {}
    for s in SRC:
        b0 = J(a.d, f"cl_{s}_nominal_before")
        cells, losses = [], []
        for c in E2C:
            rb, ra, rn = J(a.d, f"cl_{s}_{c}_before"), J(a.d, f"cl_{s}_{c}_after"), J(a.d, f"cl_{s}_{c}_afternom")
            if rb and ra:
                m, lo, hi = boot_ci(ra[0] - rb[0])
                G[(s, c)] = m
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({rb[1]}→{ra[1]})")
            else:
                cells.append("–")
            if b0 and rn:
                losses.append(b0[0].mean() - rn[0].mean())
        if losses:
            Lr[s] = float(np.mean(losses))
        L.append(f"| {s} | " + " | ".join(cells) + f" | {Lr.get(s, float('nan')):+.3f} |")
    verdict = None
    if all((s, c) in G for s in ("P", "ME", "MC") for c in E2C) and all(s in Lr for s in ("P", "ME", "MC")):
        sup = []
        for m in ("ME", "MC"):
            n = sum(G[(m, c)] - G[("P", c)] >= 0.10 for c in E2C)
            if n >= 2 and Lr[m] - Lr["P"] <= 0.05:
                sup.append(m)
        against = sum(G[("P", c)] - max(G[("ME", c)], G[("MC", c)]) >= 0.10 for c in E2C) >= 2
        verdict = f"가설 지지 ({', '.join(sup)})" if sup else ("가설 반대" if against else "중간")
        L += ["", f"**판정: {verdict}**"]
        if all(("PN", c) in G for c in E2C):
            L.append("참고 PN − P: " + ", ".join(f"{c} {G[('PN', c)] - G[('P', c)]:+.3f}" for c in E2C))
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"verdict": verdict, "G": {f"{k[0]}|{k[1]}": v for k, v in G.items()}, "L": Lr},
              open(a.md[:-3] + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
