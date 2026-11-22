"""Design 40 §9: pick one level per family by the registered rule."""
import argparse, json, os
import numpy as np

FAM = {"motor": ["mot40", "mot30", "mot25"], "leg": ["legR30", "legR20", "legR10"],
       "delay": ["delay1", "delay2", "delay3"], "tow": ["tow20", "tow30", "tow40"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d"); ap.add_argument("--md"); a = ap.parse_args()
    nom = json.load(open(os.path.join(a.d, "e0_ppo_nominal.json")))
    Jn = float(np.mean(nom["J"]))
    L = ["# 설계 40 §9 수준 선별 (PPO, 16개, 자동 생성)", "", f"nominal J {Jn:.3f}", "",
         "| 계열 | 수준 | J | J 하락 | 넘어짐 | 전진 m/s |", "|---|---|---|---|---|---|"]
    pick = {}
    for fam, levels in FAM.items():
        rows = []
        for c in levels:
            js = json.load(open(os.path.join(a.d, f"screen_ppo_{c}.json")))
            J = float(np.mean(js["J"])); fl = int(np.sum(js["fell"]))
            rows.append((c, J, Jn - J, fl))
            L.append(f"| {fam} | {c} | {J:.3f} | {Jn - J:.3f} | {fl} | {np.nanmean(js['vx_mean']):.2f} |")
        ok = [r for r in rows if r[3] <= 2 and r[2] >= 0.05]
        if ok:
            pick[fam] = (ok[-1][0], "rule 1")          # levels are ordered weak -> strong
        else:
            ok2 = [r for r in rows if r[3] <= 2]
            pick[fam] = (max(ok2, key=lambda r: r[2])[0], "rule 2") if ok2 else (levels[0], "rule 3")
    L += ["", "## 고른 수준", ""] + [f"- {f}: {p} ({r})" for f, (p, r) in pick.items()]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(pick, open(a.md[:-3] + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
