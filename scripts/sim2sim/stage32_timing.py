"""Design doc 32 part A, descriptive look that was NOT registered (docs/design32_recipe_ko.md §8.5): where does the gain
that the replay arms lose on the own condition sit - startup or mid-walk, and at which command speed?

    python3 scripts/sim2sim/stage32_timing.py --d results/stage32 --d31 results/stage31 --ref results/stage29

Per arm on the own condition (robots resampled jointly, paired bootstrap, 5000):
  startup  = probability that an episode falls within its first 125 steps
  mid-walk = falls per 1000 steps at risk with steps-since-reset >= 300
  kept     = (PRE - arm) / (PRE - DYN) of each of the two numbers
and the falls per 1000 steps of each command speed (6 robots each).
"""
import argparse, os
import numpy as np

RNG = np.random.default_rng(32); NB = 5000
CELLS = [("P", "ICE", "ice", "논문·빙판"), ("P", "MOT", "mot", "논문·모터 40 %"), ("B", "ICE", "ice", "넓은 DR·빙판")]


def load(d, name):
    f = os.path.join(d, f"diag_{name}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True); s = z["since"].astype(np.int64); fl = z["fall"].astype(bool); ok = ~z["resetting"].astype(bool)
    return {"ep": ((s == 124) | (fl & (s < 125))).sum(0).astype(float), "f_start": (fl & (s < 125)).sum(0).astype(float),
            "mid": (ok & (s >= 300)).sum(0).astype(float), "f_mid": (fl & (s >= 300)).sum(0).astype(float),
            "falls": fl.sum(0) / s.shape[0] * 1000.0, "speed": np.round(z["speeds"].astype(float), 1)}


def stat(x, kind, i=slice(None)):
    return x["f_start"][i].sum() / max(1.0, x["ep"][i].sum()) if kind == "start" else 1000.0 * x["f_mid"][i].sum() / max(1.0, x["mid"][i].sum())


def kept(pre, dyn, arm, kind):
    def one(i):
        p, q, t = stat(pre, kind, i), stat(dyn, kind, i), stat(arm, kind, i)
        return (p - t) / (p - q) if p - q > 0 else np.nan
    n = len(pre["ep"]); b = np.array([one(i) for i in RNG.integers(0, n, (NB, n))])
    return one(slice(None)), np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="results/stage32"); ap.add_argument("--d31", default="results/stage31")
    ap.add_argument("--ref", default="results/stage29"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.d, "timing.md")
    L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    P("## 설계 32-A (등록 안 함): 자기 조건에서 잃은 이득이 출발에 있나 걷는 중에 있나"); P()
    for m, ad, c, ko in CELLS:
        pre = load(a.ref, f"{m}_PRE_{c}"); dyn = load(a.ref, f"{m}_{ad}_{c}")
        arms = [("사전학습", pre), ("동역학 fine-tuning", dyn), ("토큰 모델 시드 42", load(a.d31, f"{m}_{ad}tA_{c}")),
                ("토큰 모델 시드 43", load(a.d, f"{m}_{ad}t43A_{c}")), ("토큰 모델 시드 44", load(a.d, f"{m}_{ad}t44A_{c}")),
                ("토큰 없음 + replay 둘 다", load(a.d, f"{m}_{ad}n2_{c}")), ("토큰 + 사전학습 창만", load(a.d, f"{m}_{ad}t1A_{c}")),
                ("토큰 없음 + 사전학습 창만", load(a.d, f"{m}_{ad}n1_{c}"))]
        P(f"### {ko} (자기 조건)"); P()
        P("| 팔 | 출발에서 넘어질 확률 | 출발 이득 유지 [95 % CI] | 걷는 중 1000스텝당 넘어짐 | 걷는 중 이득 유지 [95 % CI] | 속도 0.3 / 0.6 / 0.9의 넘어짐 |"); P("|---|---|---|---|---|---|")
        for name, x in arms:
            if x is None:
                continue
            ks = ["—", "—"]
            if name not in ("사전학습", "동역학 fine-tuning"):
                ks = ["{:.2f} [{:.2f}, {:.2f}]".format(*kept(pre, dyn, x, k)) if stat(pre, k) - stat(dyn, k) > 0 else "해당 없음 (이득 없음)" for k in ("start", "mid")]
            sp = " / ".join(f"{x['falls'][x['speed'] == v].mean():.2f}" for v in (0.3, 0.6, 0.9))
            P(f"| {name} | {stat(x, 'start'):.2f} | {ks[0]} | {stat(x, 'mid'):.2f} | {ks[1]} | {sp} |")
        P()
    open(out, "w").write("\n".join(L) + "\n"); print("written", out)


if __name__ == "__main__":
    main()
