"""Design doc 33 (docs/design33_two_model_ko.md §5, row "both fail"): where does the gain that the two systems lose on the
own condition sit - in the policy-start phase, right after the handover to the planner, or mid-walk?  The split itself
(three windows) was not fixed before the run; it is the descriptive reading that the decision table asks for.

    python3 scripts/sim2sim/stage33_timing.py --d results/stage33 --ref results/stage29 --d32 results/stage32

Windows by steps since reset (reset phase = steps 0-24, policy start = steps 25-149):
  start    = steps < 150          (reset phase + policy start)
  handover = steps 150-299        (first 150 planner steps after the policy start)
  mid      = steps >= 300
Per arm: falls per 1000 steps at risk in each window (hazard), and the additive share of the total falls per 1000 steps
that each window carries (falls in the window / all steps x 1000; the three shares sum to the total).
"Lost gain" of a system = system - mean of the three dynamics-only seeds, split into the three additive shares, with a
paired bootstrap over robots (5000) on the seed means.
"""
import argparse, os
import numpy as np

RNG = np.random.default_rng(33); NB = 5000
WIN = (("start", 0, 150), ("handover", 150, 300), ("mid", 300, 10**9)); WIN_KO = {"start": "출발 (정책 구간)", "handover": "넘겨받은 직후", "mid": "걷는 중"}
CELLS = [("P", "ICE", "ice", "논문·빙판"), ("P", "MOT", "mot", "논문·모터 40 %"), ("B", "ICE", "ice", "넓은 DR·빙판"), ("B", "MOT", "mot", "넓은 DR·모터 40 % (이득 칸 아님)")]


def load(d, name):
    f = os.path.join(d, f"diag_{name}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True); s = z["since"].astype(np.int64); fl = z["fall"].astype(bool); ok = ~z["resetting"].astype(bool)
    x = {"T": float(s.shape[0])}
    for k, lo, hi in WIN:
        w = (s >= lo) & (s < hi); x["f_" + k] = (fl & w).sum(0).astype(float); x["n_" + k] = (ok & w).sum(0).astype(float)
    return x


def haz(x, k, i=slice(None)):
    return 1000.0 * x["f_" + k][i].sum() / max(1.0, x["n_" + k][i].sum())


def share(x, k, i=slice(None)):
    return 1000.0 * x["f_" + k][i].mean() / x["T"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="results/stage33"); ap.add_argument("--ref", default="results/stage29")
    ap.add_argument("--d32", default="results/stage32"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.d, "timing.md")
    L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731
    P("## 설계 33: 잃은 이득이 어느 구간에 있나 (구간 나눔은 실행 전에 정하지 않음)"); P()
    P("구간은 리셋 뒤 스텝 수로 나눈다: 출발 = 150스텝 전(리셋 25 + 정책 출발 125), 넘겨받은 직후 = 150~299, 걷는 중 = 300 이후."); P()
    for m, ad, c, ko in CELLS:
        pre = load(a.ref, f"{m}_PRE_{c}"); dyn = [x for x in [load(a.ref, f"{m}_{ad}_{c}")] + [load(a.d, f"{m}_{ad}d{s}_{c}") for s in (43, 44)] if x is not None]
        sysT = [x for x in (load(a.d, f"{m}_{ad}T{s}_{c}") for s in (42, 43, 44)) if x is not None]
        sysR = [x for x in (load(a.d, f"{m}_{ad}R{s}_{c}") for s in (42, 43, 44)) if x is not None]
        arms = [("사전학습", [pre]), ("사전학습 + 정책 출발", [load(a.d, f"{m}_PREps_{c}")]), ("정책만", [load(a.d, f"{m}_POL_{c}")]),
                ("동역학 fine-tuning (시드 3개)", dyn), ("replay 한 모델 42, 정책 출발 없음 (설계 32)", [load(a.d32, f"{m}_{ad}n1_{c}")]),
                ("두 벌 + 정책 출발 + 오차 전환 (시드 3개)", sysT), ("replay 한 모델 + 정책 출발 (시드 3개)", sysR)]
        P(f"### {ko}, 자기 조건"); P()
        P("구간 안 1000스텝당 넘어짐 (시드가 여럿이면 시드 42 / 43 / 44)"); P()
        P("| 팔 | " + " | ".join(WIN_KO[k] for k, _, _ in WIN) + " |"); P("|---|---|---|---|")
        for name, xs in arms:
            xs = [x for x in xs if x is not None]
            if xs:
                P(f"| {name} | " + " | ".join(" / ".join(f"{haz(x, k):.2f}" for x in xs) for k, _, _ in WIN) + " |")
        P()
        P("전체 넘어짐(1000스텝당)에서 각 구간이 차지하는 몫 (세 몫의 합 = 전체, 시드 평균)"); P()
        P("| 팔 | " + " | ".join(WIN_KO[k] for k, _, _ in WIN) + " | 전체 |"); P("|---|---|---|---|---|")
        for name, xs in arms:
            xs = [x for x in xs if x is not None]
            if xs:
                v = [np.mean([share(x, k) for x in xs]) for k, _, _ in WIN]
                P(f"| {name} | " + " | ".join(f"{q:.2f}" for q in v) + f" | {sum(v):.2f} |")
        P()
        if dyn and (sysT or sysR):
            P("잃은 이득 = 시스템 − 동역학 fine-tuning (시드 평균끼리, 1000스텝당 넘어짐의 몫, [95 % CI])"); P()
            P("| 시스템 | " + " | ".join(WIN_KO[k] for k, _, _ in WIN) + " | 전체 |"); P("|---|---|---|---|---|")
            n = len(pre["f_mid"]); idx = RNG.integers(0, n, (NB, n))
            for name, xs in (("두 벌 + 정책 출발 + 오차 전환", sysT), ("replay 한 모델 + 정책 출발", sysR)):
                if not xs:
                    continue
                cs = []
                for ks in [(k,) for k, _, _ in WIN] + [tuple(k for k, _, _ in WIN)]:
                    f = lambda i: sum(np.mean([share(x, k, i) for x in xs]) - np.mean([share(x, k, i) for x in dyn]) for k in ks)  # noqa: E731
                    b = np.array([f(i) for i in idx]); cs.append(f"{f(slice(None)):+.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]")
                P(f"| {name} | " + " | ".join(cs) + " |")
            P()
    open(out, "w").write("\n".join(L) + "\n"); print("written", out)


if __name__ == "__main__":
    main()
