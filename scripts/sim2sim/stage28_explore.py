"""Design 28 exploratory (not registered) checks, written after the registered analysis:
  E1 reward terms: which reward terms give planner arms their realized-reward edge over the policy (broad / paper floor)
  E2 deviation split: d^2 = s + n/k fitted on base and S4 (k = 1, 4), checked against Sq (k = 1/4) -> systematic part
  E4 command tracking: the command is a path-following P-controller (x_des moving at 0.9 m/s, vy = ky*(-dy),
     wz = kyaw*(-heading error)); per arm mean command and tracking error |v - cmd| on the normal floor
  E5 path offset: with kx = ky = kyaw = 1 the mean command gives the mean heading error (-wz cmd, rad) and lateral offset
     (-vy cmd, m) from the straight path; sign consistency over the 18 robots (negative = turned / displaced to the right)
  E3 ice falls by arm: startup (< 300 steps since reset) vs mid-walk, and mid-walk falls preceded by a slip event (100 steps)
    python3 scripts/sim2sim/stage28_explore.py --d27 results/stage27 --d28 results/stage28
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage28_analyze as A  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--d27", default="results/stage27"); ap.add_argument("--d28", default="results/stage28")
ap.add_argument("--out", default=None); args = ap.parse_args()
out = args.out or os.path.join(args.d28, "explore.md")
L = []; P = lambda s="": (print(s), L.append(s))  # noqa: E731

def files(pre, pol, base):
    return {"pol": [os.path.join(args.d27, pol)], "base": [os.path.join(args.d27, base)],
            "Sq": [os.path.join(args.d28, f"diag_{pre}_Sq.npz")],
            "S4": sorted(glob.glob(os.path.join(args.d28, f"diag_{pre}_S4_p*.npz"))),
            "V0": [os.path.join(args.d28, f"diag_{pre}_V0.npz")]}
SET = {"B_floor": (2000, "diag_NB_pi.npz", "diag_NB_P.npz"), "P_floor": (2000, "diag_NP_pi.npz", "diag_NP_P.npz"),
       "B_ice": (None, "diag_AB_pi.npz", "diag_AB_P.npz")}
ARMS = ["pol", "Sq", "base", "S4", "V0"]
D = {}
for st, (trim, fp, fb) in SET.items():
    for arm, fl in files(st, fp, fb).items():
        D[(st, arm)] = A.raw_load(fl, trim)

P("## 설계 28 보조 분석 (등록 안 함)")
# E1
for st in ("B_floor", "P_floor"):
    P(f"\n### E1 {st}: 실제 보상 항목별 스텝당 평균 (걷는 중), 정책과의 차 ×1e4")
    names = [str(x) for x in D[(st, "pol")]["term_names"]]
    vals = {}
    for arm in ARMS:
        d = D[(st, arm)]; m = (~d["resetting"].astype(bool)) & (d["since"] >= 300)
        t = d["terms"].astype(np.float64)
        vals[arm] = np.array([t[..., j][m].mean() for j in range(len(names))]); vals[arm + "_tot"] = d["reward"][m].mean()
    P("| 항목 | pol | " + " | ".join(f"{a}−pol" for a in ARMS[1:]) + " |"); P("|---|" + "---|" * len(ARMS))
    for j, n in enumerate(names):
        P(f"| {n} | {vals['pol'][j]*1e4:.1f} | " + " | ".join(f"{(vals[a][j]-vals['pol'][j])*1e4:+.1f}" for a in ARMS[1:]) + " |")
    P(f"| 합(보상) | {vals['pol_tot']*1e4:.1f} | " + " | ".join(f"{(vals[a+'_tot']-vals['pol_tot'])*1e4:+.1f}" for a in ARMS[1:]) + " |")
# E4
for st in ("B_floor", "P_floor"):
    P(f"\n### E4 {st}: 명령 추종 (명령 = 직선 경로 추종 P 제어, 옆·회전 명령은 경로 이탈 보정)")
    P("| 팔 | 평균 명령 vx | 평균 명령 vy | 평균 명령 wz | |vx−명령| | |vy−명령| | |wz−명령| |"); P("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        d = D[(st, arm)]; m = (~d["resetting"].astype(bool)) & (d["since"] >= 300)
        c = d["cmd"].astype(np.float64); lv = d["lin_vel"].astype(np.float64); av = d["ang_vel"].astype(np.float64)
        P(f"| {arm} | {c[...,0][m].mean():.3f} | {c[...,1][m].mean():+.3f} | {c[...,2][m].mean():+.3f} | "
          f"{np.abs(lv[...,0]-c[...,0])[m].mean():.3f} | {np.abs(lv[...,1]-c[...,1])[m].mean():.3f} | {np.abs(av[...,2]-c[...,2])[m].mean():.3f} |")
# E5
P("\n### E5 경로 이탈 방향: 평균 방향 오차(도)와 옆 이탈(m), 음수 = 오른쪽, 괄호 = 음수인 로봇 수/18")
P("| 설정 | " + " | ".join(ARMS) + " |"); P("|---|" + "---|" * len(ARMS))
for st in SET:
    row = []
    for arm in ARMS:
        d = D[(st, arm)]; m = (~d["resetting"].astype(bool)) & (d["since"] >= 300); c = d["cmd"].astype(np.float64)
        hy = np.array([-c[m[:, i], i, 2].mean() for i in range(m.shape[1])]); dy = np.array([-c[m[:, i], i, 1].mean() for i in range(m.shape[1])])
        row.append(f"{np.degrees(hy.mean()):+.1f}° ({(hy < 0).sum()}), {dy.mean():+.2f} m ({(dy < 0).sum()})")
    P(f"| {st} | " + " | ".join(row) + " |")
# E2
P("\n### E2 정책 이탈 d² = s + n/k (k = 표본 배수; base k=1, S4 k=4로 맞추고 Sq k=1/4로 확인)")
P("| 설정 | d base | d S4 | d Sq 관측 | d Sq 예측 | 무한 표본 d (=√s) | base d² 중 잡음 몫 |"); P("|---|---|---|---|---|---|---|")
for st in SET:
    dv = {}
    for arm in ("base", "S4", "Sq"):
        d = D[(st, arm)]; m = (~d["resetting"].astype(bool)) & (d["since"] >= 300)
        dev = np.linalg.norm(d["act"].astype(np.float32) - d["act_pol"].astype(np.float32), axis=-1)
        dv[arm] = np.nanmean([dev[m[:, i], i].mean() for i in range(m.shape[1])])
    n = (dv["base"]**2 - dv["S4"]**2) * 4 / 3; s = dv["base"]**2 - n
    P(f"| {st} | {dv['base']:.3f} | {dv['S4']:.3f} | {dv['Sq']:.3f} | {np.sqrt(s + 4*n):.3f} | {np.sqrt(max(s,0)):.3f} | {n/dv['base']**2:.0%} |")
# E3
PER = {}
P("\n### E3 넓은 DR 빙판: 넘어짐 분해 (18대 × 5000스텝)")
P("| 팔 | 넘어짐 | 출발 중(<300) | 걷는 중 | 걷는 중 중 직전 100스텝에 미끄러짐 사건 | 사건 없이 넘어짐 |"); P("|---|---|---|---|---|---|")
for arm in ARMS:
    d = D[("B_ice", arm)]; a = A.m27_view(d, arm)
    c = a["contact"].astype(np.float32); slip = (a["foot_speed"] * c).sum(-1) / np.maximum(c.sum(-1), 1.0)
    ep = np.cumsum(a["since"] == 0, axis=0)
    tot = start = mid = mid_ev = 0; per_start = np.zeros(a["N"]); per_mid = np.zeros(a["N"])
    for i in range(a["N"]):
        ev_t = []
        for e in np.unique(ep[:, i]):
            idx = np.where((ep[:, i] == e) & a["valid"][:, i])[0]
            if len(idx) == 0: continue
            x = slip[idx, i].astype(np.float64); cs = np.cumsum(np.insert(x, 0, 0.0)); lo = np.maximum(np.arange(len(idx)) - 9, 0)
            ma = (cs[np.arange(len(idx)) + 1] - cs[lo]) / (np.arange(len(idx)) + 1 - lo)
            ev_t += [idx[s0] for s0 in A.runs_over(ma, A.THETA) if a["since"][idx[s0], i] >= 300]
        ev_t = np.array(ev_t)
        for t in np.where(a["fall"][:, i])[0]:
            tot += 1
            if a["since"][t, i] < 300: start += 1; per_start[i] += 1; continue
            mid += 1; per_mid[i] += 1; mid_ev += int(len(ev_t) and ((ev_t <= t) & (ev_t > t - 100)).any())
    P(f"| {arm} | {tot} | {start} | {mid} | {mid_ev} | {mid - mid_ev} |"); PER[arm] = (per_start, per_mid, a["seeds"])
rng = np.random.default_rng(28)
def bratio(x, y, B=5000):
    idx = rng.integers(0, len(x), (B, len(x))); r = x[idx].sum(1) / np.maximum(y[idx].sum(1), 1e-9)
    return x.sum() / y.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)
P("\n| 팔/base (로봇 단위 짝 부트스트랩) | 출발 중 넘어짐 | 걷는 중 넘어짐 |"); P("|---|---|---|")
for arm in ("pol", "Sq", "S4", "V0"):
    assert np.array_equal(PER[arm][2], PER["base"][2])
    a0 = bratio(PER[arm][0], PER["base"][0]); a1 = bratio(PER[arm][1], PER["base"][1])
    P(f"| {arm} | {a0[0]:.2f} [{a0[1]:.2f}, {a0[2]:.2f}] | {a1[0]:.2f} [{a1[1]:.2f}, {a1[2]:.2f}] |")
open(out, "w").write("\n".join(L) + "\n"); print("written", out)
