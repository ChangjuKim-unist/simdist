"""Design doc 27 part M (docs/design27_recovery_quality_ko.md §5.1): motion-quality profile from diag npz files.

    python3 scripts/sim2sim/stage27_metrics.py --d26 results/stage26 --d27 results/stage27 [--mid 300] [--out results/stage27/metrics.md]

Per robot (then mean over the 18 robots, 95 % bootstrap CI over robots; ratios with the robots resampled jointly by seed),
over the mid-walk phase (released steps, steps since reset >= --mid) unless a table says otherwise:

  safety   falls_1k (all steps), slip mean of the feet in contact
  task     speed_err |vx - 0.9|, lat_speed |vy|, yaw_rate |wz|
  smooth   act_rate |a_t - a_t-1|, act_jerk |a_t - 2 a_t-1 + a_t-2| (12-d action, consecutive released steps of one
           episode), hf_ratio = share of the action power at 10-25 Hz (128-step Hann windows inside contiguous stretches,
           mean removed, DC excluded, mean over joints and windows), joint_acc (reward term dof_acc_l2, sign flipped;
           direct sum qdd^2 when joint velocities are logged)
  effort   torque_sq (reward term dof_torques_l2, sign flipped; direct sum tau^2 when torques are logged), power
           sum |tau * qd| [W], cot = mean power / (mass * 9.81 * mean vx) (logged runs only)
  body     rp_rate RMS |w_xy|, vz RMS, tilt mean |g_xy|, height_sd
  gait     contact runs per foot; a run of <= 2 steps is a short contact (short_cps: per second, all feet); touchdown =
           start of a run of >= 3 steps; stride = interval between touchdowns of one foot (<= 100 steps) inside a contiguous
           stretch: stride_T [s], stride_cv (SD / mean over the robot's strides); duty = contact fraction; diag_sync = share
           of steps with FL = RR and FR = RL contact states; gait_term = reward term "gait"
  foot     impact = mean peak contact force in the 3 steps from a touchdown [N] (logged runs only)

Reward-term metrics carry the term weight, so only their ratios are reported.
"""
import argparse
import os

import numpy as np

DT = 0.02
RNG = np.random.default_rng(0)
NBOOT = 5000
TARGET_V = 0.9

METRICS = [  # name, Korean label, better direction (-1 lower, +1 higher, 0 = closer to reference)
    ("falls_1k", "넘어짐/1000스텝", -1), ("slip", "접지 발 미끄럼 m/s", -1),
    ("speed_err", "전진 속도 오차 |vx−0.9| m/s", -1), ("lat_speed", "옆 속도 |vy| m/s", -1), ("yaw_rate", "회전 속도 |wz| rad/s", -1),
    ("act_rate", "행동 변화 |Δa|", -1), ("act_jerk", "행동 2차 차분 |Δ²a|", -1), ("hf_ratio", "행동 고주파(10–25 Hz) 비율", -1),
    ("joint_acc", "관절 가속도²(항)", -1), ("joint_acc_d", "관절 가속도² (rad/s²)²", -1),
    ("torque_sq", "토크²(항)", -1), ("torque_sq_d", "토크² N²m²", -1), ("power", "기계적 일률 W", -1), ("cot", "이동 비용 CoT", -1),
    ("rp_rate", "몸통 롤·피치 속도 RMS rad/s", -1), ("vz", "몸통 상하 속도 RMS m/s", -1), ("tilt", "몸통 기울기 |g_xy|", -1),
    ("height_sd", "몸통 높이 SD m", -1),
    ("stride_T", "보폭 주기 s (기준 0.5)", 0), ("stride_cv", "보폭 주기 변동계수", -1), ("duty", "접지 비율 (기준 0.5)", 0),
    ("diag_sync", "대각 발 동기율", +1), ("short_cps", "짧은 접지(≤40 ms)/s", -1), ("gait_term", "걸음 보상항(항)", +1),
    ("impact", "착지 충격 N", -1),
]
LABEL = {m: k for m, k, _ in METRICS}
BETTER = {m: b for m, _, b in METRICS}


def load(path):
    d = dict(np.load(path, allow_pickle=True))
    a = {"name": os.path.basename(path)[5:-4]}
    a["since"] = d["since"].astype(np.int64); a["valid"] = ~d["resetting"].astype(bool)
    a["S"], a["N"] = a["since"].shape
    for k in ("act", "lin_vel", "ang_vel", "grav", "height", "foot_speed", "terms"):
        a[k] = d[k].astype(np.float32)
    a["contact"] = d["contact"].astype(bool); a["fall"] = d["fall"].astype(bool)
    a["term_names"] = [str(x) for x in d["term_names"]]; a["foot_names"] = [str(x) for x in d["foot_names"]]
    a["seeds"] = d["seeds"]
    for k in ("joint_vel", "torque", "foot_force"):
        a[k] = d[k].astype(np.float32) if k in d else None
    a["mass"] = d["robot_mass"].astype(np.float64) if "robot_mass" in d else None
    # consecutive: step t continues step t-1 of the same episode (both released)
    cons = np.zeros_like(a["valid"]); cons[1:] = a["valid"][1:] & a["valid"][:-1] & (a["since"][1:] == a["since"][:-1] + 1)
    a["cons"] = cons
    return a


def phase(a, ph, mid):
    s, v = a["since"], a["valid"]
    if ph == "mid":
        return v & (s >= mid)
    if ph == "startup":
        return v & (s >= 25) & (s <= 99)
    raise ValueError(ph)


def segments(m_col, cons_col):
    """contiguous stretches (start, end exclusive) of mask m where consecutive steps continue each other."""
    segs = []; t = 0; S = len(m_col)
    while t < S:
        if not m_col[t]:
            t += 1; continue
        u = t + 1
        while u < S and m_col[u] and cons_col[u]:
            u += 1
        segs.append((t, u)); t = u
    return segs


def term(a, name):
    return a["terms"][..., a["term_names"].index(name)] if name in a["term_names"] else None


def robot_metrics(a, m):
    """dict metric -> (N,) per-robot value over mask m (S, N)."""
    N = a["N"]; out = {k: np.full(N, np.nan) for k, _, _ in METRICS}
    c = a["contact"]; fn = a["foot_names"]
    pair = (fn.index("FL_foot"), fn.index("RR_foot"), fn.index("FR_foot"), fn.index("RL_foot"))
    freqs = np.fft.rfftfreq(128, d=DT); hf = freqs >= 10.0; hann = np.hanning(128)[:, None]
    gt, at = term(a, "gait"), term(a, "dof_acc_l2"); tt = term(a, "dof_torques_l2")
    for i in range(N):
        mi = m[:, i]
        if mi.sum() < 10:
            continue
        out["falls_1k"][i] = a["fall"][:, i].sum() / a["S"] * 1000.0
        ci = c[:, i].astype(np.float32)
        slip = (a["foot_speed"][:, i] * ci).sum(-1) / np.maximum(ci.sum(-1), 1.0)
        out["slip"][i] = slip[mi].mean()
        lv, av = a["lin_vel"][:, i], a["ang_vel"][:, i]
        out["speed_err"][i] = np.abs(lv[mi, 0] - TARGET_V).mean(); out["lat_speed"][i] = np.abs(lv[mi, 1]).mean()
        out["yaw_rate"][i] = np.abs(av[mi, 2]).mean()
        out["rp_rate"][i] = np.sqrt((av[mi, 0] ** 2 + av[mi, 1] ** 2).mean()); out["vz"][i] = np.sqrt((lv[mi, 2] ** 2).mean())
        out["tilt"][i] = np.linalg.norm(a["grav"][mi, i, :2], axis=-1).mean(); out["height_sd"][i] = a["height"][mi, i].std()
        act = a["act"][:, i]; cons = a["cons"][:, i]
        m1 = mi.copy(); m1[0] = False; m1 &= cons
        if m1.any():
            out["act_rate"][i] = np.linalg.norm(act[1:][m1[1:]] - act[:-1][m1[1:]], axis=-1).mean()
        m2 = m1.copy(); m2[1:] &= m1[:-1]; m2[:2] = False
        if m2.any():
            idx = np.where(m2)[0]
            out["act_jerk"][i] = np.linalg.norm(act[idx] - 2 * act[idx - 1] + act[idx - 2], axis=-1).mean()
        segs = segments(mi, cons)
        hfs = []
        for s0, s1 in segs:
            for w in range(s0, s1 - 127, 128):
                x = act[w:w + 128] - act[w:w + 128].mean(0)
                p = np.abs(np.fft.rfft(x * hann, axis=0)) ** 2
                tot = p[1:].sum(0)
                ok = tot > 0
                if ok.any():
                    hfs.append((p[hf].sum(0)[ok] / tot[ok]).mean())
        if hfs:
            out["hf_ratio"][i] = float(np.mean(hfs))
        if at is not None:
            out["joint_acc"][i] = -at[mi, i].mean()
        if tt is not None:
            out["torque_sq"][i] = -tt[mi, i].mean()
        if gt is not None:
            out["gait_term"][i] = gt[mi, i].mean()
        if a["joint_vel"] is not None:
            qd = a["joint_vel"][:, i]
            if m1.any():
                out["joint_acc_d"][i] = (((qd[1:][m1[1:]] - qd[:-1][m1[1:]]) / DT) ** 2).sum(-1).mean()
            tau = a["torque"][:, i]
            out["torque_sq_d"][i] = (tau[mi] ** 2).sum(-1).mean()
            pw = np.abs(tau * qd).sum(-1)
            out["power"][i] = pw[mi].mean()
            vbar = lv[mi, 0].mean()
            if a["mass"] is not None and vbar > 0.05:
                out["cot"][i] = pw[mi].mean() / (a["mass"][i] * 9.81 * vbar)
        # gait
        out["duty"][i] = c[mi, i].mean()
        out["diag_sync"][i] = ((c[mi, i, pair[0]] == c[mi, i, pair[1]]) & (c[mi, i, pair[2]] == c[mi, i, pair[3]])).mean()
        strides, n_short, imp = [], 0, []
        for s0, s1 in segs:
            for f in range(4):
                x = c[s0:s1, i, f].astype(np.int8)
                if len(x) < 3:
                    continue
                d = np.diff(np.concatenate([[0], x, [0]]))
                st, en = np.where(d == 1)[0], np.where(d == -1)[0]
                # drop runs cut by the stretch edges
                runs = [(u, v) for u, v in zip(st, en) if u > 0 and v < len(x)]
                n_short += sum(1 for u, v in runs if v - u <= 2)
                td = [u for u, v in runs if v - u >= 3]
                for k in range(1, len(td)):
                    if td[k] - td[k - 1] <= 100:
                        strides.append(td[k] - td[k - 1])
                if a["foot_force"] is not None:
                    ff = a["foot_force"][s0:s1, i, f]
                    imp += [ff[u:u + 3].max() for u in td]
        out["short_cps"][i] = n_short / (mi.sum() * DT)
        if len(strides) >= 5:
            st = np.asarray(strides, np.float64) * DT
            out["stride_T"][i] = st.mean(); out["stride_cv"][i] = st.std() / st.mean()
        if imp:
            out["impact"][i] = float(np.mean(imp))
    return out


def boot_mean(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = x[RNG.integers(0, len(x), (NBOOT, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def boot_ratio(x, y):
    """mean(y) / mean(x) with robots resampled jointly (same seed order)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.mean() == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), (NBOOT, len(x)))
    den = x[idx].mean(1)
    b = np.where(den > 0, y[idx].mean(1) / np.where(den > 0, den, 1.0), np.nan)
    return y.mean() / x.mean(), np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def fmt(v, nd=3):
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d26", default="results/stage26"); ap.add_argument("--d27", default="results/stage27")
    ap.add_argument("--mid", type=int, default=300); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = {"D1": "diag_D1_paper_plan_off.npz", "D3": "diag_D3_paperFR_plan_off.npz", "D4": "diag_D4_paper_pol_off.npz",
             "D5": "diag_D5_paper_plan_r50.npz", "D6": "diag_D6_broad_plan_r50.npz", "D7": "diag_D7_broad_pol_r50.npz",
             "D8": "diag_D8_paper_pol_r50.npz"}
    paths = {k: os.path.join(args.d26, v) for k, v in files.items()}
    for k in ("AB_P", "AB_pi", "AP_P", "AP_pi", "NB_P", "NB_pi", "NP_P", "NP_pi"):
        paths[k] = os.path.join(args.d27, f"diag_{k}.npz")
    paths = {k: p for k, p in paths.items() if os.path.exists(p)}
    print("arms:", sorted(paths))
    arms = {k: load(p) for k, p in paths.items()}
    M = {}  # (arm, phase) -> metrics
    for k, a in arms.items():
        M[(k, "mid")] = robot_metrics(a, phase(a, "mid", args.mid))
        if k in ("D1", "D3", "D4"):
            M[(k, "startup")] = robot_metrics(a, phase(a, "startup", args.mid))
    lines = []; P = lambda s="": (print(s), lines.append(s))  # noqa: E731
    res = {"pred": {}}

    def table(title, A, B, ph="mid", note=""):
        if (A, ph) not in M or (B, ph) not in M:
            P(f"\n### {title}\n\n(팔 없음: {A} 또는 {B})"); return None
        P(f"\n### {title}\n")
        if note:
            P(note + "\n")
        P(f"| 지표 | {A} | {B} | {B}/{A} [95 % CI] | 좋은 쪽 |")
        P("|---|---|---|---|---|")
        out = {}
        for m, lab, better in METRICS:
            x, y = M[(A, ph)][m], M[(B, ph)][m]
            if not (np.isfinite(x).any() and np.isfinite(y).any()):
                continue
            r = boot_ratio(x, y); out[m] = r
            mx, my = np.nanmean(x), np.nanmean(y)
            arrow = {-1: "낮을수록", +1: "높을수록", 0: "기준에 가까울수록"}[better]
            P(f"| {lab} | {fmt(mx)} | {fmt(my)} | {fmt(r[0], 2)} [{fmt(r[1], 2)}, {fmt(r[2], 2)}] | {arrow} |")
        return out

    P("## 움직임 질 지표 (설계 27 §5.1)\n")
    P(f"걷는 중 = 리셋 후 {args.mid}스텝 이후. 비는 18대 로봇 단위 짝 부트스트랩. '(항)'은 보상 항 값(가중치 포함)이라 비만 본다.")
    T = {}
    T["M1"] = table("M1. 논문 모델, 빙판, 램프: 정책(D8) → 계획기(D5)", "D8", "D5")
    T["M2"] = table("M2. 넓은 DR, 빙판, 램프: 정책(D7) → 계획기(D6)", "D7", "D6")
    T["M3a"] = table("M3a. 재현(논문 모델, 빙판): 정책(AP_pi) → 계획기(AP_P)", "AP_pi", "AP_P")
    T["M3b"] = table("M3b. 재현(넓은 DR, 빙판): 정책(AB_pi) → 계획기(AB_P)", "AB_pi", "AB_P")
    T["M4a"] = table("M4a. 넓은 DR 정책: 보통 바닥(NB_pi) → 빙판(AB_pi)", "NB_pi", "AB_pi")
    T["M4b"] = table("M4b. 넓은 DR 계획기: 보통 바닥(NB_P) → 빙판(AB_P)", "NB_P", "AB_P")
    T["M4c"] = table("M4c. 논문 모델 정책: 보통 바닥(NP_pi) → 빙판(AP_pi)", "NP_pi", "AP_pi")
    T["M4d"] = table("M4d. 논문 모델 계획기: 보통 바닥(NP_P) → 빙판(AP_P)", "NP_P", "AP_P")
    T["M4e"] = table("M4e. 보통 바닥: 넓은 DR 정책(NB_pi) → 계획기(NB_P)", "NB_pi", "NB_P")
    T["M4f"] = table("M4f. 보통 바닥: 논문 모델 정책(NP_pi) → 계획기(NP_P)", "NP_pi", "NP_P")
    T["M5a"] = table("M5a. 빙판 계획기: 논문 모델(D5) → 넓은 DR(D6)", "D5", "D6")
    T["M5b"] = table("M5b. 빙판 정책: 논문 모델(D8) → 넓은 DR(D7)", "D8", "D7")
    T["M6a"] = table("M6a. 출발(s 25–99), 기존 출발: 논문 정책(D4) → 논문 계획기(D1)", "D4", "D1", ph="startup",
                     note="출발 구간은 75스텝이라 고주파 비율(128스텝 창)은 없다.")
    T["M6b"] = table("M6b. 출발(s 25–99), 기존 출발: 논문 계획기(D1) → 마찰 적응 계획기(D3)", "D1", "D3", ph="startup")

    P("\n## 예측 판정 (§6.1)\n")

    def r_of(t, m):
        return T.get(t, {}).get(m, (np.nan, np.nan, np.nan)) if T.get(t) else (np.nan, np.nan, np.nan)
    pred = res["pred"]
    for key, m, thr, desc in (("PM1", "act_jerk", 1.2, "행동 2차 차분"), ("PM2", "hf_ratio", 1.2, "행동 고주파 비율"),
                              ("PM3", "torque_sq", 1.1, "토크²(항)"), ("PM4", "stride_cv", 1.2, "보폭 주기 변동계수")):
        r1, r2 = r_of("M1", m), r_of("M2", m)
        ok = bool(np.isfinite(r1[0]) and np.isfinite(r2[0]) and r1[0] >= thr and r2[0] >= thr)
        pred[key] = {"paper": r1, "broad": r2, "holds": ok}
        P(f"{key} {desc} 계획기/정책: 논문 {fmt(r1[0], 2)} [{fmt(r1[1], 2)}, {fmt(r1[2], 2)}], 넓은 DR {fmt(r2[0], 2)} [{fmt(r2[1], 2)}, {fmt(r2[2], 2)}] (기준 둘 다 ≥ {thr}) → {'성립' if ok else '기각'}")
    r = r_of("M5a", "act_jerk"); ok = bool(np.isfinite(r[0]) and r[0] <= 1.1); pred["PM5"] = {"value": r, "holds": ok}
    P(f"PM5 행동 2차 차분 넓은 DR 계획기/논문 계획기 = {fmt(r[0], 2)} [{fmt(r[1], 2)}, {fmt(r[2], 2)}] (기준 ≤ 1.1) → {'성립' if ok else '기각'}")
    s_b, s_p = r_of("M4a", "slip"), r_of("M4c", "slip"); c_b, c_p = r_of("M4a", "stride_cv"), r_of("M4c", "stride_cv")
    vals = [s_b[0], s_p[0], c_b[0], c_p[0]]
    ok = bool(all(np.isfinite(vals)) and s_b[0] >= 3 and s_p[0] >= 3 and c_b[0] >= 1.3 and c_p[0] >= 1.3)
    pred["PM6"] = {"slip_broad": s_b, "slip_paper": s_p, "cv_broad": c_b, "cv_paper": c_p, "holds": ok}
    P(f"PM6 정책 빙판/보통 바닥: 미끄럼 넓은 DR {fmt(s_b[0], 2)}, 논문 {fmt(s_p[0], 2)} (≥ 3); 보폭 변동계수 넓은 DR {fmt(c_b[0], 2)}, 논문 {fmt(c_p[0], 2)} (≥ 1.3) → {'성립' if ok else '기각'}")

    out = args.out or os.path.join(args.d27, "metrics.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    import json

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o
    per_arm = {f"{k}|{ph}": {m: clean(boot_mean(v)) for m, v in mm.items()} for (k, ph), mm in M.items()}
    json.dump(clean({"pred": pred, "tables": T, "per_arm": per_arm}), open(out[:-3] + ".json", "w"), indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
