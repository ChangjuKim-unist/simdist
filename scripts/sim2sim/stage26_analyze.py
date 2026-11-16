"""Design doc 26 analysis (docs/design26_why_planner_falls_ko.md §5-§7): why the planner falls.

    python3 scripts/sim2sim/stage26_analyze.py results/stage26 [--out results/stage26/analysis.md]

Reads diag_<arm>.npz written by simulate_batch.py output.diag_npz (arrays (S, N, ...)) and prints / writes:
  - per arm and phase (startup s = 25-99, early s = 25-49, mid-walk s >= 300): deviation d = |a - a_pol|, model preference
    dJ = J(exec) - J(policy plan), optimism O = R_hat(exec) - R_real (25-step discounted, 0 after a fall), one-step latent
    error, contact slip, startup 1 s max slip, forward speed / command
  - slip events (theta = 95th percentile of the D7 mid-walk 10-step moving-average slip), events per 1k mid-walk steps,
    fall within 100 steps after an event, decomposition of mid-walk falls
  - precursors of mid-walk falls (fall at s >= 200): first feature over its control 95th percentile for 5 steps in the
    150 steps before the fall, AUC of the [-100, -25) window mean vs control windows, command surge (vx > 1.0 for 5 steps)
  - the registered predictions PA1-PA8, PB1-PB5 with robot-level bootstrap CIs
"""
import argparse
import glob
import json
import os

import numpy as np

H = 25  # model horizon / realised-return window
RNG = np.random.default_rng(0)
NBOOT = 5000


def load(path):
    d = dict(np.load(path))
    a = {}
    a["name"] = os.path.basename(path)[5:-4]
    a["since"] = d["since"].astype(np.int64)
    a["valid"] = ~d["resetting"]
    a["dev"] = np.linalg.norm(d["act"].astype(np.float32) - d["act_pol"].astype(np.float32), axis=-1)
    a["dJ"] = (d["R_exec"] + d["V_exec"]) - (d["R_pol"] + d["V_pol"])
    a["R_hat"] = d["R_exec"].astype(np.float64)
    a["lat"] = d["lat_err"].astype(np.float64)
    c = d["contact"].astype(np.float32)
    a["slip"] = (d["foot_speed"].astype(np.float32) * c).sum(-1) / np.maximum(c.sum(-1), 1.0)
    a["tilt"] = np.linalg.norm(d["grav"][..., :2].astype(np.float32), axis=-1)
    a["cmdvx"] = d["cmd"][..., 0].astype(np.float32)
    a["vx"] = d["lin_vel"][..., 0].astype(np.float32)
    a["reward"] = d["reward"].astype(np.float64)
    a["fall"] = d["fall"].astype(bool)
    a["reset"] = d["reset"].astype(bool)
    a["seeds"] = d["seeds"]
    gamma = float(d["discount"]); assert int(d["T"]) == H, int(d["T"])
    a["gamma"] = gamma
    S, N = a["since"].shape
    a["S"], a["N"] = S, N
    # episode id per (step, robot): a new episode starts where since == 0
    a["ep"] = np.cumsum(a["since"] == 0, axis=0)
    # realised 25-step discounted return (0 after a fall); NaN if a timeout reset is inside the window or the run ends
    disc = gamma ** np.arange(H)
    R = np.full((S, N), np.nan)
    for i in range(N):
        r, f, rs = a["reward"][:, i], a["fall"][:, i], a["reset"][:, i]
        for t in range(S - H + 1):
            w_r = r[t:t + H]; w_reset = rs[t:t + H]
            if w_reset.any():
                k = int(np.argmax(w_reset))
                if not f[t + k]:
                    continue  # timeout inside the window
                R[t, i] = float((w_r[:k + 1] * disc[:k + 1]).sum())
            else:
                R[t, i] = float((w_r * disc).sum())
    a["R_real"] = R
    a["O"] = a["R_hat"] - R
    # 10-step trailing moving averages of slip and tilt inside an episode, over released steps only
    for k in ("slip", "tilt"):
        ma = np.full((S, N), np.nan)
        for i in range(N):
            x = a[k][:, i]; v = a["valid"][:, i]; ep = a["ep"][:, i]
            for e in np.unique(ep):
                idx = np.where((ep == e) & v)[0]
                if len(idx) == 0:
                    continue
                cs = np.cumsum(np.insert(x[idx].astype(np.float64), 0, 0.0))
                lo = np.maximum(np.arange(len(idx)) - 9, 0)
                ma[idx, i] = (cs[np.arange(len(idx)) + 1] - cs[lo]) / (np.arange(len(idx)) + 1 - lo)
        a[k + "_ma"] = ma
    return a


def phase_mask(a, ph):
    s = a["since"]; v = a["valid"]
    if ph == "startup":
        return v & (s >= 25) & (s <= 99)
    if ph == "early":
        return v & (s >= 25) & (s <= 49)
    if ph == "mid":
        return v & (s >= 300)
    raise ValueError(ph)


def robot_means(a, key, ph):
    """(N,) per-robot mean of a[key] over the phase (NaN if the robot has no such step)."""
    m = phase_mask(a, ph); x = a[key]
    out = np.full(a["N"], np.nan)
    for i in range(a["N"]):
        xi = x[m[:, i], i]; xi = xi[np.isfinite(xi)]
        if len(xi):
            out[i] = xi.mean()
    return out


def startup_max_slip(a):
    """(N,) per-robot mean over episodes of the max 10-step MA slip in s = 25-74."""
    out = np.full(a["N"], np.nan)
    for i in range(a["N"]):
        vals = []
        for e in np.unique(a["ep"][:, i]):
            idx = np.where((a["ep"][:, i] == e) & (a["since"][:, i] >= 25) & (a["since"][:, i] <= 74) & a["valid"][:, i])[0]
            x = a["slip_ma"][idx, i]; x = x[np.isfinite(x)]
            if len(x):
                vals.append(x.max())
        if vals:
            out[i] = np.mean(vals)
    return out


def boot_mean(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = x[RNG.integers(0, len(x), (NBOOT, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def boot_ratio(x, y):
    """mean(x) / mean(y) over robots resampled jointly by seed index (both (N,), same seeds)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), (NBOOT, len(x)))
    b = x[idx].mean(1) / y[idx].mean(1)
    return x.mean() / y.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def runs_over(x, thr, k=5):
    """start indices of runs where x > thr for >= k consecutive steps."""
    above = np.nan_to_num(x, nan=-np.inf) > thr
    starts = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= k:
                starts.append(i)
            i = j
        else:
            i += 1
    return starts


def slip_events(a, theta):
    """per robot: list of (event start step) in mid-walk (start s >= 300), and per-robot mid-walk step counts."""
    ev = []; mid_steps = np.zeros(a["N"])
    m = phase_mask(a, "mid")
    for i in range(a["N"]):
        mid_steps[i] = m[:, i].sum()
        evs = []
        for e in np.unique(a["ep"][:, i]):
            idx = np.where((a["ep"][:, i] == e) & a["valid"][:, i])[0]
            if len(idx) == 0:
                continue
            for st in runs_over(a["slip_ma"][idx, i], theta):
                t = idx[st]
                if a["since"][t, i] >= 300:
                    evs.append(t)
        ev.append(evs)
    return ev, mid_steps


def control_mask(a, min_s=200, gap=200):
    """steps with s >= min_s and no fall within +-gap steps of the same robot."""
    S, N = a["S"], a["N"]
    near = np.zeros((S, N), bool)
    for i in range(N):
        for t in np.where(a["fall"][:, i])[0]:
            near[max(0, t - gap):t + gap + 1, i] = True
    return a["valid"] & (a["since"] >= min_s) & ~near


FEATS = ["dev", "dJ", "slip_ma", "cmdvx", "tilt_ma", "lat"]


def precursors(a):
    S, N = a["S"], a["N"]
    cm = control_mask(a)
    thr = {f: np.nanpercentile(a[f][cm], 95) for f in FEATS}
    falls = [(t, i) for i in range(N) for t in np.where(a["fall"][:, i])[0] if a["since"][t, i] >= 200]
    first = {f: 0 for f in FEATS + ["none"]}
    surge_fall = 0
    rows = []
    for t, i in falls:
        lo = t - 150
        cross = {}
        for f in FEATS:
            st = runs_over(a[f][lo:t, i], thr[f])
            if st:
                cross[f] = st[0]
        if cross:
            mn = min(cross.values())
            for f, v in cross.items():
                if v == mn:
                    first[f] += 1
        else:
            first["none"] += 1
        if runs_over(a["cmdvx"][t - 100:t, i], 1.0):
            surge_fall += 1
        rows.append((t, i, cross))
    # control anchors every 25 steps
    anchors = [(t, i) for i in range(N) for t in range(100, S, 25) if cm[t, i] and a["since"][t, i] >= 200 and cm[t - 100:t, i].all()]
    surge_ctrl = sum(1 for t, i in anchors if runs_over(a["cmdvx"][t - 100:t, i], 1.0))
    auc = {}
    for f in FEATS:
        xf = np.array([np.nanmean(a[f][t - 100:t - 25, i]) for t, i in falls]) if falls else np.array([])
        xc = np.array([np.nanmean(a[f][t - 100:t - 25, i]) for t, i in anchors]) if anchors else np.array([])
        xf, xc = xf[np.isfinite(xf)], xc[np.isfinite(xc)]
        if len(xf) and len(xc):
            gt = (xf[:, None] > xc[None, :]).mean() + 0.5 * (xf[:, None] == xc[None, :]).mean()
            auc[f] = (float(gt), float(xf.mean()), float(xc.mean()))
    nf = len(falls)
    return {"n_falls": nf, "n_ctrl": len(anchors), "thr": {k: float(v) for k, v in thr.items()},
            "first": {k: v / nf if nf else np.nan for k, v in first.items()}, "first_counts": first,
            "surge_fall": surge_fall / nf if nf else np.nan, "surge_ctrl": surge_ctrl / len(anchors) if anchors else np.nan,
            "auc": auc}


def falls_per_1k(a):
    return a["fall"].sum(0) / a["S"] * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    arms = {}
    for p in sorted(glob.glob(os.path.join(args.dir, "diag_*.npz"))):
        a = load(p); arms[a["name"].split("_")[0]] = a
        print("loaded", a["name"], a["S"], a["N"], flush=True)
    L = []; res = {}
    P = lambda s: (L.append(s), print(s, flush=True))  # noqa: E731
    fmt = lambda t, nd=3: f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"  # noqa: E731

    # ---- per arm / phase table -------------------------------------------------------------------------------
    P("## 팔별 요약 (로봇 18대 평균 [95 % CI])\n")
    P("| 팔 | 넘어짐/1k | 구간 | 이탈 d | ΔJ | 낙관 O | 잠재 오차 | 접지 미끄럼 | 전진 속도 | 명령 vx |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for k, a in arms.items():
        res[k] = {"falls": boot_mean(falls_per_1k(a))}
        for ph in ("early", "startup", "mid"):
            row = {key: boot_mean(robot_means(a, key, ph)) for key in ("dev", "dJ", "O", "lat", "slip", "vx", "cmdvx")}
            res[k][ph] = row
            P(f"| {a['name']} | {res[k]['falls'][0]:.2f} | {ph} | {fmt(row['dev'])} | {fmt(row['dJ'])} | {fmt(row['O'])} | "
              f"{fmt(row['lat'])} | {fmt(row['slip'])} | {row['vx'][0]:.2f} | {row['cmdvx'][0]:.2f} |")
        res[k]["startup_max_slip"] = boot_mean(startup_max_slip(a))
    P("\n출발 1초 최대 미끄럼(에피소드 평균, m/s): " + ", ".join(f"{k} {fmt(res[k]['startup_max_slip'])}" for k in arms))

    # ---- QA predictions ---------------------------------------------------------------------------------------
    P("\n## QA 예측\n")
    pred = {}
    def rm(k, key, ph): return robot_means(arms[k], key, ph)
    if {"D1", "D5"} <= set(arms):
        r = boot_ratio(rm("D1", "dev", "startup"), rm("D5", "dev", "mid")); pred["PA1"] = (r, r[0] >= 1.5)
        P(f"PA1 이탈 d D1 출발 / D5 걷는 중 = {fmt(r, 2)} (기준 ≥ 1.5) → {'성립' if r[0] >= 1.5 else '기각'}")
        dj1 = res["D1"]["startup"]["dJ"]; r = boot_ratio(rm("D1", "dJ", "startup"), rm("D5", "dJ", "mid"))
        ok = dj1[0] > 0 and (res["D5"]["mid"]["dJ"][0] <= 0 or r[0] >= 2.0)
        pred["PA2"] = ((dj1, r), ok)
        P(f"PA2 ΔJ D1 출발 = {fmt(dj1)}, D5 걷는 중 = {fmt(res['D5']['mid']['dJ'])}, 비 = {fmt(r, 2)} (기준 > 0 그리고 ≥ 2) → {'성립' if ok else '기각'}")
        r = boot_ratio(rm("D1", "O", "startup"), rm("D5", "O", "mid")); pred["PA3"] = (r, r[0] >= 2.0)
        P(f"PA3 낙관 O D1 출발 = {fmt(res['D1']['startup']['O'])}, D5 걷는 중 = {fmt(res['D5']['mid']['O'])}, 비 = {fmt(r, 2)} (기준 ≥ 2) → {'성립' if r[0] >= 2.0 else '기각'}")
    if {"D1", "D2"} <= set(arms):
        rd = boot_ratio(rm("D2", "dev", "early"), rm("D1", "dev", "early")); rj = boot_ratio(rm("D2", "dJ", "early"), rm("D1", "dJ", "early"))
        ok = 0.7 <= rd[0] <= 1.3 and 0.7 <= rj[0] <= 1.3; pred["PA4"] = ((rd, rj), ok)
        P(f"PA4 출발 직후 D2/D1: 이탈 d {fmt(rd, 2)}, ΔJ {fmt(rj, 2)} (기준 둘 다 0.7~1.3) → {'성립' if ok else '기각'}")
    if {"D1", "D3"} <= set(arms):
        r = boot_ratio(rm("D3", "dev", "startup"), rm("D1", "dev", "startup")); f3 = res["D3"]["falls"]
        ok = r[0] <= 0.7 and f3[0] <= 1.0; pred["PA5"] = ((r, f3), ok)
        P(f"PA5 이탈 d D3/D1 출발 = {fmt(r, 2)}, D3 넘어짐 {fmt(f3, 2)} (기준 ≤ 0.7, ≤ 1.0) → {'성립' if ok else '기각'}")
    if {"D1", "D4"} <= set(arms):
        r = boot_ratio(startup_max_slip(arms["D4"]), startup_max_slip(arms["D1"])); pred["PA6"] = (r, r[0] <= 0.7)
        P(f"PA6 출발 1초 최대 미끄럼 D4/D1 = {fmt(r, 2)} (기준 ≤ 0.7) → {'성립' if r[0] <= 0.7 else '기각'}")
    if {"D1", "D2", "D3"} <= set(arms):
        r2 = boot_ratio(rm("D1", "lat", "startup"), rm("D2", "lat", "startup")); r3 = boot_ratio(rm("D1", "lat", "startup"), rm("D3", "lat", "startup"))
        ok = r2[0] >= 1.5 and r3[0] >= 1.5; pred["PA7"] = ((r2, r3), ok)
        P(f"PA7 잠재 오차 출발 D1/D2 = {fmt(r2, 2)}, D1/D3 = {fmt(r3, 2)} (기준 둘 다 ≥ 1.5) → {'성립' if ok else '기각'}")
    if {"D1", "D9"} <= set(arms):
        rd = boot_ratio(rm("D9", "dev", "startup"), rm("D1", "dev", "startup")); rj = boot_ratio(rm("D9", "dJ", "startup"), rm("D1", "dJ", "startup"))
        ok = rd[0] >= 1.2 and res["D1"]["startup"]["dJ"][0] > 0 and rj[0] >= 1.2; pred["PA8"] = ((rd, rj), ok)
        P(f"PA8 출발 D9/D1: 이탈 d {fmt(rd, 2)}, ΔJ {fmt(rj, 2)} (기준 둘 다 ≥ 1.2, D1 ΔJ > 0) → {'성립' if ok else '기각'}")

    # ---- QB: slip events, decomposition, precursors ----------------------------------------------------------------
    P("\n## QB 미끄러짐 사건과 분해\n")
    if "D7" in arms:
        theta = float(np.nanpercentile(arms["D7"]["slip_ma"][phase_mask(arms["D7"], "mid")], 95))
        P(f"θ (D7 걷는 중 10스텝 이동평균 미끄럼 95 백분위) = {theta:.3f} m/s\n")
        P("| 팔 | 걷는 중 넘어짐/1k | 사건/1k | 사건 뒤 100스텝 안 넘어짐 비율 | 사건 없이 난 넘어짐/1k |")
        P("|---|---|---|---|---|")
        dec = {}
        for k, a in arms.items():
            ev, mid_steps = slip_events(a, theta)
            m = phase_mask(a, "mid")
            mid_falls = np.array([(a["fall"][:, i] & m[:, i]).sum() for i in range(a["N"])], float)
            n_ev = np.array([len(e) for e in ev], float)
            ev_fall = np.array([sum(1 for t in e if a["fall"][t:t + 100, i].any()) for i, e in enumerate(ev)], float)
            no_ev = np.zeros(a["N"])
            for i in range(a["N"]):
                for t in np.where(a["fall"][:, i] & m[:, i])[0]:
                    if not any(t - 100 <= s0 <= t for s0 in ev[i]):
                        no_ev[i] += 1
            ok = mid_steps > 0
            per1k = lambda x: x[ok] / mid_steps[ok] * 1000.0  # noqa: E731
            idx = RNG.integers(0, ok.sum(), (NBOOT, ok.sum()))
            p = ev_fall[ok].sum() / max(n_ev[ok].sum(), 1)
            pb = ev_fall[ok][idx].sum(1) / np.maximum(n_ev[ok][idx].sum(1), 1)
            dec[k] = {"mid_falls": boot_mean(per1k(mid_falls)), "events": boot_mean(per1k(n_ev)),
                      "p_fall": (p, np.percentile(pb, 2.5), np.percentile(pb, 97.5)), "no_event": boot_mean(per1k(no_ev)),
                      "n_events": int(n_ev.sum()), "n_ev_falls": int(ev_fall.sum())}
            P(f"| {a['name']} | {fmt(dec[k]['mid_falls'], 2)} | {fmt(dec[k]['events'], 2)} | {fmt(dec[k]['p_fall'], 3)} "
              f"({dec[k]['n_ev_falls']}/{dec[k]['n_events']}) | {fmt(dec[k]['no_event'], 2)} |")
        res["decomp"] = dec; res["theta"] = theta
        if {"D6", "D7"} <= set(arms):
            r1 = dec["D6"]["events"][0] / max(dec["D7"]["events"][0], 1e-9)
            r2 = dec["D6"]["p_fall"][0] / max(dec["D7"]["p_fall"][0], 1e-9)
            pred["PB1"] = (r1, r1 >= 2.0); pred["PB2"] = (r2, r2 >= 2.0)
            P(f"\nPB1 사건/1k D6/D7 = {r1:.2f} (기준 ≥ 2) → {'성립' if r1 >= 2 else '기각'}")
            P(f"PB2 사건 뒤 넘어짐 비율 D6/D7 = {r2:.2f} (기준 ≥ 2) → {'성립' if r2 >= 2 else '기각'}")
    P("\n## QB 넘어짐 전조 (넘어진 스텝의 s ≥ 200)\n")
    for k in ("D6", "D5", "D1", "D9", "D3"):
        if k not in arms:
            continue
        pr = precursors(arms[k]); res.setdefault(k, {})["precursors"] = pr
        P(f"**{arms[k]['name']}**: 넘어짐 {pr['n_falls']}회, 대조 창 {pr['n_ctrl']}개")
        P("- 첫 신호 비율: " + ", ".join(f"{f} {v:.2f}" for f, v in pr["first"].items()))
        P(f"- 명령 급증(2초 안 vx > 1.0, 5스텝): 넘어짐 {pr['surge_fall']:.2f}, 대조 {pr['surge_ctrl']:.2f}")
        P("- AUC (넘어지기 2초~0.5초 전 평균 vs 대조; 넘어짐 평균 / 대조 평균): " + ", ".join(f"{f} {v[0]:.2f} ({v[1]:.3f}/{v[2]:.3f})" for f, v in pr["auc"].items()))
        P(f"- 기준(대조 95 백분위): " + ", ".join(f"{f} {v:.3f}" for f, v in pr["thr"].items()) + "\n")
    if "D6" in arms:
        pr = res["D6"]["precursors"]
        ok = pr["surge_fall"] >= 0.5 and pr["surge_ctrl"] <= 0.2; pred["PB3"] = ((pr["surge_fall"], pr["surge_ctrl"]), ok)
        P(f"PB3 명령 급증 D6 넘어짐 {pr['surge_fall']:.2f}, 대조 {pr['surge_ctrl']:.2f} (기준 ≥ 0.5, ≤ 0.2) → {'성립' if ok else '기각'}")
        # first signal is d or slip: count falls whose earliest crossing includes dev or slip_ma
        n = pr["n_falls"]; cnt = 0
        a = arms["D6"]; cm = control_mask(a); thr = pr["thr"]
        for t, i in [(t, i) for i in range(a["N"]) for t in np.where(a["fall"][:, i])[0] if a["since"][t, i] >= 200]:
            cross = {}
            for f in FEATS:
                st = runs_over(a[f][t - 150:t, i], thr[f])
                if st:
                    cross[f] = st[0]
            if cross and min(cross.values()) in (cross.get("dev"), cross.get("slip_ma")):
                cnt += 1
        frac = cnt / n if n else np.nan; pred["PB4"] = (frac, frac >= 0.6)
        P(f"PB4 첫 신호가 이탈 d 또는 미끄럼인 비율 = {frac:.2f} (기준 ≥ 0.6) → {'성립' if frac >= 0.6 else '기각'}")
    if {"D5", "D6"} <= set(arms):
        r = boot_ratio(rm("D6", "dev", "mid"), rm("D5", "dev", "mid")); pred["PB5"] = (r, r[0] >= 1.3)
        P(f"PB5 걷는 중 이탈 d D6/D5 = {fmt(r, 2)} (기준 ≥ 1.3) → {'성립' if r[0] >= 1.3 else '기각'}")

    # ---- startup time course (for figures) --------------------------------------------------------------------
    P("\n## 출발 시간 경과 (리셋 후 스텝별 평균, 정지 구간 제외)\n")
    P("| 팔 | s | 이탈 d | ΔJ | 미끄럼 | 전진 속도 | 명령 vx | 넘어짐 비율 |")
    P("|---|---|---|---|---|---|---|---|")
    course = {}
    for k, a in arms.items():
        course[k] = {}
        for s0 in range(25, 150, 10):
            m = a["valid"] & (a["since"] >= s0) & (a["since"] < s0 + 10)
            if m.sum() == 0:
                continue
            row = [float(np.nanmean(a[f][m])) for f in ("dev", "dJ", "slip", "vx", "cmdvx")] + [float(a["fall"][m].mean())]
            course[k][s0] = row
            if k in ("D1", "D2", "D3", "D4", "D9"):
                P(f"| {a['name']} | {s0}-{s0 + 9} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    res["course"] = course

    out = args.out or os.path.join(args.dir, "analysis.md")
    open(out, "w").write("\n".join(L) + "\n")
    js = {k: v for k, v in res.items()}
    js["pred"] = {k: {"value": v[0], "holds": bool(v[1])} for k, v in pred.items()}
    json.dump(js, open(out.replace(".md", ".json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print("written", out)


if __name__ == "__main__":
    main()
