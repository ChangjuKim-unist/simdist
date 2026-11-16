"""Design doc 28 (docs/design28_planner_motion_source_ko.md §4-§5): search noise vs terminal value as the source of the
planner's unnatural motion.

    python3 scripts/sim2sim/stage28_analyze.py --d27 results/stage27 --d28 results/stage28 [--out results/stage28/analysis.md]

Arms per setting: pol (design 27 policy run), base (design 27 planner run), Sq / S4 (1/4 or 4x samples + elites), V0 (value
weight 0). S4 diag files come in parts (diag_<setting>_S4_p0..4.npz, 4 robots each) and are merged here. Normal-floor runs
use the first 2000 steps of every arm (design 27 runs were 5000 steps). Mid-walk = released steps with steps since reset
>= 300. Motion metrics = stage27_metrics.robot_metrics; extra per-robot metrics: dev |a - a_pol|, realized reward per step,
model-score differences executed - policy plan (dJ = dR + dV, as logged: V0 logs dV = 0), reward optimism r0_hat - r,
and on ice slip events (design 26 §5) per 1k mid-walk steps and the share followed by a fall within 100 steps.
Ratios: robots resampled jointly by seed (95 % bootstrap CI).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage27_metrics as M27  # noqa: E402

THETA = 0.9553579986095428
MID = 300
RNG = np.random.default_rng(28)
NBOOT = 5000
STEP_KEYS = None  # raw keys with a leading step axis are found by shape


def raw_load(paths, trim=None):
    """load one diag npz or merge several (robot axis), sort robots by seed, optionally keep the first `trim` steps."""
    ds = [dict(np.load(p, allow_pickle=True)) for p in paths]
    S = ds[0]["since"].shape[0]
    assert all(d["since"].shape[0] == S for d in ds), [d["since"].shape for d in ds]
    out = {}
    for k in ds[0]:
        v = ds[0][k]
        if k in ("speeds", "seeds", "robot_mass"):
            out[k] = np.concatenate([d[k] for d in ds], 0)
        elif v.ndim >= 2 and v.shape[0] == S and k != "term_names":
            out[k] = np.concatenate([d[k] for d in ds], 1)
        else:
            out[k] = v
    order = np.argsort(out["seeds"], kind="stable")
    for k in list(out):
        v = out[k]
        if k in ("speeds", "seeds", "robot_mass"):
            out[k] = v[order]
        elif isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == S and k != "term_names":
            out[k] = v[:, order]
    if trim is not None and S > trim:
        for k in list(out):
            v = out[k]
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == S and k != "term_names":
                out[k] = v[:trim]
    return out


def m27_view(d, name):
    """the dict stage27_metrics.load builds, from an in-memory diag dict."""
    a = {"name": name}
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
    cons = np.zeros_like(a["valid"]); cons[1:] = a["valid"][1:] & a["valid"][:-1] & (a["since"][1:] == a["since"][:-1] + 1)
    a["cons"] = cons
    return a


def runs_over(x, thr, k=5):
    above = np.nan_to_num(x, nan=-np.inf) > thr
    starts, i = [], 0
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


def extra_metrics(d, a, m):
    """per-robot extra metrics over the mid-walk mask m."""
    N = a["N"]; out = {k: np.full(N, np.nan) for k in ("dev", "reward", "dJ", "dR", "dV", "opt", "events_1k", "ev_n", "ev_f")}
    dev = np.linalg.norm(d["act"].astype(np.float32) - d["act_pol"].astype(np.float32), axis=-1)
    rew = d["reward"].astype(np.float64)
    dR = d["R_exec"].astype(np.float64) - d["R_pol"].astype(np.float64)
    dV = d["V_exec"].astype(np.float64) - d["V_pol"].astype(np.float64)
    opt = d["r0_exec"].astype(np.float64) - rew
    c = a["contact"].astype(np.float32)
    slip = (a["foot_speed"] * c).sum(-1) / np.maximum(c.sum(-1), 1.0)
    ep = np.cumsum(a["since"] == 0, axis=0)
    for i in range(N):
        mi = m[:, i]
        if mi.sum() < 10:
            continue
        out["dev"][i] = dev[mi, i].mean(); out["reward"][i] = rew[mi, i].mean()
        out["dR"][i] = dR[mi, i].mean(); out["dV"][i] = dV[mi, i].mean(); out["dJ"][i] = out["dR"][i] + out["dV"][i]
        out["opt"][i] = opt[mi, i].mean()
        # slip events (design 26 §5): 10-step trailing MA inside an episode over released steps, > theta for 5 steps, s >= 300
        n_ev = n_f = 0
        for e in np.unique(ep[:, i]):
            idx = np.where((ep[:, i] == e) & a["valid"][:, i])[0]
            if len(idx) == 0:
                continue
            x = slip[idx, i].astype(np.float64)
            cs = np.cumsum(np.insert(x, 0, 0.0)); lo = np.maximum(np.arange(len(idx)) - 9, 0)
            ma = (cs[np.arange(len(idx)) + 1] - cs[lo]) / (np.arange(len(idx)) + 1 - lo)
            for st in runs_over(ma, THETA):
                t = idx[st]
                if a["since"][t, i] >= MID:
                    n_ev += 1; n_f += int(a["fall"][t:t + 100, i].any())
        out["ev_n"][i] = n_ev; out["ev_f"][i] = n_f
        out["events_1k"][i] = n_ev / mi.sum() * 1000.0
    return out


def boot_prop_ratio(nx, fx, ny, fy):
    """(sum fy / sum ny) / (sum fx / sum nx) with robots resampled jointly."""
    def prop(n, f, idx=None):
        if idx is None:
            return f.sum() / max(n.sum(), 1)
        return f[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    px, py = prop(nx, fx), prop(ny, fy)
    idx = RNG.integers(0, len(nx), (NBOOT, len(nx)))
    bx, by = prop(nx, fx, idx), prop(ny, fy, idx)
    b = np.where(bx > 0, by / np.where(bx > 0, bx, 1.0), np.nan)
    r = py / px if px > 0 else np.nan
    return px, py, (r, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5))


def boot_excess_reduction(jp, jb, js):
    """1 - mean(js - jp) / mean(jb - jp): share of the planner jitter excess removed (robots resampled jointly)."""
    ok = np.isfinite(jp) & np.isfinite(jb) & np.isfinite(js)
    jp, jb, js = jp[ok], jb[ok], js[ok]
    pt = 1 - (js - jp).mean() / (jb - jp).mean()
    idx = RNG.integers(0, len(jp), (NBOOT, len(jp)))
    den = (jb[idx] - jp[idx]).mean(1)
    b = np.where(den > 0, 1 - (js[idx] - jp[idx]).mean(1) / np.where(den > 0, den, 1.0), np.nan)
    return pt, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


LAB = dict(M27.LABEL)
LAB.update({"dev": "정책 이탈 d = |a − a_pol|", "reward": "실제 보상/스텝", "dJ": "ΔĴ 실행 − 정책 계획", "dR": "ΔĴ 중 보상 몫",
            "dV": "ΔĴ 중 가치 몫", "opt": "보상 낙관 r̂0 − r", "events_1k": "미끄러짐 사건/1k(걷는 중)"})
FLOOR_ROWS = ["dev", "act_jerk", "hf_ratio", "act_rate", "stride_cv", "stride_T", "short_cps", "diag_sync", "duty", "lat_speed",
              "yaw_rate", "speed_err", "height_sd", "rp_rate", "vz", "tilt", "slip", "torque_sq_d", "power", "reward", "dJ", "dR",
              "dV", "opt"]
ICE_ROWS = ["falls_1k", "events_1k"] + FLOOR_ROWS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d27", default="results/stage27"); ap.add_argument("--d28", default="results/stage28")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_md = args.out or os.path.join(args.d28, "analysis.md")
    settings = {  # setting -> (trim, policy file, base file, prefix of new arms, rows)
        "B_floor": (2000, "diag_NB_pi.npz", "diag_NB_P.npz", "B_floor", FLOOR_ROWS),
        "P_floor": (2000, "diag_NP_pi.npz", "diag_NP_P.npz", "P_floor", FLOOR_ROWS),
        "B_ice": (None, "diag_AB_pi.npz", "diag_AB_P.npz", "B_ice", ICE_ROWS),
    }
    title = {"B_floor": "넓은 DR, 보통 바닥(마찰 0.8, 처음 2000스텝)", "P_floor": "논문 모델, 보통 바닥(마찰 0.8, 처음 2000스텝)",
             "B_ice": "넓은 DR, 빙판(마찰 0.15, 5000스텝)"}
    R = {}  # (setting, arm) -> metrics dict of per-robot arrays
    for st, (trim, fpol, fbase, pre, _) in settings.items():
        files = {"pol": [os.path.join(args.d27, fpol)], "base": [os.path.join(args.d27, fbase)],
                 "Sq": [os.path.join(args.d28, f"diag_{pre}_Sq.npz")], "V0": [os.path.join(args.d28, f"diag_{pre}_V0.npz")],
                 "S4": sorted(glob.glob(os.path.join(args.d28, f"diag_{pre}_S4_p*.npz")))}
        for arm, fl in files.items():
            if not fl or not all(os.path.exists(f) for f in fl):
                print(f"missing {st} {arm}: {fl}"); continue
            d = raw_load(fl, trim)
            a = m27_view(d, f"{st}_{arm}")
            m = a["valid"] & (a["since"] >= MID)
            met = M27.robot_metrics(a, m); met.update(extra_metrics(d, a, m))
            met["seeds"] = np.asarray(a["seeds"]); R[(st, arm)] = met
            print(f"{st} {arm}: {a['N']} robots, {a['S']} steps, {int(m.sum())} mid-walk robot-steps")
    lines = []; P = lambda s="": (print(s), lines.append(s))  # noqa: E731
    res = {"pred": {}, "tables": {}}

    def same_seeds(st, *arms):
        s0 = R[(st, arms[0])]["seeds"]
        return all(np.array_equal(R[(st, x)]["seeds"], s0) for x in arms)

    def ratio(st, x, y, k):
        return M27.boot_ratio(R[(st, x)][k], R[(st, y)][k])

    def table(st, arms, ref, rows, head):
        have = [x for x in arms if (st, x) in R]
        if ref not in have:
            P(f"\n### {head}\n\n(기준 팔 없음)"); return {}
        assert same_seeds(st, *have), f"seed order differs in {st}"
        cmp_arms = [x for x in have if x not in (ref, "pol")]
        P(f"\n### {head}\n")
        P("| 지표 | " + " | ".join(have) + " | " + " | ".join(f"{x}/{ref} [95 % CI]" for x in cmp_arms) + " |")
        P("|---|" + "---|" * (len(have) + len(cmp_arms)))
        tab = {}
        for k in rows:
            vals = [R[(st, x)][k] for x in have]
            if not any(np.isfinite(v).any() for v in vals):
                continue
            cells = [M27.fmt(np.nanmean(v), 4 if k in ("reward", "opt") else 3) for v in vals]
            rs = []
            for x in cmp_arms:
                r = ratio(st, ref, x, k) if k not in ("dJ", "dR", "dV", "opt") else (np.nan, np.nan, np.nan)
                rs.append(r); tab[(k, x)] = r
                cells.append("—" if not np.isfinite(r[0]) else f"{M27.fmt(r[0], 2)} [{M27.fmt(r[1], 2)}, {M27.fmt(r[2], 2)}]")
            P(f"| {LAB.get(k, k)} | " + " | ".join(cells) + " |")
        return tab

    P("## 설계 28: 계획기의 부자연스러운 움직임의 출처\n")
    P("팔: pol 정책 헤드(설계 27), base 기본 계획기(설계 27), Sq 표본·elite ¼, S4 표본·elite 4배, V0 가치 가중 0. "
      "걷는 중 = 리셋 후 300스텝 이후. 비는 18대 로봇 단위 짝 부트스트랩. ΔĴ·낙관은 비 대신 값만 본다.")
    T = {}
    for st in settings:
        rows = settings[st][4]
        T[(st, "noise")] = table(st, ["pol", "Sq", "base", "S4"], "base", rows, f"{title[st]}: 탐색 잡음 (Sq, S4 대 base)")
        T[(st, "value")] = table(st, ["pol", "base", "V0"], "base", rows, f"{title[st]}: 가치 항 (V0 대 base)")

    # jitter excess over the policy
    P("\n### 떨림 초과분 (행동 2차 차분, 계획기 팔 − 같은 설정의 정책)\n")
    P("| 설정 | base | Sq | S4 | V0 | S4가 없앤 몫 [95 % CI] | Sq가 없앤 몫 | V0가 없앤 몫 |")
    P("|---|---|---|---|---|---|---|---|")
    EX = {}
    for st in settings:
        if (st, "pol") not in R or (st, "base") not in R:
            continue
        jp = R[(st, "pol")]["act_jerk"]
        ex = {x: (np.nanmean(R[(st, x)]["act_jerk"] - jp) if (st, x) in R else np.nan) for x in ("base", "Sq", "S4", "V0")}
        red = {x: (boot_excess_reduction(jp, R[(st, "base")]["act_jerk"], R[(st, x)]["act_jerk"]) if (st, x) in R
                   else (np.nan,) * 3) for x in ("S4", "Sq", "V0")}
        EX[st] = {"excess": ex, "reduction": red}
        P(f"| {st} | " + " | ".join(M27.fmt(ex[x]) for x in ("base", "Sq", "S4", "V0")) + " | "
          + " | ".join(f"{M27.fmt(red[x][0], 2)} [{M27.fmt(red[x][1], 2)}, {M27.fmt(red[x][2], 2)}]" for x in ("S4", "Sq", "V0")) + " |")

    # ice: falls after slip events
    PE = {}
    if ("B_ice", "base") in R:
        P("\n### 넓은 DR 빙판: 미끄러짐 사건 뒤 100스텝 안 넘어짐 비율\n")
        P("| 팔 | 사건 수 | 사건 뒤 넘어짐 비율 | 팔/base [95 % CI] |")
        P("|---|---|---|---|")
        b = R[("B_ice", "base")]
        for x in ("pol", "base", "Sq", "S4", "V0"):
            if ("B_ice", x) not in R:
                continue
            y = R[("B_ice", x)]
            px, py, r = boot_prop_ratio(b["ev_n"], b["ev_f"], y["ev_n"], y["ev_f"])
            PE[x] = {"n": int(np.nansum(y["ev_n"])), "p": py, "ratio": r}
            P(f"| {x} | {int(np.nansum(y['ev_n']))} | {M27.fmt(py)} | " + ("—" if x == "base" else f"{M27.fmt(r[0], 2)} [{M27.fmt(r[1], 2)}, {M27.fmt(r[2], 2)}]") + " |")

    # predictions §5
    P("\n## 예측 판정 (§5)\n")
    pred = res["pred"]

    def rr(st, x, k):
        return ratio(st, "base", x, k) if (st, x) in R and (st, "base") in R else (np.nan,) * 3

    def f3(r):
        return f"{M27.fmt(r[0], 2)} [{M27.fmt(r[1], 2)}, {M27.fmt(r[2], 2)}]"
    r4, rq = rr("B_floor", "S4", "dev"), rr("B_floor", "Sq", "dev")
    ok = bool(np.isfinite(r4[0]) and np.isfinite(rq[0]) and r4[0] <= 0.85 and rq[0] >= 1.2)
    pred["PN1"] = {"S4": r4, "Sq": rq, "holds": ok}
    P(f"PN1 넓은 DR 보통 바닥 d: S4/base {f3(r4)} (≤ 0.85), Sq/base {f3(rq)} (≥ 1.2) → {'성립' if ok else '기각'}")
    ex = EX.get("B_floor", {})
    red4 = ex.get("reduction", {}).get("S4", (np.nan,) * 3); exs = ex.get("excess", {})
    ok = bool(np.isfinite(red4[0]) and red4[0] >= 0.40 and np.isfinite(exs.get("Sq", np.nan)) and exs["Sq"] > exs["base"])
    pred["PN2"] = {"S4_reduction": red4, "excess": exs, "holds": ok}
    P(f"PN2 넓은 DR 보통 바닥 떨림 초과분: S4가 없앤 몫 {f3(red4)} (≥ 0.40), 초과분 Sq {M27.fmt(exs.get('Sq', np.nan))} > base {M27.fmt(exs.get('base', np.nan))} → {'성립' if ok else '기각'}")
    rc, rs = rr("B_floor", "S4", "stride_cv"), rr("B_floor", "S4", "short_cps")
    ok = bool(np.isfinite(rc[0]) and np.isfinite(rs[0]) and rc[0] <= 0.8 and rs[0] <= 0.7)
    pred["PN3"] = {"stride_cv": rc, "short_cps": rs, "holds": ok}
    P(f"PN3 넓은 DR 보통 바닥 S4/base: 보폭 변동계수 {f3(rc)} (≤ 0.8), 짧은 접지 {f3(rs)} (≤ 0.7) → {'성립' if ok else '기각'}")
    r = rr("P_floor", "S4", "dev"); ok = bool(np.isfinite(r[0]) and r[0] <= 0.85); pred["PN4"] = {"S4": r, "holds": ok}
    P(f"PN4 논문 모델 보통 바닥 d: S4/base {f3(r)} (≤ 0.85) → {'성립' if ok else '기각'}")
    r = rr("B_floor", "V0", "dev"); ok = bool(np.isfinite(r[0]) and r[0] <= 0.85); pred["PV1"] = {"V0": r, "holds": ok}
    P(f"PV1 넓은 DR 보통 바닥 d: V0/base {f3(r)} (≤ 0.85) → {'성립' if ok else '기각'}")
    r = rr("B_floor", "V0", "stride_cv"); ok = bool(np.isfinite(r[0]) and r[0] <= 0.8); pred["PV2"] = {"V0": r, "holds": ok}
    P(f"PV2 넓은 DR 보통 바닥 보폭 변동계수: V0/base {f3(r)} (≤ 0.8) → {'성립' if ok else '기각'}")
    r = rr("B_ice", "S4", "falls_1k"); ok = bool(np.isfinite(r[0]) and r[0] <= 0.7 and r[2] < 1.0); pred["PC1"] = {"S4": r, "holds": ok}
    P(f"PC1 넓은 DR 빙판 넘어짐/1k: S4/base {f3(r)} (≤ 0.7, CI 상한 < 1) → {'성립' if ok else '기각'}")
    r = PE.get("S4", {}).get("ratio", (np.nan,) * 3); ok = bool(np.isfinite(r[0]) and r[0] <= 0.7); pred["PC2"] = {"S4": r, "holds": ok}
    P(f"PC2 넓은 DR 빙판 사건 뒤 넘어짐 비율: S4/base {f3(r)} (≤ 0.7) → {'성립' if ok else '기각'}")
    pr1 = {}; ok = ("B_floor", "pol") in R
    for x in ("base", "S4", "Sq", "V0"):
        if ("B_floor", x) not in R:
            ok = False; continue
        rw = M27.boot_ratio(R[("B_floor", "pol")]["reward"], R[("B_floor", x)]["reward"])
        cv = M27.boot_ratio(R[("B_floor", "pol")]["stride_cv"], R[("B_floor", x)]["stride_cv"])
        pr1[x] = {"reward": rw, "stride_cv": cv}
        ok &= bool(np.isfinite(rw[0]) and 0.95 <= rw[0] <= 1.10 and np.isfinite(cv[0]) and cv[0] >= 3.0)
    pred["PR1"] = {"arms": pr1, "holds": bool(ok)}
    P("PR1 넓은 DR 보통 바닥, 계획기 팔/정책: " + ", ".join(f"{x} 보상 {M27.fmt(v['reward'][0], 3)}·보폭 변동계수 {M27.fmt(v['stride_cv'][0], 2)}" for x, v in pr1.items())
      + f" (보상 0.95~1.10, 변동계수 ≥ 3) → {'성립' if ok else '기각'}")
    if ("B_floor", "S4") in R and ("B_floor", "base") in R:
        dj4, djb = np.nanmean(R[("B_floor", "S4")]["dJ"]), np.nanmean(R[("B_floor", "base")]["dJ"])
        rw = rr("B_floor", "S4", "reward"); ok = bool(dj4 > djb and np.isfinite(rw[0]) and rw[0] <= 1.01)
    else:
        dj4 = djb = np.nan; rw = (np.nan,) * 3; ok = False
    pred["PO1"] = {"dJ_S4": dj4, "dJ_base": djb, "reward_S4_base": rw, "holds": ok}
    P(f"PO1 넓은 DR 보통 바닥: ΔĴ S4 {M27.fmt(dj4)} > base {M27.fmt(djb)}, 실제 보상 S4/base {f3(rw)} (≤ 1.01) → {'성립' if ok else '기각'}")
    v = {"noise_jitter": pred["PN1"]["holds"] and pred["PN2"]["holds"], "noise_gait": pred["PN3"]["holds"],
         "objective_gait": pred["PN1"]["holds"] and not pred["PN3"]["holds"] and not pred["PV2"]["holds"],
         "value": pred["PV1"]["holds"], "reward_gap": pred["PR1"]["holds"], "exploit": pred["PO1"]["holds"]}
    red_ice = EX.get("B_ice", {}).get("reduction", {}).get("S4", (np.nan,) * 3)
    v["chain"] = bool((pred["PC1"]["holds"] or pred["PC2"]["holds"]) and np.isfinite(red_ice[0]) and red_ice[0] > 0)
    res["verdict"] = v
    P(f"\n판정: 떨림 = 탐색 잡음 {v['noise_jitter']}, 불규칙한 걸음 = 잡음 {v['noise_gait']}, 불규칙한 걸음 = 점수의 최적 {v['objective_gait']}, "
      f"가치 항 탓 {v['value']}, 사슬 확인 {v['chain']}, 보상의 빈틈 {v['reward_gap']}, 점수 착취 {v['exploit']}")

    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    open(out_md, "w").write("\n".join(lines) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(x) for k, x in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    res["tables"] = {f"{k[0]}|{k[1]}": {f"{m}|{x}": r for (m, x), r in t.items()} for k, t in T.items()}
    res["excess"] = EX; res["events"] = PE
    json.dump(clean(res), open(os.path.splitext(out_md)[0] + ".json", "w"), indent=1)
    print(f"written {out_md}")


if __name__ == "__main__":
    main()
