"""Design doc 27 part R (docs/design27_recovery_quality_ko.md §5.2): recovery after slips, branched from the same states.

    python3 scripts/sim2sim/stage27_branch.py --dir results/stage27 [--out results/stage27/branch.md]

Reads <dir>/steps_<src>/pts_*.npz (branch_eval.py branch.steps_dir) for the sources AB_P, AB_pi (broad DR, anchor driven by
the planner / the policy head) and AP_P, AP_pi (paper model) and, for stage 2, <dir>/k/steps_<src>/.

Outcome per branch: fell100 = fall within 100 steps of the branch point. Per branch point the clones are averaged; rates
and paired differences (same branch point) get 95 % CIs from a cluster bootstrap over robots ((source, seed) clusters).
Candidates: A0 = policy head, Fcl = SimDist planner replanning every step, Nrep = policy head + a replayed stretch of the
planner's own deviation recorded after other slips (design 26), Pcalfpol / Ponlycalf (stage 2 K).
"""
import argparse
import glob
import json
import os

import numpy as np

RNG = np.random.default_rng(0)
NBOOT = 5000
CALF = np.array([2, 5, 8, 11]); NONCALF = np.array([j for j in range(12) if j not in CALF])
JOINTS = [f"{l}_{j}" for l in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
SOURCES = {"AB_P": "넓은 DR, 계획기로 걷다 미끄러짐", "AB_pi": "넓은 DR, 정책으로 걷다 미끄러짐",
           "AP_P": "논문 모델, 계획기로 걷다 미끄러짐", "AP_pi": "논문 모델, 정책으로 걷다 미끄러짐"}
GLAB = {"A0": "정책 헤드", "Fcl": "계획기", "Ncl": "정책 + AR(1) 잡음", "Nrep": "정책 + 계획기 이탈 재생", "Pcalfpol": "계획기, 종아리만 정책", "Ponlycalf": "정책, 종아리만 계획기"}


def load_source(d):
    fs = sorted(glob.glob(os.path.join(d, "pts_*.npz")))
    if not fs:
        return None
    parts = [dict(np.load(f, allow_pickle=True)) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def row_metrics(z, theta):
    """per-row metrics (dict of (R,) arrays) for windows w25 = steps 0-24 and w100 = steps 0-99, over alive steps."""
    R, H = z["alive"].shape
    al = z["alive"].astype(bool)
    act, pol = z["act"].astype(np.float32), z["act_pol"].astype(np.float32)
    dlt = act - pol
    out = {}
    jerk = np.full((R, H), np.nan, np.float32)
    jerk[:, 2:] = np.linalg.norm(act[:, 2:] - 2 * act[:, 1:-1] + act[:, :-2], axis=-1)
    jerk[:, 2:][~(al[:, 2:] & al[:, 1:-1] & al[:, :-2])] = np.nan
    feats = {"d": np.linalg.norm(dlt, axis=-1), "calf_dev": np.abs(dlt[..., CALF]).mean(-1), "noncalf_dev": np.abs(dlt[..., NONCALF]).mean(-1),
             "slip": z["slip"].astype(np.float32), "vx": z["lin_vel"][..., 0].astype(np.float32), "act_jerk": jerk,
             "tau2": z["tau2"].astype(np.float32), "power": z["power"].astype(np.float32), "lat": z["lat_err"].astype(np.float32),
             "ncontact": z["contact"].sum(-1).astype(np.float32), "rp_rate": np.linalg.norm(z["ang_vel"][..., :2].astype(np.float32), axis=-1)}
    for w, hi in (("w25", 25), ("w100", 100)):
        m = al[:, :hi]
        for k, x in feats.items():
            xx = np.where(m, x[:, :hi], np.nan)
            with np.errstate(invalid="ignore"):
                out[f"{k}_{w}"] = np.nanmean(xx, axis=1)
        tl = np.where(m, z["tilt"][:, :hi], np.nan)
        with np.errstate(invalid="ignore"):
            out[f"tilt_max_{w}"] = np.nanmax(tl, axis=1)
    # recovery: first k with the 10-step trailing mean slip (inside the branch) <= theta for 25 consecutive alive steps
    slip = z["slip"].astype(np.float64)
    cs = np.concatenate([np.zeros((R, 1)), np.cumsum(slip, 1)], 1)
    lo = np.maximum(np.arange(H) - 9, 0)
    ma = (cs[:, 1:] - cs[:, lo]) / (np.arange(H) + 1 - lo)
    ok = (ma <= theta) & al
    rec = np.full(R, np.nan)
    for r in range(R):
        run = 0
        for t in range(H):
            run = run + 1 if ok[r, t] else 0
            if run == 25:
                rec[r] = t - 24; break
    out["rec_time"] = rec
    out["rec100"] = (np.isfinite(rec) & (rec + 25 <= 100)).astype(np.float64)
    out["fell100"] = (z["fell"].astype(bool) & (z["fall_step"] >= 0) & (z["fall_step"] < 100)).astype(np.float64)
    out["delta_w25"] = np.array([np.nanmean(np.where(al[r, :25, None], dlt[r, :25], np.nan), axis=0) for r in range(R)])
    out["absdelta_w100"] = np.array([np.nanmean(np.where(al[r, :100, None], np.abs(dlt[r, :100]), np.nan), axis=0) for r in range(R)])
    return out


def point_table(src, z, met, kind):
    """dict group -> {"keys": [(src, seed, step)], "cluster": [...], metric -> (P,) per-point mean over clones}."""
    sel = (z["is_event"] == (1 if kind == "event" else 0))
    res = {}
    for g in np.unique(z["group"][sel]):
        m = sel & (z["group"] == g)
        keys = sorted(set(zip(z["seed"][m].tolist(), z["step"][m].tolist())))
        kidx = {k: i for i, k in enumerate(keys)}
        rows = np.where(m)[0]
        pi = np.array([kidx[(z["seed"][r], z["step"][r])] for r in rows])
        d = {"keys": [(src, s, t) for s, t in keys], "cluster": [(src, s) for s, t in keys]}
        for k, v in met.items():
            if v.ndim == 1:
                acc = np.zeros(len(keys)); cnt = np.zeros(len(keys))
                ok = np.isfinite(v[rows])
                np.add.at(acc, pi[ok], v[rows][ok]); np.add.at(cnt, pi[ok], 1)
                d[k] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            else:
                acc = np.zeros((len(keys), v.shape[1])); cnt = np.zeros((len(keys), 1))
                vv = v[rows]; ok = np.isfinite(vv).all(1)
                np.add.at(acc, pi[ok], vv[ok]); np.add.at(cnt, pi[ok], 1)
                d[k] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        res[g] = d
    return res


def merge(tables):
    """pool point tables of several sources: group -> concatenated."""
    out = {}
    for t in tables:
        for g, d in t.items():
            if g not in out:
                out[g] = {k: (list(v) if isinstance(v, list) else v.copy()) for k, v in d.items()}
            else:
                for k, v in d.items():
                    out[g][k] = out[g][k] + list(v) if isinstance(v, list) else np.concatenate([out[g][k], v])
    return out


def cluster_boot(values, clusters, fn=np.nanmean):
    values = np.asarray(values, np.float64); cl = np.array([hash(c) for c in clusters])
    u = np.unique(cl); groups = [np.where(cl == c)[0] for c in u]
    est = fn(values)
    bs = []
    for _ in range(NBOOT):
        pick = RNG.integers(0, len(u), len(u))
        idx = np.concatenate([groups[p] for p in pick])
        bs.append(fn(values[idx]))
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def paired(T, g1, g2, k):
    """per-point difference g2 - g1 on matched keys, cluster bootstrap."""
    if g1 not in T or g2 not in T:
        return (np.nan, np.nan, np.nan), 0
    i1 = {key: i for i, key in enumerate(T[g1]["keys"])}
    common = [(i1[key], j) for j, key in enumerate(T[g2]["keys"]) if key in i1]
    if not common:
        return (np.nan, np.nan, np.nan), 0
    a = np.array([T[g1][k][i] for i, _ in common]); b = np.array([T[g2][k][j] for _, j in common])
    cl = [T[g2]["cluster"][j] for _, j in common]
    ok = np.isfinite(a) & np.isfinite(b)
    return cluster_boot((b - a)[ok], [c for c, o in zip(cl, ok) if o]), int(ok.sum())


def rate(T, g, k="fell100"):
    if g not in T:
        return (np.nan, np.nan, np.nan), 0
    v = np.asarray(T[g][k]); ok = np.isfinite(v)
    return cluster_boot(v[ok], [c for c, o in zip(T[g]["cluster"], ok) if o]), int(ok.sum())


def diff_indep(Ta, Tb, g, k="fell100"):
    """rate(Tb, g) - rate(Ta, g), independent cluster bootstraps."""
    va, ca = np.asarray(Ta[g][k]), Ta[g]["cluster"]; vb, cb = np.asarray(Tb[g][k]), Tb[g]["cluster"]
    def bs(v, c):
        cl = np.array([hash(x) for x in c]); u = np.unique(cl); grp = [np.where(cl == x)[0] for x in u]
        return [np.nanmean(v[np.concatenate([grp[p] for p in RNG.integers(0, len(u), len(u))])]) for _ in range(NBOOT)]
    d = np.array(bs(vb, cb)) - np.array(bs(va, ca))
    return np.nanmean(vb) - np.nanmean(va), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)


def auc(x, y):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def f3(t, nd=3):
    return "—" if not np.isfinite(t[0]) else f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def r3(t, nd=3):
    return "—" if not np.isfinite(t[0]) else f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/stage27"); ap.add_argument("--out", default=None)
    ap.add_argument("--theta", type=float, default=None)
    args = ap.parse_args()
    theta = args.theta if args.theta is not None else json.load(open(os.path.join(args.dir, "noise_params.json")))["theta"]
    lines = []; P = lambda s="": (print(s), lines.append(s))  # noqa: E731
    res = {"theta": theta, "pred": {}, "rates": {}}
    data, tabs = {}, {}
    for src in SOURCES:
        z = load_source(os.path.join(args.dir, f"steps_{src}"))
        if z is None:
            continue
        met = row_metrics(z, theta)
        data[src] = (z, met)
        tabs[(src, "event")] = point_table(src, z, met, "event"); tabs[(src, "control")] = point_table(src, z, met, "control")
    P("## 미끄러진 순간에서 갈라 실행 (설계 27 §5.2)\n")
    P(f"θ = {theta:.4f} m/s. 넘어짐 = 갈라진 뒤 100스텝 안. 분기점마다 복제본 평균, 95 % CI는 로봇 단위 군집 부트스트랩.\n")
    # fidelity: unperturbed clone 0 of the anchor's own controller vs the anchor's continuation
    P("### 복원 충실도 (참고)\n")
    for src, (z, met) in data.items():
        own = "Fcl" if src.endswith("_P") else "A0"
        m = (z["group"] == own) & (z["clone"] == 0)
        P(f"- {src}: 앵커 자신의 제어기({GLAB[own]}) 복제본 0의 150스텝 안 넘어짐 {z['fell'][m].mean():.3f} (n {int(m.sum())})")
    P("\n### 넘어짐 비율 (분기점 평균)\n")
    P("| 출발 상태 | 종류 | 분기점 | " + " | ".join(GLAB[g] for g in ("A0", "Fcl", "Nrep")) + " |")
    P("|---|---|---|---|---|---|")
    for (src, kind), T in tabs.items():
        cells = []
        for g in ("A0", "Fcl", "Nrep"):
            r, n = rate(T, g); cells.append(r3(r)); res["rates"][f"{src}|{kind}|{g}"] = r
        n = len(T.get("A0", {}).get("keys", []))
        P(f"| {SOURCES[src]} | {'미끄러짐' if kind == 'event' else '평상시'} | {n} | " + " | ".join(cells) + " |")
    pool = {}
    for model, srcs in (("broad", ("AB_P", "AB_pi")), ("paper", ("AP_P", "AP_pi"))):
        for kind in ("event", "control"):
            ts = [tabs[(s, kind)] for s in srcs if (s, kind) in tabs]
            if ts:
                pool[(model, kind)] = merge(ts)
    P("\n### 같은 분기점 짝 비교 (뒤 − 앞, 넘어짐 비율)\n")
    P("| 모델 | 종류 | 계획기 − 정책 | 이탈 재생 − 정책 | 계획기 − 이탈 재생 |")
    P("|---|---|---|---|---|")
    PD = {}
    for (model, kind), T in pool.items():
        a, _ = paired(T, "A0", "Fcl", "fell100"); b, _ = paired(T, "A0", "Nrep", "fell100"); c, _ = paired(T, "Nrep", "Fcl", "fell100")
        PD[(model, kind)] = (a, b, c)
        P(f"| {model} | {'미끄러짐' if kind == 'event' else '평상시'} | {f3(a)} | {f3(b)} | {f3(c)} |")
    pred = res["pred"]
    P("\n### 예측 판정 (§6.2)\n")
    a, b, c = PD.get(("broad", "event"), ((np.nan,) * 3,) * 3)
    pred["PR1"] = {"value": a, "holds": bool(np.isfinite(a[0]) and a[0] >= 0.10 and a[1] > 0)}
    P(f"PR1 넓은 DR 미끄러짐, 계획기 − 정책 = {f3(a)} (기준 ≥ +0.10, CI > 0) → {'성립' if pred['PR1']['holds'] else '기각'}")
    if ("AB_P", "event") in tabs and ("AB_pi", "event") in tabs:
        s = diff_indep(tabs[("AB_pi", "event")], tabs[("AB_P", "event")], "A0")
    else:
        s = (np.nan,) * 3
    pred["PR2"] = {"value": s, "holds": bool(np.isfinite(s[0]) and s[0] <= 0.05)}
    P(f"PR2 넓은 DR, 정책 회복의 넘어짐: 계획기가 만든 미끄러짐 − 정책이 만든 미끄러짐 = {f3(s)} (기준 ≤ +0.05) → {'성립' if pred['PR2']['holds'] else '기각'}")
    pred["PR3"] = {"noise_minus_pol": b, "plan_minus_noise": c,
                   "holds": bool(np.isfinite(b[0]) and np.isfinite(c[0]) and b[0] <= 0.05 and c[0] >= 0.10 and c[1] > 0)}
    P(f"PR3 넓은 DR 미끄러짐: 이탈 재생 − 정책 = {f3(b)} (≤ +0.05), 계획기 − 이탈 재생 = {f3(c)} (≥ +0.10, CI > 0) → {'성립' if pred['PR3']['holds'] else '기각'}")
    Tc = pool.get(("broad", "control"))
    ra, _ = rate(Tc, "A0") if Tc else ((np.nan,) * 3, 0); rf, _ = rate(Tc, "Fcl") if Tc else ((np.nan,) * 3, 0)
    pred["PR4"] = {"A0": ra, "Fcl": rf, "holds": bool(np.isfinite(ra[0]) and np.isfinite(rf[0]) and ra[0] <= 0.03 and rf[0] <= 0.03)}
    P(f"PR4 넓은 DR 평상시: 정책 {r3(ra)}, 계획기 {r3(rf)} (둘 다 ≤ 0.03) → {'성립' if pred['PR4']['holds'] else '기각'}")
    pa = PD.get(("paper", "event"), ((np.nan,) * 3,))[0]
    pred["PR5"] = {"value": pa, "holds": bool(np.isfinite(pa[0]) and abs(pa[0]) <= 0.05)}
    P(f"PR5 논문 모델 미끄러짐, 계획기 − 정책 = {f3(pa)} (기준 |차| ≤ 0.05) → {'성립' if pred['PR5']['holds'] else '기각'}")
    # verdicts (§6.2)
    v = {}
    v["recovery_action"] = pred["PR1"]["holds"]
    v["state_effect"] = bool(np.isfinite(s[0]) and s[0] >= 0.10 and s[1] > 0)
    if pred["PR3"]["holds"]:
        v["size_or_direction"] = "direction"
    elif np.isfinite(b[0]) and b[0] >= 0.10 and b[1] > 0 and np.isfinite(c[0]) and abs(c[0]) <= 0.05:
        v["size_or_direction"] = "size"
    else:
        v["size_or_direction"] = "mixed"
    v["slip_specific"] = pred["PR4"]["holds"]; v["protocol_check"] = pred["PR5"]["holds"]
    # stage-2 condition: direction AND calf concentration of the broad planner's deviation (events, steps 0-99)
    calf_ratio = np.nan
    if ("broad", "event") in pool and "Fcl" in pool[("broad", "event")]:
        ad = np.nanmean(pool[("broad", "event")]["Fcl"]["absdelta_w100"], axis=0)
        calf_ratio = float(ad[CALF].mean() / ad[NONCALF].mean())
    v["calf_ratio"] = calf_ratio
    v["stage2_K"] = bool(v["size_or_direction"] == "direction" and np.isfinite(calf_ratio) and calf_ratio >= 1.5)
    res["verdict"] = v
    P(f"\n판정: 회복 행동 탓 {v['recovery_action']}, 상태 탓 {v['state_effect']}, 크기/방향 {v['size_or_direction']}, "
      f"미끄러짐 특이적 {v['slip_specific']}, 프로토콜 점검 {v['protocol_check']}, 계획기 이탈의 종아리/나머지 = {calf_ratio:.2f} → 2단계 K {'실행' if v['stage2_K'] else '미실행'}")
    # mechanism tables
    keys25 = ["d_w25", "calf_dev_w25", "noncalf_dev_w25", "tilt_max_w25", "slip_w25", "vx_w25", "act_jerk_w25", "tau2_w25", "power_w25",
              "lat_w25", "ncontact_w25", "rp_rate_w25"]
    keys100 = ["tilt_max_w100", "slip_w100", "vx_w100", "act_jerk_w100", "tau2_w100", "power_w100", "lat_w100", "rec100", "rec_time"]
    KLAB = {"d": "정책과의 행동 차 |a−a_pol|", "calf_dev": "종아리 관절 이탈 평균", "noncalf_dev": "나머지 관절 이탈 평균", "tilt_max": "최대 몸통 기울기",
            "slip": "접지 발 미끄럼 m/s", "vx": "전진 속도 m/s", "act_jerk": "행동 2차 차분", "tau2": "토크² N²m²", "power": "기계적 일률 W",
            "lat": "모델 1스텝 예측 오차", "ncontact": "접지 발 수", "rp_rate": "롤·피치 속도 rad/s", "rec100": "100스텝 안 미끄럼 회복 비율",
            "rec_time": "미끄럼 회복까지 스텝"}
    for model in ("broad", "paper"):
        T = pool.get((model, "event"))
        if not T:
            continue
        P(f"\n### 회복 과정 비교 — {model}, 미끄러짐 분기점 (정책, 계획기, 이탈 재생의 평균과 계획기 − 정책 짝 차이)\n")
        P("| 지표 | 구간 | 정책 | 계획기 | 이탈 재생 | 계획기 − 정책 [95 % CI] |")
        P("|---|---|---|---|---|---|")
        for k in keys25 + keys100:
            base = k.rsplit("_w", 1)[0] if "_w" in k else k
            win = "0–24" if k.endswith("w25") else ("0–99" if k.endswith("w100") else "0–149")
            mv = [np.nanmean(T[g][k]) if g in T else np.nan for g in ("A0", "Fcl", "Nrep")]
            dd, _ = paired(T, "A0", "Fcl", k)
            P(f"| {KLAB.get(base, base)} | {win} | " + " | ".join("—" if not np.isfinite(x) else f"{x:.3f}" for x in mv) + f" | {f3(dd)} |")
            res.setdefault("mech", {})[f"{model}|{k}"] = {"means": mv, "plan_minus_pol": dd}
    # AUC within the broad planner branches: metric (0-24) of falling vs not falling branches (rows, not points)
    if "AB_P" in data or "AB_pi" in data:
        P("\n### 넓은 DR 계획기 분기에서 넘어진 것과 안 넘어진 것 (0–24스텝, 행 단위)\n")
        P("| 지표 | 넘어짐 평균 | 안 넘어짐 평균 | AUC |")
        P("|---|---|---|---|")
        rows = [(z, met) for s, (z, met) in data.items() if s.startswith("AB")]
        for k in keys25:
            xs, ys = [], []
            for z, met in rows:
                m = (z["group"] == "Fcl") & (z["is_event"] == 1)
                f = met["fell100"][m] > 0.5
                xs.append(met[k][m][f]); ys.append(met[k][m][~f])
            x, y = np.concatenate(xs), np.concatenate(ys)
            base = k.rsplit("_w", 1)[0]
            P(f"| {KLAB.get(base, base)} | {np.nanmean(x):.3f} | {np.nanmean(y):.3f} | {auc(x, y):.2f} |")
        P("\n### 넓은 DR 계획기의 관절별 이탈 (미끄러짐 분기, 0–24스텝 부호 있는 평균 / 0–99스텝 절대 평균)\n")
        P("| 관절 | 부호 평균: 넘어짐 | 부호 평균: 안 넘어짐 | 절대 평균 |")
        P("|---|---|---|---|")
        dl, ab, fl = [], [], []
        for z, met in rows:
            m = (z["group"] == "Fcl") & (z["is_event"] == 1)
            dl.append(met["delta_w25"][m]); ab.append(met["absdelta_w100"][m]); fl.append(met["fell100"][m] > 0.5)
        dl, ab, fl = np.concatenate(dl), np.concatenate(ab), np.concatenate(fl)
        for j in range(12):
            P(f"| {JOINTS[j]} | {np.nanmean(dl[fl, j]):+.3f} | {np.nanmean(dl[~fl, j]):+.3f} | {np.nanmean(ab[:, j]):.3f} |")
    # time course (broad events): 10-step bins
    for model, srcs in (("broad", ("AB_P", "AB_pi")), ("paper", ("AP_P", "AP_pi"))):
        rows = [data[s] for s in srcs if s in data]
        if not rows:
            continue
        P(f"\n### 시간 경과 — {model}, 미끄러짐 분기 (10스텝 구간 평균: 살아 있는 비율 / 기울기 / 미끄럼 / 전진 속도 / 정책과의 차)\n")
        P("| 스텝 | " + " | ".join(GLAB[g] for g in ("A0", "Fcl", "Nrep")) + " |")
        P("|---|---|---|---|")
        for b0 in range(0, 100, 10):
            cells = []
            for g in ("A0", "Fcl", "Nrep"):
                al, tl, sl, vx, dd = [], [], [], [], []
                for z, met in rows:
                    m = (z["group"] == g) & (z["is_event"] == 1)
                    if not m.any():
                        continue
                    A = z["alive"][m, b0:b0 + 10].astype(bool)
                    al.append(A.mean())
                    tl.append(z["tilt"][m, b0:b0 + 10][A]); sl.append(z["slip"][m, b0:b0 + 10][A])
                    vx.append(z["lin_vel"][m, b0:b0 + 10, 0].astype(np.float32)[A])
                    dd.append(np.linalg.norm(z["act"][m, b0:b0 + 10].astype(np.float32) - z["act_pol"][m, b0:b0 + 10].astype(np.float32), axis=-1)[A])
                if not al:
                    cells.append("—"); continue
                cat = lambda L: np.concatenate(L) if L else np.array([np.nan])  # noqa: E731
                cells.append(f"{np.mean(al):.2f} / {np.nanmean(cat(tl)):.3f} / {np.nanmean(cat(sl)):.2f} / {np.nanmean(cat(vx)):.2f} / {np.nanmean(cat(dd)):.2f}")
            P(f"| {b0}–{b0 + 9} | " + " | ".join(cells) + " |")
    # stage 2 (K) if present
    c2 = os.path.join(args.dir, "k")
    if os.path.isdir(c2):
        tabs2 = []
        for src in ("AB_P", "AB_pi"):
            z = load_source(os.path.join(c2, f"steps_{src}"))
            if z is not None:
                tabs2.append(point_table(src, z, row_metrics(z, theta), "event"))
        if tabs2:
            T2 = merge(tabs2)
            P("\n### 2단계 K: 종아리 관절 나누기 (넓은 DR, 미끄러짐 분기점)\n")
            P("| 후보 | 넘어짐 비율 | 후보 − 정책 [95 % CI] |")
            P("|---|---|---|")
            for g in ("A0", "Fcl", "Pcalfpol", "Ponlycalf"):
                r, _ = rate(T2, g); dd, _ = paired(T2, "A0", g, "fell100") if g != "A0" else (((0.0, 0.0, 0.0)), 0)
                P(f"| {GLAB[g]} | {r3(r)} | {f3(dd)} |")
            a1, _ = paired(T2, "A0", "Pcalfpol", "fell100"); a2, _ = paired(T2, "A0", "Ponlycalf", "fell100")
            ok = bool(np.isfinite(a1[0]) and np.isfinite(a2[0]) and a1[0] <= 0.05 and a2[0] >= 0.10)
            pred["PK"] = {"calfpol_minus_pol": a1, "onlycalf_minus_pol": a2, "holds": ok}
            P(f"\nPK 종아리만 정책 − 정책 = {f3(a1)} (≤ +0.05), 종아리만 계획기 − 정책 = {f3(a2)} (≥ +0.10) → {'성립' if ok else '기각'}")
    out = args.out or os.path.join(args.dir, "branch.md")
    open(out, "w").write("\n".join(lines) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(x) for k, x in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    json.dump(clean(res), open(out[:-3] + ".json", "w"), indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
