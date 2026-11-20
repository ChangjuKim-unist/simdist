"""Design doc 35 (docs/design35_headroom_gates_ko.md): task score S and the registered gate readings.

    python scripts/sim2sim/stage35_score.py --d results/stage35 --d34 results/stage34 [--pick_heads]

Task score S (registered before any run, §4 of the doc), per control step, from the diagnostics:
    S = dt * [1.5 * exp(-((vx - v*)^2 + vy^2) / 0.25) + 0.75 * exp(-wz^2 / 0.25)]
        + lin_vel_z_l2 + ang_vel_xy_l2 + dof_torques_l2 + dof_acc_l2 + action_rate_l2   (logged reward terms, weighted, x dt)
v* = the robot's target speed (set-point, not the outer-loop command), body-frame velocities. Counted steps: not resetting,
at least MID = 300 steps since the reset (mid-walk; the startup is handled outside the model). The style terms (gait,
swing height, flat orientation, hip deviation) are reported apart and are not in S.
Comparisons are paired by robot (same speed and seed); 95 % intervals from 5000 bootstrap resamples of the 18 robots.
"""
import argparse, os
import numpy as np

MID = 300; DT = 0.02; NB = 5000
PEN = ["lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "action_rate_l2"]
STYLE = ["gait", "feet_height_gait", "flat_orientation_l2", "joint_deviation_l1"]
_cache = {}


def load(path):
    if path in _cache:
        return _cache[path]
    if not os.path.exists(path):
        _cache[path] = None; return None
    z = np.load(path, allow_pickle=True)
    names = list(z["term_names"]); terms = z["terms"].astype(np.float64)
    s = z["since"].astype(np.int64); ok = (~z["resetting"].astype(bool)) & (s >= MID)
    v = z["lin_vel"].astype(np.float64); w = z["ang_vel"][..., 2].astype(np.float64); sp = z["speeds"].astype(np.float64)
    track = DT * (1.5 * np.exp(-(((v[..., 0] - sp[None]) ** 2) + v[..., 1] ** 2) / 0.25) + 0.75 * np.exp(-(w ** 2) / 0.25))
    pen = sum(terms[..., names.index(k)] for k in PEN if k in names)
    sty = sum(terms[..., names.index(k)] for k in STYLE if k in names)
    S = track + pen
    T = s.shape[0]; n = np.maximum(ok.sum(0), 1).astype(np.float64)
    m = lambda x: (x * ok).sum(0) / n  # noqa: E731  per-robot mean over counted steps
    fl = z["fall"].astype(bool)
    r = {"S": m(S), "track": m(track), "pen": m(pen), "style": m(sty), "reward": m(z["reward"].astype(np.float64)), "n": n,
         "falls": 1000.0 * fl.sum(0) / T, "haz": 1000.0 * (fl & (s >= MID)).sum(0) / n, "share": ok.mean(0),
         "vratio": m(v[..., 0]) / sp, "vy": m(np.abs(v[..., 1])), "wz_sd": np.sqrt(np.maximum(m(w ** 2) - m(w) ** 2, 0.0)),
         "gx": m(z["grav"][..., 0].astype(np.float64)), "gy": m(z["grav"][..., 1].astype(np.float64))}
    _cache[path] = r
    return r


def pooled(r, k):
    return float((r[k] * r["n"]).sum() / r["n"].sum()) if k not in ("falls", "share") else float(r[k].mean())


def diff(a, b, k="S", ref=None, seed=35):
    """paired difference a - b of the robot-pooled metric k, in % of ref (a scalar); bootstrap over robots"""
    rng = np.random.default_rng(seed); N = len(a["n"]); idx = rng.integers(0, N, (NB, N))
    def pm(r, i):
        return (r[k][i] * r["n"][i]).sum(-1) / r["n"][i].sum(-1)
    d = pm(a, np.arange(N)) - pm(b, np.arange(N)); bd = pm(a, idx) - pm(b, idx)
    sc = 100.0 / ref if ref else 1.0
    return d * sc, np.percentile(bd, 2.5) * sc, np.percentile(bd, 97.5) * sc


def helps(a, b, ref, thr):
    """registered 'helps' rule: S gain >= thr % with the interval above 0, or falls down >= 30 % and >= 0.3 per 1000 steps
    with S not below -2 %"""
    d, lo, hi = diff(a, b, ref=ref)
    fa, fb = pooled(a, "falls"), pooled(b, "falls")
    by_s = d >= thr and lo > 0
    by_f = fb > 0 and fa <= 0.7 * fb and fb - fa >= 0.3 and d > -2.0
    return (by_s or by_f), d, lo, hi, fa, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="results/stage35"); ap.add_argument("--d34", default="results/stage34")
    ap.add_argument("--conds", default="legR50 slope5 tow20 mot50")
    ap.add_argument("--heads", default="go1_paper_broadhead go1_paper_targethead go1_paper_targethead_s2 go1_paper_payloadhead "
                    "go1_paper_fr_blanket go1_paper_rw_t05 go1_paper_rw_t06 go1_paper_rw_w02 go1_paper_rw_w06 go1_broadwm")
    ap.add_argument("--pick_heads", action="store_true", help="only write best_heads.txt (P2, best head alone per level)")
    ap.add_argument("--mid", type=int, default=300, help="testing only (short smoke runs)")
    a = ap.parse_args(); C = a.conds.split(); H = a.heads.split()
    global MID
    MID = a.mid
    f34 = lambda n: load(os.path.join(a.d34, f"diag_{n}.npz"))  # noqa: E731
    f35 = lambda n: load(os.path.join(a.d, f"diag_{n}.npz"))  # noqa: E731
    out = []; P = out.append

    def gap_row(label, nom, x):
        if nom is None or x is None:
            P(f"| {label} | - | - | - |"); return None
        g = -diff(x, nom, ref=pooled(nom, "S"))[0]
        P(f"| {label} | {pooled(x, 'S'):.4f} | {g:.1f} % | {pooled(x, 'falls'):.2f} |")
        return g

    # ---- P2 pick (also used by the full report) ----
    best = {}
    for c in C:
        pol = f34(f"{c}_POL"); nom = f34("nominal_POL")
        cand = [(h, f35(f"{c}_H_{h}")) for h in H]
        cand = [(h, r) for h, r in cand if r is not None and pol is not None and nom is not None]
        if cand:
            best[c] = max(cand, key=lambda t: diff(t[1], pol, ref=pooled(nom, "S"))[0])[0]
    bh = os.path.join(a.d, "best_heads.txt")
    if not a.pick_heads and os.path.exists(bh):  # the heads that were really run inside the planner (stage 2)
        best = dict(l.split() for l in open(bh) if l.strip())
    if a.pick_heads:
        with open(os.path.join(a.d, "best_heads.txt"), "w") as fh:
            for c, h in best.items():
                fh.write(f"{c} {h}\n"); print(f"best {c}: {h}")
        return

    P("# Design 35 report (generated by scripts/sim2sim/stage35_score.py)\n")
    P("S = task score per step (set-point tracking + effort/smoothness penalties), mid-walk (>= 300 steps since reset). "
      "Gaps and changes in % of the nominal reference S. Falls = per 1000 steps, whole run.\n")
    # ---- P0 ----
    P("## P0: gap to nominal without the integral term (design-34 runs)\n")
    P("| level, arm | S | gap to nominal pretrained planner | falls |"); P("|---|---|---|---|")
    nomP = f34("nominal_PRE"); p0 = {}
    gap_row("nominal, pretrained planner", nomP, nomP)
    for c in C:
        gap_row(f"{c}, policy alone", f34("nominal_POL"), f34(f"{c}_POL"))
        gap_row(f"{c}, pretrained planner", nomP, f34(f"{c}_PRE"))
        p0[c] = gap_row(f"{c}, adapted planner", nomP, f34(f"{c}_ADP"))
    keep = [c for c in C if p0.get(c) is not None and p0[c] >= 8.0]
    P(f"\nP0 rule (gap of the adapted planner >= 8 %): kept {keep}, dropped {[c for c in C if c not in keep]}\n")
    # ---- P1 ----
    P("## P1: integral term on the model-side command\n")
    P("| level, arm | S | gap to nominal pretrained planner + integral | falls |"); P("|---|---|---|---|")
    nomI = f35("nominal_PREI"); p1 = {}
    gap_row("nominal, pretrained planner + integral", nomI, nomI)
    for c in C:
        p1[c] = gap_row(f"{c}, adapted planner + integral", nomI, f35(f"{c}_ADPI"))
    P("\n| level | change of S by the integral term, adapted planner | same, policy alone | registered reward change, adapted planner |")
    P("|---|---|---|---|")
    loop_ok = 0; n_loop = 0
    for c in ["nominal"] + C:
        x0, x1 = (f34("nominal_PRE"), nomI) if c == "nominal" else (f34(f"{c}_ADP"), f35(f"{c}_ADPI"))
        q0, q1 = f34(f"{c}_POL"), f35(f"{c}_POLI")
        ref = pooled(nomP, "S") if nomP else None
        cell = lambda u, v, k="S": "-" if (u is None or v is None or ref is None) else "{:+.1f} % [{:+.1f}, {:+.1f}]".format(*diff(v, u, k=k, ref=ref if k == "S" else pooled(nomP, "reward")))  # noqa: E731
        P(f"| {c} | {cell(x0, x1)} | {cell(q0, q1)} | {cell(x0, x1, 'reward')} |")
        if c != "nominal" and x0 is not None and x1 is not None and ref:
            n_loop += 1; loop_ok += diff(x1, x0, ref=ref)[0] >= -1.0
    small = [c for c in C if p1.get(c) is not None and p1[c] < 8.0]
    P(f"\nP1 rule: gap < 8 % with the integral term in {small} -> {'PAUSE the idea (>= 3 of 4)' if len(small) >= 3 else 'continue'}")
    if nomI is not None and nomP is not None:
        dn = diff(nomI, nomP, ref=pooled(nomP, "S"))[0]
        adopt = loop_ok >= 3 and dn >= -1.0
        P(f"Integral term kept in the baseline: {adopt} (not worse than -1 % in {loop_ok} of {n_loop} levels, nominal {dn:+.1f} %)\n")
    # ---- P2 ----
    P("## P2: existing policy heads alone (no integral term)\n")
    nomPol = f34("nominal_POL"); refP = pooled(nomPol, "S") if nomPol else None
    P("| head | change on nominal | " + " | ".join(C) + " |"); P("|---|---|" + "---|" * len(C))
    nh = {c: 0 for c in C}
    for h in H:
        cells = []
        hn = f35(f"nominal_H_{h}")
        nc = "-" if hn is None or nomPol is None else "{:+.1f} %".format(diff(hn, nomPol, ref=refP)[0])
        for c in C:
            x, b = f35(f"{c}_H_{h}"), f34(f"{c}_POL")
            if x is None or b is None or refP is None:
                cells.append("-"); continue
            ok, d, lo, hi, fa, fb = helps(x, b, refP, 5.0); nh[c] += ok
            cells.append(f"{d:+.1f} % [{lo:+.1f}, {hi:+.1f}], falls {fb:.2f}->{fa:.2f}{' **helps**' if ok else ''}")
        P(f"| {h} | {nc} | " + " | ".join(cells) + " |")
    P("\n| level | head used (best alone) | adapted planner + that head vs adapted planner |"); P("|---|---|---|")
    for c in C:
        x, b = f35(f"{c}_ADPH"), f34(f"{c}_ADP")
        if x is None or b is None or nomP is None:
            P(f"| {c} | {best.get(c, '-')} | - |"); continue
        ok, d, lo, hi, fa, fb = helps(x, b, pooled(nomP, "S"), 5.0)
        P(f"| {c} | {best.get(c, '-')} | {d:+.1f} % [{lo:+.1f}, {hi:+.1f}], falls {fb:.2f}->{fa:.2f}{' **helps**' if ok else ''} |")
    P(f"\nP2 reading: levels where some head alone helps: {[c for c in C if nh[c]]}\n")
    # ---- P3 ----
    P("## P3: specialist experts (fine-tuned inside the level) against the paper expert and the nominal continuation\n")
    xp_nom = f35("nominal_X_paper"); refX = pooled(xp_nom, "S") if xp_nom else None
    P("| level | paper expert S | nominal continuation S | specialist S | headroom (specialist - better reference) | falls ref -> specialist | nominal cost of the specialist |")
    P("|---|---|---|---|---|---|---|")
    small3 = []
    for c in C:
        xp, xb, xs = f35(f"{c}_X_paper"), f35(f"{c}_X_b0c"), f35(f"{c}_X_spec_{c}")
        if None in (xp, xb, xs) or refX is None:
            P(f"| {c} | - | - | - | - | - | - |"); small3.append(c); continue
        ref = xp if pooled(xp, "S") >= pooled(xb, "S") else xb
        ok, d, lo, hi, fa, fb = helps(xs, ref, refX, 10.0)
        sn = f35(f"nominal_X_spec_{c}"); nc = "-" if sn is None else "{:+.1f} %".format(diff(sn, xp_nom, ref=refX)[0])
        P(f"| {c} | {pooled(xp, 'S'):.4f} | {pooled(xb, 'S'):.4f} | {pooled(xs, 'S'):.4f} | {d:+.1f} % [{lo:+.1f}, {hi:+.1f}]{' **headroom**' if ok else ''} | {fb:.2f} -> {fa:.2f} | {nc} |")
        if not ok:
            small3.append(c)
    P(f"\nP3 rule: no headroom in {small3} -> {'STOP behaviour-side work on these levels (>= 3 of 4)' if len(small3) >= 3 else 'continue'}\n")
    # ---- P4 ----
    P("## P4: single-style experts trained at nominal, zero-shot on the levels (against the nominal continuation)\n")
    xb_nom = f35("nominal_X_b0c"); refB = pooled(xb_nom, "S") if xb_nom else None
    P("| style | nominal cost | " + " | ".join(C) + " |"); P("|---|---|" + "---|" * len(C))
    fam_help = set(); cheap_help = False
    for st in ("sty_pp", "sty_pm", "sty_rp", "sty_rm"):
        sn = f35(f"nominal_X_{st}")
        cost = None if sn is None or xb_nom is None else -diff(sn, xb_nom, ref=refB)[0]
        cells = []; any_h = False
        for c in C:
            x, b = f35(f"{c}_X_{st}"), f35(f"{c}_X_b0c")
            if x is None or b is None or refB is None:
                cells.append("-"); continue
            ok, d, lo, hi, fa, fb = helps(x, b, refB, 5.0)
            if ok:
                fam_help.add(c); any_h = True
            cells.append(f"{d:+.1f} % [{lo:+.1f}, {hi:+.1f}], falls {fb:.2f}->{fa:.2f}{' **helps**' if ok else ''}")
        if any_h and cost is not None and cost <= 3.0:
            cheap_help = True
        P(f"| {st} | {'-' if cost is None else f'{cost:.1f} %'} | " + " | ".join(cells) + " |")
    kill = len(fam_help) < 2 or not cheap_help
    P(f"\nP4 rule: levels helped by some style {sorted(fam_help)}; a helpful style with nominal cost <= 3 %: {cheap_help} -> "
      f"{'KILL the head-set idea' if kill else 'pass'}\n")
    # ---- motion detail ----
    P("## Motion detail (pooled): S parts, registered reward, style terms, speed ratio, |vy|, yaw-rate SD, tilt\n")
    P("| run | S | tracking | penalties | style terms | reward | falls | speed ratio | abs vy | yaw-rate SD | gx | gy |")
    P("|---" * 12 + "|")
    for d_, tag in ((a.d34, "34"), (a.d, "35")):
        for fn in sorted(os.listdir(d_)) if os.path.isdir(d_) else []:
            if not (fn.startswith("diag_") and fn.endswith(".npz")):
                continue
            r = load(os.path.join(d_, fn))
            if r is None:
                continue
            P(f"| {tag}:{fn[5:-4]} | {pooled(r, 'S'):.4f} | {pooled(r, 'track'):.4f} | {pooled(r, 'pen'):.4f} | {pooled(r, 'style'):.4f} | "
              f"{pooled(r, 'reward'):.4f} | {pooled(r, 'falls'):.2f} | {pooled(r, 'vratio'):.2f} | {pooled(r, 'vy'):.3f} | "
              f"{pooled(r, 'wz_sd'):.3f} | {pooled(r, 'gx'):+.3f} | {pooled(r, 'gy'):+.3f} |")
    txt = "\n".join(out)
    with open(os.path.join(a.d, "report.md"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
