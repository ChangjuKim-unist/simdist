"""Design 18 section 9 (gate-1-lite): offline selector test on the check-3 branch data.

Inputs per condition: the branch CSV (branch3_gen_<cond>.csv: realised 125-step return / fall of every candidate x clone)
and the feature npz (feat_gen_<cond>.npz from branch_eval.py mode=features: branch-point history, the candidates'
t = 0 chunks and the world model's imagination of them).

Selectors (section 9.3): always A0, fixed best candidate (chosen on the training side), J-hat max (no learning),
S_hist (history + candidate information), S_wm (S_hist + world-model imagination), and the cross-validated upper bound.
Learning (9.4): MLP; loss (a) gain regression + fall BCE, (b) pairwise ranking within a branch point; margin m chosen by
a 2-fold cross-fit inside the training side only. Splits (9.5): within-condition (anchor seeds {11,12,13} vs {14,15,16},
both folds) and leave-one-condition-out. Splits and the margin grid are fixed here.

    python scripts/sim2sim/select_offline.py --dir results/stage18 --out results/stage18/summary_select.md
"""

import argparse
import json
import os
import sys
import time

import csv

import numpy as np
import torch
import torch.nn as nn

CONDS = ["nominal", "motor0.4", "friction0.15", "mass3", "motor0.35", "friction0.1"]
IN_RANGE = ["nominal", "motor0.4", "friction0.15", "mass3"]
FOLD_SEEDS = [[11, 12, 13], [14, 15, 16]]
VAL_FRAC = 0.15  # early-stopping validation = random 15 % of the training-side branch points (fixed rng), used only for stopping
M_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]  # x sd of the training side's A0 return
# candidate order for the candidate-count sensitivity (section 9.5): first 4 / 8 / all 15
CAND_ORDER = ["A0", "G0", "D_vx0.75", "G1", "D_yaw+0.3", "G2", "D_vy+0.15", "G3", "D_vx0.5", "G4", "D_yaw-0.3", "G5", "D_vy-0.15", "G6", "G7"]
E_LABELS = ["E_2026-10-01_10-38-36", "E_2026-10-20_07-40-06"]
GAMMA = 0.99


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, np.float64)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---- data ------------------------------------------------------------------------------------------------------
def load_condition(d, cond, with_e):
    f = np.load(os.path.join(d, f"feat_gen_{cond}.npz"))
    labels = list(f["labels"]); K = len(labels)
    key = list(zip(np.round(f["speed"], 3), f["seed"].astype(int), f["step"].astype(int)))
    kidx = {k: i for i, k in enumerate(key)}
    all_labels = labels + (E_LABELS if with_e else [])
    lidx = {l: i for i, l in enumerate(all_labels)}
    rows = [r for r in csv.DictReader(open(os.path.join(d, f"branch3_gen_{cond}.csv"))) if r["label"] in lidx]
    pi = np.array([kidx.get((round(float(r["speed"]), 3), int(r["seed"]), int(r["step"])), -1) for r in rows]); ok = pi >= 0
    ci = np.array([lidx[r["label"]] for r in rows]); cl = np.array([int(r["clone"]) for r in rows])
    ret = np.array([float(r["ret"]) for r in rows], np.float32); fell = np.array([float(r["fell"]) for r in rows], np.float32)
    J = int(cl.max()) + 1
    n = len(key); KK = len(all_labels)
    R = np.full((n, KK, J), np.nan, np.float32); Fl = np.zeros((n, KK, J), np.float32)
    R[pi[ok], ci[ok], cl[ok]] = ret[ok]; Fl[pi[ok], ci[ok], cl[ok]] = fell[ok]
    keep = ~np.isnan(R).any(axis=(1, 2))
    sel = np.where(keep)[0]
    T = f["rew"].shape[-1]; disc = GAMMA ** np.arange(T)
    # features
    P = f["hist_p"][sel].reshape(len(sel), -1); Ah = f["hist_a"][sel].reshape(len(sel), -1)
    point = np.concatenate([P, Ah, f["cmd"][sel], f["speed"][sel][:, None]], -1).astype(np.float32)
    znow = f["z_now"][sel].astype(np.float32)
    onehot = np.zeros((KK, 4), np.float32)
    for i, l in enumerate(all_labels):
        onehot[i, {"A": 0, "D": 1, "G": 2, "E": 3}[l[0]]] = 1.0
    dcmd = np.zeros((len(sel), KK, 3), np.float32); dcmd[:, :K] = f["cand_cmd"][sel] - f["cmd"][sel][:, None]
    u = np.zeros((len(sel), KK, f["cand_u"].shape[-1]), np.float32); u[:, :K] = f["cand_u"][sel]
    cand = np.concatenate([np.repeat(onehot[None], len(sel), 0), dcmd, u], -1)
    wm = np.zeros((len(sel), KK, T + T + 2 + 64 + 64), np.float32)
    jh = f["jhat"][sel]
    wm[:, :K] = np.concatenate([f["rew"][sel], f["val"][sel], jh[..., None], (jh - jh[:, :1])[..., None], f["z_T"][sel], f["z_mean"][sel]], -1)
    jhat = np.full((len(sel), KK), -np.inf, np.float32); jhat[:, :K] = jh
    return {"cond": cond, "labels": all_labels, "seed": f["seed"][sel].astype(int), "speed": f["speed"][sel], "step": f["step"][sel],
            "R": R[sel], "F": Fl[sel], "point": point, "znow": znow, "cand": cand, "wm": wm, "jhat": jhat}


def concat(parts):
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("seed", "speed", "step", "R", "F", "point", "znow", "cand", "wm", "jhat")}
    out["cond"] = np.concatenate([[p["cond"]] * len(p["seed"]) for p in parts]); out["labels"] = parts[0]["labels"]
    return out


def subset(D, mask):
    out = {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(mask) else v) for k, v in D.items()}
    return out


# ---- model -----------------------------------------------------------------------------------------------------
class Scorer(nn.Module):
    def __init__(self, d_point, d_cand, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_point + d_cand, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, point, cand):  # point (B, dp), cand (B, K, dc) -> gain (B, K), fall logit (B, K)
        B, K, _ = cand.shape
        x = torch.cat([point[:, None].expand(B, K, point.shape[-1]), cand], -1)
        y = self.net(x)
        return y[..., 0], y[..., 1]


def features(D, wm):
    point = np.concatenate([D["point"], D["znow"]], -1) if wm else D["point"]
    cand = np.concatenate([D["cand"], D["wm"]], -1) if wm else D["cand"]
    return point.astype(np.float32), cand.astype(np.float32)


def standardize(train, *others):
    mp, sp = train[0].mean(0), train[0].std(0) + 1e-6
    c = train[1].reshape(-1, train[1].shape[-1]); mc, sc = c.mean(0), c.std(0) + 1e-6
    f = lambda t: ((t[0] - mp) / sp, (t[1] - mc) / sc)  # noqa: E731
    return [f(train)] + [f(o) for o in others]


def targets(D):
    cm = D["R"].mean(-1); gain = cm - cm[:, :1]; fall = (D["F"].mean(-1) >= 0.5).astype(np.float32)
    return gain.astype(np.float32), fall


def train_model(tr, va, loss_kind, seed, dev, epochs=150, patience=12, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    (ptr, ctr), gtr, ftr = tr; (pva, cva), gva, fva = va
    model = Scorer(ptr.shape[-1], ctr.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    Ptr, Ctr, Gtr, Ftr = t(ptr), t(ctr), t(gtr), t(ftr); Pva, Cva, Gva, Fva = t(pva), t(cva), t(gva), t(fva)
    n = len(Ptr); bs = 256; best = (np.inf, None); bad = 0

    def loss_fn(P, C, G, F):
        s, fl = model(P, C)
        if loss_kind == "reg":
            return ((s - G) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(fl, F)
        # pairwise ranking within a branch point: logistic loss on the sign of the gain difference
        ds = s[:, :, None] - s[:, None, :]; dg = G[:, :, None] - G[:, None, :]
        w = (dg.abs() > 1e-6).float()
        return (nn.functional.binary_cross_entropy_with_logits(ds, (dg > 0).float(), reduction="none") * w).sum() / w.sum().clamp(min=1) \
            + nn.functional.binary_cross_entropy_with_logits(fl, F)

    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = loss_fn(Ptr[idx], Ctr[idx], Gtr[idx], Ftr[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(Pva, Cva, Gva, Fva))
        if vl < best[0] - 1e-5:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best[1]); model.eval()
    return model


def predict(models, P, C, dev, calib=None):
    """Mean predicted score (B, K) over models; for ranking models the score is mapped to return units by calib."""
    with torch.no_grad():
        s = np.mean([m(torch.as_tensor(P, device=dev), torch.as_tensor(C, device=dev))[0].cpu().numpy() for m in models], axis=0)
    s = s - s[:, :1]  # relative to A0
    if calib is not None:
        s = calib[0] * s + calib[1]
    return s


def fit_calib(s, gain):
    """1-D least squares from the ranking score (relative to A0) to the realised gain, non-A0 candidates."""
    x = s[:, 1:].ravel(); y = gain[:, 1:].ravel()
    a = float(np.dot(x - x.mean(), y - y.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-9)); b = float(y.mean() - a * x.mean())
    return (max(a, 1e-6), b)


# ---- selection and scoring -------------------------------------------------------------------------------------
def select(score, m, allowed):
    """score (B, K) predicted gain relative to A0 (A0 column = 0); choose argmax over allowed columns if > m else A0."""
    s = np.where(allowed[None], score, -np.inf); s[:, 0] = 0.0
    pick = s.argmax(1); best = s[np.arange(len(s)), pick]
    return np.where(best > m, pick, 0)


def realised(D, pick):
    cm = D["R"].mean(-1); fm = D["F"].mean(-1)
    return cm[np.arange(len(pick)), pick] - cm[:, 0], fm[np.arange(len(pick)), pick], fm[:, 0]


def cv_bound(D, allowed):
    """Section 7's cross-validated upper bound over the allowed candidates: choose by the other clones, score with the held-out clone."""
    R = D["R"]; J = R.shape[-1]; ids = np.where(allowed)[0]; vals = []
    for h in range(J):
        others = [c for c in range(J) if c != h]
        sel = R[:, ids][:, :, others].mean(-1); pick = ids[sel.argmax(1)]
        vals.append(R[np.arange(len(R)), pick, h] - R[:, 0, h])
    return np.mean(vals, axis=0)


def pick_margin(score, D, allowed, sd):
    best, bm = -np.inf, 0.0
    for m in M_GRID:
        g = realised(D, select(score, m * sd, allowed))[0].mean()
        if g > best:
            best, bm = g, m * sd
    return bm


SEEDS = (0, 1, 2)


def run_learned(train, evals, wm, loss_kind, dev, seeds=None, log=print):
    seeds = seeds or SEEDS
    """train: data dict with a 'val_mask' for early stopping; evals: list of data dicts. Returns scores per eval (relative gains in
    return units) and the margin m chosen by the 2-fold cross-fit inside train."""
    gtr, ftr = targets(train)
    Ptr, Ctr = features(train, wm)
    sd = float(train["R"][:, 0].mean(-1).std())
    # cross-fit margin: split the training side by anchor seed (odd / even positions of the sorted seeds)
    seeds_sorted = sorted(set(train["seed"].tolist())); half = {s: i % 2 for i, s in enumerate(seeds_sorted)}
    fold_id = np.array([half[s] for s in train["seed"]])
    allowed_all = np.ones(train["R"].shape[1], bool)
    ms = []; rng = np.random.default_rng(12345)
    for k in (0, 1):
        a = fold_id == k; b = ~a
        va = a & (rng.random(len(a)) < VAL_FRAC)  # validation = random points of the fitting half (early stopping only)
        fit = a & ~va
        (Pf, Cf), (Pv, Cv), (Pb, Cb) = standardize((Ptr[fit], Ctr[fit]), (Ptr[va], Ctr[va]), (Ptr[b], Ctr[b]))
        mdl = train_model(((Pf, Cf), gtr[fit], ftr[fit]), ((Pv, Cv), gtr[va], ftr[va]), loss_kind, seeds[0], dev)
        calib = None
        if loss_kind == "rank":
            calib = fit_calib(predict([mdl], Pf, Cf, dev), gtr[fit])
        sb = predict([mdl], Pb, Cb, dev, calib)
        ms.append(pick_margin(sb, subset(train, b), allowed_all, sd))
    m = float(np.mean(ms))
    # final models on the full training side (validation = train['val_mask'])
    va = train["val_mask"]; fit = ~va
    std = standardize((Ptr[fit], Ctr[fit]), (Ptr[va], Ctr[va]), *[features(e, wm) for e in evals])
    (Pf, Cf), (Pv, Cv) = std[0], std[1]
    models = [train_model(((Pf, Cf), gtr[fit], ftr[fit]), ((Pv, Cv), gtr[va], ftr[va]), loss_kind, s, dev) for s in seeds]
    calib = fit_calib(predict(models, Pf, Cf, dev), gtr[fit]) if loss_kind == "rank" else None
    scores = [predict(models, P, C, dev, calib) for (P, C) in std[2:]]
    return scores, m, sd


def fixed_best(train):
    g, _ = targets(train); mg = g.mean(0); mg[0] = 0.0
    return int(mg.argmax())


def summarize(D, pick, bound):
    g, fsel, f0 = realised(D, pick); a0 = float(D["R"][:, 0].mean())
    out = {"gain": float(g.mean()), "ci": boot_ci(g), "a0": a0, "pct": float(100 * g.mean() / a0) if a0 > 0.5 else np.nan,
           "bound": float(bound.mean()), "rec": float(g.mean() / bound.mean()) if bound.mean() > 1e-9 else np.nan,
           "fall": float(fsel.mean()), "fall_a0": float(f0.mean()), "dev": float((pick != 0).mean()), "n": int(len(pick))}
    return out, g


# ---- experiments -----------------------------------------------------------------------------------------------
def evaluate_split(train, evals, dev, labels, log):
    """train: data with val_mask; evals: list of (name, data). Returns {selector: {eval name: summary}} and per-point gains."""
    K = len(labels); allowed = np.ones(K, bool)
    res = {}; gains = {}
    sets = {"all": allowed}
    for nlim, name in ((4, "4"), (8, "8")):
        a = np.zeros(K, bool)
        for l in CAND_ORDER[:nlim]:
            if l in labels: a[labels.index(l)] = True
        sets[name] = a
    bounds = {n: {s: cv_bound(e, m) for s, m in sets.items()} for n, e in evals}
    # always A0
    for n, e in evals:
        pick = np.zeros(len(e["seed"]), int); res.setdefault("A0", {})[n], gains[("A0", n)] = summarize(e, pick, bounds[n]["all"])
    # fixed best (chosen on train)
    fb = fixed_best(train)
    for n, e in evals:
        pick = np.full(len(e["seed"]), fb); res.setdefault("fixed", {})[n], gains[("fixed", n)] = summarize(e, pick, bounds[n]["all"])
        res["fixed"][n]["label"] = labels[fb]
    # J-hat max (no learning, no margin)
    for n, e in evals:
        for sname, a in sets.items():
            sc = e["jhat"] - e["jhat"][:, :1]; sc[:, 0] = 0.0
            pick = select(sc, -np.inf, a)
            key = "Jhat" if sname == "all" else f"Jhat@{sname}"
            res.setdefault(key, {})[n], gains[(key, n)] = summarize(e, pick, bounds[n][sname])
    # learned
    for wm, fname in ((False, "S_hist"), (True, "S_wm")):
        for loss_kind in ("reg", "rank"):
            t0 = time.time()
            scores, m, sd = run_learned(train, [e for _, e in evals], wm, loss_kind, dev)
            key = f"{fname}({loss_kind})"
            for (n, e), sc in zip(evals, scores):
                for sname, a in sets.items():
                    pick = select(sc, m, a)
                    kk = key if sname == "all" else f"{key}@{sname}"
                    res.setdefault(kk, {})[n], gains[(kk, n)] = summarize(e, pick, bounds[n][sname])
                    res[kk][n]["m"] = m
            log(f"    {key}: m = {m:.3f} ({m / sd:.2f} sd), {time.time() - t0:.0f} s")
    for n, e in evals:
        for sname in sets:
            key = "bound" if sname == "all" else f"bound@{sname}"
            b = bounds[n][sname]
            res.setdefault(key, {})[n] = {"gain": float(b.mean()), "ci": boot_ci(b), "a0": float(e["R"][:, 0].mean()),
                                         "pct": float(100 * b.mean() / e["R"][:, 0].mean()), "bound": float(b.mean()), "rec": 1.0,
                                         "fall": np.nan, "fall_a0": float(e["F"][:, 0].mean()), "dev": np.nan, "n": len(b)}
            gains[(key, n)] = b
    return res, gains


def within_condition(data, dev, labels, log):
    """Both folds; the eval points of both folds are pooled per condition."""
    pooled = {}; gpool = {}
    for k in (0, 1):
        tr_seeds, ev_seeds = FOLD_SEEDS[k], FOLD_SEEDS[1 - k]
        train = concat([subset(d, np.isin(d["seed"], tr_seeds)) for d in data.values()])
        train["val_mask"] = np.random.default_rng(777 + k).random(len(train["seed"])) < VAL_FRAC
        evals = [(c, subset(d, np.isin(d["seed"], ev_seeds))) for c, d in data.items()]
        log(f"  within-condition fold {k}: train seeds {tr_seeds} ({len(train['seed'])} points), eval seeds {ev_seeds}")
        res, gains = evaluate_split(train, evals, dev, labels, log)
        for (sel, n), g in gains.items():
            gpool.setdefault((sel, n), []).append(g)
        for sel, per in res.items():
            for n, s in per.items():
                pooled.setdefault(sel, {}).setdefault(n, []).append(s)
    gcat = {k: np.concatenate(v) for k, v in gpool.items()}
    out = {}
    for sel, per in pooled.items():
        out[sel] = {}
        for n, lst in per.items():
            g = gcat[(sel, n)]; b = gcat[("bound" if "@" not in sel else "bound@" + sel.split("@")[1], n)]
            a0 = float(np.mean([s["a0"] for s in lst]))
            out[sel][n] = {"gain": float(g.mean()), "ci": boot_ci(g), "a0": a0, "pct": float(100 * g.mean() / a0) if a0 > 0.5 else np.nan,
                           "bound": float(b.mean()), "rec": float(g.mean() / b.mean()) if b.mean() > 1e-9 else np.nan,
                           "fall": float(np.nanmean([s["fall"] for s in lst])), "fall_a0": float(np.mean([s["fall_a0"] for s in lst])),
                           "dev": float(np.nanmean([s["dev"] for s in lst])), "n": int(len(g)), "label": lst[0].get("label", ""), "m": lst[0].get("m", np.nan)}
    return out, gcat


def leave_one_out(data, dev, labels, log):
    out = {}; gpool = {}
    for c in CONDS:
        train = concat([d for cc, d in data.items() if cc != c])
        train["val_mask"] = np.random.default_rng(999).random(len(train["seed"])) < VAL_FRAC
        log(f"  leave-out {c}: train {len(train['seed'])} points")
        res, gains = evaluate_split(train, [(c, data[c])], dev, labels, log)
        for sel, per in res.items():
            out.setdefault(sel, {})[c] = per[c]
        for k, g in gains.items():
            gpool[k] = g
    return out, gpool


# ---- report ----------------------------------------------------------------------------------------------------
def fmt_cell(s):
    if s is None: return "—"
    if np.isnan(s.get("pct", np.nan)):
        return f"{s['gain']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]"
    return f"{s['pct']:+.1f} % [{100 * s['ci'][0] / s['a0']:+.1f}, {100 * s['ci'][1] / s['a0']:+.1f}]"


ROWS = [("A0", "항상 A0"), ("fixed", "고정 최선 후보"), ("Jhat", "Ĵ 최대"), ("S_hist(reg)", "S_hist (회귀)"), ("S_hist(rank)", "S_hist (순위)"),
        ("S_wm(reg)", "S_wm (회귀)"), ("S_wm(rank)", "S_wm (순위)"), ("bound", "정답 상한 (cv)")]


def table(res, title, note):
    L = [f"#### {title}", "", note, "",
         "| 선택기 | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for key, name in ROWS:
        if key not in res: continue
        L.append(f"| {name} | " + " | ".join(fmt_cell(res[key].get(c)) for c in CONDS) + " |")
    L += ["", "회수율 (고른 후보의 이득 / 상한, 1이 최대) · 넘어짐 (고른 후보 / A0, 작을수록 좋음) · A0에서 벗어난 비율:", "",
          "| 선택기 | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for key, name in ROWS[1:-1]:
        if key not in res: continue
        cells = []
        for c in CONDS:
            s = res[key].get(c)
            cells.append("—" if s is None else f"회수 {s['rec']:.2f} · 넘어짐 {s['fall']:.2f}/{s['fall_a0']:.2f} · 벗어남 {s['dev']:.2f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L


def table_sens(res, title):
    L = [f"#### {title}", "", "고른 후보의 이득 (A0 대비 %, 클수록 좋음) / 그 집합의 상한; 후보 집합 = 고정 순서의 앞 4 / 8 / 전체 15", "",
         "| 선택기 | 후보 수 | " + " | ".join(CONDS) + " |", "|---|---|" + "---|" * len(CONDS)]
    for key, name in (("Jhat", "Ĵ 최대"), ("S_hist(reg)", "S_hist (회귀)"), ("S_wm(reg)", "S_wm (회귀)")):
        for sname in ("4", "8", "all"):
            k = key if sname == "all" else f"{key}@{sname}"; bk = "bound" if sname == "all" else f"bound@{sname}"
            if k not in res: continue
            cells = []
            for c in CONDS:
                s, b = res[k].get(c), res[bk].get(c)
                cells.append("—" if s is None else (f"{s['pct']:+.1f} / {b['pct']:+.1f} %" if not np.isnan(s["pct"]) else f"{s['gain']:+.2f} / {b['gain']:+.2f}"))
            L.append(f"| {name} | {sname if sname != 'all' else 15} | " + " | ".join(cells) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/stage18"); ap.add_argument("--out", default="results/stage18/summary_select.md")
    ap.add_argument("--json", default=None); ap.add_argument("--no-appendix", action="store_true")
    ap.add_argument("--conds", default=None, help="comma list (tests)"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gains", default=None, help="npz of per-point gains of every selector (main part)")
    args = ap.parse_args()
    global CONDS, SEEDS
    if args.conds:
        CONDS = args.conds.split(",")
    SEEDS = tuple(range(args.seeds))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda s: print(s, flush=True)  # noqa: E731
    log(f"device {dev}")
    out = ["## 설계 18 §9 관문 1-축소판: 오프라인 선택기 판정 (확인 3 분기 데이터, 생성기 기본)", ""]
    allres = {}
    for with_e, tag in ((False, "본문 (후보 A0 + D 6 + G 8)"), (True, "부록 (후보에 E 다른 전문가 2 추가; E는 종류 원핫만, 상상 특징 없음)")):
        if with_e and args.no_appendix: break
        log(f"=== {tag}")
        data = {c: load_condition(args.dir, c, with_e) for c in CONDS}
        labels = data[CONDS[0]]["labels"]
        for c in CONDS:
            log(f"  {c}: {len(data[c]['seed'])} branch points, {len(labels)} candidates")
        log(" within-condition")
        r1, g1 = within_condition(data, dev, labels, log)
        log(" leave-one-condition-out")
        r2, g2 = leave_one_out(data, dev, labels, log)
        allres[tag] = {"within": r1, "loco": r2}
        out += [f"### {tag}", ""]
        out += table(r1, "표 1. 조건 내 분할 (앵커 시드 {11,12,13} ↔ {14,15,16} 2겹, 평가 쪽 점 합침)",
                     "고른 후보의 실현 125스텝 수익 − A0, A0 평균 대비 % [분기점 부트스트랩 95 % CI], 클수록 좋음. A0 < 0.5인 칸은 절대값.")
        out += [""]
        out += table(r2, "표 2. 조건 빼기 (5조건 학습 → 빼 둔 1조건 평가)", "같은 지표. 열 = 빼 둔(평가) 조건.")
        out += [""]
        out += table_sens(r1, "표 3. 후보 수 민감도 (조건 내 분할)")
        out += [""]
        out += table_sens(r2, "표 4. 후보 수 민감도 (조건 빼기)")
        out += [""]
        # margins and fixed labels
        out += ["여유 m (수익 단위, 학습 쪽 교차 적합으로 고름; 조건 빼기, 빼 둔 조건별): " + ", ".join(
            f"{c}: " + "/".join(f"{r2[k][c]['m']:.2f}" for k in ("S_hist(reg)", "S_hist(rank)", "S_wm(reg)", "S_wm(rank)") if k in r2 and c in r2[k]) for c in CONDS), ""]
        out += ["고정 최선 후보 (조건 빼기, 학습 쪽에서 고른 것): " + ", ".join(f"{c}: {r2['fixed'][c]['label']}" for c in CONDS if c in r2["fixed"]), ""]
        for nm, gp in (("조건 내 분할", g1), ("조건 빼기", g2)):
            for lk in ("reg", "rank"):
                cells = []
                for c in CONDS:
                    a, b = gp.get((f"S_wm({lk})", c)), gp.get((f"S_hist({lk})", c))
                    if a is None or b is None: cells.append("—"); continue
                    dlt = a - b; ci = boot_ci(dlt); a0 = r1["A0"][c]["a0"] if nm == "조건 내 분할" else r2["A0"][c]["a0"]
                    cells.append(f"{c} {100 * dlt.mean() / a0:+.1f} % [{100 * ci[0] / a0:+.1f}, {100 * ci[1] / a0:+.1f}]")
                out += [f"S_wm − S_hist ({nm}, {'회귀' if lk == 'reg' else '순위'}; 같은 분기점 짝지은 차이, A0 대비 %, CI가 0을 넘어야 상상이 기여): " + ", ".join(cells), ""]
        if args.gains:
            np.savez(args.gains, **{f"{sel}|{c}": g for (sel, c), g in g1.items()}, **{f"loco|{sel}|{c}": g for (sel, c), g in g2.items()})
        if not with_e:
            # predictions
            def rec(r, k, c):
                s = r[k].get(c); return np.nan if s is None else s["rec"]
            p1 = rec(r1, "S_wm(reg)", "friction0.1"); p1b = rec(r1, "S_wm(rank)", "friction0.1")
            p2a = rec(r2, "S_wm(reg)", "friction0.1"); p2a2 = rec(r2, "S_wm(rank)", "friction0.1")
            p2b = rec(r2, "S_hist(reg)", "friction0.1"); p2b2 = rec(r2, "S_hist(rank)", "friction0.1")
            p3 = sum(1 for c in CONDS if r2["Jhat"][c]["gain"] <= 0); n3 = len(CONDS)
            p4 = [(c, r2["S_wm(reg)"][c]["gain"], r2["S_wm(reg)"][c]["rec"]) for c in IN_RANGE if c in r2["S_wm(reg)"]]
            out += ["**예측 판정용 수치**", f"- P1 (조건 내, friction0.1, S_wm 회수율): 회귀 {p1:.2f}, 순위 {p1b:.2f} (기준 ≥ 0.5)",
                    f"- P2 (조건 빼기, friction0.1 회수율): S_wm 회귀 {p2a:.2f} / 순위 {p2a2:.2f}, S_hist 회귀 {p2b:.2f} / 순위 {p2b2:.2f} (기준 ≥ 1/3)",
                    f"- P3 (조건 빼기, Ĵ 최대 ≤ A0인 칸): {p3}/{n3} (기준 과반)",
                    "- P4 (조건 빼기, 범위 안, S_wm 회귀 이득·회수율): " + ", ".join(f"{c} {g:+.3f} (회수 {r:.2f})" for c, g, r in p4), ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(out) + "\n")
    if args.json:
        json.dump(allres, open(args.json, "w"), indent=1, default=float)
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
