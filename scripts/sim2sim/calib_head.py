"""Design doc 15: calibrated planner score.  Train a small head that maps
    (history: realised latents z_{t-24..t} + executed actions a_{t-25..t-1},
     candidate: imagined rewards r^_{0..24}, final imagined value V^_24, promised return J^, imagined latents mean/last)
to quantiles (0.1 / 0.5 / 0.9) of the REALISED discounted 25-step return (+ the model's value at the state actually reached) and the probability of a fall within 100 steps,
from the self-rollout traces of design doc 14 (simulate_batch output.trace_npz).  Reports the pre-registered C1 metrics
(Spearman of realised return vs q50 / q10 / J^ on held-out seeds, in-distribution and on the held-out axis) and saves the
weights for simdist/control/mppi_batch.py (control.calib.weights).

    python scripts/sim2sim/calib_head.py --traces "/nas_data/cjkim/simdist/traces_stage13/T_broadwm_*.npz" \
        "/nas_data/cjkim/simdist/traces_stage13/G_broadwm_*.npz" --holdout_axis friction --out results/stage14/head_broadwm_noF
"""
import argparse
import glob
import json
import os

import numpy as np

W = 25          # history window (latents z_{t-W+1..t}, actions a_{t-W..t-1})
H_RET = 25      # realised-return horizon = model horizon
K_FALL = 100
GAMMA = 0.99
TAUS = (0.1, 0.5, 0.9)


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def samples(d, stride=5):
    """Per decision step t: history features, candidate (executed plan) features, labels.  Only steps whose
    history window lies in one episode and contains no resetting step are kept."""
    z = d["z"].astype(np.float32)                  # (S, N, D)  z[t] = E(o_t)
    act = d["action"].astype(np.float32)           # (S, N, A)  executed a_t
    pl = d["pred_latents"].astype(np.float32)      # (S, N, T, D) imagined latents of the executed plan
    pr = d["pred_rewards"].astype(np.float32)      # (S, N, T)
    pv = d["pred_values"].astype(np.float32)       # (S, N, T)
    r = d["reward"].astype(np.float32)             # (S, N)
    fall, reset, resetting = d["fall"], d["reset"], d["resetting"]
    S, N, D = z.shape
    T = pr.shape[2]
    disc = GAMMA ** np.arange(T)
    jhat = (pr * disc).sum(-1) + pv[:, :, -1] * GAMMA ** (T + 1)      # (S, N) promised return as MPPI scores it
    ep = np.concatenate([np.zeros((1, N), np.int64), np.cumsum(reset, axis=0)[:-1]])
    Xh, Xc, G, F, keep = [], [], [], [], []
    for t in range(W, S - K_FALL, stride):
        valid = (ep[t - W] == ep[t]) & ~resetting[t - W:t + 1].any(axis=0)
        # realised discounted return; rewards after a reset inside the horizon count as 0 (the fall ends the episode)
        fut_r = r[t:t + H_RET]                                            # (H, N)
        cut = np.cumsum(reset[t:t + H_RET], axis=0) > 0                   # True from the first reset on
        g = (np.where(cut, 0.0, fut_r) * disc[:H_RET, None]).sum(0)
        # terminal value at the state actually reached (the model's own 1-step value from z_{t+H}), so that the label
        # has the same long-horizon term as J^ and the calibrated planner is not myopic; 0 if the episode ended
        v_real = np.where(cut[-1], 0.0, pv[t + H_RET, :, 0])
        g = g + GAMMA ** (T + 1) * v_real
        fut_reset = reset[t:t + K_FALL]; fut_fall = fall[t:t + K_FALL]
        any_reset = fut_reset.any(axis=0)
        first = np.where(any_reset, fut_reset.argmax(axis=0), 0)
        is_fall = fut_fall[first, np.arange(N)] & any_reset
        drop = any_reset & ~is_fall                                       # time-out inside the fall window: ambiguous
        hist = np.concatenate([z[t - W + 1:t + 1].transpose(1, 0, 2).reshape(N, -1),
                               act[t - W:t].transpose(1, 0, 2).reshape(N, -1)], 1)
        cand = np.concatenate([pr[t], pv[t, :, -1:], jhat[t][:, None], pl[t].mean(1), pl[t][:, -1]], 1)
        Xh.append(hist); Xc.append(cand); G.append(g); F.append(is_fall.astype(np.float32)); keep.append(valid & ~drop)
    Xh = np.concatenate(Xh).astype(np.float16); Xc = np.concatenate(Xc).astype(np.float32)
    G = np.concatenate(G); F = np.concatenate(F); keep = np.concatenate(keep)
    seeds = np.tile(d["seeds"], len(G) // N)
    return Xh[keep], Xc[keep], G[keep], F[keep], seeds[keep]


def spearman(a, b):
    def rank(x):
        o = np.argsort(x); rk = np.empty(len(x)); rk[o] = np.arange(len(x)); return rk
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(y, s):
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    pos = y > 0.5; n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train(X, G, F, Xva, seed=0, epochs=20, hidden=256, lr=1e-3):
    import jax, jax.numpy as jnp, optax
    mu, sd = X.mean(0), X.std(0) + 1e-6
    g_mu, g_sd = float(G.mean()), float(G.std() + 1e-6)
    Xn = jnp.asarray((X - mu) / sd, jnp.float32); Gn = jnp.asarray((G - g_mu) / g_sd); Fj = jnp.asarray(F)
    key = jax.random.PRNGKey(seed); d_in = X.shape[1]; h = hidden
    k1, k2, k3 = jax.random.split(key, 3)
    params = {"w1": jax.random.normal(k1, (d_in, h)) / np.sqrt(d_in), "b1": jnp.zeros(h),
              "w2": jax.random.normal(k2, (h, h)) / np.sqrt(h), "b2": jnp.zeros(h),
              "w3": jax.random.normal(k3, (h, 4)) / np.sqrt(h), "b3": jnp.zeros(4)}
    taus = jnp.asarray(TAUS)

    def f(p, x):
        hh = jax.nn.relu(x @ p["w1"] + p["b1"]); hh = jax.nn.relu(hh @ p["w2"] + p["b2"]); return hh @ p["w3"] + p["b3"]

    def loss(p, x, g, fl):
        o = f(p, x)
        u = g[:, None] - o[:, :3]
        pin = jnp.mean(jnp.maximum(taus * u, (taus - 1) * u))
        bce = optax.sigmoid_binary_cross_entropy(o[:, 3], fl).mean()
        return pin + bce

    opt = optax.adamw(lr, weight_decay=1e-4); st = opt.init(params)

    @jax.jit
    def step(p, s, x, g, fl):
        gr = jax.grad(loss)(p, x, g, fl); u, s = opt.update(gr, s, p); return optax.apply_updates(p, u), s

    n = len(X); bs = 2048; rng = np.random.default_rng(seed)
    for e in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; params, st = step(params, st, Xn[b], Gn[b], Fj[b])
    weights = {k: np.asarray(v) for k, v in params.items()}
    weights.update(mu=mu.astype(np.float32), sd=sd.astype(np.float32), g_mu=np.float32(g_mu), g_sd=np.float32(g_sd))

    def predict(Xq):
        out = np.asarray(f(params, jnp.asarray((Xq - mu) / sd, jnp.float32)))
        q = out[:, :3] * g_sd + g_mu
        return q, 1 / (1 + np.exp(-out[:, 3]))
    return weights, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--holdout_axis", default=None, help="substring of condition names to hold out (friction / motor / mass)")
    ap.add_argument("--test_seeds", default="15,16")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True, help="output prefix: <out>.npz weights, <out>.json metrics")
    args = ap.parse_args()
    test_seeds = {int(s) for s in args.test_seeds.split(",")}
    paths = sorted(sum([glob.glob(t) for t in args.traces], []))
    assert paths, "no traces"
    data = {}
    for p in paths:
        d = load(p); key = (str(d["model"]), str(d["condition"]))
        data[key] = samples(d, args.stride)
        print(f"{os.path.basename(p)}: {len(data[key][2])} samples, fall rate {data[key][3].mean():.3f}", flush=True)

    def held(cond):
        return args.holdout_axis is not None and args.holdout_axis in cond

    def stack(items, sel):
        Xh = np.concatenate([v[0][sel(v)] for v in items]); Xc = np.concatenate([v[1][sel(v)] for v in items])
        return (np.concatenate([Xh.astype(np.float32), Xc], 1), np.concatenate([v[2][sel(v)] for v in items]),
                np.concatenate([v[3][sel(v)] for v in items]))

    train_items = [v for k, v in data.items() if not held(k[1])]
    hold_items = [v for k, v in data.items() if held(k[1])]
    Xtr, Gtr, Ftr = stack(train_items, lambda v: ~np.isin(v[4], list(test_seeds)))
    Xva, Gva, Fva = stack(train_items, lambda v: np.isin(v[4], list(test_seeds)))
    print(f"train {len(Xtr)} samples ({Xtr.shape[1]} features), in-dist val {len(Xva)}", flush=True)
    weights, predict = train(Xtr, Gtr, Ftr, Xva, epochs=args.epochs)
    n_hist = train_items[0][0].shape[1]
    weights.update(n_hist=np.int64(n_hist), gamma=np.float32(GAMMA), taus=np.asarray(TAUS, np.float32))
    np.savez(args.out + ".npz", **weights)

    def report(X, G, F, name):
        q, pf = predict(X)
        jhat = X[:, n_hist + H_RET + 1]          # promised return column of the candidate block
        res = {"n": int(len(G)), "spearman_q50": spearman(G, q[:, 1]), "spearman_q10": spearman(G, q[:, 0]),
               "spearman_jhat": spearman(G, jhat), "fall_auc": auc(F, pf),
               "coverage_q10": float((G >= q[:, 0]).mean()), "coverage_q90": float((G <= q[:, 2]).mean()),
               "mean_gap_jhat": float((jhat - G).mean()), "mean_gap_q50": float((q[:, 1] - G).mean()),
               "mse_ratio_q50_vs_jhat": float(np.mean((G - q[:, 1]) ** 2) / np.mean((G - jhat) ** 2)),
               "pinball10_ratio": float(np.mean(np.maximum(0.1 * (G - q[:, 0]), -0.9 * (G - q[:, 0]))) /
                                        np.mean(np.maximum(0.1 * (G - jhat), -0.9 * (G - jhat))))}
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        return res

    metrics = {"holdout_axis": args.holdout_axis, "train_conditions": sorted({k[1] for k in data if not held(k[1])}),
               "in_dist_val": report(Xva, Gva, Fva, "in-dist val")}
    if hold_items:
        Xho, Gho, Fho = stack(hold_items, lambda v: np.ones(len(v[2]), bool))
        metrics["holdout"] = report(Xho, Gho, Fho, "holdout axis")
        metrics["holdout_per_condition"] = {}
        for k, v in data.items():
            if held(k[1]):
                X, G, F = stack([v], lambda v: np.ones(len(v[2]), bool))
                metrics["holdout_per_condition"][f"{k[0]}:{k[1]}"] = report(X, G, F, f"  {k[0]} {k[1]}")
    json.dump(metrics, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
