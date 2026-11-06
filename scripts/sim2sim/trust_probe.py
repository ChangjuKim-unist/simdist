"""Design doc 14 stage 0, S0-2: can a window of the model's OWN prediction errors (imagined - realized)
predict (i) a fall within K steps, (ii) the realized-minus-predicted 25-step return, (iii) the physics condition,
better than a window of raw state/action history?  Inputs: trace .npz files written by simulate_batch
(output.trace_npz).  Three probe inputs over a window of W steps ending at t:
  hist  : realized latents z_{t-W+1..t}  (what a history-conditioned model already sees, in latent form)
  err   : one-step and k-step prediction errors  pred_latents[t-k, :, k-1] - z[t]  for k=1..K_ERR, plus reward errors
  both  : hist + err
Split by seed (held-out seeds), train on a mix of conditions; optional held-out condition (--holdout) for generalisation.
Usage: python trust_probe.py --traces /data1/cjkim/stage13/T_broadwm_prior_*.npz --holdout mass3 --out probe.json
"""
import argparse, glob, json, numpy as np

def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}

def features(d, W=25, K_ERR=10, K_FALL=100, H_RET=25):
    z = d["z"].astype(np.float32)                 # (S, N, D)
    pl = d["pred_latents"].astype(np.float32)     # (S, N, T, D)
    pr = d["pred_rewards"]; r = d["reward"]       # (S, N, T), (S, N)
    fall = d["fall"]; reset = d["reset"]; resetting = d["resetting"]
    S, N, D = z.shape
    X_hist, X_err, y_fall, y_gap, keep = [], [], [], [], []
    # episode id per step to avoid crossing resets
    ep = np.cumsum(reset, axis=0)
    for t in range(W + K_ERR, S - K_FALL):
        same = (ep[t - W - K_ERR] == ep[t]) & (~resetting[t]) & (ep[t + K_FALL] == ep[t]) | ((ep[t - W - K_ERR] == ep[t]) & (~resetting[t]) & fall[t:t + K_FALL].any(axis=0))
        hist = z[t - W + 1:t + 1].transpose(1, 0, 2).reshape(N, -1)
        errs = [pl[t - k, :, k - 1] - z[t] for k in range(1, K_ERR + 1)]                 # (N, D) each
        rerr = [pr[t - k, :, k - 1] - r[t] for k in range(1, K_ERR + 1)]                 # (N,)
        err = np.concatenate([np.stack(errs, 1).reshape(N, -1), np.stack(rerr, 1)], 1)
        f = fall[t + 1:t + 1 + K_FALL].any(axis=0).astype(np.float32)
        # realized vs predicted 25-step return (undiscounted), only if no reset inside
        ok = ep[t + H_RET] == ep[t]
        gap = np.where(ok, r[t + 1:t + 1 + H_RET].sum(0) - pr[t].sum(1), np.nan)
        X_hist.append(hist); X_err.append(err); y_fall.append(f); y_gap.append(gap); keep.append(same)
    X_hist = np.concatenate(X_hist); X_err = np.concatenate(X_err); y_fall = np.concatenate(y_fall)
    y_gap = np.concatenate(y_gap); keep = np.concatenate(keep)
    seeds = np.tile(d["seeds"], len(y_fall) // N)
    return X_hist[keep], X_err[keep], y_fall[keep], y_gap[keep], seeds[keep]

def mlp_fit(X, y, task, Xte, seed=0, epochs=30):
    import jax, jax.numpy as jnp, optax
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = jnp.asarray((X - mu) / sd); Xt = jnp.asarray((Xte - mu) / sd); yj = jnp.asarray(y)
    key = jax.random.PRNGKey(seed); d_in = X.shape[1]; h = 256
    k1, k2 = jax.random.split(key)
    params = {"w1": jax.random.normal(k1, (d_in, h)) / np.sqrt(d_in), "b1": jnp.zeros(h), "w2": jax.random.normal(k2, (h, 1)) / np.sqrt(h), "b2": jnp.zeros(1)}
    def f(p, x): return (jax.nn.relu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"])[:, 0]
    def loss(p, x, yy):
        o = f(p, x)
        if task == "cls": return optax.sigmoid_binary_cross_entropy(o, yy).mean()
        return jnp.mean((o - yy) ** 2)
    opt = optax.adamw(1e-3, weight_decay=1e-4); st = opt.init(params)
    @jax.jit
    def step(p, s, x, yy):
        g = jax.grad(loss)(p, x, yy); u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s
    n = len(X); bs = 1024; rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; params, st = step(params, st, Xn[b], yj[b])
    return np.asarray(f(params, Xt))

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    pos = y > 0.5; n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traces", nargs="+", required=True); ap.add_argument("--holdout", default=None)
    ap.add_argument("--test_seeds", default="15,16"); ap.add_argument("--out", default=None); args = ap.parse_args()
    test_seeds = {int(s) for s in args.test_seeds.split(",")}
    data = {}
    for p in sorted(sum([glob.glob(t) for t in args.traces], [])):
        d = load(p); cond = str(d["condition"]); name = str(d["model"])
        data[(name, cond)] = features(d)
        print(p, "windows", len(data[(name, cond)][2]), "fall rate", float(data[(name, cond)][2].mean()), flush=True)
    res = {}
    conds = sorted({c for _, c in data})
    for inp in ("hist", "err", "both"):
        def getX(v): return {"hist": v[0], "err": v[1], "both": np.concatenate([v[0], v[1]], 1)}[inp]
        tr = [(k, v) for k, v in data.items() if k[1] != args.holdout]
        Xtr = np.concatenate([getX(v)[~np.isin(v[4], list(test_seeds))] for _, v in tr])
        ytr = np.concatenate([v[2][~np.isin(v[4], list(test_seeds))] for _, v in tr])
        gtr = np.concatenate([v[3][~np.isin(v[4], list(test_seeds))] for _, v in tr])
        for k, v in data.items():
            te = np.isin(v[4], list(test_seeds)) if k[1] != args.holdout else np.ones(len(v[2]), bool)
            Xte, yte, gte = getX(v)[te], v[2][te], v[3][te]
            s = mlp_fit(Xtr, ytr, "cls", Xte)
            m = ~np.isnan(gtr); mt = ~np.isnan(gte)
            g = mlp_fit(Xtr[m], gtr[m], "reg", Xte[mt]) if mt.sum() > 10 else None
            res[f"{inp}|{k[0]}|{k[1]}"] = {"fall_auc": auc(yte, s), "n": int(len(yte)), "fall_rate": float(yte.mean()),
                                           "gap_r2": r2(gte[mt], g) if g is not None else None, "holdout": k[1] == args.holdout}
            print(inp, k, res[f"{inp}|{k[0]}|{k[1]}"], flush=True)
    if args.out: json.dump(res, open(args.out, "w"), indent=1)

if __name__ == "__main__":
    main()
