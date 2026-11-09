"""Design doc 16: conditional VAE over 25-step action chunks (the generative prior for MPPI candidates).
Condition c = [proprio history (H+1, o_p), action history (H, A), future commands (T+1, cmd)] normalised with the
dataset scaler; chunk = next T actions (normalised).  Trained on expert-flagged windows of the broad DR dataset.
Reports the pre-registered G0 metrics on held-out windows and saves the decoder for simdist/control/mppi_batch.py
(control.gen.weights).

    python scripts/sim2sim/gen_chunks.py --data datasets/sim/2026-11-05_11-39-08/processed_data_go1dr_H-25_T-25 --out results/stage15/gen_broad
"""
import argparse
import json
import os

import h5py
import numpy as np

H = 25; T = 25


def load_windows(d, n_blocks, block, stride, seed):
    from simdist.data import DATA_KEY
    f = {k: h5py.File(os.path.join(d, k + ".hdf5"), "r")[DATA_KEY] for k in ("proprio_obs", "acts", "cmds", "expert_policy_flag")}
    # episode-boundary source: terminal.hdf5 (steps-to-end countdown, DR datasets) or start_idxs.hdf5 (valid window starts
    # t_H of the training loader, paper-style datasets without terminal.hdf5)
    has_term = os.path.exists(os.path.join(d, "terminal.hdf5"))
    if has_term:
        f["terminal"] = h5py.File(os.path.join(d, "terminal.hdf5"), "r")[DATA_KEY]
    else:
        start_idxs = h5py.File(os.path.join(d, "start_idxs.hdf5"), "r")[DATA_KEY][:].ravel()
    sp = json.load(open(os.path.join(d, "scaler_params.json")))
    st = {k: (np.asarray(sp[k]["mean"], np.float32), np.asarray(sp[k]["std"], np.float32) + 1e-8) for k in ("proprio_obs", "actions", "commands")}
    N = f["acts"].shape[0]; rng = np.random.default_rng(seed)
    starts = np.sort(rng.integers(H, N - block - T - 1, n_blocks))
    C, X = [], []
    for b in starts:
        lo, hi = b - H, b + block + T + 1
        p = (f["proprio_obs"][lo:hi] - st["proprio_obs"][0]) / st["proprio_obs"][1]
        a = (f["acts"][lo:hi] - st["actions"][0]) / st["actions"][1]
        c = (f["cmds"][lo:hi] - st["commands"][0]) / st["commands"][1]
        fl = f["expert_policy_flag"][lo:hi]
        if has_term:
            ste = f["terminal"][lo:hi][:, 0]
            same_ep = lambda t: ste[t - H] - ste[t + T] == H + T
        else:
            sl = start_idxs[np.searchsorted(start_idxs, lo):np.searchsorted(start_idxs, hi)] - lo
            ok = np.zeros(hi - lo, bool); ok[sl] = True       # ok[j]: a window with t_H = lo + j stays inside one episode
            same_ep = lambda t: bool(ok[t - H])
        for t in range(H, H + block, stride):
            if not fl[t:t + T].all() or not same_ep(t):
                continue
            C.append(np.concatenate([p[t - H:t + 1].ravel(), a[t - H:t].ravel(), c[t:t + T + 1].ravel()]))
            X.append(a[t:t + T].ravel())
    return np.asarray(C, np.float32), np.asarray(X, np.float32), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_blocks", type=int, default=5000); ap.add_argument("--block", type=int, default=500); ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--du", type=int, default=16); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    C, X, st = load_windows(args.data, args.n_blocks, args.block, args.stride, args.seed)
    print(f"windows {len(C)}, cond dim {C.shape[1]}, chunk dim {X.shape[1]}", flush=True)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(C)); n_va = min(20000, len(C) // 20)
    va, tr = perm[:n_va], perm[n_va:]
    import jax, jax.numpy as jnp, optax
    dc, dx, du = C.shape[1], X.shape[1], args.du
    k = jax.random.split(jax.random.PRNGKey(args.seed), 8)
    def lin(key, i, o): return jax.random.normal(key, (i, o)) * np.sqrt(2.0 / i)
    params = {"e1": lin(k[0], dx + dc, 512), "eb1": jnp.zeros(512), "e2": lin(k[1], 512, 256), "eb2": jnp.zeros(256),
              "emu": lin(k[2], 256, du) * 0.1, "ebmu": jnp.zeros(du), "elv": lin(k[3], 256, du) * 0.1, "eblv": jnp.zeros(du),
              "d1": lin(k[4], du + dc, 512), "db1": jnp.zeros(512), "d2": lin(k[5], 512, 512), "db2": jnp.zeros(512),
              "d3": lin(k[6], 512, dx) * 0.1, "db3": jnp.zeros(dx), "dlog_sig": jnp.full(dx, -1.0)}

    def enc(p, x, c):
        h = jax.nn.relu(jnp.concatenate([x, c], -1) @ p["e1"] + p["eb1"]); h = jax.nn.relu(h @ p["e2"] + p["eb2"])
        return h @ p["emu"] + p["ebmu"], h @ p["elv"] + p["eblv"]

    def dec(p, u, c):
        h = jax.nn.relu(jnp.concatenate([u, c], -1) @ p["d1"] + p["db1"]); h = jax.nn.relu(h @ p["d2"] + p["db2"])
        return h @ p["d3"] + p["db3"]

    def loss(p, x, c, key, beta):
        mu, lv = enc(p, x, c); u = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
        m = dec(p, u, c); ls = jnp.clip(p["dlog_sig"], -3.0, 1.0)
        nll = 0.5 * jnp.sum(((x - m) / jnp.exp(ls)) ** 2 + 2 * ls, -1).mean()
        kl = 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1 - lv, -1).mean()
        return nll + beta * kl, (nll, kl)

    opt = optax.adamw(1e-3, weight_decay=1e-5); stt = opt.init(params)

    @jax.jit
    def step(p, s, x, c, key, beta):
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p, x, c, key, beta); upd, s = opt.update(g, s, p); return optax.apply_updates(p, upd), s, l, aux

    Cj, Xj = jnp.asarray(C), jnp.asarray(X); bs = 1024; key = jax.random.PRNGKey(args.seed + 1)
    for e in range(args.epochs):
        beta = args.beta * min(1.0, (e + 1) / 3.0); idx = rng.permutation(tr); tot = []
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; key, sk = jax.random.split(key)
            params, stt, l, (nll, kl) = step(params, stt, Xj[b], Cj[b], sk, beta); tot.append((float(l), float(nll), float(kl)))
        m = np.mean(tot, 0); print(f"epoch {e + 1} loss {m[0]:.2f} nll {m[1]:.2f} kl {m[2]:.2f} beta {beta:.2f}", flush=True)

    # ---- G0 metrics on held-out windows -----------------------------------------------------------------------
    p = params; xv, cv = Xj[va], Cj[va]
    mu, lv = enc(p, xv, cv); rec = dec(p, mu, cv); zero = dec(p, jnp.zeros_like(mu), cv)
    var = float(jnp.var(xv)); r2_post = 1 - float(jnp.mean((xv - rec) ** 2)) / var; r2_zero = 1 - float(jnp.mean((xv - zero) ** 2)) / var
    a_std = jnp.asarray(np.tile(st["actions"][1], T)); a_mean = jnp.asarray(np.tile(st["actions"][0], T))
    sel = va[:200]; S = 450; key = jax.random.PRNGKey(7)
    div, d_gen, d_03, d_20 = [], [], [], []
    for j in sel:
        c = jnp.repeat(Cj[j][None], S, 0); key, k1, k2, k3 = jax.random.split(key, 4)
        u = jax.random.normal(k1, (S, du)); gen = dec(p, u, c) * a_std + a_mean                       # raw units
        g0 = dec(p, jnp.zeros((1, du)), c[:1]) * a_std + a_mean
        n03 = g0 + 0.3 * jax.random.normal(k2, (S, dx)); n20 = g0 + 2.0 * jax.random.normal(k3, (S, dx))
        true = Xj[j] * a_std + a_mean
        div.append(float(jnp.std(gen, 0).mean()))
        for lst, cand in ((d_gen, gen), (d_03, n03), (d_20, n20)):
            lst.append(float(jnp.sqrt(jnp.mean((cand - true) ** 2, -1)).min()))
    res = {"n_windows": int(len(C)), "n_val": int(n_va), "r2_posterior": r2_post, "r2_u0": r2_zero, "u_extra_r2": r2_post - r2_zero,
           "sample_std_raw": float(np.mean(div)), "nearest_rms_gen": float(np.mean(d_gen)), "nearest_rms_noise0.3": float(np.mean(d_03)),
           "nearest_rms_noise2.0": float(np.mean(d_20)), "dec_sigma_mean": float(jnp.exp(jnp.clip(p["dlog_sig"], -3, 1)).mean()),
           "G0a": bool(r2_post >= 0.6 and r2_post - r2_zero >= 0.1), "G0b": bool(np.mean(div) >= 0.1),
           "G0c": bool(np.mean(d_gen) < np.mean(d_03) < np.mean(d_20))}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    json.dump(res, open(args.out + ".json", "w"), indent=1)
    np.savez(args.out + ".npz", **{k: np.asarray(v, np.float32) for k, v in params.items()}, du=np.int64(du),
             p_mean=st["proprio_obs"][0], p_std=st["proprio_obs"][1], a_mean=st["actions"][0], a_std=st["actions"][1],
             c_mean=st["commands"][0], c_std=st["commands"][1], H=np.int64(H), T=np.int64(T))


if __name__ == "__main__":
    main()
