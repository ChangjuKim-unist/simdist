"""How wrong is a model's value function in a target condition?

Reads the raw episode logs of a sim-to-sim dataset (datasets/real/<name>/raw,
written by scripts/simulate_go2.py with logging.enabled=true, which stores the
simulator reward) and compares, for every logged state, the model's value
V(z_t, c_{t+1}) on the encoded true latent with the discounted Monte-Carlo
return G_t of the logged rollout. Episodes that ended by a fall give an exact
G_t; for episodes that hit the time limit the last `min_tail` steps are dropped
so the missing tail is negligible (0.99^300 ~ 0.05).

Reported per model: Pearson / Spearman correlation (ranking), least-squares
slope and intercept of G on V and the mean bias (scale / offset), RMSE, and a
calibration curve (mean G per V-decile). Run inside the simdist container:

    python scripts/sim2sim/value_error.py --models go1_paper go1_paper_combo_hard_res_replay_b45000 \
        --dataset sim2sim_combo_hard_go1_paper --out results/sim2sim/combo_hard/value_error.json
"""

import argparse
import glob
import json
import os

import h5py
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from simdist.utils import model as model_utils, paths


def load_episodes(dataset: str, discount: float, min_tail: int, stride: int, time_limit_steps: int, T: int = 25):
    files = sorted(glob.glob(os.path.join(paths.get_real_raw_data_dir(dataset), "*", "*.hdf5")))
    assert files, f"no raw episodes for dataset {dataset}"
    P, E, C, G, F = [], [], [], [], []
    for f in files:
        with h5py.File(f, "r") as h:
            n = len(h["reward"])
            if n < 50:
                continue
            r = h["reward"][:].astype(np.float64)
            ret = np.zeros(n)
            acc = 0.0
            for t in range(n - 1, -1, -1):
                acc = r[t] + discount * acc
                ret[t] = acc
            fell = n < time_limit_steps - 50
            keep = np.arange(0, n, stride)
            if not fell:
                keep = keep[keep < n - min_tail]
            if len(keep) == 0:
                continue
            # the value head is a sequence model over T consecutive latents: build
            # windows of T states starting at the kept indices
            keep = keep[keep + T <= n]
            if len(keep) == 0:
                continue
            cmd = h["commands"][:]
            cmd_next = np.concatenate([cmd[1:], cmd[-1:]], axis=0)
            po, ex = h["proprio_obs"][:], h["extero_obs"][:]
            idx = keep[:, None] + np.arange(T)[None, :]
            P.append(po[idx]); E.append(ex[idx]); C.append(cmd_next[idx]); G.append(ret[idx])
            F.append(np.full(idx.shape, fell))
    return tuple(np.concatenate(a) for a in (P, E, C, G, F))


def model_values(model, P, E, C, batch_size=1024):
    sc = model.get_scaler().get_scaler_params()
    def scale(x, k):
        return (x - np.asarray(sc[k]["mean"])) / (np.asarray(sc[k]["std"]) + 1e-8)
    out = []
    for i in range(0, len(P), batch_size):
        p = jnp.asarray(scale(P[i:i + batch_size], "proprio_obs"))
        e = jnp.asarray(scale(E[i:i + batch_size], "extero_obs"))
        c = jnp.asarray(scale(C[i:i + batch_size], "commands"))
        z = model.encode_latent(p, e, deterministic=True)  # (B, T, lat)
        v_s = model.value_from_latents(z, c, deterministic=True, use_residual=True)  # (B, T)
        out.append(np.asarray(v_s).reshape(-1) * float(sc["values"]["std"][0]) + float(sc["values"]["mean"][0]))
    return np.concatenate(out)


def stats(v, g):
    A = np.stack([v, np.ones_like(v)], axis=1)
    slope, intercept = np.linalg.lstsq(A, g, rcond=None)[0]
    dec = np.quantile(v, np.linspace(0, 1, 11))
    calib = []
    for lo, hi in zip(dec[:-1], dec[1:]):
        m = (v >= lo) & (v <= hi)
        if m.any():
            calib.append([float(v[m].mean()), float(g[m].mean())])
    return {
        "n": int(len(v)),
        "pearson": float(np.corrcoef(v, g)[0, 1]),
        "spearman": float(spearmanr(v, g).correlation),
        "slope_G_on_V": float(slope),
        "intercept": float(intercept),
        "bias_V_minus_G": float(np.mean(v - g)),
        "rmse": float(np.sqrt(np.mean((v - g) ** 2))),
        "std_V": float(v.std()),
        "std_G": float(g.std()),
        "mean_V": float(v.mean()),
        "mean_G": float(g.mean()),
        "calibration_V_G": calib,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", required=True, help="sim2sim dataset name with raw episode logs")
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--min_tail", type=int, default=300)
    ap.add_argument("--stride", type=int, default=25, help="window start stride (windows are 25 steps)")
    ap.add_argument("--time_limit_steps", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P, E, C, G, F = load_episodes(args.dataset, args.discount, args.min_tail, args.stride, args.time_limit_steps)
    G, F = G.reshape(-1), F.reshape(-1)
    print(f"{len(G)} states from {args.dataset} ({int(F.sum())} from episodes that ended by a fall)")
    results = {"_meta": {"dataset": args.dataset, "discount": args.discount, "n": int(len(G)), "n_fell": int(F.sum())}}
    for name in args.models:
        model, _, _ = model_utils.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), name))
        v = model_values(model, P, E, C)
        results[name] = stats(v, G)
        results[name]["fallen_states"] = stats(v[F], G[F]) if F.sum() > 100 else None
        r = results[name]
        print(f"{name}: pearson {r['pearson']:.3f} spearman {r['spearman']:.3f} slope {r['slope_G_on_V']:.2f} "
              f"bias(V-G) {r['bias_V_minus_G']:+.2f} rmse {r['rmse']:.2f} (std G {r['std_G']:.2f})")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        existing = {}
        if os.path.exists(args.out):
            existing = json.load(open(args.out))
        existing.update(results)
        json.dump(existing, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
