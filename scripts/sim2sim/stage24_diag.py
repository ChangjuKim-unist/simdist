"""Design doc 24 §9: is the gate-0 failure of the FiLM context the concept or the per-episode table training?

    python scripts/sim2sim/stage24_diag.py --ckpt go1_paper_film_s42 --out results/stage24/diag_go1_paper_film_s42.json

Q1: on the gate-0 windows, latent MSE with the own table row xi_e vs another episode's row (row permutation) vs xi = 0.
Q2: per episode (>= 400 windows), freeze the network and fit only xi on the first 80 % of its windows (the adaptation
objective: latent MSE + prior_w * ||xi||^2, full-batch Adam), evaluate the relative gain over xi = 0 on the last 20 %.
Q2': the same fit on the last 20 % itself (how much an 8-d xi can move the latent MSE at all).
"""
import argparse, json, os
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader, Subset
from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", default="paper_sub8192")
ap.add_argument("--system", default="go1")
ap.add_argument("--n_val", type=int, default=4096)
ap.add_argument("--n_episodes", type=int, default=64)
ap.add_argument("--min_windows", type=int, default=400)
ap.add_argument("--n_fit", type=int, default=256)
ap.add_argument("--n_test", type=int, default=80)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--prior_w", type=float, default=0.01)
ap.add_argument("--out", required=True)
args = ap.parse_args()

model, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), args.ckpt))
assert getattr(model, "film_proj", None) is not None and model.xi_table is not None
d = int(model.film_dim)
cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
cfg["data"] = {"dataset_name": args.dataset}
ds = get_dataset(cfg); ds.eval()
n = len(ds)
gd, st = nnx.split(model)
table = np.asarray(model.xi_table.value)


def load_batch(idx, bs=256):
    out = []
    for b in DataLoader(Subset(ds, list(idx)), batch_size=bs, shuffle=False, num_workers=0):  # workers fork JAX
        out.append(mu.dataset_batch_to_jax(b))
    return jax.tree.map(lambda *a: jnp.concatenate(a, 0), *out) if len(out) > 1 else out[0]


def latent_mse(m, x, y, xi):
    x = dict(x)
    x["xi"] = jnp.broadcast_to(xi, (x["proprio_obs_hist"].shape[0], d)) if xi.ndim == 1 else xi
    pred = m(x, deterministic=True)
    ys = m.get_scaler().scale(y)
    fut = m.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    return jnp.mean((pred["latents"] - fut) ** 2)


mse = jax.jit(lambda xi, x, y: latent_mse(nnx.merge(gd, st), x, y, xi))
res = {"ckpt": args.ckpt, "dataset": args.dataset}

# film output size over the table rows: mean |gamma|, |beta|
rows_ = table[:1024]
fo = np.asarray(model.film_proj(jnp.asarray(rows_))).reshape(len(rows_), model.film_layers, 3, 2, model.latent_dim)
res["film_abs_gamma"] = float(np.abs(fo[..., 0, :]).mean()); res["film_abs_beta"] = float(np.abs(fo[..., 1, :]).mean())
res["film_kernel_norm"] = float(jnp.linalg.norm(model.film_proj.kernel.value))

# Q1: same windows as the gate-0 usage check (seed 0)
idx = np.sort(np.random.default_rng(0).choice(n, size=min(args.n_val, n), replace=False))
perm = np.random.default_rng(1).permutation(table.shape[0])
tot = np.zeros(3); cnt = 0
for s in range(0, len(idx), 256):
    b = load_batch(idx[s:s + 256])
    ep = np.asarray(b["labels"]["episode_id"]).reshape(-1)
    k = len(ep)
    tot += k * np.array([float(mse(jnp.asarray(table[ep]), b["model_in"], b["labels"])),
                         float(mse(jnp.asarray(table[perm[ep]]), b["model_in"], b["labels"])),
                         float(mse(jnp.zeros((d,)), b["model_in"], b["labels"]))])
    cnt += k
q1 = tot / cnt
res["q1"] = {"mse_own": q1[0], "mse_other_episode": q1[1], "mse_xi0": q1[2],
             "gain_own_vs_other": 1 - q1[0] / q1[1], "gain_own_vs_xi0": 1 - q1[0] / q1[2]}
print("Q1", json.dumps(res["q1"]), flush=True)

# Q2 / Q2': per-episode fits with the network frozen
starts = np.asarray(ds._data["start_idxs"][:])
ep_of_win = np.asarray(ds._data["episode_id"][:])[starts + ds.H]
ids, counts = np.unique(ep_of_win, return_counts=True)
cand = ids[counts >= args.min_windows]
chosen = np.random.default_rng(2).choice(cand, size=min(args.n_episodes, len(cand)), replace=False)


def fit(x, y):
    loss = lambda xi: latent_mse(nnx.merge(gd, st), x, y, xi) + args.prior_w * jnp.sum(xi ** 2)
    opt = optax.adam(args.lr)

    def body(c, _):
        xi, os_ = c
        g = jax.grad(loss)(xi)
        up, os_ = opt.update(g, os_, xi)
        return (optax.apply_updates(xi, up), os_), None
    xi0 = jnp.zeros((d,))
    (xi, _), _ = jax.lax.scan(body, (xi0, opt.init(xi0)), None, length=args.steps)
    return xi


fit = jax.jit(fit)
rows = []
for e in chosen:
    w = np.flatnonzero(ep_of_win == e)  # window indices of this episode, time order
    cut = int(0.8 * len(w))
    fw = w[np.linspace(0, cut - 1, args.n_fit).astype(int)]
    tw = w[np.linspace(cut, len(w) - 1, args.n_test).astype(int)]
    bf, bt = load_batch(fw), load_batch(tw)
    xi_fit = fit(bf["model_in"], bf["labels"])
    xi_up = fit(bt["model_in"], bt["labels"])
    m0 = float(mse(jnp.zeros((d,)), bt["model_in"], bt["labels"]))
    r = {"episode": int(e), "n_windows": int(len(w)), "mse_xi0": m0,
         "mse_own": float(mse(jnp.asarray(table[e]), bt["model_in"], bt["labels"])),
         "mse_fit": float(mse(xi_fit, bt["model_in"], bt["labels"])),
         "mse_fit_on_test": float(mse(xi_up, bt["model_in"], bt["labels"])),
         "xi_fit_norm": float(jnp.linalg.norm(xi_fit))}
    r["gain_fit"] = 1 - r["mse_fit"] / m0; r["gain_upper"] = 1 - r["mse_fit_on_test"] / m0; r["gain_own"] = 1 - r["mse_own"] / m0
    rows.append(r)
    print(f"ep {e} windows {len(w)} gain fit {r['gain_fit']:+.4f} upper {r['gain_upper']:+.4f} own {r['gain_own']:+.4f}", flush=True)


def summ(key):
    v = np.array([r[key] for r in rows])
    bs = np.random.default_rng(0).choice(v, size=(20000, len(v))).mean(1)
    return {"mean": float(v.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "frac_pos": float((v > 0).mean())}


res["q2"] = {k: summ(k) for k in ("gain_fit", "gain_upper", "gain_own")}
res["episodes"] = rows
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "episodes"}), flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
