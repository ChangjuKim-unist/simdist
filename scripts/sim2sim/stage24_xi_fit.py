"""Design doc 24: fit only the FiLM context xi of a frozen model to adaptation data by latent prediction error.

    python scripts/sim2sim/stage24_xi_fit.py --ckpt go1_paper_film --dataset sim2sim_very_weak_motor_go1_paper_b45000 \
        --out results/stage24/xi_motor.json

Loss = latent MSE (as SimDist fine-tuning) + prior_w * ||xi||^2 (the training prior). Adam from xi = 0, all weights
frozen. Time split: first 90 % of windows fit, last 10 % held out. Reports held-out latent MSE (lower is better) for
xi = 0 and the fitted xi, and optionally for a comparison checkpoint (e.g. the SimDist adaptation of the control).
--usage: instead, on an episode-subset dataset, compare each window's own trained xi_e with xi = 0 (is the slot used?),
on the training range of windows (the held-out tail is reported separately: its rows were never trained).
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
ap.add_argument("--dataset", required=True)
ap.add_argument("--system", default="go1")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--prior_w", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--n_val", type=int, default=4096)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--compare_ckpt", default="")
ap.add_argument("--usage", action="store_true")
ap.add_argument("--train_ratio", type=float, default=0.95, help="training_data_ratio of the FiLM run: rows of the held-out tail episodes were never trained")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(ckpt):
    m, c, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), ckpt))
    return m, c


model, cfg = load(args.ckpt)
assert getattr(model, "film_proj", None) is not None, "needs a model with a FiLM context (design doc 24)"
d = int(model.film_dim)
cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
cfg["data"] = {"dataset_name": args.dataset}
ds = get_dataset(cfg); ds.eval()
n = len(ds)


def batches(subset, bs, shuffle, seed):
    import torch
    loader = DataLoader(subset, batch_size=bs, shuffle=shuffle, num_workers=0, drop_last=shuffle,
                        generator=None if not shuffle else torch.Generator().manual_seed(seed))
    for b in loader:
        yield mu.dataset_batch_to_jax(b)


def latent_mse(m, x, y, xi):
    x = dict(x)
    if xi is not None:
        x["xi"] = jnp.broadcast_to(xi, (x["proprio_obs_hist"].shape[0], d)) if xi.ndim == 1 else xi
    pred = m(x, deterministic=True)
    ys = m.get_scaler().scale(y)
    fut = m.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    return jnp.mean((pred["latents"] - fut) ** 2)


gd, st = nnx.split(model)
res = {"ckpt": args.ckpt, "dataset": args.dataset}

if args.usage:
    # windows of the trainer's held-out tail (index >= train_ratio * n) belong to episodes whose xi_e was never
    # updated (still the random init), so the usage gate is computed on the training range only
    n_tr = int(n * args.train_ratio)

    @jax.jit
    def f(x, y):
        m = nnx.merge(gd, st)
        xi = m.xi_table.value[jnp.asarray(y["episode_id"]).reshape(-1)]
        return latent_mse(m, x, y, xi), latent_mse(m, x, y, jnp.zeros((d,)))

    def usage(lo, hi):
        idx = np.random.default_rng(args.seed).choice(np.arange(lo, hi), size=min(args.n_val, hi - lo), replace=False)
        a = b = cnt = 0.0
        eps = []
        for bt in batches(Subset(ds, np.sort(idx)), args.batch, False, 0):
            k = bt["model_in"]["proprio_obs_hist"].shape[0]
            u, v = f(bt["model_in"], bt["labels"]); a += float(u) * k; b += float(v) * k; cnt += k
            eps.append(np.asarray(bt["labels"]["episode_id"]).reshape(-1))
        return a / cnt, b / cnt, np.unique(np.concatenate(eps))

    tab = np.asarray(model.xi_table.value)
    a, b, ep_tr = usage(0, n_tr)
    ah, bh, ep_ho = usage(n_tr, n)
    res.update({"mse_own_xi": a, "mse_xi0": b, "rel_gain": 1 - a / b, "n_train_windows": n_tr,
                "heldout_mse_own_xi": ah, "heldout_mse_xi0": bh, "heldout_rel_gain": 1 - ah / bh,
                "xi_norm_train_rows": float(np.linalg.norm(tab[ep_tr], axis=1).mean()),
                "xi_norm_heldout_rows": float(np.linalg.norm(tab[ep_ho], axis=1).mean()),
                "xi_table_norm_mean": float(np.linalg.norm(tab, axis=1).mean()), "xi_table_std": tab.std(0).tolist()})
    print("RESULT", json.dumps(res), flush=True)
else:
    n_fit = int(0.9 * n)
    fit_set, val_set = Subset(ds, range(n_fit)), Subset(ds, range(n_fit, n))

    @jax.jit
    def loss_fn(xi, x, y):
        return latent_mse(nnx.merge(gd, st), x, y, xi) + args.prior_w * jnp.sum(xi ** 2)

    def val(fn):
        tot = cnt = 0.0
        for bt in batches(val_set, args.batch, False, 0):
            k = bt["model_in"]["proprio_obs_hist"].shape[0]
            tot += float(fn(bt["model_in"], bt["labels"])) * k; cnt += k
            if cnt >= args.n_val:
                break
        return tot / cnt

    mse = jax.jit(lambda xi, x, y: latent_mse(nnx.merge(gd, st), x, y, xi))
    res["val_mse_xi0"] = val(lambda x, y: mse(jnp.zeros((d,)), x, y))
    xi = jnp.zeros((d,)); opt = optax.adam(args.lr); os_ = opt.init(xi)
    g = jax.jit(jax.value_and_grad(loss_fn)); step = 0; curve = []
    while step < args.steps:
        for bt in batches(fit_set, args.batch, True, args.seed + step):
            l, gr = g(xi, bt["model_in"], bt["labels"])
            up, os_ = opt.update(gr, os_, xi); xi = optax.apply_updates(xi, up); step += 1
            if step % 100 == 0 or step == 1:
                curve.append([step, float(l), float(jnp.linalg.norm(xi))])
                print(f"step {step} loss {float(l):.5f} |xi| {float(jnp.linalg.norm(xi)):.3f}", flush=True)
            if step >= args.steps:
                break
    res["xi_fit"] = [float(v) for v in xi]
    res["val_mse_fit"] = val(lambda x, y: mse(xi, x, y))
    res["curve"] = curve
    if args.compare_ckpt:
        m2, _ = load(args.compare_ckpt); g2, s2 = nnx.split(m2)
        f2 = jax.jit(lambda x, y: latent_mse(nnx.merge(g2, s2), x, y, None))
        res["compare_ckpt"] = args.compare_ckpt; res["val_mse_compare"] = val(f2)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "curve"}), flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
