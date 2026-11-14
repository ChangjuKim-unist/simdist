"""Design doc 21, gate 0: fit only the context slot of a frozen context world model (go1_ctx, design 13) to
real-like adaptation data by latent prediction error, instead of inferring it from history.

The slot's physics part psi[:3] (standardized [motor, friction, mass]) is replaced by a free 3-vector xi
(the ``context_override`` input, design 13 oracle path); all weights stay frozen. The fitted xi is then
deployed with ``sim.context_override_xi`` in scripts/simulate_batch.py.

    python scripts/sim2sim/stage21_ctx_opt.py --ckpt go1_ctx --dataset sim2sim_friction0.15_go1_broadwm_b45000 \
        --out results/stage21/xi_friction0.15.json

Reports held-out latent MSE (lower is better) for: inferred psi, oracle psi (true physics of the data), the
fitted xi, and optionally a fully fine-tuned checkpoint (--compare_ckpt) on the same windows.
Split: first 90 % of windows (time order) fit, last 10 % held out.
"""
import argparse
import json
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader, Subset

from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="go1_ctx")
ap.add_argument("--dataset", required=True)
ap.add_argument("--system", default="go1")
ap.add_argument("--true_xi", type=float, nargs=3, default=[1.0, 0.15, 0.0], help="raw [motor, friction, mass] of the data (oracle arm)")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--n_val", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--compare_ckpt", default="")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(ckpt):
    model, cfg, step = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), ckpt))
    return model, cfg


model, cfg = load(args.ckpt)
assert getattr(model, "context", None) is not None, "needs a model with a context slot (design 13)"
cfg = dict(cfg)
cfg["system"] = dict(cfg["system"])
cfg["system"]["name"] = args.system
cfg["data"] = {"dataset_name": args.dataset}
ds = get_dataset(cfg)
ds.eval()  # no augmentation noise
n = len(ds)
n_fit = int(0.9 * n)
fit_set, val_set = Subset(ds, range(n_fit)), Subset(ds, range(n_fit, n))
print(f"windows: {n} (fit {n_fit}, held-out {n - n_fit})", flush=True)


def batches(subset, bs, shuffle, seed):
    loader = DataLoader(subset, batch_size=bs, shuffle=shuffle, num_workers=0, drop_last=shuffle,  # workers fork JAX -> corrupt batches
                        generator=None if not shuffle else __import__("torch").Generator().manual_seed(seed))
    for b in loader:
        yield mu.dataset_batch_to_jax(b)


def latent_mse(m, x, y, override):
    if override is not None:
        x = dict(x)
        x["context_override"] = jnp.broadcast_to(override, (x["proprio_obs_hist"].shape[0], 3))
    pred = m(x, deterministic=True)
    ys = m.get_scaler().scale(y)
    fut = m.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    return jnp.mean((pred["latents"] - fut) ** 2)


graphdef, state = nnx.split(model)


@jax.jit
def loss_fn(xi, x, y):
    m = nnx.merge(graphdef, state)
    return latent_mse(m, x, y, xi)


@jax.jit
def loss_none(x, y):
    m = nnx.merge(graphdef, state)
    return latent_mse(m, x, y, None)


def val_loss(fn):
    tot, cnt = 0.0, 0
    for i, b in enumerate(batches(val_set, args.batch, False, 0)):
        k = b["model_in"]["proprio_obs_hist"].shape[0]
        tot += float(fn(b["model_in"], b["labels"])) * k
        cnt += k
        if cnt >= args.n_val:
            break
    return tot / cnt


xi_true = np.asarray(model.scale_xi(np.asarray(args.true_xi, np.float32)), np.float32)
res = {"ckpt": args.ckpt, "dataset": args.dataset, "true_xi_raw": args.true_xi, "true_xi_std": xi_true.tolist()}
res["val_mse_inferred"] = val_loss(loss_none)
res["val_mse_oracle"] = val_loss(lambda x, y: loss_fn(jnp.asarray(xi_true), x, y))
print(f"held-out latent MSE  inferred {res['val_mse_inferred']:.5f}  oracle {res['val_mse_oracle']:.5f}", flush=True)

# fit xi: start at the standardized DR mean (0), Adam on latent prediction error, all weights frozen
xi = jnp.zeros((3,), jnp.float32)
opt = optax.adam(args.lr)
opt_state = opt.init(xi)
grad_fn = jax.jit(jax.value_and_grad(loss_fn))
step, curve = 0, []
while step < args.steps:
    for b in batches(fit_set, args.batch, True, args.seed + step):
        l, g = grad_fn(xi, b["model_in"], b["labels"])
        upd, opt_state = opt.update(g, opt_state, xi)
        xi = optax.apply_updates(xi, upd)
        step += 1
        if step % 100 == 0 or step == 1:
            curve.append([step, float(l)] + [float(v) for v in xi])
            print(f"step {step} fit-batch MSE {float(l):.5f} xi_std {np.round(np.asarray(xi), 3).tolist()} "
                  f"xi_raw {np.round(np.asarray(model.unscale_xi(xi)), 3).tolist()}", flush=True)
        if step >= args.steps:
            break

res["xi_fit_std"] = [float(v) for v in xi]
res["xi_fit_raw"] = [float(v) for v in model.unscale_xi(xi)]
res["val_mse_fit"] = val_loss(lambda x, y: loss_fn(xi, x, y))
res["curve"] = curve
print(f"held-out latent MSE  fitted xi {res['val_mse_fit']:.5f}", flush=True)

if args.compare_ckpt:
    m2, _ = load(args.compare_ckpt)
    g2, s2 = nnx.split(m2)
    f2 = jax.jit(lambda x, y: latent_mse(nnx.merge(g2, s2), x, y, None))
    res["compare_ckpt"] = args.compare_ckpt
    res["val_mse_compare"] = val_loss(f2)
    print(f"held-out latent MSE  {args.compare_ckpt} {res['val_mse_compare']:.5f}", flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "curve"}), flush=True)
