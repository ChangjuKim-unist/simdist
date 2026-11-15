"""Design doc 22 §12 diagnosis: where do the refit heads send a best-of-K selection?

For held-out windows, K candidate action sequences = logged actions + Gaussian noise (sigma x per-dim action std;
candidate 0 = the logged actions). Each model scores all candidates with the MPPI return J (model.inference).
Reported per model M: J_M(best_M) - J_M(logged) (how much better M believes its pick is), the cross-score
J_N(best_M) - J_N(logged) under every other model N (does anyone else agree?), the action distance of the pick,
and the across-candidate std of J (selection sharpness at MPPI temperature 0.25).

    python scripts/sim2sim/stage22_candidates.py --dataset sim2sim_very_weak_motor_go1_paper_ho_b15000 \
        --ckpts B:... C:... D:... --out results/stage22q/candidates.json
"""
import argparse, json, os
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import torch
from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5])
ap.add_argument("--K", type=int, default=256)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--out", required=True)
args = ap.parse_args()

fns, names, ds = {}, [], None
for spec in args.ckpts:
    name, ckpt = spec.split(":", 1)
    model, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), ckpt))
    if ds is None:
        cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = "go1"; cfg["data"] = {"dataset_name": args.dataset}
        ds = get_dataset(cfg); ds.eval()
        a_std = jnp.asarray(np.asarray(model.get_scaler().get_scaler_params()["actions"]["std"])).reshape(-1)
    gd, st = nnx.split(model)

    def make(gd=gd, st=st):
        @jax.jit
        def J(x):
            y = nnx.merge(gd, st).inference(x)
            T = y["rewards"].shape[-1]
            return jnp.sum(y["rewards"] * 0.99 ** jnp.arange(T), -1) + y["values"][:, -1] * 0.99 ** (T + 1)
        return J
    fns[name] = make(); names.append(name)

idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
key = jax.random.PRNGKey(0)
res = {"dataset": args.dataset, "K": args.K, "n": args.n}
for sg in args.sigmas:
    acc = {m: {"self_gain": [], "dist": [], "J_std": [], **{f"cross_{n}": [] for n in names}} for m in names}
    for i in range(0, len(idx), args.chunk):
        items = [ds[j] for j in idx[i:i + args.chunk]]
        b = mu.dataset_batch_to_jax(torch.utils.data.default_collate(items))
        x = b["model_in"]; W = x["fut_acts"].shape[0]
        xt = {k: jnp.repeat(v, args.K, axis=0) for k, v in x.items()}
        key, k = jax.random.split(key)
        noise = sg * a_std * jax.random.normal(k, xt["fut_acts"].shape)
        noise = noise.reshape(W, args.K, *noise.shape[1:]).at[:, 0].set(0.0).reshape(xt["fut_acts"].shape)
        xt["fut_acts"] = xt["fut_acts"] + noise
        Js = {m: np.asarray(fns[m](xt)).reshape(W, args.K) for m in names}
        nz = np.asarray(jnp.sqrt(jnp.mean((noise / a_std) ** 2, axis=(1, 2)))).reshape(W, args.K)
        for m in names:
            best = Js[m].argmax(1); r = np.arange(W)
            acc[m]["self_gain"] += (Js[m][r, best] - Js[m][:, 0]).tolist()
            acc[m]["dist"] += nz[r, best].tolist()
            acc[m]["J_std"] += Js[m].std(1).tolist()
            for n in names:
                acc[m][f"cross_{n}"] += (Js[n][r, best] - Js[n][:, 0]).tolist()
    res[f"sigma{sg}"] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    print(f"sigma {sg}:", json.dumps(res[f"sigma{sg}"]), flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
