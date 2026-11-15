"""Design doc 22 quick check, diagnostic (not a registered judgement): does a refit head rate perturbed action
sequences above the executed one? For held-out windows, planner score J = sum_k gamma^k r_hat_k + gamma^T V_hat(z_T)
(the MPPI return, model.inference) of the logged actions vs the logged actions + Gaussian noise (sigma x per-dim
action std of the scaler). Under weak motors a random perturbation of a walking gait is rarely better in reality,
so a head that prefers perturbed sequences more often than the frozen head is optimistic off the data.

    python scripts/sim2sim/stage22_optimism.py --dataset sim2sim_very_weak_motor_go1_paper_ho_b15000 \
        --ckpts B:... C:... D:... --out results/stage22q/optimism.json
"""
import argparse, json, os
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 1.0])
ap.add_argument("--draws", type=int, default=8)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--out", required=True)
args = ap.parse_args()
res = {"dataset": args.dataset, "models": {}}
for spec in args.ckpts:
    name, ckpt = spec.split(":", 1)
    model, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), ckpt))
    cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = "go1"; cfg["data"] = {"dataset_name": args.dataset}
    ds = get_dataset(cfg); ds.eval()
    gd, st = nnx.split(model)
    a_std = jnp.asarray(np.asarray(model.get_scaler().get_scaler_params()["actions"]["std"])).reshape(-1)

    @jax.jit
    def J(x):
        y = nnx.merge(gd, st).inference(x)
        T = y["rewards"].shape[-1]
        return jnp.sum(y["rewards"] * 0.99 ** jnp.arange(T), -1) + y["values"][:, -1] * 0.99 ** (T + 1)

    out = {f"sigma{s}": {"dJ": [], "better": []} for s in args.sigmas}
    key = jax.random.PRNGKey(0); seen = 0
    idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
    sub = [ds[i] for i in idx]
    for i in range(0, len(sub), 256):
        b = mu.dataset_batch_to_jax(__import__("torch").utils.data.default_collate(sub[i:i + 256]))
        x = b["model_in"]; j0 = J(x)
        for s in args.sigmas:
            for _ in range(args.draws):
                key, k = jax.random.split(key)
                xn = dict(x); xn["fut_acts"] = x["fut_acts"] + s * a_std * jax.random.normal(k, x["fut_acts"].shape)
                d = np.asarray(J(xn) - j0)
                out[f"sigma{s}"]["dJ"].append(d); out[f"sigma{s}"]["better"].append(d > 0)
    r = {}
    for s, v in out.items():
        dJ = np.concatenate(v["dJ"]); bt = np.concatenate(v["better"])
        r[s] = {"mean_dJ": float(dJ.mean()), "frac_perturbed_better": float(bt.mean()), "p90_dJ": float(np.percentile(dJ, 90))}
    res["models"][name] = r
    print(name, ckpt, json.dumps(r), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
