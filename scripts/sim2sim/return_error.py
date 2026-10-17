"""Decision-relevant dynamics error: the planner's return computed on PREDICTED latents vs
on the TRUE future latents, for logged windows, split into pre-fall and normal windows.

    python scripts/sim2sim/return_error.py --models go1_paper <adapted ...> --dataset <processed real dataset>

For a window the planner's return of the executed action sequence is
    R(z) = sum_k gamma^k r_k(z) + gamma^(T+1) V(z_T)
with rewards and values from the model's heads. The error |R(z_hat) - R(z_true)| is what
can flip the ranking of candidates; the latent MSE is not.
"""
import argparse, json, os
import h5py, numpy as np, jax, jax.numpy as jnp
from torch.utils.data import DataLoader, Subset
from simdist.utils import model as m, paths
from simdist.data.dataset import get_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--prefall", type=int, default=50)
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--discount", type=float, default=0.99)
ap.add_argument("--out", default=None)
args = ap.parse_args()

d = os.path.join(paths.get_real_dataset_dir(args.dataset), "processed_data_go1_H-25_T-25")
term = h5py.File(os.path.join(d, "terminal.hdf5"))["data"][:]
idx = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
end = idx + 50  # window end step t_T (H = T = 25)
near = (term[end, 0] <= args.prefall) & (term[end, 2] > 0.5)
rng = np.random.default_rng(0)
sel = {"pre-fall": rng.choice(np.where(near)[0], min(args.n, int(near.sum())), replace=False),
       "normal": rng.choice(np.where(~near)[0], args.n, replace=False)}
T = 25
disc = args.discount ** np.arange(T); final = args.discount ** (T + 1)
results = {}
for name in args.models:
    mdl, cfg, _ = m.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), name))
    c = dict(cfg); c["data"] = {"dataset_name": args.dataset}; ds = get_dataset(c); ds.eval(); sc = mdl.get_scaler()
    vp = sc.get_scaler_params()["values"]; rp = sc.get_scaler_params()["rewards"]
    res = {}
    for lab, ids in sel.items():
        errs, lat, rew_e, val_e = [], [], [], []
        for b in DataLoader(Subset(ds, ids), batch_size=200):
            b = m.dataset_batch_to_jax(b); x, y = b["model_in"], b["labels"]
            x_s, y_s = sc.scale(x), sc.scale(y)
            out = mdl(x, deterministic=True)  # predicted latents, rewards, values (scaled)
            z_true = mdl.encode_latent(y_s["proprio_obs"], y_s["extero_obs"], deterministic=True)
            r_true = mdl.reward_from_latents(x_s, z_true, deterministic=True)
            v_true = mdl.value_from_latents(z_true, x_s["fut_cmds"][:, 1:], deterministic=True, use_residual=True)
            r_hat, v_hat = out["rewards"], out["values"]
            R_hat = jnp.sum(r_hat * disc, -1) * float(rp["std"][0]) + final * (v_hat[:, -1] * float(vp["std"][0]))
            R_true = jnp.sum(r_true * disc, -1) * float(rp["std"][0]) + final * (v_true[:, -1] * float(vp["std"][0]))
            errs.append(np.asarray(jnp.abs(R_hat - R_true)))
            rew_e.append(np.asarray(jnp.abs(jnp.sum((r_hat - r_true) * disc, -1)) * float(rp["std"][0])))
            val_e.append(np.asarray(jnp.abs(v_hat[:, -1] - v_true[:, -1]) * float(vp["std"][0]) * final))
            lat.append(np.asarray(jnp.mean((out["latents"] - z_true) ** 2, axis=(1, 2))))
        e = np.concatenate(errs)
        res[lab] = {"return_abs_err_mean": float(e.mean()), "return_abs_err_p90": float(np.percentile(e, 90)),
                    "reward_part": float(np.concatenate(rew_e).mean()), "value_part": float(np.concatenate(val_e).mean()),
                    "latent_mse": float(np.concatenate(lat).mean()), "n": int(len(e))}
    results[name] = res
    print(f"{name}:")
    for lab in ("normal", "pre-fall"):
        q = res[lab]
        print(f"   {lab:8s} |R_hat - R_true| mean {q['return_abs_err_mean']:.3f} (p90 {q['return_abs_err_p90']:.3f}) = reward part {q['reward_part']:.3f} + value part {q['value_part']:.3f}; latent MSE {q['latent_mse']:.4f}")
if args.out:
    json.dump(results, open(args.out, "w"), indent=2)
