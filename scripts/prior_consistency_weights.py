"""Design doc 9: prior-consistency weights for real-data adaptation.

For every step t of a processed real-format dataset, compare the logged action a_t with the action the
deployed prior would take from the same history, pi(h_t)[0], and store
    w_t = exp(-||a_t - pi(h_t)||^2_scaled / (2 sigma^2 d))         (d = action dim)
in prior_weight.hdf5. The adaptation loss (loss.prior_weight=true) weights each window by the mean w over
its horizon, so data from a different collecting policy cannot overwrite the pretrained model where the
deployed prior does not go. sigma is set so that the deployed prior's own data gets weights near 1.

    python scripts/prior_consistency_weights.py --dataset sim2sim_very_weak_motor_go1_paper_b45000 --student student_broad --sigma 0.5
"""
import argparse, json, os
import h5py, numpy as np, jax.numpy as jnp
from simdist.control.student_prior import StudentPrior
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1"); ap.add_argument("--student", default="student_broad")
ap.add_argument("--sigma", type=float, default=0.5, help="scale of the scaled-action distance; weights ~ exp(-d^2 / (2 sigma^2))")
ap.add_argument("--chunk", type=int, default=4096)
args = ap.parse_args()
st = StudentPrior(args.student); H, T = st.H, st.T
d = os.path.join(paths.get_real_dataset_dir(args.dataset), f"processed_data_{args.system}_H-25_T-25")
P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"]; C = h5py.File(os.path.join(d, "cmds.hdf5"))["data"]
n = P.shape[0]; a_std = np.asarray(st.scaler["actions"]["std"], np.float32) + 1e-8
w = np.ones(n, np.float32); dist = np.full(n, np.nan, np.float32)
for i in range(H, n - T - 1, args.chunk):
    t = np.arange(i, min(i + args.chunk, n - T - 1)); tH = t - H
    x = {"proprio_obs_hist": np.stack([P[s:s + H + 1] for s in tH]), "acts_hist": np.stack([A[s:s + H] for s in tH]), "fut_cmds": np.stack([C[s:s + T + 1] for s in t])}
    pred = np.asarray(st(x))[:, 0]  # the prior's first action
    dd = np.mean(((np.stack([A[s] for s in t]) - pred) / a_std) ** 2, axis=-1)  # per-dim mean scaled squared distance
    dist[t] = dd; w[t] = np.exp(-dd / (2 * args.sigma ** 2))
with h5py.File(paths.get_prior_weight_path(d), "w") as f: f.create_dataset("data", data=w)
valid = ~np.isnan(dist)
stats = {"dataset": args.dataset, "student": args.student, "sigma": args.sigma, "steps": int(n), "mean_dist": float(np.nanmean(dist)), "median_dist": float(np.nanmedian(dist)),
         "mean_weight": float(w[valid].mean()), "effective_fraction": float(w[valid].sum() ** 2 / (n * (w[valid] ** 2).sum() + 1e-8)), "frac_weight_gt_0.5": float((w[valid] > 0.5).mean())}
json.dump(stats, open(os.path.join(d, "prior_weight_stats.json"), "w"), indent=2); print(json.dumps(stats, indent=2))
