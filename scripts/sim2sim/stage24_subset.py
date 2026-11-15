"""Design doc 24: episode subset of the paper pretraining data with a per-step episode id (for per-episode context xi_e).

    python scripts/sim2sim/stage24_subset.py --src datasets/sim/2026-10-02_01-18-59/processed_data_go1_H-25_T-25 \
        --dst datasets/sim/paper_sub8192/processed_data_go1_H-25_T-25 --n_episodes 8192

Episodes = maximal runs of consecutive window starts (start_idxs). Each selected episode keeps its full step range
[first t_H, last t_H + H + T]; start_idxs are re-based; episode_id.hdf5 holds the subset episode index per step.
"""
import argparse, json, os, shutil
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--n_episodes", type=int, default=8192)
ap.add_argument("--min_windows", type=int, default=200)
ap.add_argument("--H", type=int, default=25)
ap.add_argument("--T", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.dst, exist_ok=True)
s = h5py.File(f"{args.src}/start_idxs.hdf5", "r")["data"][:]
br = np.where(np.diff(s) != 1)[0]
w0, w1 = np.r_[0, br + 1], np.r_[br, len(s) - 1]
nwin = w1 - w0 + 1
elig = np.where(nwin >= args.min_windows)[0]
rng = np.random.default_rng(args.seed)
sel = np.sort(rng.choice(elig, size=args.n_episodes, replace=False))
a = s[w0[sel]]; b = s[w1[sel]] + args.H + args.T  # inclusive step range
lens = b - a + 1
print(f"episodes in source {len(w0)}, eligible (>= {args.min_windows} windows) {len(elig)}, selected {len(sel)}, steps {lens.sum()}", flush=True)
off = np.r_[0, np.cumsum(lens)[:-1]]
new_s = np.concatenate([off[k] + (s[w0[e]:w1[e] + 1] - a[k]) for k, e in enumerate(sel)])
ep_id = np.concatenate([np.full(lens[k], k, np.int32) for k in range(len(sel))])
with h5py.File(f"{args.dst}/start_idxs.hdf5", "w") as f:
    f.create_dataset("data", data=new_s.astype(np.int64))
with h5py.File(f"{args.dst}/episode_id.hdf5", "w") as f:
    f.create_dataset("data", data=ep_id)
for name in ["proprio_obs", "extero_obs", "acts", "cmds", "rewards", "values", "expert_policy_flag"]:
    src = h5py.File(f"{args.src}/{name}.hdf5", "r")["data"]
    with h5py.File(f"{args.dst}/{name}.hdf5", "w") as f:
        out = f.create_dataset("data", shape=(int(lens.sum()),) + src.shape[1:], dtype=src.dtype)
        for k in range(len(sel)):
            out[off[k]:off[k] + lens[k]] = src[a[k]:b[k] + 1]
    print(f"{name} done", flush=True)
for name in ["scaler_params.json", "system.yaml"]:
    shutil.copy(f"{args.src}/{name}", f"{args.dst}/{name}")
json.dump({"num_episodes": int(len(sel)), "num_trajectories": int(len(new_s)), "num_steps": int(lens.sum()),
           "source": args.src, "source_episode_index": sel.tolist(), "seed": args.seed, "min_windows": args.min_windows},
          open(f"{args.dst}/dataset_metrics.json", "w"))
print("done", flush=True)
