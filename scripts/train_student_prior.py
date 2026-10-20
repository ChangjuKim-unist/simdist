"""Behavior-clone a history-conditioned action-sequence prior (student) from a simulation
dataset generated with an expert (design doc 4).

    python scripts/train_student_prior.py --dataset 2026-10-02_01-18-59 --system go1 --name student_narrow
    python scripts/train_student_prior.py --dataset <go1dr dataset> --system go1dr --name student_broad

Inputs: last H+1 proprio observations, last H actions, T+1 future commands (scaled with the
dataset's scaler). Target: the T future actions, masked like SimDist's action loss (only
steps before the first non-expert action count). Saves checkpoints/students/<name>/.
"""
import argparse, json, os, pickle, time
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx
from omegaconf import OmegaConf

from simdist.control.student_prior import StudentMLP, StudentPrior, students_dir
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1"); ap.add_argument("--name", required=True)
ap.add_argument("--H", type=int, default=25); ap.add_argument("--T", type=int, default=25)
ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 512])
ap.add_argument("--num_windows", type=int, default=2_000_000); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

d = paths.get_processed_data_dir(args.dataset, args.system, args.H, args.T)
f = {k: h5py.File(os.path.join(d, fn))["data"] for k, fn in [("proprio", "proprio_obs.hdf5"), ("acts", "acts.hdf5"), ("cmds", "cmds.hdf5"), ("flags", "expert_policy_flag.hdf5"), ("start", "start_idxs.hdf5")]}
sc = json.load(open(os.path.join(d, "scaler_params.json")))
H, T = args.H, args.T
starts = f["start"][:]
rng = np.random.default_rng(args.seed)
# windows are taken from `blocks` random contiguous regions of the dataset so that the
# needed time steps fit in memory (the data file is time-contiguous, episode after episode)
blocks = 20
per_block = max(1, min(args.num_windows, len(starts)) // blocks)
P, A, C, FL = f["proprio"], f["acts"], f["cmds"], f["flags"]
Pm, Am, Cm, Fm, sel0 = [], [], [], [], []
offset = 0
for b in range(blocks):
    i0 = int(rng.integers(0, max(1, len(starts) - per_block)))
    blk = starts[i0:i0 + per_block]
    lo, hi = int(blk.min()), int(blk.max()) + H + T + 2
    Pm.append(P[lo:hi].astype(np.float32)); Am.append(A[lo:hi].astype(np.float32)); Cm.append(C[lo:hi].astype(np.float32)); Fm.append(FL[lo:hi].astype(np.float32))
    sel0.append(blk - lo + offset); offset += hi - lo
Pm, Am, Cm, Fm = (np.concatenate(z) for z in (Pm, Am, Cm, Fm)); sel0 = np.concatenate(sel0)
print(f"{len(sel0)} windows in {blocks} blocks ({Pm.shape[0]} steps in memory)", flush=True)
def gather(idx):
    tH = sel0[idx]; t = tH + H; tT = t + T
    r = np.arange(H + 1)
    p = Pm[tH[:, None] + r[None, :]]                      # (B, H+1, P)
    a = Am[tH[:, None] + np.arange(H)[None, :]]           # (B, H, A)
    c = Cm[t[:, None] + np.arange(T + 1)[None, :]]        # (B, T+1, C)
    y = Am[t[:, None] + np.arange(T)[None, :]]            # (B, T, A)
    fl = Fm[t[:, None] + np.arange(T)[None, :]]           # (B, T)
    return p, a, c, y, fl
scj = {k: (jnp.asarray(v["mean"]), jnp.asarray(v["std"]) + 1e-8) for k, v in sc.items()}
Pdim, Adim, Cdim = Pm.shape[1], Am.shape[1], Cm.shape[1]
in_dim = (H + 1) * Pdim + H * Adim + (T + 1) * Cdim
model = StudentMLP(in_dim, args.hidden, T * Adim, nnx.Rngs(args.seed))
sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, 500, args.steps)
opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=1e-4))

@nnx.jit
def step(model, opt, p, a, c, y, fl):
    def loss_fn(m):
        x = StudentPrior.features({"proprio_obs_hist": p, "acts_hist": a, "fut_cmds": c}, scj)
        pred = m(x).reshape(y.shape)
        ys = (y - scj["actions"][0]) / scj["actions"][1]
        mask = jnp.cumprod(fl, axis=-1)[:, :, None]
        return jnp.sum(((pred - ys) * mask) ** 2) / (jnp.sum(mask) * ys.shape[-1] + 1e-8)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(grads)
    return loss

t0 = time.time(); n = len(sel0); ema = None
for it in range(args.steps):
    idx = rng.integers(0, n, size=args.batch)
    p, a, c, y, fl = gather(idx)
    loss = float(step(model, opt, jnp.asarray(p), jnp.asarray(a), jnp.asarray(c), jnp.asarray(y), jnp.asarray(fl)))
    ema = loss if ema is None else 0.99 * ema + 0.01 * loss
    if it % 500 == 0 or it == args.steps - 1:
        print(f"step {it:6d} loss {loss:.4f} ema {ema:.4f} ({time.time()-t0:.0f}s)", flush=True)
out = os.path.join(students_dir(), args.name); os.makedirs(out, exist_ok=True)
_, state = nnx.split(model)
pickle.dump(state.to_pure_dict(), open(os.path.join(out, "params.pkl"), "wb"))
json.dump(sc, open(os.path.join(out, "scaler_params.json"), "w"))
json.dump({"H": H, "T": T, "act_dim": Adim, "in_dim": in_dim, "hidden": args.hidden, "dataset": args.dataset, "system": args.system, "steps": args.steps, "num_windows": int(n)}, open(os.path.join(out, "config.json"), "w"), indent=2)
print("saved", out)
