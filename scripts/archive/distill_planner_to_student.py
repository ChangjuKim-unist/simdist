"""Design doc 7 (PFI step iii): update a student prior with the planner's *executed* actions.

Reads a processed real-format log (datasets/real/<name>/processed_data_<system>_H-25_T-25) in
which the executed actions are the planner's outputs on the real/simulated robot, and
behavior-clones the student on those windows, mixed with windows of the student's original
simulation dataset (trust region: a fraction beta of each batch comes from the planner log).
The student is initialized from an existing student (--init). No imagination is used.

    python scripts/distill_planner_to_student.py --init student_broad --name student_pfi1 \
        --real sim2sim_very_weak_motor_priorfirst_b45000 --sim 2026-10-21_08-50-39 --sim_system go1dr --beta 0.3
"""
import argparse, json, os, pickle, time
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx
from simdist.control.student_prior import StudentMLP, StudentPrior, students_dir
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--real", required=True, nargs="+", help="processed real-format dataset name(s) (planner logs)"); ap.add_argument("--real_system", default="go1")
ap.add_argument("--sim", required=True, help="simulation dataset of the original student"); ap.add_argument("--sim_system", default="go1dr")
ap.add_argument("--beta", type=float, default=0.3, help="fraction of each batch from the planner logs")
ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--sim_windows", type=int, default=500_000); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
st = StudentPrior(args.init); H, T, A = st.H, st.T, st.A; sc = st.scaler
scj = {k: (jnp.asarray(v["mean"]), jnp.asarray(v["std"]) + 1e-8) for k, v in sc.items()}
rng = np.random.default_rng(args.seed)

def load(d, num_windows, flags=True):
    P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; Ac = h5py.File(os.path.join(d, "acts.hdf5"))["data"]; C = h5py.File(os.path.join(d, "cmds.hdf5"))["data"]
    FL = h5py.File(os.path.join(d, "expert_policy_flag.hdf5"))["data"] if flags and os.path.exists(os.path.join(d, "expert_policy_flag.hdf5")) else None
    starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
    blocks, per = 20, max(1, min(num_windows, len(starts)) // 20); Pm, Am, Cm, Fm, sel, off = [], [], [], [], [], 0
    for b in range(blocks):
        i0 = int(rng.integers(0, max(1, len(starts) - per))); blk = starts[i0:i0 + per]; lo, hi = int(blk.min()), int(blk.max()) + H + T + 2
        Pm.append(P[lo:hi]); Am.append(Ac[lo:hi]); Cm.append(C[lo:hi]); Fm.append(FL[lo:hi] if FL is not None else np.ones(hi - lo, np.float32)); sel.append(blk - lo + off); off += hi - lo
    return [np.concatenate(z).astype(np.float32) for z in (Pm, Am, Cm, Fm)] + [np.concatenate(sel)]
sim = load(paths.get_processed_data_dir(args.sim, args.sim_system, H, T), args.sim_windows)
reals = [load(os.path.join(paths.get_real_dataset_dir(r), f"processed_data_{args.real_system}_H-25_T-25"), 10**9, flags=False) for r in args.real]
# concatenate the planner logs
off = 0; Pr, Ar, Cr, Fr, Sr = [], [], [], [], []
for Pm, Am, Cm, Fm, sel in reals:
    Pr.append(Pm); Ar.append(Am); Cr.append(Cm); Fr.append(Fm); Sr.append(sel + off); off += len(Pm)
real = [np.concatenate(Pr), np.concatenate(Ar), np.concatenate(Cr), np.concatenate(Fr), np.concatenate(Sr)]
print(f"sim windows {len(sim[4])}, planner-log windows {len(real[4])} from {args.real}", flush=True)
def gather(data, idx):
    Pm, Am, Cm, Fm, sel = data; tH = sel[idx]; t = tH + H
    return (Pm[tH[:, None] + np.arange(H + 1)[None, :]], Am[tH[:, None] + np.arange(H)[None, :]], Cm[t[:, None] + np.arange(T + 1)[None, :]],
            Am[t[:, None] + np.arange(T)[None, :]], Fm[t[:, None] + np.arange(T)[None, :]])
model = st.model
opt = nnx.Optimizer(model, optax.adamw(optax.warmup_cosine_decay_schedule(0.0, args.lr, 200, args.steps), weight_decay=1e-4))
@nnx.jit
def step(model, opt, p, a, c, y, fl):
    def loss_fn(m):
        x = StudentPrior.features({"proprio_obs_hist": p, "acts_hist": a, "fut_cmds": c}, scj)
        pred = m(x).reshape(y.shape); ys = (y - scj["actions"][0]) / scj["actions"][1]
        mask = jnp.cumprod(fl, axis=-1)[:, :, None]
        return jnp.sum(((pred - ys) * mask) ** 2) / (jnp.sum(mask) * ys.shape[-1] + 1e-8)
    loss, grads = nnx.value_and_grad(loss_fn)(model); opt.update(grads); return loss
nb_real = int(round(args.beta * args.batch)); t0 = time.time(); ema = None
for it in range(args.steps):
    gs = gather(sim, rng.integers(0, len(sim[4]), args.batch - nb_real)); gr = gather(real, rng.integers(0, len(real[4]), nb_real))
    batch = [np.concatenate([a, b]) for a, b in zip(gs, gr)]
    loss = float(step(model, opt, *[jnp.asarray(z) for z in batch])); ema = loss if ema is None else 0.99 * ema + 0.01 * loss
    if it % 500 == 0 or it == args.steps - 1: print(f"step {it:6d} loss {loss:.4f} ema {ema:.4f} ({time.time()-t0:.0f}s)", flush=True)
out = os.path.join(students_dir(), args.name); os.makedirs(out, exist_ok=True)
pickle.dump(nnx.split(model)[1].to_pure_dict(), open(os.path.join(out, "params.pkl"), "wb"))
json.dump(sc, open(os.path.join(out, "scaler_params.json"), "w"))
cfg = dict(st.cfg); cfg.update({"init": args.init, "real": args.real, "beta": args.beta, "distill_steps": args.steps}); json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
print("saved", out)
