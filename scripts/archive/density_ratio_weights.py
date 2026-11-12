"""Design doc 9: deployment-matched weights for prior (student) training.

Trains a classifier on history-window features (the student's own inputs) to tell windows of the
deployed system's real-format logs ("real") from windows of the simulation expert dataset ("sim"), and
turns its output into a density ratio r = p(real|x) / p(sim|x) * (n_sim / n_real) for every sim window.
Weighted BC on the sim dataset (scripts/train_student_prior.py --weights <file>) then trains the prior
more on the states the deployed system actually visits, while the labels stay the expert's.
Also reports a coverage statistic: the fraction of real windows the classifier separates with
p(real) > 0.95 (states without simulation support -> widen the DR there).

    python scripts/density_ratio_weights.py --sim <go1dr dataset> --real sim2sim_very_weak_motor_priorfirst_b45000 --out results/stage4/w_motor04.npz
"""
import argparse, json, os, time
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx
from simdist.control.student_prior import StudentMLP
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True); ap.add_argument("--sim_system", default="go1dr"); ap.add_argument("--real", required=True, nargs="+"); ap.add_argument("--real_system", default="go1")
ap.add_argument("--num_sim", type=int, default=400_000); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--hidden", type=int, nargs="+", default=[64]); ap.add_argument("--clip", type=float, default=20.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--features", default="stats", choices=["all", "proprio", "stats"], help="all = the student's inputs (actions included: separates *behaviour*, degenerate); proprio = observation history only; stats = per-window summary statistics of the observations (policy-invariant, coarse)")
ap.add_argument("--alpha", type=float, default=0.5, help="tempering of the density ratio, w = r^alpha")
args = ap.parse_args(); H = T = 25; rng = np.random.default_rng(args.seed)
sd = paths.get_processed_data_dir(args.sim, args.sim_system, H, T); sc = json.load(open(os.path.join(sd, "scaler_params.json")))
def scale(x, k): return (x - np.asarray(sc[k]["mean"], np.float32)) / (np.asarray(sc[k]["std"], np.float32) + 1e-8)
def feats_from(P, A, C, tH):
    t = tH + H; B = len(tH)
    p = scale(np.stack([P[s:s + H + 1] for s in tH]), "proprio_obs")
    if args.features == "stats":  # mean / std / first-last difference of each observation dim over the window
        return np.concatenate([p.mean(1), p.std(1), p[:, -1] - p[:, 0]], -1).astype(np.float32)
    if args.features == "proprio":
        return p.reshape(B, -1).astype(np.float32)
    a = scale(np.stack([A[s:s + H] for s in tH]), "actions"); c = scale(np.stack([C[s:s + T + 1] for s in t]), "commands")
    return np.concatenate([p.reshape(B, -1), a.reshape(B, -1), c.reshape(B, -1)], -1).astype(np.float32)
# sim windows (all start indices; features for a random subset for training, weights for all)
Ps, As, Cs = (h5py.File(os.path.join(sd, f))["data"] for f in ("proprio_obs.hdf5", "acts.hdf5", "cmds.hdf5")); starts_s = h5py.File(os.path.join(sd, "start_idxs.hdf5"))["data"][:]
# real windows
Xr = []
for r in args.real:
    d = os.path.join(paths.get_real_dataset_dir(r), f"processed_data_{args.real_system}_H-25_T-25")
    P, A, C = (h5py.File(os.path.join(d, f))["data"] for f in ("proprio_obs.hdf5", "acts.hdf5", "cmds.hdf5")); st = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
    for i in range(0, len(st), 4096): Xr.append(feats_from(P, A, C, st[i:i + 4096]))
Xr = np.concatenate(Xr); n_real = len(Xr)
sub = np.sort(rng.choice(starts_s, min(args.num_sim, len(starts_s)), replace=False)); Xs = np.concatenate([feats_from(Ps, As, Cs, sub[i:i + 4096]) for i in range(0, len(sub), 4096)])
print(f"real windows {n_real}, sim windows for training {len(Xs)} (of {len(starts_s)})", flush=True)
# held-out split for calibration check
perm_r = rng.permutation(n_real); perm_s = rng.permutation(len(Xs)); nr_te, ns_te = n_real // 5, len(Xs) // 5
Xr_tr, Xr_te = Xr[perm_r[nr_te:]], Xr[perm_r[:nr_te]]; Xs_tr, Xs_te = Xs[perm_s[ns_te:]], Xs[perm_s[:ns_te]]
model = StudentMLP(Xr.shape[1], args.hidden, 1, nnx.Rngs(args.seed)); opt = nnx.Optimizer(model, optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-4, 200, args.steps), weight_decay=1e-4))
@nnx.jit
def step(model, opt, x, y):
    def loss_fn(m):
        l = m(x)[:, 0]; return jnp.mean(jnp.maximum(l, 0) - l * y + jnp.log1p(jnp.exp(-jnp.abs(l))))
    l, g = nnx.value_and_grad(loss_fn)(model); opt.update(g); return l
t0 = time.time(); half = args.batch // 2
for it in range(args.steps):  # balanced batches: the logit then estimates log p(real|x)/p(sim|x) up to the prior ratio
    x = np.concatenate([Xr_tr[rng.integers(0, len(Xr_tr), half)], Xs_tr[rng.integers(0, len(Xs_tr), half)]]); y = np.concatenate([np.ones(half), np.zeros(half)]).astype(np.float32)
    step(model, opt, jnp.asarray(x), jnp.asarray(y))
def logits(X): return np.concatenate([np.asarray(model(jnp.asarray(X[i:i + 8192])))[:, 0] for i in range(0, len(X), 8192)])
lr, ls = logits(Xr_te), logits(Xs_te); auc = float((lr[:, None] > ls[None, :]).mean())
print(f"classifier AUC (held-out) {auc:.3f}; real windows with p(real)>0.95: {float((1/(1+np.exp(-lr)) > 0.95).mean()):.3f} (no sim support -> widen DR)  ({time.time()-t0:.0f}s)", flush=True)
# density-ratio weights for every sim window (balanced-batch logit = log density ratio)
lg = np.concatenate([logits(feats_from(Ps, As, Cs, starts_s[i:i + 8192])) for i in range(0, len(starts_s), 8192)])  # chunked: 1.8M windows
r = np.exp(np.clip(lg, -args.clip, np.log(args.clip))) ** args.alpha; w = r / r.mean()
print(f"sim weights: mean 1, median {np.median(w):.3f}, max {w.max():.1f}, effective sample fraction {w.sum()**2/(len(w)*(w**2).sum()):.3f}, top-10% windows carry {np.sort(w)[-len(w)//10:].sum()/w.sum():.2f} of the mass", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); np.savez(args.out, start_idxs=starts_s, weights=w.astype(np.float32), auc=auc, features=args.features, alpha=args.alpha)
print("saved", args.out)
