"""Diagnostics for a physics axis that the history probe cannot regress (stage 0, broad dataset: motor R^2 ~ 0).
(a) Does the axis affect the dynamics at all?  next-state R^2 from history vs history + true xi_j.
(b) Is a coarse regime decodable?  AUC of "xi_j below its 30th percentile" vs "above its 70th" from the history.
    python scripts/sim2sim/visibility_diag.py --dataset 2026-10-21_08-50-39 --system go1dr --axis motor
"""
import argparse, json, os, time
import h5py, numpy as np, jax.numpy as jnp, optax
from flax import nnx
from simdist.control.student_prior import StudentMLP
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1dr"); ap.add_argument("--axis", default="motor")
ap.add_argument("--history", type=int, default=25); ap.add_argument("--num_windows", type=int, default=300_000); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512]); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
args = ap.parse_args(); AXES = ["motor", "friction", "mass"]; j = AXES.index(args.axis)
d = paths.get_processed_data_dir(args.dataset, args.system, 25, 25)
P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"]; X = h5py.File(os.path.join(d, "physics.hdf5"))["data"]
starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]; sc = json.load(open(os.path.join(d, "scaler_params.json")))
H, Hp = 25, args.history; rng = np.random.default_rng(args.seed)
blocks, per = 20, max(1, min(args.num_windows, len(starts)) // 20); Pm, Am, Xm, sel, split, off = [], [], [], [], [], 0
for b in range(blocks):
    i0 = int(rng.integers(0, max(1, len(starts) - per))); blk = starts[i0:i0 + per]; lo, hi = int(blk.min()), int(blk.max()) + 2 * H + 2
    Pm.append(P[lo:hi]); Am.append(A[lo:hi]); Xm.append(X[lo:hi]); sel.append(blk - lo + off); split.append(np.full(len(blk), b >= blocks - 4)); off += hi - lo
Pm, Am, Xm = (np.concatenate(z).astype(np.float32) for z in (Pm, Am, Xm)); sel = np.concatenate(sel); test = np.concatenate(split)
def scale(x, k): return (x - np.asarray(sc[k]["mean"], np.float32)) / (np.asarray(sc[k]["std"], np.float32) + 1e-8)
xi_all = Xm[sel + H, j]; lo_q, hi_q = np.quantile(xi_all, 0.3), np.quantile(xi_all, 0.7); xm, xs = xi_all.mean(), xi_all.std() + 1e-8
def feats(idx, with_xi):
    t = sel[idx] + H; B = len(idx)
    p = scale(Pm[(t - Hp)[:, None] + np.arange(Hp + 1)[None, :]], "proprio_obs"); a = scale(Am[(t - Hp)[:, None] + np.arange(Hp + 1)[None, :]], "actions")
    f = np.concatenate([p.reshape(B, -1), a.reshape(B, -1)], -1)
    if with_xi: f = np.concatenate([f, ((Xm[t, j] - xm) / xs)[:, None]], -1)
    return f, scale(Pm[t + 1], "proprio_obs"), Xm[t, j]
def train(target, in_dim, out_dim, idx_tr, idx_te, name, binary=False):
    model = StudentMLP(in_dim, args.hidden, out_dim, nnx.Rngs(args.seed)); opt = nnx.Optimizer(model, optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-4, 300, args.steps), weight_decay=1e-4))
    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(m):
            o = m(x)
            if binary: l = o[:, 0]; return jnp.mean(jnp.maximum(l, 0) - l * y + jnp.log1p(jnp.exp(-jnp.abs(l))))
            return jnp.mean((o - y) ** 2)
        l, g = nnx.value_and_grad(loss_fn)(model); opt.update(g); return l
    t0 = time.time()
    for it in range(args.steps):
        x, y = target(idx_tr[rng.integers(0, len(idx_tr), args.batch)]); step(model, opt, jnp.asarray(x), jnp.asarray(y))
    pr, ys = [], []
    for i in range(0, len(idx_te), 4096):
        x, y = target(idx_te[i:i + 4096]); pr.append(np.asarray(model(jnp.asarray(x)))); ys.append(y)
    pr, yy = np.concatenate(pr), np.concatenate(ys)
    if binary:
        s = pr[:, 0]; pos, neg = s[yy == 1], s[yy == 0]; auc = float((pos[:, None] > neg[None, :]).mean()); print(f"  {name}: AUC {auc:.3f} ({time.time()-t0:.0f}s)", flush=True); return auc
    r2 = float(1 - ((pr - yy) ** 2).sum() / ((yy - yy.mean(0)) ** 2).sum()); print(f"  {name}: next-state R2 {r2:.4f} ({time.time()-t0:.0f}s)", flush=True); return r2
tr, te = np.where(~test)[0], np.where(test)[0]
print(f"windows {len(sel)}, axis {args.axis}: range {xi_all.min():.3f}-{xi_all.max():.3f}, 30/70 % quantiles {lo_q:.3f}/{hi_q:.3f}", flush=True)
in_dim = feats(np.array([0]), False)[0].shape[1]; out_dim = Pm.shape[1]
r_h = train(lambda i: feats(i, False)[:2], in_dim, out_dim, tr, te, "next state | history")
r_hx = train(lambda i: feats(i, True)[:2], in_dim + 1, out_dim, tr, te, f"next state | history + {args.axis}")
mask = lambda idx: idx[(Xm[sel[idx] + H, j] <= lo_q) | (Xm[sel[idx] + H, j] >= hi_q)]
trb, teb = mask(tr), mask(te)
auc = train(lambda i: (feats(i, False)[0], (Xm[sel[i] + H, j] >= hi_q).astype(np.float32)), in_dim, 1, trb, teb, f"{args.axis} high vs low | history", binary=True)
res = {"dataset": args.dataset, "axis": args.axis, "history": Hp, "next_state_R2_history": r_h, "next_state_R2_history_plus_xi": r_hx, "dynamics_gain_from_xi": r_hx - r_h, "regime_auc_history": auc}
print(json.dumps(res, indent=2))
if args.out: json.dump(res, open(args.out, "w"), indent=2)
