"""Stage 0 of docs/research_plan_ko.md: ambiguity of the student prior per physics condition.

The BC student (no privileged input) sees only the history; where several physics conditions
produce the same history but need different expert actions, its held-out error is large
(ambiguity term E_h Var_{xi|h}[a*] of design document 5). This script scores a trained student
on a fresh expert dataset that carries per-step physics labels and bins the error by each axis
(motor scale, static friction, added mass). The per-condition error is then compared with the
student-alone fall rate measured in the batched evaluator (Spearman, plan prediction P0).

    python scripts/sim2sim/ambiguity_by_condition.py --dataset <broad sim dataset> --system go1dr --student student_broad --out results/stage0/ambiguity_broad.json
"""
import argparse, json, os
import h5py, numpy as np, jax.numpy as jnp

from simdist.control.student_prior import StudentPrior
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1dr")
ap.add_argument("--student", default="student_broad"); ap.add_argument("--num_windows", type=int, default=200_000)
ap.add_argument("--bins", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
args = ap.parse_args()
AXES = ["motor", "friction", "mass"]

st = StudentPrior(args.student); H, T = st.H, st.T
d = paths.get_processed_data_dir(args.dataset, args.system, 25, 25)
P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"]
C = h5py.File(os.path.join(d, "cmds.hdf5"))["data"]; X = h5py.File(os.path.join(d, "physics.hdf5"))["data"]
starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
rng = np.random.default_rng(args.seed)
blocks, per = 20, max(1, min(args.num_windows, len(starts)) // 20)
Pm, Am, Cm, Xm, sel, off = [], [], [], [], [], 0
for b in range(blocks):
    i0 = int(rng.integers(0, max(1, len(starts) - per))); blk = starts[i0:i0 + per]
    lo, hi = int(blk.min()), int(blk.max()) + H + T + 2
    Pm.append(P[lo:hi]); Am.append(A[lo:hi]); Cm.append(C[lo:hi]); Xm.append(X[lo:hi]); sel.append(blk - lo + off); off += hi - lo
Pm, Am, Cm, Xm = (np.concatenate(z).astype(np.float32) for z in (Pm, Am, Cm, Xm)); sel = np.concatenate(sel)
a_std = np.asarray(st.scaler["actions"]["std"], np.float32) + 1e-8
err1, errT, xi = [], [], []
for i in range(0, len(sel), 4096):
    tH = sel[i:i + 4096]; t = tH + H
    x = {"proprio_obs_hist": Pm[tH[:, None] + np.arange(H + 1)[None, :]], "acts_hist": Am[tH[:, None] + np.arange(H)[None, :]],
         "fut_cmds": Cm[t[:, None] + np.arange(T + 1)[None, :]]}
    pred = np.asarray(st(x)); y = Am[t[:, None] + np.arange(T)[None, :]]
    e = ((pred - y) / a_std) ** 2                     # scaled squared error (B, T, A)
    err1.append(e[:, 0].mean(-1)); errT.append(e.mean((1, 2))); xi.append(Xm[t])
err1, errT, xi = np.concatenate(err1), np.concatenate(errT), np.concatenate(xi)
print(f"{len(err1)} windows; overall scaled MSE first step {err1.mean():.4f}, {T}-step {errT.mean():.4f}")
res = {"dataset": args.dataset, "student": args.student, "n": int(len(err1)), "mse_first": float(err1.mean()), "mse_seq": float(errT.mean()), "axes": {}}
for j, ax in enumerate(AXES):
    v = xi[:, j]
    if v.std() < 1e-6:
        print(f"{ax}: constant {v[0]:.3f}"); continue
    edges = np.quantile(v, np.linspace(0, 1, args.bins + 1)); edges[-1] += 1e-6
    rows = []
    for k in range(args.bins):
        m = (v >= edges[k]) & (v < edges[k + 1])
        rows.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(m.sum()), "mse_first": float(err1[m].mean()), "mse_seq": float(errT[m].mean())})
    res["axes"][ax] = rows
    print(f"--- {ax}"); [print(f"  [{r['lo']:6.3f}, {r['hi']:6.3f})  n {r['n']:6d}  first {r['mse_first']:.4f}  seq {r['mse_seq']:.4f}") for r in rows]
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); json.dump(res, open(args.out, "w"), indent=2)
