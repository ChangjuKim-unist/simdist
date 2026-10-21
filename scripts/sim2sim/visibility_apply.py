"""Apply a sim-trained history -> xi probe (visibility_probe.py --save) to any processed log
(real robot or sim-to-sim condition) and compare the estimated physics parameters with a
reference log. No labels are needed on the applied logs: the probe's estimate of each axis
shifts between the condition log and the reference log only if that axis is visible in the
history, so the per-axis separation (Cohen's d of the window-level estimates, and the shift of
the mean in units of the probe's training std) is a label-free, per-axis visibility statistic.

    python scripts/sim2sim/visibility_apply.py --probe checkpoints/probes/broad --dataset sim2sim_very_weak_motor_go1_paper_b45000 --real --reference sim2sim_nominal_go1_paper --reference_real
"""
import argparse, json, os, pickle
import h5py, numpy as np, jax.numpy as jnp
from flax import nnx

from simdist.control.student_prior import StudentMLP
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1")
ap.add_argument("--real", action="store_true"); ap.add_argument("--reference", default=None); ap.add_argument("--reference_real", action="store_true")
ap.add_argument("--reference_system", default="go1"); ap.add_argument("--num_windows", type=int, default=50_000); ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = json.load(open(os.path.join(args.probe, "config.json"))); sc = json.load(open(os.path.join(args.probe, "scaler_params.json")))
model = StudentMLP(cfg["in_dim"], cfg["hidden"], len(cfg["axes"]), nnx.Rngs(0))
gd, st = nnx.split(model); st.replace_by_pure_dict(pickle.load(open(os.path.join(args.probe, "params.pkl"), "rb"))); model = nnx.merge(gd, st)
H, Hp = 25, cfg["history"]; xm, xs = np.array(cfg["xi_mean"]), np.array(cfg["xi_std"])
def scale(x, k): return (x - np.asarray(sc[k]["mean"], np.float32)) / (np.asarray(sc[k]["std"], np.float32) + 1e-8)

def estimates(dataset, real, system):
    d = os.path.join(paths.get_real_dataset_dir(dataset), f"processed_data_{system}_H-25_T-25") if real else paths.get_processed_data_dir(dataset, system, 25, 25)
    P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"][:]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"][:]; C = h5py.File(os.path.join(d, "cmds.hdf5"))["data"][:]
    starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
    X = h5py.File(os.path.join(d, "physics.hdf5"))["data"][:] if os.path.exists(os.path.join(d, "physics.hdf5")) else None
    sel = np.sort(np.random.default_rng(0).choice(starts, min(args.num_windows, len(starts)), replace=False)); out = []
    for i in range(0, len(sel), 4096):
        t = sel[i:i + 4096] + H; B = len(t)
        p = scale(P[(t - Hp)[:, None] + np.arange(Hp + 1)[None, :]], "proprio_obs"); a = scale(A[(t - Hp)[:, None] + np.arange(Hp)[None, :]], "actions")
        c = scale(C[t[:, None] + np.arange(cfg.get("horizon", 25) + 1)[None, :]], "commands")   # commands over the probe's horizon
        f = np.concatenate([p.reshape(B, -1), a.reshape(B, -1), c.reshape(B, -1)], -1)
        out.append(np.asarray(model(jnp.asarray(f))) * xs + xm)
    est = np.concatenate(out); lab = X[sel + H] if X is not None else None
    return est, lab
est, lab = estimates(args.dataset, args.real, args.system)
res = {"probe": args.probe, "dataset": args.dataset, "reference": args.reference, "axes": {}}
ref, _ = estimates(args.reference, args.reference_real, args.reference_system) if args.reference else (None, None)
for j, ax in enumerate(cfg["axes"]):
    r = {"est_mean": float(est[:, j].mean()), "est_std": float(est[:, j].std())}
    if lab is not None: r["label_mean"] = float(lab[:, j].mean())
    if ref is not None:
        r["ref_mean"] = float(ref[:, j].mean()); r["ref_std"] = float(ref[:, j].std())
        pooled = np.sqrt(0.5 * (est[:, j].var() + ref[:, j].var())) + 1e-8
        r["cohen_d"] = float((est[:, j].mean() - ref[:, j].mean()) / pooled); r["shift_in_train_std"] = float((est[:, j].mean() - ref[:, j].mean()) / xs[j])
    res["axes"][ax] = r
    print(f"{ax:9s} est {r['est_mean']:7.3f} ± {r['est_std']:.3f}" + (f"  label {r['label_mean']:7.3f}" if lab is not None else "") + (f"  ref {r['ref_mean']:7.3f} ± {r['ref_std']:.3f}  d {r['cohen_d']:+.2f}  shift/std {r['shift_in_train_std']:+.2f}" if ref is not None else ""))
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); json.dump(res, open(args.out, "w"), indent=2)
