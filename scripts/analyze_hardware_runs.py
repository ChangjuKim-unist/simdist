"""Hardware experiment table (design doc 6 §9): falls, speed tracking and distance per model x condition.

Reads the controller's episode logs in datasets/real/hw_<model>_<condition>/raw/*.hdf5 (keys: time,
proprio_obs [base lin vel = first 3 dims], commands [vx, vy, wz], terminated). One log = one run.

    python scripts/analyze_hardware_runs.py            # all hw_* datasets
    python scripts/analyze_hardware_runs.py --glob 'hw_M3_*' --csv results/hardware/table.csv
"""
import argparse, csv, glob, os
import h5py, numpy as np
from simdist.utils import paths

ap = argparse.ArgumentParser(); ap.add_argument("--glob", default="hw_*"); ap.add_argument("--csv", default=None); ap.add_argument("--min_steps", type=int, default=50)
args = ap.parse_args()
root = os.path.dirname(paths.get_real_dataset_dir("x"))
runs = []
for d in sorted(glob.glob(os.path.join(root, args.glob))):
    name = os.path.basename(d)
    if "_" not in name[3:]: continue
    model, cond = name[3:].split("_", 1)[0], name[3:].split("_", 1)[1]
    for f in sorted(glob.glob(os.path.join(d, "raw", "*.hdf5"))):
        with h5py.File(f) as h:
            t = h["time"][:]; p = h["proprio_obs"][:]; c = h["commands"][:]
            term = bool(h["terminated"][-1]) if "terminated" in h else False
        n = len(t)
        if n < args.min_steps: continue
        dt = np.diff(t, prepend=t[0]); dt[0] = np.median(dt[1:]) if n > 1 else 0.02
        vx, vcmd = p[:, 0], c[:, 0]
        runs.append(dict(model=model, condition=cond, file=os.path.basename(f), steps=n, duration=float(t[-1] - t[0]), fell=int(term),
                         speed_err=float(np.mean(np.abs(vx - vcmd))), cmd_speed=float(np.mean(vcmd)), distance=float(np.sum(vx * dt))))
if not runs: raise SystemExit(f"no runs under {root}/{args.glob}")
rng = np.random.default_rng(0)
def ci(x, f=np.mean):
    x = np.asarray(x); b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]; return f(x), np.percentile(b, 2.5), np.percentile(b, 97.5)
groups = sorted({(r["model"], r["condition"]) for r in runs}, key=lambda k: (k[1], k[0]))
print(f"{'condition':12s} {'model':16s} runs  falls  falls/min [CI]        speed err [CI]        distance/run [CI]")
for m, cnd in groups:
    rs = [r for r in runs if r["model"] == m and r["condition"] == cnd]
    minutes = sum(r["duration"] for r in rs) / 60; falls = sum(r["fell"] for r in rs)
    fr = ci([r["fell"] for r in rs], lambda x: x.sum() / max(1e-9, sum(rr["duration"] for rr in rs) / 60 * len(x) / len(rs)))
    se = ci([r["speed_err"] for r in rs]); di = ci([r["distance"] for r in rs])
    print(f"{cnd:12s} {m:16s} {len(rs):4d}  {falls:5d}  {fr[0]:5.2f} [{fr[1]:.2f},{fr[2]:.2f}]   {se[0]:.3f} [{se[1]:.3f},{se[2]:.3f}]   {di[0]:5.1f} [{di[1]:.1f},{di[2]:.1f}]   ({minutes:.1f} min)")
if args.csv:
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys())); w.writeheader(); w.writerows(runs)
    print("wrote", args.csv)
