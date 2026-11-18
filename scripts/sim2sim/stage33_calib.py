"""Design doc 33 (docs/design33_two_model_ko.md §2.3): threshold of the online one-step error switch.

    python scripts/sim2sim/stage33_calib.py --diag <diag of the calibration run> --out <theta.txt> [--q 95]

The calibration run drives the adapted model on its own condition with the policy start and no switch, on robots that
are not used in the evaluation. theta = q-th percentile of the window mean of the one-step latent error (the number the
switch compares, recorded as err_mean) over the steps where the planner acted (mode 0).
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True); ap.add_argument("--out", required=True); ap.add_argument("--q", type=float, default=95.0)
a = ap.parse_args()
z = np.load(a.diag, allow_pickle=True)
em = z["err_mean"].astype(np.float64); ok = (z["mode"] == 0) & (~z["resetting"].astype(bool)) & np.isfinite(em)
assert ok.sum() >= 100, f"too few planner steps with a full error window: {int(ok.sum())}"
theta = float(np.percentile(em[ok], a.q))
open(a.out, "w").write(f"{theta:.6f}\n")
print(f"theta {theta:.6f} (q {a.q}, {int(ok.sum())} steps, median {np.median(em[ok]):.4f}) -> {a.out}", flush=True)
