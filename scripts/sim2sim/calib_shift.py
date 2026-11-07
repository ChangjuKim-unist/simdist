"""Design doc 15 §7 distribution-shift check: score a trained calibration head on the executed candidates of other traces
(e.g. the calibrated planner's own runs at 16 iterations) and report coverage / gap / rank metrics against the realised return.

    python scripts/sim2sim/calib_shift.py --head results/stage14/head_broadwm_noF.npz --traces results/stage14/traces_shift/*.npz \
        /nas_data/cjkim/simdist/traces_stage13/T_broadwm_planner_friction0.15.npz --out results/stage14/shift_check.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_head import H_RET, auc, load, samples, spearman  # noqa: E402


def predict(w, X):
    Xn = (X - w["mu"]) / w["sd"]
    h = np.maximum(Xn @ w["w1"] + w["b1"], 0); h = np.maximum(h @ w["w2"] + w["b2"], 0); o = h @ w["w3"] + w["b3"]
    return o[:, :3] * float(w["g_sd"]) + float(w["g_mu"]), 1 / (1 + np.exp(-o[:, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    w = load(args.head); n_hist = int(w["n_hist"])
    out = {}
    for p in sorted(sum([glob.glob(t) for t in args.traces], [])):
        d = load(p); Xh, Xc, G, F, _ = samples(d, args.stride)
        X = np.concatenate([Xh.astype(np.float32), Xc], 1)
        q, pf = predict(w, X); jhat = X[:, n_hist + H_RET + 1]
        falls1k = float(d["fall"].sum() / d["fall"].shape[1] / d["fall"].shape[0] * 1000)
        res = {"model": str(d["model"]), "condition": str(d["condition"]), "n": int(len(G)), "falls_per_1k": round(falls1k, 2),
               "mean_G": float(G.mean()), "mean_jhat": float(jhat.mean()), "mean_q10": float(q[:, 0].mean()), "mean_q50": float(q[:, 1].mean()),
               "gap_jhat": float((jhat - G).mean()), "gap_q10": float((q[:, 0] - G).mean()), "gap_q50": float((q[:, 1] - G).mean()),
               "coverage_q10": float((G >= q[:, 0]).mean()), "coverage_q90": float((G <= q[:, 2]).mean()),
               "spearman_jhat": spearman(G, jhat), "spearman_q10": spearman(G, q[:, 0]), "spearman_q50": spearman(G, q[:, 1]),
               "mse_ratio_q50_vs_jhat": float(np.mean((G - q[:, 1]) ** 2) / np.mean((G - jhat) ** 2)), "fall_auc": auc(F, pf)}
        name = os.path.basename(p).replace(".npz", "")
        out[name] = res
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
