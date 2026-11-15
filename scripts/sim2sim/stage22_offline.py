"""Design doc 22 quick check: does the value of the last predicted latent foresee falls on held-out logs?

    python scripts/sim2sim/stage22_offline.py --dataset sim2sim_very_weak_motor_go1_paper_ho_b15000 \
        --ckpts B:go1_paper_very_weak_motor_dyn_b45000_s42 C:... D:... --out results/stage22q/offline.json

For every window (eval mode, no augmentation): V(z_hat_T) from the model's own prediction and V(E(o_T)) on the
encoded true latent, in raw value units. Label "fall soon": the episode ends by a fall within 50 steps after
the window end. AUC = P(V of a fall-soon window < V of a safe window) (higher = the value foresees falls).
Spearman of V(z_hat_T) with the logged reward-to-go at T on unmasked windows. Block bootstrap (500 windows)
for AUC differences against the first checkpoint and between C and D.
"""
import argparse
import json
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--ckpts", nargs="+", required=True, help="NAME:checkpoint")
ap.add_argument("--system", default="go1")
ap.add_argument("--fall_within", type=int, default=50)
ap.add_argument("--trunc_mask", type=int, default=300)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def rank(a):
    r = np.empty(len(a)); r[np.argsort(a, kind="mergesort")] = np.arange(len(a)); return r


def auc(score_low_is_fall, fall):
    s = -np.asarray(score_low_is_fall); f = np.asarray(fall, bool)
    if f.all() or (~f).all():
        return float("nan")
    r = rank(s) + 1
    n1, n0 = f.sum(), (~f).sum()
    return float((r[f].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


res = {"dataset": args.dataset, "models": {}}
labels = None
vals = {}
for spec in args.ckpts:
    name, ckpt = spec.split(":", 1)
    model, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), ckpt))
    cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
    cfg["data"] = {"dataset_name": args.dataset}
    ds = get_dataset(cfg); ds.eval()
    gd, st = nnx.split(model)
    vp = model.get_scaler().get_scaler_params()["values"]
    v_mean, v_std = np.asarray(vp["mean"]).reshape(()), np.asarray(vp["std"]).reshape(()) + 1e-8

    @jax.jit
    def f(x, y):
        m = nnx.merge(gd, st)
        out = m(x, deterministic=True)
        ys = m.get_scaler().scale(y); xs = m.get_scaler().scale(x)
        fut = m.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
        vt = m.value_from_latents(fut, xs["fut_cmds"][:, 1:], deterministic=True, use_residual=True)
        return out["values"][:, -1], vt[:, -1]

    vp_all, vt_all, lab = [], [], []
    for b in DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0):
        b = mu.dataset_batch_to_jax(b)
        a, c = f(b["model_in"], b["labels"])
        vp_all.append(np.asarray(a)); vt_all.append(np.asarray(c))
        if labels is None:
            L = b["labels"]
            lab.append(np.stack([np.asarray(L["steps_to_end"])[:, -1], np.asarray(L["terminated"])[:, -1],
                                 np.asarray(L["rtg"])[:, -1]], 1))
    if labels is None:
        labels = np.concatenate(lab)
    vals[name] = (np.concatenate(vp_all) * v_std + v_mean, np.concatenate(vt_all) * v_std + v_mean)
    print(f"{name}: {ckpt} windows {len(vals[name][0])}", flush=True)

ste, term, rtg = labels[:, 0], labels[:, 1] > 0.5, labels[:, 2]
fall = term & (ste <= args.fall_within)
valid = term | (ste >= args.trunc_mask)
res["n_windows"], res["n_fall_soon"] = int(len(fall)), int(fall.sum())
res["rtg_mean_safe"], res["rtg_mean_fall_soon"] = float(rtg[valid & ~fall].mean()), float(rtg[fall].mean())
for name, (vpred, vtrue) in vals.items():
    res["models"][name] = {
        "auc_pred": auc(vpred, fall), "auc_true": auc(vtrue, fall),
        "spearman_pred_rtg": spearman(vpred[valid], rtg[valid]),
        "mean_v_pred_safe": float(vpred[valid & ~fall].mean()), "mean_v_pred_fall_soon": float(vpred[fall].mean()),
        "rms_gap_pred_true": float(np.sqrt(np.mean((vpred - vtrue) ** 2))),
        "rms_err_pred_rtg": float(np.sqrt(np.mean((vpred[valid] - rtg[valid]) ** 2))),
    }

rng = np.random.default_rng(0)
blocks = np.array_split(np.arange(len(fall)), max(1, len(fall) // 500))
names = list(vals)
pairs = [(names[0], n) for n in names[1:]] + ([("D", "C")] if "C" in vals and "D" in vals else [])
res["auc_diff"] = {}
for a, b in pairs:
    d = []
    for _ in range(2000):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        d.append(auc(vals[b][0][idx], fall[idx]) - auc(vals[a][0][idx], fall[idx]))
    d = np.asarray(d)
    res["auc_diff"][f"{b}-{a}"] = [res["models"][b]["auc_pred"] - res["models"][a]["auc_pred"],
                                   float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
