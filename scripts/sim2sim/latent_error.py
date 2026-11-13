"""Model-level forgetting metric (design doc 20): latent prediction error of a checkpoint on held-out logs.

For every processed window of each dataset the model predicts the 25 future latents from the history and the
logged actions; the error is the mean squared difference to the encoder's latent of the true future observation
(scaled units, the same quantity as the training loss ``latent_dynamics``). Reported per dataset: the mean over
all horizon steps, the first step (1-step error) and the last step (25-step error).

    python scripts/sim2sim/latent_error.py --ckpt go1_broadwm --datasets sim2sim_friction0.15_go1_broadwm_heldout_b15000 ... \
        --out results/stage20/latent_error.csv [--tag V0] [--per_window results/stage20/pw_<ckpt>_<dataset>.npz]

``--per_window`` writes the first-step error of every window with its time index ``t`` (appendix: 2-second-window
model selection). Append-only CSV; rows with the same (tag, ckpt, dataset) are skipped.
"""
import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from torch.utils.data import DataLoader

from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--system", default="go1")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--batch_size", type=int, default=256)
ap.add_argument("--per_window", default="", help="npz path pattern with {ckpt} and {dataset}")
ap.add_argument("--max_windows", type=int, default=0, help="0 = all")
args = ap.parse_args()

model, cfg, step = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), args.ckpt))
cfg = dict(cfg)
cfg["system"] = dict(cfg["system"])
cfg["system"]["name"] = args.system


@nnx.jit
def errors(model, x, y):
    y_pred = model(x, deterministic=True)
    y_s = model.get_scaler().scale(y)
    fut = model.encode_latent(y_s["proprio_obs"], y_s["extero_obs"], deterministic=True)
    return jnp.mean((y_pred["latents"] - fut) ** 2, axis=-1)  # (B, T)


done = set()
if os.path.exists(args.out):
    for r in csv.DictReader(open(args.out)):
        done.add((r["tag"], r["ckpt"], r["dataset"]))
new_file = not os.path.exists(args.out)
fh = open(args.out, "a", newline="")
w = csv.writer(fh)
if new_file:
    w.writerow(["tag", "ckpt", "step", "dataset", "n_windows", "mse_mean", "mse_h1", "mse_h25", "mse_h5"])

for ds in args.datasets:
    if (args.tag, args.ckpt, ds) in done:
        print(f"skip {args.ckpt} on {ds} (done)")
        continue
    c = dict(cfg)
    c["data"] = {"dataset_name": ds}
    dataset = get_dataset(c)
    dataset.eval()
    n = len(dataset) if args.max_windows <= 0 else min(len(dataset), args.max_windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    per_step, ts, seen = [], [], 0
    for batch in loader:
        b = mu.dataset_batch_to_jax(batch)
        e = np.asarray(errors(model, b["model_in"], b["labels"]))
        per_step.append(e)
        ts.append(np.asarray(batch["metadata"]["t"]))
        seen += e.shape[0]
        if seen >= n:
            break
    e = np.concatenate(per_step)[:n]
    t = np.concatenate(ts)[:n]
    row = [args.tag, args.ckpt, step, ds, int(e.shape[0]), float(e.mean()), float(e[:, 0].mean()), float(e[:, -1].mean()), float(e[:, :5].mean())]
    w.writerow(row)
    fh.flush()
    print("RESULT", ",".join(str(x) for x in row), flush=True)
    if args.per_window:
        p = args.per_window.format(ckpt=args.ckpt, dataset=ds)
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        np.savez(p, t=t, err_h1=e[:, 0].astype(np.float32), err_mean=e.mean(axis=1).astype(np.float32))
fh.close()
