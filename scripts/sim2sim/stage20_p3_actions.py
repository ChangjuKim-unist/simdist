"""Design doc 20, P3 check without the simulator: the policy head's actions must be bit-identical to the
pretrained model's on the same inputs for every adapted checkpoint (only the latent dynamics changed).

    python scripts/sim2sim/stage20_p3_actions.py --base go1_broadwm --ckpts go1_broadwm_friction0.15_dyn_b45000_s42 ... \
        --dataset sim2sim_friction0.15_go1_paper_b45000 --out results/stage20/p3_actions.csv

Columns: base, ckpt, step, n_windows, max_abs_action_diff (expect 0), max_abs_latent_diff (expect > 0 for an adapted
checkpoint), max_abs_value_diff, max_abs_reward_diff.
"""
import argparse
import csv
import os

import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from torch.utils.data import DataLoader

from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--system", default="go1")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def outputs(ckpt):
    model, cfg, step = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), ckpt))
    cfg = dict(cfg)
    cfg["system"] = dict(cfg["system"])
    cfg["system"]["name"] = args.system
    cfg["data"] = {"dataset_name": args.dataset}
    dataset = get_dataset(cfg)
    dataset.eval()
    loader = DataLoader(dataset, batch_size=args.n, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    b = mu.dataset_batch_to_jax(batch)
    fn = nnx.jit(lambda m, x: m(x, deterministic=True))
    y = fn(model, b["model_in"])
    return step, {k: np.asarray(v) for k, v in y.items() if k in ("actions", "latents", "values", "rewards")}


step0, ref = outputs(args.base)
new = not os.path.exists(args.out)
fh = open(args.out, "a", newline="")
w = csv.writer(fh)
if new:
    w.writerow(["base", "ckpt", "step", "n_windows", "max_abs_action_diff", "max_abs_latent_diff", "max_abs_value_diff", "max_abs_reward_diff"])
for ck in args.ckpts:
    step, y = outputs(ck)
    d = {k: float(np.max(np.abs(y[k] - ref[k]))) for k in ref}
    row = [args.base, ck, step, int(ref["actions"].shape[0]), d["actions"], d["latents"], d["values"], d["rewards"]]
    w.writerow(row)
    fh.flush()
    print("RESULT", ",".join(str(x) for x in row), flush=True)
fh.close()
