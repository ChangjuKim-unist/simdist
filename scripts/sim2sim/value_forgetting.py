"""Measure how far an adapted model's value function moved from the pretrained one.

Samples windows from a processed dataset (normally the nominal simulation
dataset the model was pretrained on) and compares the value predictions of the
pretrained and the adapted checkpoint on the encoded true latents:

- rmse_vs_base:   RMSE between adapted and pretrained values (raw reward units)
- rmse_vs_label:  RMSE of each model against the dataset's value labels (the
                  simulation critic), i.e. the forgetting of the sim value function
- corr_vs_label:  Pearson correlation of each model's values with the labels

Run inside the simdist container, e.g.

    python scripts/sim2sim/value_forgetting.py --base go1_paper --adapted go1_res_lowfric \
        --dataset 2026-10-02_01-18-59 --num_windows 4096 --out results/forgetting.json
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import default_collate

from simdist.data.dataset import get_dataset
from simdist.utils import model as model_utils, paths


def load(name: str):
    ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), name)
    model, cfg, step = model_utils.load_model_from_ckpt(ckpt_dir)
    return model, cfg, step


def values_on_true_latents(model, batch):
    x = jax.tree.map(jnp.asarray, batch["model_in"])
    y = jax.tree.map(jnp.asarray, batch["labels"])
    scaler = model.get_scaler()
    x_s, y_s = scaler.scale(x), scaler.scale(y)
    latents = model.encode_latent(y_s["proprio_obs"], y_s["extero_obs"], deterministic=True)
    v_s = model.value_from_latents(
        latents, x_s["fut_cmds"][:, 1:], deterministic=True, use_residual=True
    )
    vp = scaler.get_scaler_params()["values"]
    return np.asarray(v_s * vp["std"] + vp["mean"]), np.asarray(y["values"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="pretrained checkpoint name")
    ap.add_argument("--adapted", required=True, nargs="+", help="adapted checkpoint name(s)")
    ap.add_argument("--dataset", required=True, help="processed dataset name (nominal sim data)")
    ap.add_argument("--num_windows", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="json file to append the results to")
    args = ap.parse_args()

    base_model, base_cfg, _ = load(args.base)
    cfg = dict(base_cfg)
    cfg["data"] = {"dataset_name": args.dataset}
    dataset = get_dataset(cfg)
    dataset.eval()
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=min(args.num_windows, len(dataset)), replace=False)

    models = {args.base: base_model}
    for name in args.adapted:
        models[name], _, _ = load(name)

    preds = {k: [] for k in models}
    labels = []
    for i in range(0, len(idxs), args.batch_size):
        batch = default_collate([dataset[int(j)] for j in idxs[i : i + args.batch_size]])
        batch = jax.tree.map(lambda t: t.numpy() if torch.is_tensor(t) else t, batch)
        for k, m in models.items():
            v, lab = values_on_true_latents(m, batch)
            preds[k].append(v.reshape(-1))
        labels.append(lab.reshape(-1))
    labels = np.concatenate(labels)
    preds = {k: np.concatenate(v) for k, v in preds.items()}

    results = {}
    base = preds[args.base]
    for k, v in preds.items():
        results[k] = {
            "rmse_vs_base": float(np.sqrt(np.mean((v - base) ** 2))),
            "rmse_vs_label": float(np.sqrt(np.mean((v - labels) ** 2))),
            "corr_vs_label": float(np.corrcoef(v, labels)[0, 1]),
            "mean_value": float(v.mean()),
        }
    results["_meta"] = {
        "dataset": args.dataset,
        "num_windows": int(len(idxs)),
        "label_std": float(labels.std()),
    }
    print(json.dumps(results, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        existing = {}
        if os.path.exists(args.out):
            with open(args.out) as f:
                existing = json.load(f)
        existing.update({k: v for k, v in results.items() if k != "_meta"})
        existing["_meta"] = results["_meta"]
        with open(args.out, "w") as f:
            json.dump(existing, f, indent=2)


if __name__ == "__main__":
    main()
