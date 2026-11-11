"""Upstream-vs-fork equivalence check (docs/audit_upstream_ko.md, section 3).

Runs the same checkpoint on the same inputs with the original SimDist code and with this
fork (all optional additions off) and reports the maximum absolute difference of the model
outputs and of the planner's action sequences. Also checks the batched planner
(simdist/control/mppi_batch.py) against the single-robot planner with the same random key,
and the checkpoint's scaler statistics against the processed dataset's scaler_params.json.

    # original code (PYTHONPATH puts the upstream worktree first)
    PYTHONPATH=/workspace/upstream python scripts/audit/equivalence_check.py --role upstream --out results/audit/upstream.npz
    # fork
    python scripts/audit/equivalence_check.py --role fork --ref results/audit/upstream.npz --out results/audit/fork.npz

Inputs are real windows of the processed pretraining dataset (h5 files read directly, so
both code bases see identical arrays).
"""

import argparse
import json
import os

import h5py
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--role", choices=["upstream", "fork"], required=True)
parser.add_argument("--ckpt", default="/workspace/simdist/checkpoints/models/go1_paper")
parser.add_argument("--data", default="/workspace/simdist/datasets/sim/2026-10-02_01-18-59/processed_data_go1_H-25_T-25")
parser.add_argument("--ref", default=None, help="npz written by the other role, to compare against")
parser.add_argument("--out", required=True)
parser.add_argument("--windows", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import jax
import jax.numpy as jnp
import simdist
from simdist.utils import model as model_utils, paths
from simdist.control.mppi import MppiController

print(f"role={args.role} simdist package: {os.path.dirname(simdist.__file__)}  jax {jax.__version__} {jax.devices()}")

H = T = 25
# ---- inputs: real windows ----------------------------------------------------------
def h5(name):
    return h5py.File(os.path.join(args.data, name), "r")["data"]

P, E, A, C = h5("proprio_obs.hdf5"), h5("extero_obs.hdf5"), h5("acts.hdf5"), h5("cmds.hdf5")
n = P.shape[0]
rng = np.random.default_rng(0)
ts = np.sort(rng.integers(H + 2, n - T - 2, size=args.windows))
x = {
    "proprio_obs_hist": np.stack([P[t - H : t + 1] for t in ts]).astype(np.float32),  # (K, H+1, 35)
    "extero_obs": np.stack([E[t] for t in ts]).astype(np.float32),  # (K, 315)
    "acts_hist": np.stack([A[t - H : t] for t in ts]).astype(np.float32),  # (K, H, 12)
    "fut_acts": np.stack([A[t : t + T] for t in ts]).astype(np.float32),  # (K, T, 12)
    "fut_cmds": np.stack([C[t : t + T + 1] for t in ts]).astype(np.float32),  # (K, T+1, 3)
}
print("window starts:", ts.tolist())

# ---- model -------------------------------------------------------------------------
model, model_cfg, step = model_utils.load_model_from_ckpt(args.ckpt, None)
print(f"checkpoint step {step}, system {model_cfg['system']['name']}")
y = model.inference({k: jnp.asarray(v) for k, v in x.items()})
out = {f"model_{k}": np.asarray(v) for k, v in y.items()}

# scaler statistics of the checkpoint vs the dataset's scaler_params.json
sp = model.get_scaler().get_scaler_params()
ds_sp = json.load(open(os.path.join(args.data, "scaler_params.json")))
for key, dkey in [("proprio_obs", "proprio_obs"), ("extero_obs", "extero_obs"), ("actions", "actions"),
                  ("commands", "commands"), ("rewards", "rewards"), ("values", "values")]:
    if key in sp:
        for stat in ("mean", "std"):
            d = float(np.max(np.abs(np.asarray(sp[key][stat]).ravel() - np.asarray(ds_sp[dkey][stat]).ravel())))
            out[f"scaler_diff_{key}_{stat}"] = d
            print(f"scaler {key}.{stat}: checkpoint vs dataset max abs diff {d:.3e}")
    else:
        print(f"scaler key {key} not in checkpoint scaler: {list(sp.keys())}")

# ---- single-robot planner (same class in both code bases) --------------------------
ctrl_cfg = yaml.safe_load(open(os.path.join(paths.get_config_dir(), "control", "mppi.yaml")))
ctrl_cfg["seed"] = args.seed
print("control cfg:", {k: v for k, v in ctrl_cfg.items() if not isinstance(v, dict)})
t0 = int(ts[0])


def ctrl_in(i):  # measurement i steps after the window start t0 - H
    t = t0 - H + i
    return {"proprio_obs": P[t].astype(np.float32), "extero_obs": E[t].astype(np.float32),
            "prev_action": A[t - 1].astype(np.float32)}


ctrl = MppiController(model, model_cfg, ctrl_cfg)
ctrl.initialize(ctrl_in(0), C[t0 - H].astype(np.float32))
ctrl.reset(ctrl_in(0), C[t0 - H].astype(np.float32))
for i in range(1, H + 1):
    ctrl.update(ctrl_in(i))
ctrl.set_fut_cmd(C[t0].astype(np.float32))
key0 = np.array(ctrl.mppi_state.key_data)
mi = ctrl._make_model_inputs(ctrl.dummy_actions)
for k in ("proprio_obs_hist", "extero_obs", "acts_hist"):  # fut_cmds: the controller holds the command constant
    assert np.allclose(np.asarray(mi[k]), x[k][0]), f"controller buffer differs from window 0 for {k}"
act1 = np.array(ctrl.run_control()["actions"])
ctrl.update(ctrl_in(H + 1))
act2 = np.array(ctrl.run_control()["actions"])
out.update(planner_key0=key0, planner_act1=act1, planner_act2=act2)
print(f"planner step 1 mean |a| {np.abs(act1).mean():.4f}, step 2 {np.abs(act2).mean():.4f}")

# ---- fork only: batched planner with the same key, prior-only path ------------------
if args.role == "fork":
    from simdist.control.mppi_batch import BatchedMppiController

    b = BatchedMppiController(model, model_cfg, ctrl_cfg, 1, [args.seed])
    c0 = ctrl_in(0)
    b.reset([0], c0["proprio_obs"][None], c0["extero_obs"][None], c0["prev_action"][None], C[t0 - H][None].astype(np.float32))
    for i in range(1, H + 1):
        ci = ctrl_in(i)
        b.update(ci["proprio_obs"][None], ci["extero_obs"][None], ci["prev_action"][None])
    b.set_fut_cmd(C[t0][None].astype(np.float32))
    b.state = b.state.replace(key_data=jnp.asarray(key0[None]))
    bx = b._model_inputs()
    for k in ("proprio_obs_hist", "extero_obs", "acts_hist"):
        assert np.allclose(np.asarray(bx[k])[0], x[k][0]), f"batched buffer differs for {k}"
    assert np.allclose(np.asarray(bx["fut_cmds"])[0], np.asarray(mi["fut_cmds"])), "batched fut_cmds differ from the single controller"
    bact1 = b.run_control()[0]
    ci = ctrl_in(H + 1)
    b.update(ci["proprio_obs"][None], ci["extero_obs"][None], ci["prev_action"][None])
    bact2 = b.run_control()[0]
    out["batched_act1_diff"] = float(np.max(np.abs(bact1 - act1)))
    out["batched_act2_diff"] = float(np.max(np.abs(bact2 - act2)))
    print(f"batched vs single planner: step 1 max abs diff {out['batched_act1_diff']:.3e}, step 2 {out['batched_act2_diff']:.3e}")
    # prior_only: the executed plan must be the policy head's own action sequence
    cfg_po = dict(ctrl_cfg); cfg_po["prior_only"] = True
    bp = BatchedMppiController(model, model_cfg, cfg_po, 1, [args.seed])
    bp.reset([0], c0["proprio_obs"][None], c0["extero_obs"][None], c0["prev_action"][None], C[t0 - H][None].astype(np.float32))
    for i in range(1, H + 1):
        ci = ctrl_in(i)
        bp.update(ci["proprio_obs"][None], ci["extero_obs"][None], ci["prev_action"][None])
    bp.set_fut_cmd(C[t0][None].astype(np.float32))
    pa = bp.run_control()[0]
    # reference: the policy head on the controller's own inputs (constant command over the horizon)
    head_ref = None  # set below from the prior-only single controller's own buffer (same history as bp)
    sp_ = MppiController(model, model_cfg, cfg_po)
    sp_.initialize(ctrl_in(0), C[t0 - H].astype(np.float32)); sp_.reset(ctrl_in(0), C[t0 - H].astype(np.float32))
    for i in range(1, H + 1):
        sp_.update(ctrl_in(i))
    sp_.set_fut_cmd(C[t0].astype(np.float32))
    mi1 = {k: jnp.asarray(np.asarray(v))[None] for k, v in sp_._make_model_inputs(sp_.dummy_actions).items()}
    head_ref = np.asarray(model.inference(mi1)["actions"][0])
    out["prior_only_diff"] = float(np.max(np.abs(pa - head_ref)))
    print(f"prior_only plan vs policy head actions (window 0): max abs diff {out['prior_only_diff']:.3e}")
    out["prior_only_single_diff"] = float(np.max(np.abs(np.array(sp_.run_control()["actions"]) - head_ref)))
    print(f"single-robot prior_only vs policy head: max abs diff {out['prior_only_single_diff']:.3e}")

# ---- compare with the other role ---------------------------------------------------
if args.ref:
    ref = np.load(args.ref)
    print(f"\ncompare with {args.ref}:")
    for k in ("model_latents", "model_rewards", "model_values", "model_actions", "planner_key0", "planner_act1", "planner_act2"):
        d = float(np.max(np.abs(np.asarray(out[k], np.float64) - np.asarray(ref[k], np.float64))))
        scale = float(np.max(np.abs(np.asarray(ref[k], np.float64))))
        out[f"diff_{k}"] = d
        print(f"  {k:16s} max abs diff {d:.3e}  (max |ref| {scale:.3e})")

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, **out)
print(f"written {args.out}")
