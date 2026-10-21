"""Stage 0 of docs/research_plan_ko.md: how visible is each physics axis from the robot's history?

For a processed simulation dataset that carries per-step physics labels xi = [motor scale,
static friction, added mass] (physics.hdf5), train a probe that regresses xi from the same
inputs the planner prior sees -- the last H+1 proprioceptive observations, the last H actions
and the commands -- and report the held-out R^2 per axis ("visibility" V_j). Also reports the
action-sensitivity proxy: the R^2 gain of predicting the expert's next action when xi_j is
added to the history (how much the right action depends on the axis beyond what the history
already tells).

    python scripts/sim2sim/visibility_probe.py --dataset <sim dataset> --system go1dr --out results/stage0/visibility_broad.json
    python scripts/sim2sim/visibility_probe.py --dataset <sim dataset> --system go1 --history 5   # shorter history
"""
import argparse, json, os, pickle, time
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx

from simdist.control.student_prior import StudentMLP
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1dr")
ap.add_argument("--history", type=int, default=25, help="history steps used as probe input (<= 25)")
ap.add_argument("--num_windows", type=int, default=600_000); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
ap.add_argument("--save", default=None, help="directory to save the history -> xi probe (params, scaler, config) for scripts/sim2sim/visibility_apply.py")
ap.add_argument("--horizon", type=int, default=25, help="action-sensitivity target: the expert's next `horizon` actions (the student's output); 1 = next action only, which is almost fully determined by the last action")
args = ap.parse_args()
AXES = ["motor", "friction", "mass"]

d = paths.get_processed_data_dir(args.dataset, args.system, 25, 25)
P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"]
C = h5py.File(os.path.join(d, "cmds.hdf5"))["data"]; X = h5py.File(os.path.join(d, "physics.hdf5"))["data"]
starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
sc = json.load(open(os.path.join(d, "scaler_params.json")))
H, Hp = 25, args.history
rng = np.random.default_rng(args.seed)
# windows from random contiguous blocks (see train_student_prior.py); held-out = last 20 % of blocks
blocks, per = 20, max(1, min(args.num_windows, len(starts)) // 20)
Pm, Am, Cm, Xm, sel, split = [], [], [], [], [], []
off = 0
for b in range(blocks):
    i0 = int(rng.integers(0, max(1, len(starts) - per))); blk = starts[i0:i0 + per]
    lo, hi = int(blk.min()), int(blk.max()) + 2 * H + args.horizon + 2
    Pm.append(P[lo:hi]); Am.append(A[lo:hi]); Cm.append(C[lo:hi]); Xm.append(X[lo:hi])
    sel.append(blk - lo + off); split.append(np.full(len(blk), b >= blocks - 4)); off += hi - lo
Pm, Am, Cm, Xm = (np.concatenate(z).astype(np.float32) for z in (Pm, Am, Cm, Xm)); sel = np.concatenate(sel); test = np.concatenate(split)
print(f"windows {len(sel)} (test {int(test.sum())}); physics ranges:", {a: (float(Xm[:, i].min()), float(Xm[:, i].max())) for i, a in enumerate(AXES)}, flush=True)

def scale(x, k):
    return (x - np.asarray(sc[k]["mean"], np.float32)) / (np.asarray(sc[k]["std"], np.float32) + 1e-8)
def feats(idx):
    tH = sel[idx]; t = tH + H
    p = scale(Pm[(t - Hp)[:, None] + np.arange(Hp + 1)[None, :]], "proprio_obs")   # last Hp+1 obs
    a = scale(Am[(t - Hp)[:, None] + np.arange(Hp)[None, :]], "actions")           # last Hp actions
    c = scale(Cm[t[:, None] + np.arange(args.horizon + 1)[None, :]], "commands")       # commands over the horizon
    B = len(idx)
    y = scale(Am[t[:, None] + np.arange(args.horizon)[None, :]], "actions").reshape(B, -1)   # next `horizon` expert actions
    return np.concatenate([p.reshape(B, -1), a.reshape(B, -1), c.reshape(B, -1)], -1), Xm[t], y
xmean, xstd = Xm.mean(0), Xm.std(0) + 1e-8

def train_probe(target_fn, in_dim, out_dim, name):
    model = StudentMLP(in_dim, args.hidden, out_dim, nnx.Rngs(args.seed))
    opt = nnx.Optimizer(model, optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-4, 300, args.steps), weight_decay=1e-4))
    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(m): return jnp.mean((m(x) - y) ** 2)
        l, g = nnx.value_and_grad(loss_fn)(model); opt.update(g); return l
    tr = np.where(~test)[0]; te = np.where(test)[0]
    t0 = time.time()
    for it in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]; x, y = target_fn(idx)
        step(model, opt, jnp.asarray(x), jnp.asarray(y))
    # held-out R^2 per output dim
    preds, ys = [], []
    for i in range(0, len(te), 4096):
        x, y = target_fn(te[i:i + 4096]); preds.append(np.asarray(model(jnp.asarray(x)))); ys.append(y)
    pr, yy = np.concatenate(preds), np.concatenate(ys)
    r2 = 1 - ((pr - yy) ** 2).sum(0) / (((yy - yy.mean(0)) ** 2).sum(0) + 1e-8)
    print(f"  {name}: R2 per dim {np.round(r2, 3).tolist()[:6]}{'...' if len(r2) > 6 else ''} ({time.time()-t0:.0f}s)", flush=True)
    return r2, model

print("--- visibility: history -> xi", flush=True)
def tgt_xi(idx):
    f, x, _ = feats(idx); return f, (x - xmean) / xstd
in_dim = feats(np.array([0]))[0].shape[1]
V, probe = train_probe(tgt_xi, in_dim, 3, "V_j (history -> xi)")
if args.save:
    os.makedirs(args.save, exist_ok=True)
    pickle.dump(nnx.split(probe)[1].to_pure_dict(), open(os.path.join(args.save, "params.pkl"), "wb"))
    json.dump(sc, open(os.path.join(args.save, "scaler_params.json"), "w"))
    json.dump({"in_dim": in_dim, "hidden": args.hidden, "history": Hp, "horizon": args.horizon, "axes": AXES, "xi_mean": xmean.tolist(), "xi_std": xstd.tolist(),
               "dataset": args.dataset, "system": args.system}, open(os.path.join(args.save, "config.json"), "w"), indent=2)
    print(f"saved probe to {args.save}", flush=True)

print("--- action sensitivity proxy: next expert action from history (+ xi_j)", flush=True)
def tgt_act(idx, axes=()):
    f, x, a = feats(idx)
    if axes: f = np.concatenate([f, ((x - xmean) / xstd)[:, list(axes)]], -1)
    return f, a
A_dim = Am.shape[1] * args.horizon
def seq_r2(r2):   # mean over the sequence and over the last step (where the history helps least)
    return float(r2.mean()), float(r2[-Am.shape[1]:].mean())
base = seq_r2(train_probe(lambda i: tgt_act(i), in_dim, A_dim, "action | history")[0])
S = {}
for j, ax in enumerate(AXES):
    r2 = seq_r2(train_probe(lambda i, j=j: tgt_act(i, (j,)), in_dim + 1, A_dim, f"action | history + {ax}")[0])
    S[ax] = {"mean": r2[0] - base[0], "last_step": r2[1] - base[1]}
res = {"dataset": args.dataset, "system": args.system, "history": Hp, "visibility_R2": {a: float(V[i]) for i, a in enumerate(AXES)},
       "horizon": args.horizon, "action_R2_history": {"mean": base[0], "last_step": base[1]}, "action_R2_gain_from_axis": S,
       "physics_std": {a: float(xstd[i]) for i, a in enumerate(AXES)}, "n_train": int((~test).sum()), "n_test": int(test.sum())}
print(json.dumps(res, indent=2))
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
