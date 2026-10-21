"""Label-free visibility of a dynamics shift (docs/research_plan_ko.md, stage 0).

On any processed dataset (real-robot logs or a sim-to-sim condition), train two predictors of
the next proprioceptive observation from the same scaled inputs:
    blind:   (o_t, a_t)                   -> o_{t+1}
    history: (o_{t-Hp..t}, a_{t-Hp..t})   -> o_{t+1}
The held-out MSE gap  G = (MSE_blind - MSE_hist) / MSE_blind  measures how much of the
dynamics the history explains beyond the instantaneous state: a shift that is visible in the
history makes G large. No physics labels are needed, so the same statistic can be computed on
real Go1 logs. Validated in simulation against the labelled probe (visibility_probe.py).

    python scripts/sim2sim/visibility_labelfree.py --dataset sim2sim_very_weak_motor_go1_paper_b45000 --real --out results/stage0/lf_weak.json
    python scripts/sim2sim/visibility_labelfree.py --dataset <sim dataset> --system go1dr --out ...
"""
import argparse, json, os, time
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx

from simdist.control.student_prior import StudentMLP
from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--system", default="go1")
ap.add_argument("--real", action="store_true", help="dataset lives in datasets/real (processed with system)")
ap.add_argument("--history", type=int, default=25); ap.add_argument("--num_windows", type=int, default=400_000)
ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

if args.real:
    d = os.path.join(paths.get_real_dataset_dir(args.dataset), f"processed_data_{args.system}_H-25_T-25")
else:
    d = paths.get_processed_data_dir(args.dataset, args.system, 25, 25)
P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"]
starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
sc = json.load(open(os.path.join(d, "scaler_params.json")))
H, Hp = 25, args.history
rng = np.random.default_rng(args.seed)
blocks, per = 20, max(1, min(args.num_windows, len(starts)) // 20)
Pm, Am, sel, split, off = [], [], [], [], 0
for b in range(blocks):
    i0 = int(rng.integers(0, max(1, len(starts) - per))); blk = starts[i0:i0 + per]
    lo, hi = int(blk.min()), int(blk.max()) + 2 * H + 2
    Pm.append(P[lo:hi]); Am.append(A[lo:hi]); sel.append(blk - lo + off); split.append(np.full(len(blk), b >= blocks - 4)); off += hi - lo
Pm, Am = (np.concatenate(z).astype(np.float32) for z in (Pm, Am)); sel = np.concatenate(sel); test = np.concatenate(split)
def scale(x, k): return (x - np.asarray(sc[k]["mean"], np.float32)) / (np.asarray(sc[k]["std"], np.float32) + 1e-8)
def feats(idx, hist):
    t = sel[idx] + H; B = len(idx)
    y = scale(Pm[t + 1], "proprio_obs")
    if hist:
        p = scale(Pm[(t - Hp)[:, None] + np.arange(Hp + 1)[None, :]], "proprio_obs"); a = scale(Am[(t - Hp)[:, None] + np.arange(Hp + 1)[None, :]], "actions")
    else:
        p = scale(Pm[t], "proprio_obs")[:, None]; a = scale(Am[t], "actions")[:, None]
    return np.concatenate([p.reshape(B, -1), a.reshape(B, -1)], -1), y

def fit(hist, name):
    in_dim = feats(np.array([0]), hist)[0].shape[1]; out_dim = Pm.shape[1]
    model = StudentMLP(in_dim, args.hidden, out_dim, nnx.Rngs(args.seed))
    opt = nnx.Optimizer(model, optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-4, 300, args.steps), weight_decay=1e-4))
    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(m): return jnp.mean((m(x) - y) ** 2)
        l, g = nnx.value_and_grad(loss_fn)(model); opt.update(g); return l
    tr, te = np.where(~test)[0], np.where(test)[0]; t0 = time.time()
    for it in range(args.steps):
        x, y = feats(tr[rng.integers(0, len(tr), args.batch)], hist); step(model, opt, jnp.asarray(x), jnp.asarray(y))
    se = []
    for i in range(0, len(te), 4096):
        x, y = feats(te[i:i + 4096], hist); se.append(np.asarray((model(jnp.asarray(x)) - y) ** 2))
    mse = float(np.concatenate(se).mean()); print(f"  {name}: held-out MSE {mse:.4f} ({time.time()-t0:.0f}s)", flush=True); return mse
print(f"windows {len(sel)} (test {int(test.sum())}), history {Hp}", flush=True)
m_blind = fit(False, "blind (o_t, a_t)"); m_hist = fit(True, f"history ({Hp} steps)")
res = {"dataset": args.dataset, "history": Hp, "mse_blind": m_blind, "mse_history": m_hist, "visibility_gap": (m_blind - m_hist) / m_blind}
print(json.dumps(res, indent=2))
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); json.dump(res, open(args.out, "w"), indent=2)
