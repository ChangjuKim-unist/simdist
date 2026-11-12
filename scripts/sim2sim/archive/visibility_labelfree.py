"""Label-free visibility of a dynamics shift (docs/research_plan_ko.md, stage 0).

On any processed dataset (real-robot logs or a sim-to-sim condition), train two predictors of
the next proprioceptive observation from the same scaled inputs:
    blind:   (o_t, a_t)                   -> o_{t+1}
    history: (o_{t-Hp..t}, a_{t-Hp..t})   -> o_{t+1}
The held-out MSE gap  G = (MSE_blind - MSE_hist) / MSE_blind  measures how much of the
dynamics the history explains beyond the instantaneous state: a shift that is visible in the
history makes G large. No physics labels are needed, so the same statistic can be computed on
real Go1 logs. Validated in simulation against the labelled probe (visibility_probe.py).

A single-condition dataset cannot reveal a shift (the regime is constant), so the shift is
measured RELATIVE TO A REFERENCE: with --mix <nominal dataset> the predictors are trained on a
50/50 mixture of the shifted and the reference data; the history predictor can tell the regimes
apart only if the shift is visible in the history. Also reports the AUC of a regime classifier
(shifted vs reference) from the history versus from the instantaneous state.

    python scripts/sim2sim/visibility_labelfree.py --dataset sim2sim_very_weak_motor_go1_paper_b45000 --real --mix 2026-10-02_01-18-59 --out results/stage0/lf_weak.json
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
ap.add_argument("--mix", default=None, help="reference (nominal) dataset to mix 50/50 with; sim dataset unless --mix_real")
ap.add_argument("--mix_real", action="store_true"); ap.add_argument("--mix_system", default="go1")
ap.add_argument("--probe", default="both", choices=["mlp", "ridge", "both"], help="MLP (early-stopped on a validation split) and/or closed-form ridge regression")
ap.add_argument("--out", default=None)
args = ap.parse_args()

H, Hp = 25, args.history
rng = np.random.default_rng(args.seed)

def load(dataset, real, system, num_windows):
    if real:
        d = os.path.join(paths.get_real_dataset_dir(dataset), f"processed_data_{system}_H-25_T-25")
    else:
        d = paths.get_processed_data_dir(dataset, system, 25, 25)
    P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"]
    starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]
    blocks, per = 20, max(1, min(num_windows, len(starts)) // 20)
    Pm, Am, sel, split, off = [], [], [], [], 0
    for b in range(blocks):
        i0 = int(rng.integers(0, max(1, len(starts) - per))); blk = starts[i0:i0 + per]
        lo, hi = int(blk.min()), int(blk.max()) + 2 * H + 2
        Pm.append(P[lo:hi]); Am.append(A[lo:hi]); sel.append(blk - lo + off)
        split.append(np.full(len(blk), 2 if b >= blocks - 4 else (1 if b >= blocks - 7 else 0))); off += hi - lo   # 0 train / 1 val / 2 test blocks
    Pm, Am = (np.concatenate(z).astype(np.float32) for z in (Pm, Am)); sel = np.concatenate(sel); test = np.concatenate(split)
    return Pm, Am, sel, test, json.load(open(os.path.join(d, "scaler_params.json")))

Pm, Am, sel, test, sc = load(args.dataset, args.real, args.system, args.num_windows)
regime = np.zeros(len(sel), np.float32)
if args.mix:
    P2, A2, sel2, test2, _ = load(args.mix, args.mix_real, args.mix_system, args.num_windows)
    off = len(Pm); Pm = np.concatenate([Pm, P2]); Am = np.concatenate([Am, A2])
    sel = np.concatenate([sel, sel2 + off]); test = np.concatenate([test, test2]); regime = np.concatenate([regime, np.ones(len(sel2), np.float32)])
    print(f"mixed with reference {args.mix}: {int((regime==0).sum())} shifted + {int((regime==1).sum())} reference windows", flush=True)
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
    tr, va, te = np.where(test == 0)[0], np.where(test == 1)[0], np.where(test == 2)[0]; t0 = time.time()
    def mse_on(idx):
        se = []
        for i in range(0, len(idx), 4096):
            x, y = feats(idx[i:i + 4096], hist); se.append(np.asarray((model(jnp.asarray(x)) - y) ** 2))
        return float(np.concatenate(se).mean())
    best, best_it, best_test = np.inf, 0, np.nan
    for it in range(args.steps):
        x, y = feats(tr[rng.integers(0, len(tr), args.batch)], hist); step(model, opt, jnp.asarray(x), jnp.asarray(y))
        if (it + 1) % 250 == 0:            # early stopping on the validation blocks (never on the test blocks)
            v = mse_on(va)
            if v < best: best, best_it, best_test = v, it + 1, mse_on(te)
    print(f"  {name}: MLP held-out MSE {best_test:.4f} (best val {best:.4f} at step {best_it}; {time.time()-t0:.0f}s)", flush=True); return best_test

def fit_ridge(hist, name):
    """Closed-form ridge regression; lambda chosen on the validation blocks. Linear, so it cannot overfit the way the MLP does."""
    tr, va, te = np.where(test == 0)[0], np.where(test == 1)[0], np.where(test == 2)[0]; t0 = time.time()
    X, Y = feats(tr, hist); Xv, Yv = feats(va, hist); Xt, Yt = feats(te, hist)
    mu, sd = X.mean(0), X.std(0) + 1e-6; X, Xv, Xt = (X - mu) / sd, (Xv - mu) / sd, (Xt - mu) / sd
    ym = Y.mean(0); G = X.T @ X; XY = X.T @ (Y - ym)
    best = (np.inf, None)
    for lam in [1e-2, 1e-1, 1, 10, 100, 1000]:
        W = np.linalg.solve(G + lam * len(X) * np.eye(len(G), dtype=G.dtype), XY)
        v = float(((Xv @ W + ym - Yv) ** 2).mean())
        if v < best[0]: best = (v, lam, float(((Xt @ W + ym - Yt) ** 2).mean()))
    print(f"  {name}: ridge held-out MSE {best[2]:.4f} (lambda {best[1]}; {time.time()-t0:.0f}s)", flush=True); return best[2]
print(f"windows {len(sel)} (train {int((test==0).sum())} / val {int((test==1).sum())} / test {int((test==2).sum())}), history {Hp}", flush=True)
res = {"dataset": args.dataset, "mix": args.mix, "history": Hp}
if args.probe in ("ridge", "both"):
    rb, rh = fit_ridge(False, "blind (o_t, a_t)"), fit_ridge(True, f"history ({Hp} steps)")
    res.update(ridge_mse_blind=rb, ridge_mse_history=rh, ridge_visibility_gap=(rb - rh) / rb)
if args.probe in ("mlp", "both"):
    m_blind = fit(False, "blind (o_t, a_t)"); m_hist = fit(True, f"history ({Hp} steps)")
    res.update(mse_blind=m_blind, mse_history=m_hist, visibility_gap=(m_blind - m_hist) / m_blind)

def regime_auc(hist, name):
    """AUC of a shifted-vs-reference classifier from the history or from the instantaneous state."""
    in_dim = feats(np.array([0]), hist)[0].shape[1]
    model = StudentMLP(in_dim, args.hidden, 1, nnx.Rngs(args.seed))
    opt = nnx.Optimizer(model, optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-4, 300, args.steps), weight_decay=1e-4))
    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(m):
            logit = m(x)[:, 0]; return jnp.mean(jnp.maximum(logit, 0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit))))
        l, g = nnx.value_and_grad(loss_fn)(model); opt.update(g); return l
    tr, te = np.where(test == 0)[0], np.where(test == 2)[0]
    for it in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]; x, _ = feats(idx, hist); step(model, opt, jnp.asarray(x), jnp.asarray(regime[idx]))
    sc_, y_ = [], []
    for i in range(0, len(te), 4096):
        x, _ = feats(te[i:i + 4096], hist); sc_.append(np.asarray(model(jnp.asarray(x)))[:, 0]); y_.append(regime[te[i:i + 4096]])
    sc_, y_ = np.concatenate(sc_), np.concatenate(y_)
    pos, neg = sc_[y_ == 1], sc_[y_ == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    print(f"  regime AUC {name}: {auc:.3f}", flush=True); return auc
if args.mix:
    res["regime_auc_blind"] = regime_auc(False, "blind"); res["regime_auc_history"] = regime_auc(True, "history")
print(json.dumps(res, indent=2))
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); json.dump(res, open(args.out, "w"), indent=2)
