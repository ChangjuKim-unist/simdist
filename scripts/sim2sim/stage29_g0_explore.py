"""EXPLORATORY (not registered), after the G0 verdict (fail): is the near-zero physics share real or an estimator artefact?
Adds  C5 = global ridge on [phi, physics labels, phi x physics] (true continuous physics, lambda by 5-fold CV on A over a wide grid)
      vs C3 = global ridge on phi (same CV), and C4b = per-group ridge with its own CV lambda.
Original docstring of the registered G0 script follows.

Design doc 29 gate G0 (docs/design29_gate_physcode_ko.md §2): how much of the broad-DR model's latent prediction error
can a correction fitted on OTHER episodes of the same physics group remove, per scene (startup vs steady)?

    python scripts/sim2sim/stage29_g0.py --out results/stage29/g0   (inside the container; local GPU is enough)

Residual r = predicted latents - encoded future observations (25 x 64, the training-loss space) with the data's own
actions. Per scene, fitted on episode half A and scored on half B (episode-level bootstrap):
  C1 global constant mean r_k, C2 group constant, C3 global ridge on phi = [z_t, mean of the next 5 actions, cmd_t, 1],
  C4 per-group ridge (same lambda, chosen by 5-fold CV on A for C3).
  physics share (constant) = 1 - MSE(C2)/MSE(C1); physics share (linear) = 1 - MSE(C4)/MSE(C3).
Groups: motor tertiles (0.592, 0.848) x friction tertiles (0.497, 0.858) x mass halves (0.883) = 18; labels only group.
"""
import argparse, json, os
import flax.nnx as nnx
import h5py
import jax, jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, Subset
from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="go1_broadwm")
ap.add_argument("--dataset", default="2026-11-05_11-39-08")
ap.add_argument("--system", default="go1dr")
ap.add_argument("--n_episodes", type=int, default=4000)
ap.add_argument("--per_scene", type=int, default=8)
ap.add_argument("--min_len", type=int, default=400)
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
MOTOR_CUT = (0.592, 0.848); FRIC_CUT = (0.497, 0.858); MASS_CUT = 0.883
SCENES = {"startup": (25, 125), "steady": (300, None)}
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
rng = np.random.default_rng(29)

model, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), args.ckpt))
cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
cfg["data"] = {"dataset_name": args.dataset}
ds = get_dataset(cfg); ds.eval()
H, T = ds.H, ds.T
ddir = paths.get_processed_data_dir(args.dataset, args.system, H, T)


def h5(name):
    f = h5py.File(os.path.join(ddir, name + ".hdf5"), "r")
    return f[list(f.keys())[0]]


ste = np.asarray(h5("terminal")[:, 0])
phys = np.asarray(h5("physics")[:])
starts_win = np.asarray(ds._data["start_idxs"][:])
print(f"steps {len(ste)}, windows {len(starts_win)}", flush=True)

# episodes: index 0 and every index where steps-to-end jumps up start an episode; steps-to-end counts down to 1, so the
# episode length is ste[start] (checked: physics labels are constant on [start, start + ste[start]))
jump = np.concatenate([[0], np.where(ste[1:] > ste[:-1])[0] + 1])
lens = ste[jump].astype(np.int64)
ok = (jump + lens <= len(ste)) & (lens >= args.min_len)
ok &= np.array([o and bool(np.all(phys[s] == phys[s + l - 1])) for o, s, l in zip(ok, jump, lens)])  # physics constant inside
ep_start, ep_len = jump[ok], lens[ok]
print(f"episodes with a visible start and length >= {args.min_len}: {len(ep_start)}", flush=True)
sel = np.sort(rng.choice(len(ep_start), size=min(args.n_episodes, len(ep_start)), replace=False))
ep_start, ep_len = ep_start[sel], ep_len[sel]
half = rng.permutation(len(ep_start)) < len(ep_start) // 2  # True = A (fit), False = B (eval)


def group_of(p):
    return int(np.searchsorted(MOTOR_CUT, p[0])) * 6 + int(np.searchsorted(FRIC_CUT, p[1])) * 2 + int(p[2] >= MASS_CUT)


rows = []  # (window index, episode, group, scene, tau)
for e, (s, l) in enumerate(zip(ep_start, ep_len)):
    g = group_of(phys[s])
    for sc, (lo, hi) in SCENES.items():
        hi_ = l - T - 1 if hi is None else min(hi, l - T - 1)
        if hi_ <= lo:
            continue
        taus = np.sort(rng.choice(np.arange(lo, hi_), size=min(args.per_scene, hi_ - lo), replace=False))
        for tau in taus:
            t_H = s + tau - H
            wi = int(np.searchsorted(starts_win, t_H))
            if wi < len(starts_win) and starts_win[wi] == t_H:
                rows.append((wi, e, g, sc, int(tau)))
print(f"windows: {len(rows)}", flush=True)


gd, gs = nnx.split(model)


@jax.jit
def resid(x, y):
    model = nnx.merge(gd, gs)
    pred = model(x, deterministic=True)
    sc = model.get_scaler()
    ys = sc.scale(y); xs = sc.scale(x)
    fut = model.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    z = model.encode_latent(xs["proprio_obs_hist"][:, -1], xs["extero_obs"], deterministic=True)
    return pred["latents"] - fut, z


R, PHI = [], []
widx = [r[0] for r in rows]
for b0 in range(0, len(widx), 256):
    for b in DataLoader(Subset(ds, widx[b0:b0 + 256]), batch_size=256, shuffle=False, num_workers=0):
        b = mu.dataset_batch_to_jax(b)
        r, z = resid(b["model_in"], b["labels"])
        fa = np.asarray(b["model_in"]["fut_acts"])[:, :5].mean(1); cm = np.asarray(b["model_in"]["fut_cmds"])[:, 0]
        R.append(np.asarray(r, np.float32).reshape(len(fa), -1)); PHI.append(np.concatenate([np.asarray(z), fa, cm], -1))
    if b0 % 5120 == 0:
        print(f"  {b0}/{len(widx)}", flush=True)
R = np.concatenate(R); PHI = np.concatenate(PHI).astype(np.float64)
EP = np.array([r[1] for r in rows]); G = np.array([r[2] for r in rows]); SC = np.array([r[3] for r in rows])
A = half[EP]
D = R.shape[1] // T



PH = np.array([phys[ep_start[r[1]]] for r in rows], np.float64)
GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def ridge(X, Y, lam):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    reg = lam * np.eye(Xb.shape[1]); reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def apply(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def cv_lam(X, Y, eps):
    ue = np.unique(eps); folds = rng.permutation(len(ue)) % 5; best = None
    for lam in GRID:
        err = 0.0
        for f in range(5):
            te = np.isin(eps, ue[folds == f]); tr = ~te
            err += ((Y[te] - apply(ridge(X[tr], Y[tr], lam), X[te])) ** 2).sum()
        if best is None or err < best[1]:
            best = (lam, err)
    return best[0]


res = {"note": "exploratory, not registered", "scenes": {}}
for sc in SCENES:
    m = SC == sc; mA, mB = m & A, m & ~A
    mu_x, sd_x = PHI[mA].mean(0), PHI[mA].std(0) + 1e-6
    Xs = (PHI - mu_x) / sd_x
    pz = (PH - PH[mA].mean(0)) / (PH[mA].std(0) + 1e-6)
    Xp = np.concatenate([Xs, pz, (Xs[:, :, None] * pz[:, None, :]).reshape(len(Xs), -1)], 1)
    l3 = cv_lam(Xs[mA], R[mA], EP[mA]); l5 = cv_lam(Xp[mA], R[mA], EP[mA])
    e3 = ((R[mB] - apply(ridge(Xs[mA], R[mA], l3), Xs[mB])) ** 2).mean(1)
    e5 = ((R[mB] - apply(ridge(Xp[mA], R[mA], l5), Xp[mB])) ** 2).mean(1)
    p4 = np.zeros_like(R[mB]); gB = G[mB]
    for g in np.unique(G[m]):
        a = mA & (G == g); bs = gB == g
        lg = cv_lam(Xs[a], R[a], EP[a]); p4[bs] = apply(ridge(Xs[a], R[a], lg), Xs[mB][bs])
    e4 = ((R[mB] - p4) ** 2).mean(1)
    epB = EP[mB]; ue = np.unique(epB)
    s3 = np.zeros(len(ep_start)); s4 = np.zeros(len(ep_start)); s5 = np.zeros(len(ep_start))
    np.add.at(s3, epB, e3); np.add.at(s4, epB, e4); np.add.at(s5, epB, e5)
    b5 = []; b4 = []
    for _ in range(2000):
        bs = rng.choice(ue, size=len(ue), replace=True)
        b5.append(1 - s5[bs].sum() / s3[bs].sum()); b4.append(1 - s4[bs].sum() / s3[bs].sum())
    out = {"lambda_C3": l3, "lambda_C5": l5, "mse_C3": float(e3.mean()), "mse_C5": float(e5.mean()), "mse_C4b": float(e4.mean()),
           "share_true_physics": float(1 - e5.mean() / e3.mean()), "ci_true_physics": [float(np.percentile(b5, 2.5)), float(np.percentile(b5, 97.5))],
           "share_group_cv": float(1 - e4.mean() / e3.mean()), "ci_group_cv": [float(np.percentile(b4, 2.5)), float(np.percentile(b4, 97.5))]}
    # low-friction episodes only (friction tertile 0): does true physics help where it should matter most?
    lowf = (PH[mB][:, 1] < FRIC_CUT[0])
    out["share_true_physics_lowfric"] = float(1 - e5[lowf].mean() / e3[lowf].mean())
    lowm = (PH[mB][:, 0] < MOTOR_CUT[0])
    out["share_true_physics_lowmotor"] = float(1 - e5[lowm].mean() / e3[lowm].mean())
    res["scenes"][sc] = out
    print(sc, json.dumps(out), flush=True)
json.dump(res, open(args.out + "_explore.json", "w"), indent=1)
L = ["## 설계 29 G0 보조 확인 (등록 안 함): 실제 물리값을 넣은 선형 보정", "",
     "몫 = 1 − MSE(보정)/MSE(물리 없는 전체 선형 보정), B 에피소드, 95 % CI 에피소드 부트스트랩. λ는 A 안 5겹 교차검증 {1, 10, 100, 1000, 10000}.", "",
     "| 장면 | 실제 물리값 + 상호작용 [CI] | 묶음별 선형(λ 따로) [CI] | 낮은 마찰 에피소드만 | 낮은 모터 에피소드만 |", "|---|---|---|---|---|"]
for sc, o in res["scenes"].items():
    L.append(f"| {sc} | {o['share_true_physics']:.3f} [{o['ci_true_physics'][0]:.3f}, {o['ci_true_physics'][1]:.3f}] | {o['share_group_cv']:.3f} [{o['ci_group_cv'][0]:.3f}, {o['ci_group_cv'][1]:.3f}] | {o['share_true_physics_lowfric']:.3f} | {o['share_true_physics_lowmotor']:.3f} |")
open(args.out + "_explore.md", "w").write("\n".join(L) + "\n"); print("written", args.out + "_explore.md")
