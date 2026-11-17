"""Design doc 29 gate G0 (docs/design29_gate_physcode_ko.md §2): how much of the broad-DR model's latent prediction error
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


def ridge(X, Y, lam):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    reg = lam * np.eye(Xb.shape[1]); reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def apply(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def sse_per_ep(E, eps):  # sum of squared error per evaluation episode, and window counts
    s = np.zeros(len(ep_start)); c = np.zeros(len(ep_start))
    np.add.at(s, eps, (E ** 2).mean(1)); np.add.at(c, eps, 1)
    return s, c


res = {"ckpt": args.ckpt, "dataset": args.dataset, "n_episodes": int(len(ep_start)), "windows": int(len(rows)), "scenes": {}}
for sc in SCENES:
    m = SC == sc; mA, mB = m & A, m & ~A
    X, Y = PHI, R
    mu_x, sd_x = X[mA].mean(0), X[mA].std(0) + 1e-6
    Xs = (X - mu_x) / sd_x
    # lambda for the global ridge by 5-fold CV over A episodes
    epsA = np.unique(EP[mA]); folds = rng.permutation(len(epsA)) % 5; cv = []
    for lam in LAMBDAS:
        err = 0.0
        for f in range(5):
            te = mA & np.isin(EP, epsA[folds == f]); tr = mA & ~te
            err += ((Y[te] - apply(ridge(Xs[tr], Y[tr], lam), Xs[te])) ** 2).sum()
        cv.append(err)
    lam = LAMBDAS[int(np.argmin(cv))]
    c1 = Y[mA].mean(0)
    pred = {"none": np.zeros_like(Y[mB]), "C1": np.broadcast_to(c1, Y[mB].shape), "C3": apply(ridge(Xs[mA], Y[mA], lam), Xs[mB])}
    c2 = np.zeros_like(Y[mB]); c4 = np.zeros_like(Y[mB])
    gB = G[mB]
    for g in np.unique(G[m]):
        a = mA & (G == g); bsel = gB == g
        if a.sum() < 30 or bsel.sum() == 0:  # too few fit windows: fall back to the global fit
            c2[bsel] = c1; c4[bsel] = pred["C3"][bsel]; continue
        c2[bsel] = Y[a].mean(0)
        c4[bsel] = apply(ridge(Xs[a], Y[a], lam), Xs[mB][bsel])
    pred["C2"] = c2; pred["C4"] = c4
    epB = EP[mB]
    S = {k: sse_per_ep(Y[mB] - v, epB) for k, v in pred.items()}
    # horizon breakdown (k = 1 and k = 25)
    Yr = Y[mB].reshape(-1, T, D)

    def share(num, den, sl=None):
        if sl is None:
            return 1 - S[num][0].sum() / S[den][0].sum()
        en = ((Yr[:, sl] - pred[num].reshape(-1, T, D)[:, sl]) ** 2).mean(); ed = ((Yr[:, sl] - pred[den].reshape(-1, T, D)[:, sl]) ** 2).mean()
        return 1 - en / ed

    eps_eval = np.unique(epB); boots = {"const": [], "linear": []}
    for _ in range(args.nboot):
        bs = rng.choice(eps_eval, size=len(eps_eval), replace=True)
        boots["const"].append(1 - S["C2"][0][bs].sum() / S["C1"][0][bs].sum())
        boots["linear"].append(1 - S["C4"][0][bs].sum() / S["C3"][0][bs].sum())
    out = {"windows_fit": int(mA.sum()), "windows_eval": int(mB.sum()), "lambda": lam,
           "mse": {k: float(S[k][0].sum() / S[k][1].sum()) for k in S},
           "share_const": float(share("C2", "C1")), "share_const_ci": [float(np.percentile(boots["const"], 2.5)), float(np.percentile(boots["const"], 97.5))],
           "share_linear": float(share("C4", "C3")), "share_linear_ci": [float(np.percentile(boots["linear"], 2.5)), float(np.percentile(boots["linear"], 97.5))],
           "share_const_k1": float(share("C2", "C1", 0)), "share_const_k25": float(share("C2", "C1", T - 1)),
           "share_linear_k1": float(share("C4", "C3", 0)), "share_linear_k25": float(share("C4", "C3", T - 1)),
           "by_group": {}}
    for g in np.unique(gB):
        bsel = gB == g
        e1 = ((Y[mB][bsel] - pred["C1"][bsel]) ** 2).mean(); e2 = ((Y[mB][bsel] - pred["C2"][bsel]) ** 2).mean()
        e3 = ((Y[mB][bsel] - pred["C3"][bsel]) ** 2).mean(); e4 = ((Y[mB][bsel] - pred["C4"][bsel]) ** 2).mean()
        out["by_group"][int(g)] = {"n": int(bsel.sum()), "motor_bin": int(g // 6), "fric_bin": int((g % 6) // 2), "mass_bin": int(g % 2),
                                   "share_const": float(1 - e2 / e1), "share_linear": float(1 - e4 / e3), "mse_C3": float(e3)}
    res["scenes"][sc] = out
    print(sc, json.dumps({k: v for k, v in out.items() if k != "by_group"}), flush=True)

st = res["scenes"]["startup"]
best = max(st["share_const"], st["share_linear"]); lo = st["share_const_ci"][0] if st["share_const"] >= st["share_linear"] else st["share_linear_ci"][0]
res["G0_pass"] = bool(best >= 0.10 and lo > 0.05)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(res, open(args.out + ".json", "w"), indent=1)
L = ["## 설계 29 G0: 물리 묶음 보정이 줄이는 잠재 예측 오차 몫", "",
     f"모델 {args.ckpt}, 데이터 {args.dataset}, 에피소드 {res['n_episodes']} (A/B 반씩), 창 {res['windows']}. 몫 = 1 − MSE(묶음 보정)/MSE(전체 보정), B 에피소드에서, 95 % CI는 에피소드 부트스트랩.", "",
     "| 장면 | 창 (맞춤/평가) | 보정 전 MSE | 물리 몫 상수 [CI] | 물리 몫 선형 [CI] | k=1 상수/선형 | k=25 상수/선형 |", "|---|---|---|---|---|---|---|"]
for sc, o in res["scenes"].items():
    L.append(f"| {sc} | {o['windows_fit']}/{o['windows_eval']} | {o['mse']['none']:.4f} | {o['share_const']:.3f} [{o['share_const_ci'][0]:.3f}, {o['share_const_ci'][1]:.3f}] | "
             f"{o['share_linear']:.3f} [{o['share_linear_ci'][0]:.3f}, {o['share_linear_ci'][1]:.3f}] | {o['share_const_k1']:.3f}/{o['share_linear_k1']:.3f} | {o['share_const_k25']:.3f}/{o['share_linear_k25']:.3f} |")
L += ["", f"G0 통과(출발 몫 최댓값 ≥ 0.10, CI 하한 > 0.05): **{res['G0_pass']}**", "",
      "묶음별(출발 장면): 모터 구간 0~2(낮음→높음), 마찰 구간 0~2, 질량 0/1", "", "| 묶음 | 모터 | 마찰 | 질량 | 창 | 몫 상수 | 몫 선형 |", "|---|---|---|---|---|---|---|"]
for g, o in sorted(res["scenes"]["startup"]["by_group"].items()):
    L.append(f"| {g} | {o['motor_bin']} | {o['fric_bin']} | {o['mass_bin']} | {o['n']} | {o['share_const']:.3f} | {o['share_linear']:.3f} |")
open(args.out + ".md", "w").write("\n".join(L) + "\n")
print("written", args.out + ".md", "G0_pass", res["G0_pass"], flush=True)
