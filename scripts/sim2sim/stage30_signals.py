"""Design doc 30 part S (docs/design30_support_signal_ko.md §2): can an offline signal tell "this is not the condition the
world model was adapted to", so that a planner could fall back to the pretrained model there?

    python scripts/sim2sim/stage30_signals.py --base go1_paper --members go1_paper_friction0.15_dyn_b45000_s42 ... \
        --train sim2sim_friction0.15_go1_paper_b45000 --neg sim2sim_nominal_go1_paper_b45000 \
        --own sim2sim_friction0.15_go1_paper_ho_b15000 --others nominal=... other=... payload=... --out results/stage30/sig_P_ice

Per window (25-step history ending at t0, 25 logged future actions) of every dataset:
  z = encoder latent at t0, phi = [z, mean of the next 5 scaled actions, command]; k-step latent error (mean over 25 steps
  and 64 dims) of the pretrained model and of the first adapted member; variance across the adapted members' predictions.
Signals (higher = less reason to trust the adapted model here):
  S0 err    realised error of the adapted model minus that of the pretrained model (baseline: error-based selection)
  S1 dis    disagreement of the K adapted members (variance of predicted latents)
  S2 knn    mean distance to the 10 nearest adaptation-training windows in standardised phi
  S3 mde    MLP on phi regressing log(err_adapted / err_pretrained), trained on adaptation-train + in-coverage planner data
  S4 clf    MLP on phi, probability that the window is NOT from the adaptation data (negatives: in-coverage planner data)
Each signal is also averaged over the last 50 windows of the same episode (avg50, what a run-time gate would use).
Scenes: startup = 25 <= tau0 < 125, midwalk = tau0 >= 300 (tau0 = steps from episode start to t0).
Rule (split conformal): theta = 95th percentile of the signal on the calibration blocks of the own-condition held-out set
(blocks of 500 consecutive steps, even = calibration, odd = test); rejection rate on another set = share above theta.
Action part (descriptive): on own-condition test windows a constant per-joint offset N(0, sigma^2) is added to the 25
future actions; share of windows whose S1/S2/S3/S4 then exceeds the midwalk threshold.
"""
import argparse, json, os
import flax.nnx as nnx
import h5py
import jax, jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader, Subset
from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--members", nargs="+", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--neg", required=True)
ap.add_argument("--own", required=True)
ap.add_argument("--others", nargs="+", required=True, help="name=dataset")
ap.add_argument("--system", default="go1")
ap.add_argument("--max_ref", type=int, default=20000)
ap.add_argument("--max_eval", type=int, default=0)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rng = np.random.default_rng(30)
SCENES = {"startup": (25, 125), "midwalk": (300, 10 ** 9)}
SIGMAS = [0.05, 0.1, 0.2, 0.4]
BLOCK = 500; KNN = 10; Q = 0.95

base, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), args.base))
cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
members = [mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), m))[0] for m in args.members]
splits = [nnx.split(m) for m in [base] + members]
K = len(members)


@jax.jit
def forward(x, y, fut_acts):
    x = dict(x); x["fut_acts"] = fut_acts
    b = nnx.merge(*splits[0]); sc = b.get_scaler()
    ys = sc.scale(y); xs = sc.scale(x)
    fut = b.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    z = b.encode_latent(xs["proprio_obs_hist"][:, -1], xs["extero_obs"], deterministic=True)
    pb = b(x, deterministic=True)["latents"]
    pm = jnp.stack([nnx.merge(*s)(x, deterministic=True)["latents"] for s in splits[1:]])  # (K, B, T, D)
    e_pre = jnp.mean((pb - fut) ** 2, axis=(1, 2)); e_ad = jnp.mean((pm[0] - fut) ** 2, axis=(1, 2))
    dis = jnp.mean(jnp.var(pm, axis=0), axis=(1, 2))
    a5 = jnp.mean(xs["fut_acts"][:, :5], axis=1)
    return z, a5, xs["fut_cmds"][:, 0], e_pre, e_ad, dis


def load(name, cap):
    c = dict(cfg); c["data"] = {"dataset_name": name}
    ds = get_dataset(c); ds.eval()
    H, T = ds.H, ds.T
    ddir = paths.get_processed_data_dir(name, args.system, H, T)
    f = h5py.File(os.path.join(ddir, "terminal.hdf5"), "r"); ste = np.asarray(f[list(f.keys())[0]][:, 0])
    starts = np.asarray(ds._data["start_idxs"][:]).astype(np.int64)
    jump = np.concatenate([[0], np.where(ste[1:] > ste[:-1])[0] + 1])  # episode starts (steps-to-end jumps up)
    idx = np.arange(len(starts))
    if cap and len(idx) > cap:
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    out = {k: [] for k in ("z", "a5", "cmd", "e_pre", "e_ad", "dis")}
    for b0 in range(0, len(idx), 256):
        for b in DataLoader(Subset(ds, idx[b0:b0 + 256].tolist()), batch_size=256, shuffle=False, num_workers=0):
            b = mu.dataset_batch_to_jax(b)
            r = forward(b["model_in"], b["labels"], b["model_in"]["fut_acts"])
            for k, v in zip(out, r):
                out[k].append(np.asarray(v, np.float32))
    d = {k: np.concatenate(v) for k, v in out.items()}
    t0 = starts[idx] + H
    ep = np.searchsorted(jump, starts[idx], side="right") - 1
    d["t0"] = t0; d["ep"] = ep; d["tau"] = t0 - jump[ep]; d["block"] = starts[idx] // BLOCK
    d["phi"] = np.concatenate([d["z"], d["a5"], d["cmd"]], -1).astype(np.float32)
    d["_ds"] = ds; d["_idx"] = idx
    print(f"{name}: windows {len(idx)}, episodes {len(np.unique(ep))}, e_pre {d['e_pre'].mean():.4f}, e_ad {d['e_ad'].mean():.4f}, dis {d['dis'].mean():.5f}", flush=True)
    return d


D = {"train": load(args.train, args.max_ref), "neg": load(args.neg, args.max_ref), "own": load(args.own, args.max_eval)}
others = dict(o.split("=", 1) for o in args.others)
for k, v in others.items():
    D[k] = load(v, args.max_eval)
mu_phi = D["train"]["phi"].mean(0); sd_phi = D["train"]["phi"].std(0) + 1e-6
norm = lambda p: (p - mu_phi) / sd_phi  # noqa: E731
REF = jnp.asarray(norm(D["train"]["phi"]))


@jax.jit
def knn_chunk(q):
    d2 = jnp.sum(q ** 2, 1)[:, None] - 2.0 * q @ REF.T + jnp.sum(REF ** 2, 1)[None]
    return jnp.mean(jnp.sqrt(jnp.maximum(-jax.lax.top_k(-d2, KNN + 1)[0], 0.0)), axis=1)  # +1: a training point is its own neighbour


def knn(phi):
    q = norm(phi)
    return np.concatenate([np.asarray(knn_chunk(jnp.asarray(q[i:i + 2048]))) for i in range(0, len(q), 2048)])


def mlp_init(key, din):
    k1, k2, k3 = jax.random.split(key, 3)
    g = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)  # noqa: E731
    return {"w1": g(k1, din, 64), "b1": jnp.zeros(64), "w2": g(k2, 64, 64), "b2": jnp.zeros(64), "w3": g(k3, 64, 1) * 0.1, "b3": jnp.zeros(1)}


def mlp(p, x):
    h = jnp.tanh(x @ p["w1"] + p["b1"]); h = jnp.tanh(h @ p["w2"] + p["b2"])
    return (h @ p["w3"] + p["b3"])[:, 0]


def fit(X, Y, kind, steps=3000, seed=30):
    p = mlp_init(jax.random.PRNGKey(seed), X.shape[1]); opt = optax.adamw(1e-3, weight_decay=1e-4); st = opt.init(p)
    X = jnp.asarray(X); Y = jnp.asarray(Y)

    def loss(p, xb, yb):
        o = mlp(p, xb)
        return jnp.mean((o - yb) ** 2) if kind == "reg" else jnp.mean(optax.sigmoid_binary_cross_entropy(o, yb))

    @jax.jit
    def step(p, st, key):
        i = jax.random.randint(key, (256,), 0, X.shape[0])
        g = jax.grad(loss)(p, X[i], Y[i]); u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        key, k = jax.random.split(key); p, st = step(p, st, k)
    return p


Xtr = np.concatenate([norm(D["train"]["phi"]), norm(D["neg"]["phi"])])
ratio = lambda d: np.log(np.maximum(d["e_ad"], 1e-8) / np.maximum(d["e_pre"], 1e-8))  # noqa: E731
p_mde = fit(Xtr, np.concatenate([ratio(D["train"]), ratio(D["neg"])]), "reg")
p_clf = fit(Xtr, np.concatenate([np.zeros(len(D["train"]["phi"])), np.ones(len(D["neg"]["phi"]))]).astype(np.float32), "clf")
mde = lambda phi: np.asarray(mlp(p_mde, jnp.asarray(norm(phi))))  # noqa: E731
clf = lambda phi: np.asarray(jax.nn.sigmoid(mlp(p_clf, jnp.asarray(norm(phi)))))  # noqa: E731
SIG = ["err", "dis", "knn", "mde", "clf"]


def signals(d):
    return {"err": d["e_ad"] - d["e_pre"], "dis": d["dis"], "knn": knn(d["phi"]), "mde": mde(d["phi"]), "clf": clf(d["phi"])}


def avg50(d, s):
    """mean of s over the windows of the same episode with t0 in (t0 - 50, t0]"""
    order = np.lexsort((d["t0"], d["ep"])); big = int(d["t0"].max()) + 100
    key = d["ep"][order].astype(np.int64) * big + d["t0"][order]
    cs = np.concatenate([[0.0], np.cumsum(s[order].astype(np.float64))])
    lo = np.searchsorted(key, key - 50, side="right"); hi = np.arange(len(key)) + 1
    lo = np.maximum(lo, np.searchsorted(key, d["ep"][order].astype(np.int64) * big, side="left"))  # stay inside the episode
    out = np.empty(len(key)); out[order] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return out


for k in ["own"] + list(others):
    s = signals(D[k]); D[k]["sig"] = {}
    for n in SIG:
        D[k]["sig"][n] = s[n].astype(np.float64); D[k]["sig"][n + "_avg50"] = avg50(D[k], s[n])
ALL = SIG + [n + "_avg50" for n in SIG]


def auc(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


own = D["own"]; cal = own["block"] % 2 == 0
R = {"args": vars(args), "n": {k: int(len(D[k]["e_pre"])) for k in D}, "errors": {k: {"e_pre": float(D[k]["e_pre"].mean()), "e_ad": float(D[k]["e_ad"].mean()), "dis": float(D[k]["dis"].mean())} for k in D},
     "scenes": {}, "action": {}}
L = []; P = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
P(f"## 설계 30 S: 지원 범위 신호 ({args.base}, 적응 모델 {K}개, 적응 데이터 {args.train})"); P()
P("k-step 잠재 오차 평균 (사전학습 / 적응 모델), 적응 모델끼리의 불일치:"); P()
P("| 데이터 | 창 수 | 사전학습 오차 | 적응 모델 오차 | 불일치 |"); P("|---|---|---|---|---|")
for k in D:
    P(f"| {k} | {len(D[k]['e_pre'])} | {D[k]['e_pre'].mean():.4f} | {D[k]['e_ad'].mean():.4f} | {D[k]['dis'].mean():.5f} |")
P()
THETA = {}
for sc, (lo, hi) in SCENES.items():
    R["scenes"][sc] = {}
    m_own = (own["tau"] >= lo) & (own["tau"] < hi)
    P(f"### {sc}: 자기 조건 95 % 수용 문턱에서 다른 데이터를 거부한 비율 [95 % CI] / AUC"); P()
    P("| 신호 | 자기 조건 시험 반쪽 거부율 | " + " | ".join(others) + " |"); P("|---|---|" + "---|" * len(others))
    for n in ALL:
        c = own["sig"][n][m_own & cal]; t = own["sig"][n][m_own & ~cal]
        if len(c) < 20 or len(t) < 20:
            continue
        th = float(np.quantile(c, Q)); THETA[(sc, n)] = th
        row = {"theta": th, "own_reject": float((t > th).mean()), "n_cal": int(len(c)), "n_test": int(len(t))}
        cells = []
        for k in others:
            d = D[k]; m = (d["tau"] >= lo) & (d["tau"] < hi); v = d["sig"][n][m]; blk = d["block"][m]
            if len(v) < 20:
                cells.append("표본 부족"); row[k] = None; continue
            ub = np.unique(blk); per = {b: v[blk == b] for b in ub}
            bs = [np.mean(np.concatenate([per[b] for b in rng.choice(ub, len(ub))]) > th) for _ in range(args.nboot)]
            rej = float((v > th).mean()); a = auc(t, v)
            row[k] = {"reject": rej, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "auc": a, "n": int(len(v)), "blocks": int(len(ub))}
            cells.append(f"{rej:.2f} [{row[k]['ci'][0]:.2f}, {row[k]['ci'][1]:.2f}] / {a:.2f}")
        R["scenes"][sc][n] = row
        P(f"| {n} | {row['own_reject']:.2f} | " + " | ".join(cells) + " |")
    P()

# action part: constant per-joint offset on the logged future actions, own-condition test windows (midwalk)
m = (own["tau"] >= 300) & ~cal
pick = np.where(m)[0]
pick = np.sort(rng.choice(pick, size=min(2000, len(pick)), replace=False)) if len(pick) else pick
if len(pick):
    ds = own["_ds"]; sub = own["_idx"][pick]
    P("### 행동 지원: 자기 조건 걷는 중 창에서 미래 행동 25개에 관절별 일정한 오프셋 N(0, σ²)을 더했을 때 문턱을 넘는 비율"); P()
    P("| σ (행동 단위) | " + " | ".join(n for n in ("dis", "knn", "mde", "clf")) + " | 불일치 중앙값 (σ = 0 대비 배) |"); P("|---|---|---|---|---|---|")
    base_gap = None
    for sg in [0.0] + SIGMAS:
        acc = {k: [] for k in ("dis", "phi")}
        key = np.random.default_rng(300)
        for b0 in range(0, len(sub), 256):
            for b in DataLoader(Subset(ds, sub[b0:b0 + 256].tolist()), batch_size=256, shuffle=False, num_workers=0):
                b = mu.dataset_batch_to_jax(b); fa = b["model_in"]["fut_acts"]
                off = jnp.asarray(key.normal(0.0, sg, (fa.shape[0], 1, fa.shape[-1])).astype(np.float32)) if sg > 0 else 0.0
                z, a5, cm, e_pre, e_ad, dis = forward(b["model_in"], b["labels"], fa + off)
                acc["dis"].append(np.asarray(dis)); acc["phi"].append(np.concatenate([np.asarray(z), np.asarray(a5), np.asarray(cm)], -1))
        dis = np.concatenate(acc["dis"]); phi = np.concatenate(acc["phi"])
        s = {"dis": dis, "knn": knn(phi), "mde": mde(phi), "clf": clf(phi)}
        fr = {n: float((s[n] > THETA[("midwalk", n)]).mean()) if ("midwalk", n) in THETA else float("nan") for n in s}
        med = float(np.median(dis))
        base_gap = med if sg == 0 else base_gap
        R["action"][str(sg)] = {"exceed": fr, "dis_median": med}
        P(f"| {sg} | " + " | ".join(f"{fr[n]:.2f}" for n in ("dis", "knn", "mde", "clf")) + f" | {med / max(base_gap, 1e-12):.2f} |")
    P()
json.dump(R, open(args.out + ".json", "w"), indent=1)
open(args.out + ".md", "w").write("\n".join(L) + "\n")
print("written", args.out + ".md")
