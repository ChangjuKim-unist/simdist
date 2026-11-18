"""Design doc 31 gate G3 (docs/design31_token_gate_ko.md §3.3): with ONE fine-tuned model that carries a condition token
(xi = token on the adaptation data, xi = 0 on simulation replay), does the error difference between the two tokens tell
"this is not the adapted condition"?

    python scripts/sim2sim/stage31_signals.py --base go1_paper --dyn go1_paper_friction0.15_dyn_b45000_s42 \
        --tok go1_paper_friction0.15_tok_b45000_s42 --own sim2sim_friction0.15_go1_paper_ho_b15000 \
        --others nominal=... other=... payload=... --out results/stage31/sig_P_ice

Per window (25-step history, 25 logged future actions): k-step latent error (mean over 25 steps x 64 dims) of
  e_pre  pretrained model            e_dyn  dynamics-only fine-tuned model (design 20 recipe, reference)
  e_A    token model, adapted token  e_S    token model, simulation token (xi = 0)
Signals (higher = less reason to use the adapted token):
  tok    e_A - e_S     (registered signal)
  ref    e_dyn - e_pre (the "err" baseline of design 30, recomputed on the same windows)
  absA   e_A           (descriptive)
Same rule as design 30: 50-window average within the episode, scenes startup (25 <= tau < 125) / midwalk (tau >= 300),
split conformal threshold = 95th percentile on the even 500-step blocks of the own-condition held-out set, rejection
rate on the other sets with a block bootstrap, AUC against the odd blocks of the own set.
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
ap.add_argument("--base", required=True)
ap.add_argument("--dyn", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--token", type=float, nargs="+", default=[1.0, 0.0, 0.0, 0.0])
ap.add_argument("--own", required=True)
ap.add_argument("--others", nargs="+", required=True, help="name=dataset")
ap.add_argument("--system", default="go1")
ap.add_argument("--max_eval", type=int, default=0)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rng = np.random.default_rng(31)
SCENES = {"startup": (25, 125), "midwalk": (300, 10 ** 9)}
BLOCK = 500; Q = 0.95

ld = lambda n: mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), n))  # noqa: E731
base, cfg, _ = ld(args.base)
cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
tokm = ld(args.tok)[0]
assert getattr(tokm, "film_proj", None) is not None, "--tok must be a model with a FiLM context (variant tok)"
sp_base, sp_dyn, sp_tok = nnx.split(base), nnx.split(ld(args.dyn)[0]), nnx.split(tokm)
TOKEN = jnp.asarray(args.token, jnp.float32)


@jax.jit
def forward(x, y):
    b = nnx.merge(*sp_base); sc = b.get_scaler()
    ys = sc.scale(y)
    fut = b.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    err = lambda p: jnp.mean((p - fut) ** 2, axis=(1, 2))  # noqa: E731
    t = nnx.merge(*sp_tok); n = x["fut_acts"].shape[0]
    xa = dict(x); xa["xi"] = jnp.tile(TOKEN[None], (n, 1))
    xs = dict(x); xs["xi"] = jnp.zeros((n, TOKEN.shape[0]), jnp.float32)
    return (err(b(x, deterministic=True)["latents"]), err(nnx.merge(*sp_dyn)(x, deterministic=True)["latents"]),
            err(t(xa, deterministic=True)["latents"]), err(t(xs, deterministic=True)["latents"]))


def load(name):
    c = dict(cfg); c["data"] = {"dataset_name": name}
    ds = get_dataset(c); ds.eval()
    H, T = ds.H, ds.T
    ddir = paths.get_processed_data_dir(name, args.system, H, T)
    f = h5py.File(os.path.join(ddir, "terminal.hdf5"), "r"); ste = np.asarray(f[list(f.keys())[0]][:, 0])
    starts = np.asarray(ds._data["start_idxs"][:]).astype(np.int64)
    jump = np.concatenate([[0], np.where(ste[1:] > ste[:-1])[0] + 1])  # episode starts (steps-to-end jumps up)
    idx = np.arange(len(starts))
    if args.max_eval and len(idx) > args.max_eval:
        idx = np.sort(rng.choice(idx, size=args.max_eval, replace=False))
    out = {k: [] for k in ("e_pre", "e_dyn", "e_A", "e_S")}
    for b0 in range(0, len(idx), 256):
        for b in DataLoader(Subset(ds, idx[b0:b0 + 256].tolist()), batch_size=256, shuffle=False, num_workers=0):
            b = mu.dataset_batch_to_jax(b)
            for k, v in zip(out, forward(b["model_in"], b["labels"])):
                out[k].append(np.asarray(v, np.float64))
    d = {k: np.concatenate(v) for k, v in out.items()}
    t0 = starts[idx] + H; ep = np.searchsorted(jump, starts[idx], side="right") - 1
    d["t0"] = t0; d["ep"] = ep; d["tau"] = t0 - jump[ep]; d["block"] = starts[idx] // BLOCK
    print(f"{name}: windows {len(idx)}, episodes {len(np.unique(ep))}, " + ", ".join(f"{k} {d[k].mean():.4f}" for k in out), flush=True)
    return d


def avg50(d, s):
    """mean of s over the windows of the same episode with t0 in (t0 - 50, t0]"""
    order = np.lexsort((d["t0"], d["ep"])); big = int(d["t0"].max()) + 100
    key = d["ep"][order].astype(np.int64) * big + d["t0"][order]
    cs = np.concatenate([[0.0], np.cumsum(s[order].astype(np.float64))])
    lo = np.searchsorted(key, key - 50, side="right"); hi = np.arange(len(key)) + 1
    lo = np.maximum(lo, np.searchsorted(key, d["ep"][order].astype(np.int64) * big, side="left"))  # stay inside the episode
    out = np.empty(len(key)); out[order] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return out


def auc(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


others = dict(o.split("=", 1) for o in args.others)
D = {"own": load(args.own)}
for k, v in others.items():
    D[k] = load(v)
SIG = ["tok", "ref", "absA"]
for d in D.values():
    raw = {"tok": d["e_A"] - d["e_S"], "ref": d["e_dyn"] - d["e_pre"], "absA": d["e_A"]}
    d["sig"] = {}
    for n in SIG:
        d["sig"][n] = raw[n]; d["sig"][n + "_avg50"] = avg50(d, raw[n])
ALL = SIG + [n + "_avg50" for n in SIG]

own = D["own"]; cal = own["block"] % 2 == 0
R = {"args": vars(args), "n": {k: int(len(D[k]["e_pre"])) for k in D},
     "errors": {k: {e: float(D[k][e].mean()) for e in ("e_pre", "e_dyn", "e_A", "e_S")} for k in D}, "scenes": {}}
L = []; P = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
P(f"## 설계 31 G3: 토큰 오차 차이 신호 ({args.base}, 토큰 모델 {args.tok})"); P()
P("k-step 잠재 오차 평균 (25스텝 × 64차원, 낮을수록 좋음)"); P()
P("| 데이터 | 창 수 | 사전학습 | 동역학 fine-tuning | 토큰 모델, 적응 토큰 | 토큰 모델, 시뮬 토큰 |"); P("|---|---|---|---|---|---|")
for k in D:
    P(f"| {k} | {len(D[k]['e_pre'])} | {D[k]['e_pre'].mean():.4f} | {D[k]['e_dyn'].mean():.4f} | {D[k]['e_A'].mean():.4f} | {D[k]['e_S'].mean():.4f} |")
P()
for sc, (lo, hi) in SCENES.items():
    R["scenes"][sc] = {}
    m_own = (own["tau"] >= lo) & (own["tau"] < hi)
    P(f"### {sc}: 자기 조건 95 % 수용 문턱에서 다른 데이터를 거부한 비율 [95 % CI] / AUC"); P()
    P("| 신호 | 자기 조건 시험 반쪽 거부율 | " + " | ".join(others) + " |"); P("|---|---|" + "---|" * len(others))
    for n in ALL:
        c = own["sig"][n][m_own & cal]; t = own["sig"][n][m_own & ~cal]
        if len(c) < 20 or len(t) < 20:
            continue
        th = float(np.quantile(c, Q))
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
json.dump(R, open(args.out + ".json", "w"), indent=1)
open(args.out + ".md", "w").write("\n".join(L) + "\n")
print("written", args.out + ".md")
