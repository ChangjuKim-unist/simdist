"""Design doc 30 part A (docs/design30_support_signal_ko.md §3): can an action-space adapter on a FROZEN pretrained world
model explain the adaptation data as well as fine-tuning the latent dynamics?

    python scripts/sim2sim/stage30_adapter.py --base go1_paper --ft go1_paper_very_weak_motor_dyn_b45000_s42 \
        --train sim2sim_very_weak_motor_go1_paper_b45000 --evals own=..._ho_b15000 nominal=... other=... --out results/stage30/adp_P_mot

The pretrained model predicts with transformed future actions a' = g(z, a) (scaled action units, identity at init):
  A1 per-joint affine      a' = a * (1 + w) + b                      (24 parameters)
  A2 state-dependent MLP   a' = a + MLP([z, a]), 2 x 64 tanh, zero-init output; z = encoder latent at the history end
Loss = the fine-tuning loss (mean squared error of the 25 predicted latents to the encoded real future), adapter parameters
only, Adam 1e-3, 3000 steps, batch 64 (the fine-tuning budget of design 20).
Metric per evaluation set: k-step latent error (mean over 25 steps x 64 dims). Recovered share on the own condition =
(E_pre - E_adapter) / (E_pre - E_finetuned), block bootstrap (500-step blocks). On the other sets the ratio
E_adapter / E_pre shows what an always-on adapter does away from its data.
"""
import argparse, json, os
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader, Subset
from simdist.data.dataset import get_dataset
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ft", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--evals", nargs="+", required=True, help="name=dataset (first = own condition)")
ap.add_argument("--system", default="go1")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_eval", type=int, default=0)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rng = np.random.default_rng(30); BLOCK = 500

base, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), args.base))
cfg = dict(cfg); cfg["system"] = dict(cfg["system"]); cfg["system"]["name"] = args.system
ft = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), args.ft))[0]
sb, sf = nnx.split(base), nnx.split(ft)


def dataset(name):
    c = dict(cfg); c["data"] = {"dataset_name": name}
    ds = get_dataset(c); ds.eval()
    return ds


def g_apply(kind, p, z, a):  # a: (B, T, 12) scaled; z: (B, D)
    if kind == "none":
        return a
    if kind == "A1":
        return a * (1.0 + p["w"]) + p["b"]
    zz = jnp.broadcast_to(z[:, None], (a.shape[0], a.shape[1], z.shape[-1]))
    h = jnp.tanh(jnp.concatenate([zz, a], -1) @ p["w1"] + p["b1"]); h = jnp.tanh(h @ p["w2"] + p["b2"])
    return a + h @ p["w3"] + p["b3"]


def errors(kind, p, split, x, y):
    m = nnx.merge(*split); sc = m.get_scaler()
    xs = sc.scale(x); ys = sc.scale(y)
    fut = m.encode_latent(ys["proprio_obs"], ys["extero_obs"], deterministic=True)
    z = m.encode_latent(xs["proprio_obs_hist"][:, -1], xs["extero_obs"], deterministic=True)
    a2 = g_apply(kind, p, z, xs["fut_acts"])
    x2 = dict(x); x2["fut_acts"] = sc.unscale({"fut_acts": a2})["fut_acts"]
    pred = m(x2, deterministic=True)["latents"]
    return jnp.mean((pred - fut) ** 2, axis=(1, 2))  # (B,)


def init(kind, key, dz, da):
    if kind == "A1":
        return {"w": jnp.zeros(da), "b": jnp.zeros(da)}
    k1, k2 = jax.random.split(key)
    g = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(1.0 / a)  # noqa: E731
    return {"w1": g(k1, dz + da, 64), "b1": jnp.zeros(64), "w2": g(k2, 64, 64), "b2": jnp.zeros(64), "w3": jnp.zeros((64, da)), "b3": jnp.zeros(da)}


def train(kind, ds):
    b0 = mu.dataset_batch_to_jax(next(iter(DataLoader(Subset(ds, [0, 1]), batch_size=2, num_workers=0))))
    da = b0["model_in"]["fut_acts"].shape[-1]
    p = init(kind, jax.random.PRNGKey(42), 64, da); opt = optax.adam(1e-3); st = opt.init(p)

    @jax.jit
    def step(p, st, x, y):
        l, g = jax.value_and_grad(lambda q: jnp.mean(errors(kind, q, sb, x, y)))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    order = np.random.default_rng(42).permutation(len(ds)); i = 0; hist = []
    for s in range(args.steps):
        if i + args.batch > len(order):
            order = np.random.default_rng(42 + s).permutation(len(ds)); i = 0
        idx = np.sort(order[i:i + args.batch]).tolist(); i += args.batch
        b = mu.dataset_batch_to_jax(next(iter(DataLoader(Subset(ds, idx), batch_size=args.batch, num_workers=0))))
        p, st, l = step(p, st, b["model_in"], b["labels"]); hist.append(float(l))
        if s % 500 == 0 or s == args.steps - 1:
            print(f"  {kind} step {s} loss {np.mean(hist[-100:]):.5f}", flush=True)
    return p


def evaluate(ds, arms):
    n = len(ds) if args.max_eval <= 0 else min(len(ds), args.max_eval)
    idx = np.arange(len(ds)) if n == len(ds) else np.sort(rng.choice(len(ds), n, replace=False))
    starts = np.asarray(ds._data["start_idxs"][:]).astype(np.int64)[idx]
    out = {k: [] for k in arms}
    fns = {k: jax.jit(lambda x, y, kind=kind, p=p, sp=sp: errors(kind, p, sp, x, y)) for k, (kind, p, sp) in arms.items()}
    for b0 in range(0, len(idx), 256):
        b = mu.dataset_batch_to_jax(next(iter(DataLoader(Subset(ds, idx[b0:b0 + 256].tolist()), batch_size=256, num_workers=0))))
        for k in arms:
            out[k].append(np.asarray(fns[k](b["model_in"], b["labels"])))
    return {k: np.concatenate(v) for k, v in out.items()}, starts // BLOCK


evals = [e.split("=", 1) for e in args.evals]
tr = dataset(args.train)
print("training adapters on", args.train, len(tr), flush=True)
P1 = train("A1", tr); P2 = train("A2", tr)
arms = {"pre": ("none", None, sb), "ft": ("none", None, sf), "A1": ("A1", P1, sb), "A2": ("A2", P2, sb)}
R = {"args": vars(args), "A1_scale": (1.0 + np.asarray(P1["w"])).tolist(), "A1_bias": np.asarray(P1["b"]).tolist(), "evals": {}}
L = []; P = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
P(f"## 설계 30 A: 행동 어댑터 ({args.base}, 적응 데이터 {args.train}, 비교 fine-tuning {args.ft})"); P()
P("k-step 잠재 오차 (25스텝 × 64차원 평균, 낮을수록 좋음)"); P()
P("| 평가 데이터 | 사전학습 | 동역학 fine-tuning | A1 관절별 affine | A2 상태 의존 MLP | 회복 몫 A1 [95 % CI] | 회복 몫 A2 [95 % CI] | 사전학습 대비 오차 비 (fine-tuning / A1 / A2) |"); P("|---|---|---|---|---|---|---|---|")
OWN = evals[0][0]
for name, dsn in evals:
    e, blk = evaluate(dataset(dsn), arms)
    ub = np.unique(blk); per = {b: np.where(blk == b)[0] for b in ub}
    m = {k: float(v.mean()) for k, v in e.items()}

    def share(ix, a):
        d = e["pre"][ix].mean() - e["ft"][ix].mean()
        return (e["pre"][ix].mean() - e[a][ix].mean()) / d if abs(d) > 1e-9 else np.nan
    row = {"mean": m, "n": int(len(blk)), "blocks": int(len(ub))}
    cells = []
    for a in ("A1", "A2"):
        bs = [share(np.concatenate([per[b] for b in rng.choice(ub, len(ub))]), a) for _ in range(args.nboot)]
        row["share_" + a] = [float(share(np.arange(len(blk)), a)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        cells.append("{:.2f} [{:.2f}, {:.2f}]".format(*row["share_" + a]))
    row["ratio"] = {a: m[a] / m["pre"] for a in ("ft", "A1", "A2")}
    R["evals"][name] = row
    if name != OWN:  # the recovered share is defined on the own condition only
        cells = ["—", "—"]
    P(f"| {name} | {m['pre']:.4f} | {m['ft']:.4f} | {m['A1']:.4f} | {m['A2']:.4f} | {cells[0]} | {cells[1]} | {row['ratio']['ft']:.2f} / {row['ratio']['A1']:.2f} / {row['ratio']['A2']:.2f} |")
P(); P("A1이 배운 관절별 배율 (1 = 변화 없음): " + ", ".join(f"{v:.2f}" for v in R["A1_scale"])); P()
np.savez(args.out + "_params.npz", **{"A1_" + k: np.asarray(v) for k, v in P1.items()}, **{"A2_" + k: np.asarray(v) for k, v in P2.items()})
json.dump(R, open(args.out + ".json", "w"), indent=1)
open(args.out + ".md", "w").write("\n".join(L) + "\n")
print("written", args.out + ".md")
