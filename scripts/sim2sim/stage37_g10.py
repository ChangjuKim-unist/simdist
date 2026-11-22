"""Design doc 37 G1-0 (docs/design37_expert_free_wm_ko.md): which part of the planner score fails to rank the planner's own
final candidates - the 25-step predicted reward sum or the terminal value?

  score (container, JAX): re-run the model on the design-36 C3 anchors (stored history hist_p / hist_a / fut_cmds and the
      64 recorded candidates m_acts per branch point) and save per candidate R = sum_k g^k r_hat_k, Vt = g^(T+1) V_hat(z_T),
      the recomputed score R + Vt and the end latent z_T.  The height scan is not stored in the anchor; on the plane terrain
      it is uniform, clip(base_z - 0 - 0.3, -2, 2) (isaaclab mdp.height_scan, HEIGHT_SCAN_OFFSET 0.3, no corruption).
      Fidelity check against the recorded score m_score.
        python scripts/sim2sim/stage37_g10.py score --anchor X_anchor.npz --ckpt go1_paper --out X_g10.npz
  report (numpy only): join with the C3 branch csv (realised ret, ret_chunk, fell) and write the registered tables.
        python scripts/sim2sim/stage37_g10.py report --d results/stage37 --c3 results/stage36 --md results/stage37/g10.md
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

CONDS = [("nominal", "평소 (논문 모델)"), ("tow20", "끄는 짐 20 N (적응)"), ("mot50", "모터 전체 50 % (적응)"), ("legR50", "뒷다리 하나 50 % (적응)")]
GAMMA, HS_OFFSET, NB, SEED = 0.99, 0.3, 5000, 37


def score(a):
    import jax, jax.numpy as jnp
    import flax.nnx as nnx
    from simdist.utils import model as mu, paths
    z = np.load(a.anchor, allow_pickle=True)
    model, cfg, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), a.ckpt))
    gd, st = nnx.split(model)
    P, C, T, A = z["m_acts"].shape
    n_ext = int(a.extero_dim)
    ext = np.clip(z["root_pose"][:, 2:3] - HS_OFFSET, -2.0, 2.0).astype(np.float32) * np.ones((1, n_ext), np.float32)
    disc = jnp.asarray(GAMMA ** np.arange(T), jnp.float32); fd = GAMMA ** (T + 1)

    @jax.jit
    def f(x):
        y = nnx.merge(gd, st).inference(x)
        R = jnp.sum(y["rewards"] * disc, -1); Vt = y["values"][:, -1] * fd
        return R, Vt, y["latents"][:, -1]

    out = {k: [] for k in ("R", "Vt", "zT")}
    flat = [(p, c) for p in range(P) for c in range(C)]
    for i in range(0, len(flat), a.batch):
        idx = flat[i:i + a.batch]; pp = np.array([p for p, _ in idx]); cc = np.array([c for _, c in idx])
        x = {"proprio_obs_hist": jnp.asarray(z["hist_p"][pp], jnp.float32), "extero_obs": jnp.asarray(ext[pp]),
             "acts_hist": jnp.asarray(z["hist_a"][pp], jnp.float32), "fut_acts": jnp.asarray(z["m_acts"][pp, cc], jnp.float32),
             "fut_cmds": jnp.asarray(z["fut_cmds"][pp], jnp.float32)}
        R, Vt, zT = f(x)
        out["R"].append(np.asarray(R)); out["Vt"].append(np.asarray(Vt)); out["zT"].append(np.asarray(zT).reshape(len(idx), -1))
    R = np.concatenate(out["R"]).reshape(P, C); Vt = np.concatenate(out["Vt"]).reshape(P, C)
    zT = np.concatenate(out["zT"]).reshape(P, C, -1).astype(np.float32)
    rec = R + Vt; ms = z["m_score"]
    rho = np.mean([spearman(rec[p], ms[p]) for p in range(P)])
    print(f"fidelity {a.anchor}: within-point Spearman(recomputed, recorded) {rho:.4f}, "
          f"median |diff| {np.median(np.abs(rec - ms)):.4f}, score SD {ms.std():.4f}", flush=True)
    np.savez_compressed(a.out, R=R, Vt=Vt, zT=zT, rec=rec, m_score=ms, speed=z["speed"], seed=z["seed"], step=z["step"],
                        m_kind=z["m_kind"], m_iter=z["m_iter"], fidelity=rho)


def rankdata(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); xs = x[o]
    i = 0
    while i < len(x):  # average ties
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0; i = j + 1
    return r


def spearman(a, b):
    a, b = rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float))
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def auc(s, fell):
    """P(score of a non-falling candidate > score of a falling one); nan if the point has no mixed outcome"""
    s, fell = np.asarray(s), np.asarray(fell).astype(bool)
    if fell.all() or (~fell).all():
        return np.nan
    a, b = s[~fell][:, None], s[fell][None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def boot(v):
    v = np.asarray([x for x in v if np.isfinite(x)]); rng = np.random.default_rng(SEED)
    b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)


def load_join(g, c3csv):
    """per point: arrays over the 34 last-iteration sampled candidates (and all 62 sampled) of R, Vt, rec, ret, ret_chunk, fell"""
    key = {(round(float(s), 2), int(d), int(t)): i for i, (s, d, t) in enumerate(zip(g["speed"], g["seed"], g["step"]))}
    rows = defaultdict(dict)
    for r in csv.DictReader(open(c3csv)):
        if r["group"] != "M" or r["m_kind"] not in ("gauss", "policy", "top"):
            continue
        p = key[(round(float(r["speed"]), 2), int(r["seed"]), int(r["step"]))]
        rows[p][int(r["cand"])] = (float(r["ret"]), float(r["ret_chunk"]), int(r["fell"]), r["m_iter"] == "last")
    pts = []
    for p, d in rows.items():
        cs = sorted(d); last = np.array([d[c][3] for c in cs])
        pts.append({"R": g["R"][p, cs], "Vt": g["Vt"][p, cs], "rec": g["rec"][p, cs], "ret": np.array([d[c][0] for c in cs]),
                    "chunk": np.array([d[c][1] for c in cs]), "fell": np.array([d[c][2] for c in cs]), "last": last,
                    "seed": int(g["seed"][p])})
    return pts


def report(a):
    out, J = [], {}
    P = lambda s="": (print(s), out.append(s))  # noqa: E731
    fmt = lambda b: f"{b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]"  # noqa: E731
    P("## 설계 37 G1-0: planner 점수의 어느 부분이 마지막 후보 순위를 틀리나"); P()
    P("점수 = R + V항. R = 25스텝 예측 보상의 할인 합, V항 = γ^(T+1) × 끝 value. 실제 덩어리 수익 = 개루프 25스텝 동안 받은 보상 합. "
      "실제 이어 가기 수익 = 그 뒤 정책 head 100스텝 보상 합(넘어지면 거기서 끊김). 마지막 반복 후보 34개, 분기점마다 계산해 분기점 평균, "
      "[95 % 구간]은 분기점 bootstrap 5000번."); P()
    G = {}
    for c, _ in CONDS:
        g = np.load(os.path.join(a.d, f"g10_{c}.npz")); G[c] = (float(g["fidelity"]), load_join(g, os.path.join(a.c3, f"c3_{c}_branch.csv")))
    P("### 재구성 충실도 (기준: 분기점 안 Spearman(다시 계산한 점수, 기록된 점수) ≥ 0.99)"); P()
    P("| 조건 | Spearman | 통과 |"); P("|---|---|---|")
    fid_ok = True
    for c, ko in CONDS:
        f = G[c][0]; fid_ok &= f >= 0.99; P(f"| {ko} | {f:.4f} | {'예' if f >= 0.99 else '아니오'} |")
    P()
    tabs = {"rho_R": ("R 대 실제 덩어리 수익", lambda q: spearman(q["R"], q["chunk"])),
            "rho_V": ("V항 대 실제 이어 가기 수익", lambda q: spearman(q["Vt"], q["ret"] - q["chunk"])),
            "rho_R_ret": ("R 대 실제 125스텝 수익", lambda q: spearman(q["R"], q["ret"])),
            "rho_V_ret": ("V항 대 실제 125스텝 수익", lambda q: spearman(q["Vt"], q["ret"])),
            "rho_S_ret": ("점수 전체 대 실제 125스텝 수익", lambda q: spearman(q["rec"], q["ret"])),
            "auc_R": ("R의 넘어짐 AUC", lambda q: auc(q["R"], q["fell"])),
            "auc_V": ("V항의 넘어짐 AUC", lambda q: auc(q["Vt"], q["fell"])),
            "vshare": ("점수 분산 중 V항의 몫", lambda q: float(np.var(q["Vt"]) / max(np.var(q["R"]) + np.var(q["Vt"]), 1e-12)))}
    P("### 마지막 반복 후보 안: 항별 순위 상관, 넘어짐 AUC, 분산 몫"); P()
    P("| 조건 | " + " | ".join(t for t, _ in tabs.values()) + " |"); P("|---" * (1 + len(tabs)) + "|")
    for c, ko in CONDS:
        pts = [{k: (v[q["last"]] if isinstance(v, np.ndarray) else v) for k, v in q.items()} for q in G[c][1]]
        J[c] = {k: boot([fn(q) for q in pts]) for k, (_, fn) in tabs.items()}
        P(f"| {ko} | " + " | ".join(fmt(J[c][k]) for k in tabs) + " |")
    P()
    nV = sum(J[c]["rho_V"][0] < 0.1 and J[c]["rho_R"][0] >= 0.3 for c, _ in CONDS)
    nR = sum(J[c]["rho_R"][0] < 0.1 for c, _ in CONDS)
    nB = sum(J[c]["rho_R"][0] < 0.1 and J[c]["rho_V"][0] < 0.1 for c, _ in CONDS)
    verdict = ("재구성 실패" if not fid_ok else "둘 다" if nB >= 3 else "value 쪽" if nV >= 3 else "보상 쪽" if nR >= 3 else "중간")
    P(f"- 판정 (value 쪽 {nV}/4, 보상 쪽 {nR}/4, 둘 다 {nB}/4): **{verdict}**")
    os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump({"verdict": verdict, "fidelity": {c: G[c][0] for c, _ in CONDS}, "tables": J}, open(a.md.replace(".md", ".json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("--anchor", required=True); s.add_argument("--ckpt", required=True)
    s.add_argument("--out", required=True); s.add_argument("--batch", type=int, default=1024); s.add_argument("--extero_dim", type=int, default=315)
    r = sp.add_parser("report"); r.add_argument("--d", default="results/stage37"); r.add_argument("--c3", default="results/stage36")
    r.add_argument("--md", default="results/stage37/g10.md")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else report(a)


if __name__ == "__main__":
    main()
