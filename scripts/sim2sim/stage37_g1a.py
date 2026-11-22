"""Design doc 37 G1-A (docs/design37_expert_free_wm_ko.md §4.1): an offline correction head Delta(z_T) on the frozen model's
end latent; new planner score = R + g^(T+1) [V(z_T) + Delta(z_T)] = recorded score + g^(T+1) Delta(z_T).

  arms   A3  MSE: V(z_T) + Delta(z_T) -> realised continuation return (ret - ret_chunk) / g^(T+1)
              (62 sampled candidates + the executed-plan / head-chunk clones, each clone one sample)
         A1  pairwise logistic loss on the new score for same-point sampled pairs whose realised ret differs by more than the
              condition's clone-noise SD (pooled SD of the mean / head clones); score differences divided by the training SD
              of the recorded score
  labels O   same condition, train seeds 11-13 -> test seeds 14-16
         N   nominal (paper model) seeds 11-13 only -> each adapted condition's seeds 14-16 (shared encoder / value head)
  Delta  MLP (standardised input, 64 x 2 ReLU), AdamW 1e-3, weight decay 1e-4, 3000 steps, minibatch 256, init seeds 0/1/2
  test   (i) change of the last-iteration Spearman(score, ret), (ii) last-iteration fall AUC of the new score, (iii) change of the
         62-candidate Spearman, (iv) ret of the new top-scored sampled candidate - ret of the old one (% of mean |head|)

Inputs: results/stage37/g10_{c}.npz (stage37_g10.py score: R, Vt, zT, rec per anchor point x 64 candidates) and the C3 csv.
    python scripts/sim2sim/stage37_g1a.py --d results/stage37 --c3 results/stage36 --md results/stage37/g1a.md   (container, JAX)
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage37_g10 import CONDS, spearman, auc, GAMMA  # noqa: E402

ADAPTED = ["tow20", "mot50", "legR50"]
TRAIN_SEEDS, TEST_SEEDS = {11, 12, 13}, {14, 15, 16}
NB, BSEED, STEPS, MB, LR, WD, HID, INITS = 5000, 37, 3000, 256, 1e-3, 1e-4, 64, (0, 1, 2)


def load(d, c3, c):
    g = np.load(os.path.join(d, f"g10_{c}.npz"))
    T = 25; fd = GAMMA ** (T + 1)
    key = {(round(float(s), 2), int(sd), int(t)): i for i, (s, sd, t) in enumerate(zip(g["speed"], g["seed"], g["step"]))}
    rows = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(c3, f"c3_{c}_branch.csv"))):
        if r["group"] != "M":
            continue
        p = key[(round(float(r["speed"]), 2), int(r["seed"]), int(r["step"]))]
        rows[p].append((int(r["cand"]), r["m_kind"], r["m_iter"], float(r["ret"]), float(r["ret_chunk"]), int(r["fell"])))
    pts = []
    for p, rr in rows.items():
        smp = sorted([x for x in rr if x[1] in ("gauss", "policy", "top")]); cs = [x[0] for x in smp]
        cl = [x for x in rr if x[1] in ("mean", "head")]
        pts.append({"seed": int(g["seed"][p]), "cand": np.array(cs), "z": g["zT"][p, cs], "rec": g["rec"][p, cs].astype(np.float64),
                    "V": g["Vt"][p, cs] / fd, "ret": np.array([x[3] for x in smp]), "chunk": np.array([x[4] for x in smp]),
                    "fell": np.array([x[5] for x in smp]), "last": np.array([x[2] == "last" for x in smp]),
                    "cl_z": g["zT"][p, [x[0] for x in cl]], "cl_V": g["Vt"][p, [x[0] for x in cl]] / fd,
                    "cl_cont": np.array([x[3] - x[4] for x in cl]), "cl_kind": [x[1] for x in cl], "cl_ret": np.array([x[3] for x in cl])})
    v = [np.var(q["cl_ret"][[k == kind for k in q["cl_kind"]]], ddof=1) for q in pts for kind in ("mean", "head")]
    head = np.mean([abs(q["cl_ret"][[k == "head" for k in q["cl_kind"]]].mean()) for q in pts])
    return pts, float(np.sqrt(np.mean(v))), float(head), fd


def train(arm, pts, noise_sd, fd, init):
    import jax, jax.numpy as jnp, optax
    Z = np.concatenate([q["z"] for q in pts] + ([q["cl_z"] for q in pts] if arm == "A3" else []))
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    k = jax.random.PRNGKey(init); k1, k2, k3, kd = jax.random.split(k, 4); D = Z.shape[1]
    he = lambda kk, i, o: jax.random.normal(kk, (i, o)) * np.sqrt(2.0 / i)  # noqa: E731
    params = {"w1": he(k1, D, HID), "b1": jnp.zeros(HID), "w2": he(k2, HID, HID), "b2": jnp.zeros(HID), "w3": he(k3, HID, 1) * 0.01, "b3": jnp.zeros(1)}

    def f(p, z):
        h = jax.nn.relu(((z - mu) / sd) @ p["w1"] + p["b1"]); h = jax.nn.relu(h @ p["w2"] + p["b2"]); return (h @ p["w3"] + p["b3"])[..., 0]

    rng = np.random.default_rng(init)
    if arm == "A3":
        V = np.concatenate([q["V"] for q in pts] + [q["cl_V"] for q in pts])
        tgt = np.concatenate([(q["ret"] - q["chunk"]) / fd for q in pts] + [q["cl_cont"] / fd for q in pts]) - V  # Delta target
        t_sd = tgt.std() + 1e-6
        def loss(p, idx):
            return jnp.mean((f(p, Z[idx]) - tgt[idx]) ** 2) / t_sd ** 2
        draw = lambda: rng.integers(0, len(Z), MB)  # noqa: E731
    else:
        I, Jx = [], []; off = 0
        for q in pts:
            r = q["ret"]; a, b = np.nonzero(r[:, None] - r[None, :] > noise_sd); I.append(a + off); Jx.append(b + off); off += len(r)
        I, Jx = np.concatenate(I), np.concatenate(Jx); rec = np.concatenate([q["rec"] for q in pts]); s_sd = rec.std() + 1e-6
        def loss(p, idx):
            si = rec[I[idx]] + fd * f(p, Z[I[idx]]); sj = rec[Jx[idx]] + fd * f(p, Z[Jx[idx]])
            return jnp.mean(jax.nn.softplus(-(si - sj) / s_sd))
        draw = lambda: rng.integers(0, len(I), MB)  # noqa: E731
    opt = optax.adamw(LR, weight_decay=WD); st = opt.init(params)

    @jax.jit
    def step(p, s, idx):
        l, gr = jax.value_and_grad(loss)(p, idx); u, s = opt.update(gr, s, p); return optax.apply_updates(p, u), s, l
    for _ in range(STEPS):
        params, st, l = step(params, st, jnp.asarray(draw()))
    return lambda z: np.asarray(f(params, jnp.asarray(z)))


def metrics(pts, deltas, fd, head):
    """per test point, averaged over init seeds: (i) d last-iter Spearman, (ii) last-iter fall AUC, (iii) d 62-cand Spearman,
    (iv) ret(new top) - ret(old top) in % of mean |head|"""
    out = {k: [] for k in ("i", "ii", "iii", "iv", "old_i", "old_ii")}
    for qi, q in enumerate(pts):
        L = q["last"]; per = {k: [] for k in ("i", "ii", "iii", "iv")}
        for dl in deltas:
            s = q["rec"] + fd * dl[qi]
            per["i"].append(spearman(s[L], q["ret"][L]) - spearman(q["rec"][L], q["ret"][L]))
            per["ii"].append(auc(s[L], q["fell"][L]))
            per["iii"].append(spearman(s, q["ret"]) - spearman(q["rec"], q["ret"]))
            per["iv"].append(100.0 * (q["ret"][np.argmax(s)] - q["ret"][np.argmax(q["rec"])]) / head)
        for k, v in per.items():
            out[k].append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
        out["old_i"].append(spearman(q["rec"][L], q["ret"][L])); out["old_ii"].append(auc(q["rec"][L], q["fell"][L]))
    return out


def boot(v):
    v = np.asarray([x for x in v if np.isfinite(x)]); rng = np.random.default_rng(BSEED)
    b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="results/stage37"); ap.add_argument("--c3", default="results/stage36")
    ap.add_argument("--md", default="results/stage37/g1a.md"); a = ap.parse_args()
    data = {c: load(a.d, a.c3, c) for c, _ in CONDS}
    split = lambda pts, S: [q for q in pts if q["seed"] in S]  # noqa: E731
    R = {}
    for arm in ("A3", "A1"):
        nom_tr = split(data["nominal"][0], TRAIN_SEEDS)
        headsN = [train(arm, nom_tr, data["nominal"][1], data["nominal"][3], s) for s in INITS]
        for c, _ in CONDS:
            pts, nsd, head, fd = data[c]; tr, te = split(pts, TRAIN_SEEDS), split(pts, TEST_SEEDS)
            headsO = [train(arm, tr, nsd, fd, s) for s in INITS]
            for lab, hs in (("O", headsO), ("N", headsN)):
                if lab == "N" and c == "nominal":
                    continue
                m = metrics(te, [[h(q["z"]) for q in te] for h in hs], fd, head)
                R[(arm, lab, c)] = {k: boot(v) for k, v in m.items()}
                print(arm, lab, c, {k: round(v[0], 3) for k, v in R[(arm, lab, c)].items()}, flush=True)
    out = []; P = lambda s="": (print(s), out.append(s))  # noqa: E731
    fmt = lambda b, d=2: f"{b[0]:+.{d}f} [{b[1]:+.{d}f}, {b[2]:+.{d}f}]"  # noqa: E731
    P("## 설계 37 G1-A: 끝 latent 교정 head Δ (오프라인, 시드 14–16 분기점에서 시험)"); P()
    P("(i) 마지막 반복 후보 안 Spearman(점수, 실제 125스텝 수익)의 변화, (ii) 마지막 반복 후보 안 넘어짐 AUC(0.5 = 우연), "
      "(iii) 표본 후보 62개 전체 Spearman의 변화, (iv) 새 점수 1위 후보 − 원래 점수 1위 후보의 실제 수익(head 절댓값 평균 대비 %). "
      "초기화 시드 3개 평균, [95 % 구간]은 시험 분기점 bootstrap."); P()
    P("| 조건 | 원래 점수 (i) 수준 | 원래 점수 넘어짐 AUC |"); P("|---|---|---|")
    for c, ko in CONDS:
        r = R[("A3", "O", c)]; P(f"| {ko} | {r['old_i'][0]:.2f} | {r['old_ii'][0]:.2f} |")
    P()
    verdict = {}
    for arm, ako in (("A3", "A3 value 정합 (MSE)"), ("A1", "A1 순위 (pairwise)")):
        for lab, lko in (("O", "같은 조건 라벨 O"), ("N", "평소 라벨만 N")):
            P(f"### {ako}, {lko}"); P()
            P("| 조건 | (i) | (ii) | (iii) | (iv) % |"); P("|---|---|---|---|---|")
            for c, ko in CONDS:
                if (arm, lab, c) in R:
                    r = R[(arm, lab, c)]; P(f"| {ko} | {fmt(r['i'])} | {r['ii'][0]:.2f} [{r['ii'][1]:.2f}, {r['ii'][2]:.2f}] | {fmt(r['iii'])} | {fmt(r['iv'], 1)} |")
            P()
        okO = {c: (R[(arm, "O", c)]["i"][0] >= 0.2 or R[(arm, "O", c)]["ii"][0] >= 0.65) and R[(arm, "O", c)]["iii"][0] > -0.05 for c in ADAPTED}
        succ = sum(okO.values()) >= 2
        dep_c = [c for c in ADAPTED if okO[c] and R[(arm, "O", c)]["i"][0] > 0 and R[(arm, "N", c)]["i"][0] >= 0.5 * R[(arm, "O", c)]["i"][0]]
        dep = succ and len(dep_c) >= 2
        nomO = (R[(arm, "O", "nominal")]["i"][0] >= 0.2 or R[(arm, "O", "nominal")]["ii"][0] >= 0.65) and R[(arm, "O", "nominal")]["iii"][0] > -0.05
        verdict[arm] = {"success_O": succ, "conds_O": [c for c in ADAPTED if okO[c]], "deployable": dep, "conds_N": dep_c, "nominal_O": nomO}
        P(f"- {ako}: 교정 성공(O) **{'예' if succ else '아니오'}** (기준 충족 적응 조건 {sum(okO.values())}/3: {', '.join(c for c in ADAPTED if okO[c]) or '없음'}); "
          f"배치 가능 **{'예' if dep else '아니오'}** (N이 O 개선의 절반 이상인 조건 {len(dep_c)}/3, O 개선 > 0인 조건만 셈); 평소 조건 기준 충족 {'예' if nomO else '아니오'}")
        P()
    both_fail = not verdict["A3"]["success_O"] and not verdict["A1"]["success_O"]
    P(f"- 종합: {'두 팔 모두 실패 → 후보 순위 교정으로 G2에 가지 않음' if both_fail else '성공한 팔 있음 (§4.1 7항에 따라 진행)'}")
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump({"verdict": verdict, "results": {"|".join(k): v for k, v in R.items()}}, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
