"""Design doc 20 appendix: can a 2-second window of 1-step prediction error pick the model that matches the condition?

Input: per-window npz files written by latent_error.py (``--per_window``) for every (model, log). Logs are
condition-specific (nominal / friction0.15 / motor0.4) and come from two collection policies. For each log the
windows are grouped into chunks of ``--chunk`` consecutive steps (100 = 2 s at 50 Hz); the model with the smallest
mean error in the chunk is the selected model; accuracy = fraction of chunks whose selected model is the one
adapted to the log's condition (``nominal`` -> the original model).

    python scripts/sim2sim/stage20_appendix.py --dir results/stage20/pw --models go1_paper:nominal \
        go1_paper_friction0.15_dyn_b45000_s42:friction0.15 go1_paper_very_weak_motor_dyn_b45000_s42:motor0.4 \
        --logs sim2sim_nominal_go1_paper_prior_b15000:nominal:prior ... --out results/stage20/appendix.md
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--models", nargs="+", required=True, help="ckpt:condition")
ap.add_argument("--logs", nargs="+", required=True, help="dataset:condition:policy")
ap.add_argument("--chunk", type=int, default=100)
ap.add_argument("--metric", default="err_h1", choices=["err_h1", "err_mean"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

models = [m.split(":") for m in a.models]
conds = [c for _, c in models]
lines = [f"# 부록: 2초 창 예측 오차로 모델 고르기 (chunk {a.chunk}스텝, 지표 {a.metric})", "",
         "선택 = 창에서 평균 1스텝 잠재 예측 오차가 가장 작은 모델. 정답 = 기록의 조건에 적응한 모델(nominal → 원본).", ""]
lines.append("| 기록 조건 | 수집 정책 | 창 수 | 정확도 | " + " | ".join(f"→{c}" for c in conds) + " |")
lines.append("|---|---|---|---|" + "---|" * len(conds))
summary = {}
for spec in a.logs:
    ds, cond, pol = spec.split(":")
    errs = []
    for ckpt, _ in models:
        p = os.path.join(a.dir, f"pw_{ckpt}_{ds}.npz")
        d = np.load(p)
        t, e = d["t"], d[a.metric]
        order = np.argsort(t)
        errs.append((t[order], e[order]))
    t0 = errs[0][0]
    assert all(np.array_equal(t0, t) for t, _ in errs), "window time indices differ between models"
    chunk_id = t0 // a.chunk
    ids = np.unique(chunk_id)
    E = np.stack([e for _, e in errs], axis=1)  # (N, M)
    means = np.stack([E[chunk_id == i].mean(axis=0) for i in ids])  # (C, M)
    pick = means.argmin(axis=1)
    truth = conds.index(cond)
    acc = float((pick == truth).mean())
    counts = [int((pick == j).sum()) for j in range(len(conds))]
    summary[(cond, pol)] = (acc, len(ids))
    lines.append(f"| {cond} | {pol} | {len(ids)} | {acc:.2f} | " + " | ".join(str(c) for c in counts) + " |")
lines.append("")
for cond in sorted({c for c, _ in summary}):
    accs = [summary[k][0] for k in summary if k[0] == cond]
    lines.append(f"- {cond}: 수집 정책 평균 정확도 {np.mean(accs):.2f} (예측: 마찰 0.15 ≥ 0.8, 모터 0.4 < 0.6)")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
open(a.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
