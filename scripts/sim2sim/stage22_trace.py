"""Design doc 22 §12 diagnosis 2: promised vs realized return of the executed plan (output.trace_npz).

Promised J_t = sum_k gamma^k r_hat_k + gamma^(T+1) V_hat(z_T) of the plan executed at step t (the MPPI return).
Realized G_t = discounted reward-to-go of the robot from t until its episode ends (fall/reset), gamma 0.99.
Steps in the reset stabilization period and the last ``tail`` steps (G truncated) are excluded.

    python3 scripts/sim2sim/stage22_trace.py results/stage22q/diag_trace_B.npz results/stage22q/diag_trace_C.npz
"""
import sys
import numpy as np

G_ = 0.99
tail = 300
for path in sys.argv[1:]:
    d = np.load(path)
    pr, pv = d["pred_rewards"].astype(np.float64), d["pred_values"].astype(np.float64)
    r, reset, resetting, fall = d["reward"].astype(np.float64), d["reset"] > 0, d["resetting"] > 0, d["fall"] > 0
    S, N, T = pr.shape
    J = (pr * G_ ** np.arange(T)).sum(-1) + pv[..., -1] * G_ ** (T + 1)
    G = np.zeros((S, N)); acc = np.zeros(N)
    soon = np.zeros((S, N), bool); cnt = np.full(N, 10 ** 9)
    for t in range(S - 1, -1, -1):
        acc = r[t] + G_ * acc * (~reset[t])
        G[t] = acc
        cnt = np.where(fall[t], 0, cnt + 1)
        soon[t] = cnt <= 50
    ok = ~resetting
    ok[S - tail:] = False
    gap = (J - G)[ok]
    j, g, s = J[ok], G[ok], soon[ok]

    def auc(score_low_is_bad, bad):
        o = np.argsort(np.argsort(-score_low_is_bad)) + 1.0
        n1, n0 = bad.sum(), (~bad).sum()
        return (o[bad].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

    print(f"{path}: steps {S} robots {N} used {ok.sum()} falls {int(fall.sum())}")
    print(f"  promised J mean {j.mean():.3f} sd {j.std():.3f} | realized G mean {g.mean():.3f} sd {g.std():.3f}")
    print(f"  J - G mean {gap.mean():+.3f} (safe {(j - g)[~s].mean():+.3f}, fall within 50 {(j - g)[s].mean():+.3f})")
    print(f"  corr(J, G) {np.corrcoef(j, g)[0, 1]:.3f} | fall-within-50 AUC of J {auc(j, s):.3f} | frac fall-soon {s.mean():.3f}")
