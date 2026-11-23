"""Design 40 toy: reference-free MPPI with the TRUE MJX model (arXiv 2511.19204 style).

Cubic Hermite spline over joint position + velocity nodes, DIAL-style annealing (Eq. 8),
min-max normalised weights, execute the best simulated sample, warm start = best shifted.
Arms: 'free' (reference-free cost, paper Table II walking weights) and 'clock'
(+ SimDist trot clock: contact match + swing-height terms, weight w_g).

  PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs python scripts/toy/toy_mppi.py \
      --cond nominal --arm free --episodes 16 --out results/stage40/e0_free_nominal
"""
import argparse, json, os, time

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

import toy_common as C

# ---- planner constants (design 40 §4; defaults below = registered E0 settings) ----
H_S, CTRL_DT = 0.9, 0.02
J = int(round(H_S / CTRL_DT))          # 45 control steps
K = int(os.environ.get("TOY_K", 10))   # spline nodes (10 -> dt_node = 0.1 s)
DTN = H_S / (K - 1)
W = dict(h=1e2, orient=10.0, q=0.0, cvel=0.5, cforce=5e-2, H=2.5e3)   # paper Table II, walking
Z_DES = 0.27
BIAS = jnp.array([0.0, 0.5, 0.5, 0.0])   # SimDist trot clock, FR FL RR RL
PERIOD, R_GAIT, SWING_H = 0.5, 0.5, 0.09


def hermite_mats(times):
    """Matrices so that q(t) = A_q th_q + A_v th_v and qd(t) = D_q th_q + D_v th_v."""
    n = len(times)
    Aq, Av, Dq, Dv = (np.zeros((n, K)) for _ in range(4))
    for r, t in enumerate(times):
        t = min(max(t, 0.0), H_S)
        k = min(int(t / DTN), K - 2)
        s = (t - k * DTN) / DTN
        h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
        d00, d10, d01, d11 = 6*s**2 - 6*s, 3*s**2 - 4*s + 1, -6*s**2 + 6*s, 3*s**2 - 2*s
        Aq[r, k], Aq[r, k + 1] = h00, h01
        Av[r, k], Av[r, k + 1] = h10 * DTN, h11 * DTN
        Dq[r, k], Dq[r, k + 1] = d00 / DTN, d01 / DTN
        Dv[r, k], Dv[r, k + 1] = d10, d11
    return tuple(jnp.asarray(x, jnp.float32) for x in (Aq, Av, Dq, Dv))


CTRL_T = np.arange(J) * CTRL_DT
NS, SIM_DT = 5, 0.004                    # physics substeps per control step (Playground sim_dt)
FINE = os.environ.get("TOY_FINE", "0") == "1"   # 1: PD target follows the spline at every 4 ms substep
_T = (CTRL_T[:, None] + (np.arange(NS)[None] * SIM_DT if FINE else np.zeros((1, NS)))).ravel()
A_q, A_v, D_q, D_v = hermite_mats(_T)    # rows = J*NS
S_q, S_v, SD_q, SD_v = hermite_mats(np.arange(K) * DTN + CTRL_DT)   # warm-start shift by one step


def quat_angle2(q, yaw_des):
    """squared rotation angle between body quat q (wxyz) and yaw-only desired orientation."""
    c, s = jnp.cos(yaw_des / 2), jnp.sin(yaw_des / 2)
    # conj(q_des) * q with q_des = (c, 0, 0, s)
    w = c * q[0] + s * q[3]
    x = c * q[1] + s * q[2]
    y = c * q[2] - s * q[1]
    z = c * q[3] - s * q[0]
    ang = 2 * jnp.arctan2(jnp.sqrt(x * x + y * y + z * z), jnp.abs(w))
    return ang ** 2


def make_planner(F, mx, n_sub, args):
    lo, hi = F.q_lo, F.q_hi
    f0 = F.mass * 9.81 / 4.0
    kd_kp = F.kd / F.kp
    I, N = args.iters, args.samples
    ii = jnp.arange(I, 0, -1)                                   # i = I .. 1
    kk = jnp.arange(K)
    var = jnp.exp(-(I - ii)[:, None] / (args.beta1 * I) - (K - kk)[None, :] / (args.beta2 * K))
    std = jnp.sqrt(var)                                         # (I, K)

    box = jnp.tile(jnp.array([args.box_hip, args.box_thigh, args.box_knee]), 4)
    blo = jnp.maximum(lo, F.default_q - box) if args.box_hip > 0 else lo     # design 40 v2 joint box (off by default)
    bhi = jnp.minimum(hi, F.default_q + box) if args.box_hip > 0 else hi

    def clamp(thq, thv):
        thq = jnp.clip(thq, blo, bhi)
        lim = jnp.minimum(bhi - thq, thq - blo) / (DTN / 2)
        return thq, jnp.clip(thv, -lim, lim)

    def traj(thq, thv):
        q = A_q @ thq + A_v @ thv              # (J*NS, 12)
        qd = D_q @ thq + D_v @ thv
        u = jnp.clip(q, lo, hi) + kd_kp * qd   # PD with velocity target via position offset
        return u.reshape(J, NS, 12)

    def step_cost(d, t, yaw0):
        con = F.contact(d)
        fv = F.feet_vel(d)
        c = W["h"] * jnp.abs(d.qpos[2] - Z_DES)
        c += W["orient"] * quat_angle2(d.qpos[3:7], yaw0)
        c += args.wq * jnp.sum((d.qpos[7:] - F.default_q) ** 2)
        c += W["cvel"] * jnp.sum(jnp.abs(fv) * con[:, None])
        c += W["cforce"] * jnp.sum(jnp.abs(F.feet_fn(d) - f0))
        if args.arm == "clock":
            tf = jnp.mod(t / PERIOD + BIAS, 1.0)
            des = tf < R_GAIT
            fh = F.feet_pos(d)[:, 2] - 0.023
            sw = 1.0 - jnp.exp(-((fh - SWING_H) ** 2) / (2 * (SWING_H / 4) ** 2))
            c += args.w_clock * (jnp.sum(con != des) + jnp.sum(sw * (~des)))
        return c

    def rollout(d0, ctrl, t0, yaw0, p_target):
        def body(carry, x):
            d, acc = carry
            u, j = x
            d = substeps(d, u)
            acc = acc + args.run_scale * step_cost(d, t0 + (j + 1) * CTRL_DT, yaw0)
            return (d, acc), None
        (dT, acc), _ = jax.lax.scan(body, (d0, 0.0), (ctrl, jnp.arange(J)))
        term = W["H"] * jnp.sum(jnp.abs(dT.qpos[:2] - p_target))
        return acc + term

    def substeps(d, u):
        for k in range(NS):
            d = mjx_env.step(mx, d, u[k], 1)
        return d

    rollout_b = jax.vmap(rollout, in_axes=(None, 0, None, None, None))

    def plan(d0, thq, thv, key, t0, yaw0):
        # target displacement over the horizon along the desired heading
        ts = t0 + CTRL_T
        dist = jnp.sum(C.v_des(ts)) * CTRL_DT
        p_target = d0.qpos[:2] + dist * jnp.array([jnp.cos(yaw0), jnp.sin(yaw0)])

        def it(carry, x):
            thq, thv, best_q, best_v, best_s = carry
            sd, k = x
            k1, k2 = jax.random.split(k)
            eq = jax.random.normal(k1, (N, K, 12)) * (args.sigma_q * sd)[None, :, None]
            ev = jax.random.normal(k2, (N, K, 12)) * (args.sigma_v * sd)[None, :, None]
            eq, ev = eq.at[0].set(0.0), ev.at[0].set(0.0)       # sample 0 = nominal
            Q, V = jax.vmap(clamp)(thq[None] + eq, thv[None] + ev)
            ctrl = jax.vmap(traj)(Q, V)
            S = rollout_b(d0, ctrl, t0, yaw0, p_target)
            S = jnp.where(jnp.isfinite(S), S, 1e9)
            Sh = (S - S.min()) / (S.max() - S.min() + 1e-9)
            w = jax.nn.softmax(-Sh / args.lam)
            thq_n = jnp.einsum("n,nkj->kj", w, Q)
            thv_n = jnp.einsum("n,nkj->kj", w, V)
            b = jnp.argmin(S)
            better = S[b] < best_s
            best_q = jnp.where(better, Q[b], best_q)
            best_v = jnp.where(better, V[b], best_v)
            best_s = jnp.minimum(S[b], best_s)
            return (thq_n, thv_n, best_q, best_v, best_s), (S.mean(), S.min())

        keys = jax.random.split(key, I)
        (thq, thv, bq, bv, bs), stats = jax.lax.scan(it, (thq, thv, thq, thv, jnp.inf), (std, keys))
        if args.exec == "mean":
            return thq, thv, bs, stats
        return bq, bv, bs, stats

    return plan, traj, substeps


def run(args):
    env = C.load_env()
    F = C.Feat(env)
    mx = C.condition_model(env, args.cond)
    n_sub = int(round(CTRL_DT / env._config.sim_dt))
    assert n_sub == NS
    plan, traj, substeps = make_planner(F, mx, n_sub, args)

    def episode_step(carry, step):
        d, thq, thv, key, yaw0 = carry
        key, kp = jax.random.split(key)
        t0 = step * CTRL_DT
        bq, bv, bs, stats = plan(d, thq, thv, kp, t0, yaw0)
        u = traj(bq, bv)[0]                       # (NS, 12)
        d = substeps(d, u)
        # warm start: best trajectory shifted by one control step
        thq_n = S_q @ bq + S_v @ bv
        thv_n = SD_q @ bq + SD_v @ bv
        rec = C.step_log(F, d)
        rec["ctrl"] = u[0]
        rec["best_cost"] = bs
        return (d, thq_n, thv_n, key, yaw0), rec

    def episode(seed):
        st = env.reset(jax.random.PRNGKey(seed))
        d = st.data
        q0 = d.qpos[7:]
        thq = jnp.tile(q0[None], (K, 1))
        thv = jnp.zeros((K, 12))
        yaw0 = F.yaw(d)
        key = jax.random.PRNGKey(1000 + seed)
        _, recs = jax.lax.scan(episode_step, (d, thq, thv, key, yaw0), jnp.arange(args.T))
        return recs

    seeds = jnp.arange(args.seed0, args.seed0 + args.episodes)
    t = time.time()
    recs = jax.jit(jax.vmap(episode))(seeds)
    recs = jax.tree_util.tree_map(np.asarray, recs)
    wall = time.time() - t
    summ = C.summarise(recs, recs["ctrl"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out + ".npz", **recs)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summ.items()}
    js.update(cond=args.cond, arm=args.arm, wall_s=wall, args=vars(args))
    with open(args.out + ".json", "w") as f:
        json.dump(js, f, indent=1)
    print(f"{args.arm} {args.cond}: J={np.mean(summ['J']):.3f} fell={np.mean(summ['fell']):.2f} "
          f"vx={np.nanmean(summ['vx_mean']):.3f} duty={np.nanmean(summ['duty'], 0).round(2)} wall={wall:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="nominal", choices=C.CONDITIONS)
    ap.add_argument("--arm", default="free", choices=("free", "clock"))
    ap.add_argument("--w_clock", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--sigma_q", type=float, default=0.2)
    ap.add_argument("--sigma_v", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=1.0)
    ap.add_argument("--beta2", type=float, default=1.0)
    ap.add_argument("--exec", default="best", choices=("best", "mean"))
    ap.add_argument("--wq", type=float, default=0.3)
    ap.add_argument("--run_scale", type=float, default=5.0,
                    help="running cost x this (paper sums over fine sim steps; 10 = 2 ms steps)")
    ap.add_argument("--box_hip", type=float, default=0.0, help="v2 joint box (rad) around default pose; 0 = off")
    ap.add_argument("--box_thigh", type=float, default=0.0)
    ap.add_argument("--box_knee", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
