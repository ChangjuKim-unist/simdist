"""Design 40 E1/E2 toy: expert-free state world model, data sources, WM-MPPI.

Data sources (all under Playground's Go1 domain randomization, command v ~ U(0.2, 0.9) per episode):
  ppo        blind DR PPO policy (the 'expert')
  ppo_noise  PPO + per-episode Gaussian action noise (SimDist-style perturbation)
  rand       smooth random joint targets around the default pose
  mpc        true-model reference-free MPPI: executed transitions ('mpc_exec')
             and K re-simulated candidate rollouts per control step ('mpc_cand')
Every record stores the applied joint target u (12) and the state features below.

World model: MLP f(x, u) -> (x' - x, heads). x (34) = [z, gravity in body (3), q - q0 (12),
body lin vel (3), body ang vel (3), qd (12)]; heads (25) = contact (4, logits), normal force / f0 (4),
foot vel in body frame (12), d(xy) in yaw frame (2), d yaw (1), foot height (2 of 4 unused -> keep 4) ...
see HEAD_* below. The planner cost is computed from predicted features with the E0 weights.

Modes:
  gen      --src ppo|ppo_noise|rand --n_ep E --T T --out file.npz
  genmpc   --n_ep E --T T --cand K --out file.npz    (true-model MPPI, 256 x 6 by default)
  train    --data a.npz [b.npz ...] --max_trans N --out wm.pkl
  evalcl   --wm wm.pkl --cond c --out prefix          (WM-MPPI closed loop on the true env)
  rank     --wm wm.pkl --anchors file.npz --out f.json (candidate ranking vs true sim)
"""
import argparse, functools, json, os, pickle, time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

import toy_common as C
import toy_mppi as TM          # spline machinery and weights (module constants only)

DT = 0.02
XD = 34
HEAD = dict(contact=(0, 4), fn=(4, 8), fvel=(8, 20), dxy=(20, 22), dyaw=(22, 23), fh=(23, 27))
HD = 27
Z_DES = TM.Z_DES


# --------------------------------------------------------------------------- features
def features(F, d, d_prev_pose=None):
    R = d.xmat[F.torso]
    g = R.T @ jnp.array([0.0, 0.0, -1.0])
    vb = R.T @ d.qvel[0:3]
    wb = d.qvel[3:6]
    x = jnp.concatenate([d.qpos[2:3], g, d.qpos[7:] - F.default_q, vb, wb, d.qvel[6:]])
    fvb = (F.feet_vel(d) @ R)                       # rows: world vel -> body frame
    heads = jnp.concatenate([F.contact(d).astype(jnp.float32), F.feet_fn(d) / (F.mass * 9.81 / 4.0),
                             fvb.reshape(-1), jnp.zeros(3), F.feet_pos(d)[:, 2]])
    pose = jnp.concatenate([d.qpos[0:2], F.yaw(d)[None]])
    return x, heads, pose


def pose_delta_heads(pose0, pose1):
    """d(xy) in the yaw frame of pose0 and d yaw (wrapped)."""
    c, s = jnp.cos(pose0[..., 2]), jnp.sin(pose0[..., 2])
    dx, dy = pose1[..., 0] - pose0[..., 0], pose1[..., 1] - pose0[..., 1]
    dyaw = jnp.arctan2(jnp.sin(pose1[..., 2] - pose0[..., 2]), jnp.cos(pose1[..., 2] - pose0[..., 2]))
    return jnp.stack([c * dx + s * dy, -s * dx + c * dy, dyaw], -1)


# --------------------------------------------------------------------------- env helpers
def dr_models(env, n, seed):
    rnd = registry.get_domain_randomizer(C.ENV)
    mv, in_axes = rnd(env.mjx_model, jax.random.split(jax.random.PRNGKey(seed), n))
    return mv, in_axes


def with_model(env, mx, fn, *a):
    old = env._mjx_model
    env._mjx_model = mx
    try:
        return fn(*a)
    finally:
        env._mjx_model = old


# --------------------------------------------------------------------------- data generation
def box_limits(F, a, lo, hi):
    """Design 40 v2: optional joint box around the default pose for spline nodes (off when box_hip <= 0)."""
    if a.box_hip <= 0:
        return lo, hi
    box = jnp.tile(jnp.array([a.box_hip, a.box_thigh, a.box_knee]), 4)
    return jnp.maximum(lo, F.default_q - box), jnp.minimum(hi, F.default_q + box)


def gen_policy_data(a):
    from toy_ppo_eval import load_policy
    env = C.load_env()
    F = C.Feat(env)
    pi = load_policy(env, a.params) if a.src != "rand" else None
    mv, in_axes = dr_models(env, a.n_ep, a.seed)
    act_scale = env._config.action_scale

    def episode(mx, seed):
        def inner():
            key = jax.random.PRNGKey(seed)
            key, k1, k2, k3 = jax.random.split(key, 4)
            vcmd = jax.random.uniform(k1, minval=0.2, maxval=0.9)
            sig = jax.random.uniform(k2, minval=0.0, maxval=a.noise)          # per-episode noise level
            st = env.reset(k3)

            def setc(st, t):
                info = dict(st.info)
                info["command"] = jnp.array([vcmd * jnp.clip(t / C.RAMP_S, 0, 1), 0.0, 0.0])
                info["steps_until_next_cmd"] = jnp.array(10 ** 6, jnp.int32)
                return st.replace(info=info)

            st = setc(st, 0.0)
            ou = jnp.zeros(12)

            def body(carry, step):
                st, key, ou = carry
                key, k, kn = jax.random.split(key, 3)
                if a.src == "rand":
                    ou = 0.9 * ou + 0.3 * jax.random.normal(kn, (12,))     # OU, stationary SD ~0.69
                    act = jnp.clip(ou, -1.5, 1.5)
                else:
                    act, _ = pi(st.obs, k)
                    if a.src == "ppo_noise":
                        act = act + sig * jax.random.normal(kn, (12,))
                x, h, p = features(F, st.data)
                st = env.step(st, act)
                st = setc(st, (step + 1) * env.dt)
                x1, h1, p1 = features(F, st.data)
                h1 = h1.at[20:23].set(pose_delta_heads(p, p1))
                u = F.default_q + act * act_scale
                return (st, key, ou), dict(x=x, u=u, x1=x1, h1=h1, fell=F.fallen(st.data))

            _, rec = jax.lax.scan(body, (st, key, ou), jnp.arange(a.T))
            return rec
        return with_model(env, mx, inner)

    t = time.time()
    rec = jax.jit(jax.vmap(episode, in_axes=(in_axes, 0)))(mv, jnp.arange(a.n_ep) + a.seed * 100000)
    rec = jax.tree_util.tree_map(np.asarray, rec)
    np.savez_compressed(a.out, src=a.src, **rec)
    print(f"gen {a.src}: {a.n_ep}x{a.T} fell-frac {rec['fell'][:, -1].mean():.2f} {time.time() - t:.0f}s", flush=True)


def gen_mpc_data(a):
    """True-model MPPI under DR; records executed transitions and K re-simulated candidates per step.
    Design 41: --drive ppo|ppo_noise executes the PPO action (MPPI only shadows and records its candidates)."""
    env = C.load_env()
    F = C.Feat(env)
    drive = a.drive
    if drive != "mpc":
        assert a.cond == "dr", "policy-driven shadow data only under DR"
        from toy_ppo_eval import load_policy
        pi = load_policy(env, a.params)
        act_scale = env._config.action_scale
    tow = C.condition_extra(a.cond)[1] if a.cond != "dr" else 0.0
    if a.cond == "dr":
        mv, in_axes = dr_models(env, a.n_ep, a.seed)
    else:                                   # fixed true condition, no DR (test anchors)
        mv = C.condition_model(env, a.cond)
        in_axes = jax.tree_util.tree_map(lambda _: None, mv)
    n_sub = TM.NS
    J, K = TM.J, TM.K
    I, N = a.iters, a.samples
    ii = jnp.arange(I, 0, -1)
    var = jnp.exp(-(I - ii)[:, None] / I - (K - jnp.arange(K))[None, :] / K)
    std = jnp.sqrt(var)
    lo, hi = F.q_lo, F.q_hi
    blo, bhi = box_limits(F, a, lo, hi)
    kd_kp = F.kd / F.kp
    f0 = F.mass * 9.81 / 4.0

    def clamp(thq, thv):
        thq = jnp.clip(thq, blo, bhi)
        lim = jnp.minimum(bhi - thq, thq - blo) / (TM.DTN / 2)
        return thq, jnp.clip(thv, -lim, lim)

    def traj(thq, thv):
        q = TM.A_q @ thq + TM.A_v @ thv
        qd = TM.D_q @ thq + TM.D_v @ thv
        return (jnp.clip(q, lo, hi) + kd_kp * qd).reshape(J, n_sub, 12)

    def setc(st, t, vcmd):
        info = dict(st.info)
        info["command"] = jnp.array([vcmd * jnp.clip(t / C.RAMP_S, 0, 1), 0.0, 0.0])
        info["steps_until_next_cmd"] = jnp.array(10 ** 6, jnp.int32)
        return st.replace(info=info)

    def episode(mx, seed):
        return with_model(env, mx, episode_inner, mx, seed)

    def episode_inner(mx, seed):
        key = jax.random.PRNGKey(seed)
        key, k1, k3 = jax.random.split(key, 3)
        vcmd = jax.random.uniform(k1, minval=0.2, maxval=0.9)
        sig = jax.random.uniform(jax.random.fold_in(key, 7), minval=0.0, maxval=a.noise)
        st0 = setc(env.reset(k3), 0.0, vcmd)
        d = st0.data
        yaw0 = F.yaw(d)

        def substeps(d, u):
            if tow > 0:                     # true model includes the tow force
                d = d.replace(xfrc_applied=C.tow_xfrc(F, d, tow))
            for k in range(n_sub):
                d = mjx_env.step(mx, d, u[k], 1)
            return d

        def step_cost(d):
            con = F.contact(d)
            c = TM.W["h"] * jnp.abs(d.qpos[2] - Z_DES)
            c += TM.W["orient"] * TM.quat_angle2(d.qpos[3:7], yaw0)
            c += a.wq * jnp.sum((d.qpos[7:] - F.default_q) ** 2)
            c += TM.W["cvel"] * jnp.sum(jnp.abs(F.feet_vel(d)) * con[:, None])
            c += TM.W["cforce"] * jnp.sum(jnp.abs(F.feet_fn(d) - f0))
            return c

        def rollout(d0, ctrl, p_target):
            def body(carry, u):
                d, acc = carry
                d = substeps(d, u)
                return (d, acc + a.run_scale * step_cost(d)), None
            (dT, acc), _ = jax.lax.scan(body, (d0, 0.0), ctrl)
            return acc + TM.W["H"] * jnp.sum(jnp.abs(dT.qpos[:2] - p_target))

        def rollout_rec(d0, ctrl):
            def body(d, u):
                x, h, p = features(F, d)
                d = substeps(d, u)
                x1, h1, p1 = features(F, d)
                h1 = h1.at[20:23].set(pose_delta_heads(p, p1))
                return d, dict(x=x, u=u[0], x1=x1, h1=h1, fell=F.fallen(d))
            _, r = jax.lax.scan(body, d0, ctrl)
            return r

        def plan_step(carry, step):
            st, thq, thv, key = carry
            d = st.data
            key, kp, kc = jax.random.split(key, 3)
            t0 = step * DT
            vd = vcmd * jnp.clip((t0 + TM.CTRL_T) / C.RAMP_S, 0, 1)
            p_target = d.qpos[:2] + jnp.sum(vd) * DT * jnp.array([jnp.cos(yaw0), jnp.sin(yaw0)])

            def it(c, xs):
                thq, thv, bq, bv, bs, _, _ = c
                sd, k = xs
                k1, k2 = jax.random.split(k)
                eq = jax.random.normal(k1, (N, K, 12)) * (a.sigma_q * sd)[None, :, None]
                ev = jax.random.normal(k2, (N, K, 12)) * (a.sigma_v * sd)[None, :, None]
                eq, ev = eq.at[0].set(0.0), ev.at[0].set(0.0)
                Q, V = jax.vmap(clamp)(thq[None] + eq, thv[None] + ev)
                ctrl = jax.vmap(traj)(Q, V)
                S = jax.vmap(rollout, in_axes=(None, 0, None))(d, ctrl, p_target)
                S = jnp.where(jnp.isfinite(S), S, 1e9)
                w = jax.nn.softmax(-((S - S.min()) / (S.max() - S.min() + 1e-9)) / a.lam)
                b = jnp.argmin(S)
                better = S[b] < bs
                return (jnp.einsum("n,nkj->kj", w, Q), jnp.einsum("n,nkj->kj", w, V),
                        jnp.where(better, Q[b], bq), jnp.where(better, V[b], bv), jnp.minimum(S[b], bs),
                        Q, V), None

            init = (thq, thv, thq, thv, jnp.inf, jnp.zeros((N, K, 12)), jnp.zeros((N, K, 12)))
            (thq_n, thv_n, bq, bv, bs, Ql, Vl), _ = jax.lax.scan(it, init, (std, jax.random.split(kp, I)))
            # K candidates of the last iteration (uniform without the nominal), re-simulated with records
            idx = jax.random.choice(kc, jnp.arange(1, N), (a.cand,), replace=False)
            cand = jax.vmap(lambda q, v: rollout_rec(d, traj(q, v)))(Ql[idx], Vl[idx])
            x, h, p = features(F, d)
            if drive == "mpc":
                u = traj(bq, bv)[0]
                d1 = substeps(d, u)
                st1, u0 = st.replace(data=d1), u[0]
            else:                           # PPO (+ per-episode noise) acts; the planner only shadows
                act, _ = pi(st.obs, jax.random.fold_in(kc, 1))
                if drive == "ppo_noise":
                    act = act + sig * jax.random.normal(jax.random.fold_in(kc, 2), (12,))
                st1 = setc(env.step(st, act), (step + 1) * DT, vcmd)
                d1, u0 = st1.data, F.default_q + act * act_scale
            x1, h1, p1 = features(F, d1)
            h1 = h1.at[20:23].set(pose_delta_heads(p, p1))
            exe = dict(x=x, u=u0, x1=x1, h1=h1, fell=F.fallen(d1))
            thq_w = TM.S_q @ bq + TM.S_v @ bv
            thv_w = TM.SD_q @ bq + TM.SD_v @ bv
            return (st1, thq_w, thv_w, key), (exe, cand)

        thq = jnp.tile(d.qpos[7:][None], (K, 1))
        _, (exe, cand) = jax.lax.scan(plan_step, (st0, thq, jnp.zeros((K, 12)), key), jnp.arange(a.T))
        return exe, cand

    t = time.time()
    exe, cand = jax.jit(jax.vmap(episode, in_axes=(in_axes, 0)))(mv, jnp.arange(a.n_ep) + a.seed * 100000)
    exe = jax.tree_util.tree_map(np.asarray, exe)
    cand = jax.tree_util.tree_map(lambda z: np.asarray(z).astype(np.float16) if z.dtype != bool else np.asarray(z), cand)
    np.savez_compressed(a.out, src=drive, **{f"exe_{k}": v for k, v in exe.items()},
                        **{f"cand_{k}": v for k, v in cand.items()})
    print(f"genmpc: {a.n_ep}x{a.T} exec fell-frac {exe['fell'][:, -1].mean():.2f} "
          f"vx {exe['x'][:, 100:, 16].mean():.2f} {time.time() - t:.0f}s", flush=True)


# --------------------------------------------------------------------------- world model
class WM(nn.Module):
    hidden: int = 512

    @nn.compact
    def __call__(self, xn, un):
        h = jnp.concatenate([xn, un], -1)
        for _ in range(3):
            h = nn.silu(nn.Dense(self.hidden)(h))
        return nn.Dense(XD + HD)(h)


def load_windows(paths, L, max_trans, which, seed=0, trunc_fall=False):
    """Returns windows dict x (B, L+1, XD), u (B, L, 12), h (B, L, HD) from episode / candidate arrays.
    trunc_fall (design 41): keep only windows that end at or before the first fallen transition of each episode."""
    rng = np.random.default_rng(seed)
    X, U, H = [], [], []
    for p in paths:
        z = np.load(p)
        keys = []
        if "x" in z:
            keys.append("")
        else:
            if "exec" in which:
                keys.append("exe_")
            if "cand" in which:
                keys.append("cand_")
        for pre in keys:
            x, u, x1, h1 = (z[pre + k].astype(np.float32) for k in ("x", "u", "x1", "h1"))
            x = x.reshape(-1, *x.shape[-2:])          # (E', T, XD) ; candidates: (E*T*K, J, XD)
            u, x1, h1 = (v.reshape(-1, *v.shape[-2:]) for v in (u, x1, h1))
            T = x.shape[1]
            keep_upto = np.full(len(x), T)            # windows must end at or before this many steps
            if trunc_fall and (pre + "fell") in z:
                fl = z[pre + "fell"].astype(bool).reshape(len(x), T)
                keep_upto = np.where(fl.any(1), fl.argmax(1) + 1, T)
            starts = np.arange(0, T - L + 1, L)       # non-overlapping windows
            for s0 in starts:
                m = s0 + L <= keep_upto
                X.append(np.concatenate([x[m, s0:s0 + 1], x1[m, s0:s0 + L]], 1))
                U.append(u[m, s0:s0 + L])
                H.append(h1[m, s0:s0 + L])
    X, U, H = np.concatenate(X), np.concatenate(U), np.concatenate(H)
    ok = np.isfinite(X).all((1, 2)) & np.isfinite(H).all((1, 2))
    X, U, H = X[ok], U[ok], H[ok]
    nw = min(len(X), max_trans // L)
    sel = rng.choice(len(X), nw, replace=False)
    return X[sel], U[sel], H[sel]


def train(a):
    X, U, H = load_windows(a.data, a.L, a.max_trans, a.which, a.seed, trunc_fall=a.trunc_fall)
    print(f"train windows {len(X)} x {a.L} = {len(X) * a.L} transitions", flush=True)
    if a.replay:                            # fine-tuning: mix new windows with replayed pretraining windows
        n_rep = int(len(X) * a.replay_frac / (1 - a.replay_frac))
        Xr, Ur, Hr = load_windows(a.replay, a.L, n_rep * a.L, a.replay_which, a.seed + 7)
        X, U, H = np.concatenate([X, Xr]), np.concatenate([U, Ur]), np.concatenate([H, Hr])
        print(f"  + replay windows {len(Xr)}", flush=True)
    xm, xs = X.reshape(-1, XD).mean(0), X.reshape(-1, XD).std(0) + 1e-3
    um, us = U.reshape(-1, 12).mean(0), U.reshape(-1, 12).std(0) + 1e-3
    D = np.diff(X, axis=1).reshape(-1, XD)
    dm, ds = D.mean(0), D.std(0) + 1e-4
    hm, hs = H.reshape(-1, HD).mean(0), H.reshape(-1, HD).std(0) + 1e-3
    hm[0:4], hs[0:4] = 0.0, 1.0                       # contact logits unnormalised
    norm = {k: jnp.asarray(v) for k, v in dict(xm=xm, xs=xs, um=um, us=us, dm=dm, ds=ds, hm=hm, hs=hs).items()}
    net = WM()
    params = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, XD)), jnp.zeros((1, 12)))
    if a.init:                              # keep the pretrained normalisation and weights
        with open(a.init, "rb") as f:
            w0 = pickle.load(f)
        params = w0["params"]
        norm = {k: jnp.asarray(v) for k, v in w0["norm"].items()}
    opt = optax.adamw(a.lr, weight_decay=1e-5)
    ost = opt.init(params)
    Xj, Uj, Hj = jnp.asarray(X), jnp.asarray(U), jnp.asarray(H)

    def loss_fn(params, xw, uw, hw):
        def body(x, k):
            o = net.apply(params, (x - norm["xm"]) / norm["xs"], (uw[:, k] - norm["um"]) / norm["us"])
            dx = o[:, :XD] * norm["ds"] + norm["dm"]
            x1 = x + dx
            lx = jnp.mean(((x1 - xw[:, k + 1]) / norm["xs"]) ** 2)
            ho = o[:, XD:]
            lc = jnp.mean(optax.sigmoid_binary_cross_entropy(ho[:, 0:4], hw[:, k, 0:4]))
            lh = jnp.mean(((ho[:, 4:] - (hw[:, k, 4:] - norm["hm"][4:]) / norm["hs"][4:])) ** 2)
            return x1, lx + lc + lh
        _, ls = jax.lax.scan(body, xw[:, 0], jnp.arange(a.L))
        return ls.mean()

    @jax.jit
    def upd(params, ost, key):
        i = jax.random.randint(key, (a.batch,), 0, Xj.shape[0])
        l, g = jax.value_and_grad(loss_fn)(params, Xj[i], Uj[i], Hj[i])
        up, ost = opt.update(g, ost, params)
        return optax.apply_updates(params, up), ost, l

    key = jax.random.PRNGKey(a.seed + 1)
    t = time.time()
    for s in range(a.steps):
        key, k = jax.random.split(key)
        params, ost, l = upd(params, ost, k)
        if s % 2000 == 0 or s == a.steps - 1:
            print(f"step {s} loss {float(l):.4f} {time.time() - t:.0f}s", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params), "norm": jax.device_get(norm),
                     "data": a.data, "which": a.which, "n_trans": int(len(X) * a.L)}, f)


def wm_step_fn(wm):
    net = WM()
    params, norm = wm["params"], {k: jnp.asarray(v) for k, v in wm["norm"].items()}

    def step(x, u):
        o = net.apply(params, ((x - norm["xm"]) / norm["xs"])[None], ((u - norm["um"]) / norm["us"])[None])[0]
        x1 = x + o[:XD] * norm["ds"] + norm["dm"]
        h = o[XD:]
        h = jnp.concatenate([jax.nn.sigmoid(h[0:4]), h[4:] * norm["hs"][4:] + norm["hm"][4:]])
        return x1, h
    return step


# --------------------------------------------------------------------------- WM-MPPI
def eval_closed_loop(a):
    env = C.load_env()
    F = C.Feat(env)
    mx = C.condition_model(env, a.cond)
    delay, tow = C.condition_extra(a.cond)
    f0 = F.mass * 9.81 / 4.0
    if not a.oracle:
        with open(a.wm, "rb") as f:
            wm = pickle.load(f)
        step = wm_step_fn(wm)
    J, K, n_sub = TM.J, TM.K, TM.NS
    I, N = a.iters, a.samples
    ii = jnp.arange(I, 0, -1)
    std = jnp.sqrt(jnp.exp(-(I - ii)[:, None] / I - (K - jnp.arange(K))[None, :] / K))
    lo, hi = F.q_lo, F.q_hi
    blo, bhi = box_limits(F, a, lo, hi)
    kd_kp = F.kd / F.kp

    def clamp(thq, thv):
        thq = jnp.clip(thq, blo, bhi)
        lim = jnp.minimum(bhi - thq, thq - blo) / (TM.DTN / 2)
        return thq, jnp.clip(thv, -lim, lim)

    def traj(thq, thv):
        q = TM.A_q @ thq + TM.A_v @ thv
        qd = TM.D_q @ thq + TM.D_v @ thv
        return (jnp.clip(q, lo, hi) + kd_kp * qd).reshape(J, n_sub, 12)

    def rollout(x0, pose0, ctrl, yaw0, p_target):
        def body(c, u):
            x, pose, acc = c
            x1, h = step(x, u[0])
            cyaw, syaw = jnp.cos(pose[2]), jnp.sin(pose[2])
            dxy = h[20:22]
            pose1 = jnp.array([pose[0] + cyaw * dxy[0] - syaw * dxy[1],
                               pose[1] + syaw * dxy[0] + cyaw * dxy[1], pose[2] + h[22]])
            con = h[0:4]
            yerr = jnp.arctan2(jnp.sin(pose1[2] - yaw0), jnp.cos(pose1[2] - yaw0))
            tilt2 = jnp.arccos(jnp.clip(-x1[3], -1.0, 1.0)) ** 2
            cst = TM.W["h"] * jnp.abs(x1[0] - Z_DES) + TM.W["orient"] * (tilt2 + yerr ** 2)
            cst += a.wq * jnp.sum(x1[4:16] ** 2)
            cst += TM.W["cvel"] * jnp.sum(jnp.abs(h[8:20].reshape(4, 3)) * con[:, None])
            cst += TM.W["cforce"] * f0 * jnp.sum(jnp.abs(h[4:8] - 1.0))
            return (x1, pose1, acc + a.run_scale * cst), None
        (xT, pT, acc), _ = jax.lax.scan(body, (x0, pose0, 0.0), ctrl)
        return acc + TM.W["H"] * jnp.sum(jnp.abs(pT[:2] - p_target))

    def true_rollout(d0, ctrl, yaw0, p_target):          # design 41 oracle: true condition model + physical cost
        def body(carry, u):
            d, acc = carry
            if tow > 0:
                d = d.replace(xfrc_applied=C.tow_xfrc(F, d, tow))
            for k in range(n_sub):
                d = mjx_env.step(mx, d, u[k], 1)
            con = F.contact(d)
            c = TM.W["h"] * jnp.abs(d.qpos[2] - Z_DES)
            c += TM.W["orient"] * TM.quat_angle2(d.qpos[3:7], yaw0)
            c += a.wq * jnp.sum((d.qpos[7:] - F.default_q) ** 2)
            c += TM.W["cvel"] * jnp.sum(jnp.abs(F.feet_vel(d)) * con[:, None])
            c += TM.W["cforce"] * jnp.sum(jnp.abs(F.feet_fn(d) - f0))
            return (d, acc + a.run_scale * c), None
        (dT, acc), _ = jax.lax.scan(body, (d0, 0.0), ctrl)
        return acc + TM.W["H"] * jnp.sum(jnp.abs(dT.qpos[:2] - p_target))

    def plan(d, thq, thv, key, t0, yaw0):
        x0, _, pose0 = features(F, d)
        vd = C.v_des(t0 + TM.CTRL_T)
        p_target = pose0[:2] + jnp.sum(vd) * DT * jnp.array([jnp.cos(yaw0), jnp.sin(yaw0)])
        if a.oracle:
            score = lambda q, v: true_rollout(d, traj(q, v), yaw0, p_target)
        else:
            score = lambda q, v: rollout(x0, pose0, traj(q, v), yaw0, p_target)

        def it(c, xs):
            thq, thv, bq, bv, bs = c
            sd, k = xs
            k1, k2 = jax.random.split(k)
            eq = jax.random.normal(k1, (N, K, 12)) * (a.sigma_q * sd)[None, :, None]
            ev = jax.random.normal(k2, (N, K, 12)) * (a.sigma_v * sd)[None, :, None]
            eq, ev = eq.at[0].set(0.0), ev.at[0].set(0.0)
            Q, V = jax.vmap(clamp)(thq[None] + eq, thv[None] + ev)
            S = jax.vmap(score)(Q, V)
            S = jnp.where(jnp.isfinite(S), S, 1e9)
            w = jax.nn.softmax(-((S - S.min()) / (S.max() - S.min() + 1e-9)) / a.lam)
            b = jnp.argmin(S)
            better = S[b] < bs
            return (jnp.einsum("n,nkj->kj", w, Q), jnp.einsum("n,nkj->kj", w, V),
                    jnp.where(better, Q[b], bq), jnp.where(better, V[b], bv), jnp.minimum(S[b], bs)), None

        if a.record_cand > 0:                             # design 43: also return the last-iteration samples
            def it2(c, xs):
                (thq, thv, bq, bv, bs), _ = it(c[:5], xs)
                sd, k = xs
                k1, k2 = jax.random.split(k)
                eq = jax.random.normal(k1, (N, K, 12)) * (a.sigma_q * sd)[None, :, None]
                ev = jax.random.normal(k2, (N, K, 12)) * (a.sigma_v * sd)[None, :, None]
                eq, ev = eq.at[0].set(0.0), ev.at[0].set(0.0)
                Q, V = jax.vmap(clamp)(c[0][None] + eq, c[1][None] + ev)   # same samples as it() drew
                return (thq, thv, bq, bv, bs, Q, V), None
            init = (thq, thv, thq, thv, jnp.inf, jnp.zeros((N, K, 12)), jnp.zeros((N, K, 12)))
            (_, _, bq, bv, bs, Ql, Vl), _ = jax.lax.scan(it2, init, (std, jax.random.split(key, I)))
            return bq, bv, bs, Ql, Vl
        (_, _, bq, bv, bs), _ = jax.lax.scan(it, (thq, thv, thq, thv, jnp.inf), (std, jax.random.split(key, I)))
        return bq, bv, bs

    def substeps(d, u):
        for k in range(n_sub):
            d = mjx_env.step(mx, d, u[k], 1)
        return d

    def episode(seed):
        d = env.reset(jax.random.PRNGKey(seed)).data
        yaw0 = F.yaw(d)

        def true_rec(d0, ctrl):                           # design 43: true condition model, recorded like genmpc candidates
            def body(d, u):
                x, h, p = features(F, d)
                if tow > 0:
                    d = d.replace(xfrc_applied=C.tow_xfrc(F, d, tow))
                d = substeps(d, u)
                x1, h1, p1 = features(F, d)
                h1 = h1.at[20:23].set(pose_delta_heads(p, p1))
                return d, dict(x=x, u=u[0], x1=x1, h1=h1)
            return jax.lax.scan(body, d0, ctrl)[1]

        def es(c, stp):
            d, thq, thv, key, buf = c
            key, kp = jax.random.split(key)
            if a.record_cand > 0:
                bq, bv, bs, Ql, Vl = plan(d, thq, thv, kp, stp * DT, yaw0)
                idx = jax.random.choice(jax.random.fold_in(kp, 77), jnp.arange(1, N), (a.record_cand,), replace=False)
                cand = jax.vmap(lambda q, v: true_rec(d, traj(q, v)))(Ql[idx], Vl[idx])
            else:
                bq, bv, bs = plan(d, thq, thv, kp, stp * DT, yaw0)
            u = traj(bq, bv)[0]
            buf = jnp.concatenate([buf[1:], u[None]], 0)
            ue = buf[0]                                   # executed target (delayed by `delay` steps)
            x, h, p = features(F, d)
            if tow > 0:
                d = d.replace(xfrc_applied=C.tow_xfrc(F, d, tow))
            d = substeps(d, ue)
            x1, h1, p1 = features(F, d)
            h1 = h1.at[20:23].set(pose_delta_heads(p, p1))
            rec = C.step_log(F, d)
            rec["ctrl"] = ue[0]
            rec.update(tr_x=x, tr_u=ue[0], tr_x1=x1, tr_h1=h1)
            if a.record_cand > 0:
                rec.update({"cd_" + k: v for k, v in cand.items()})
            return (d, TM.S_q @ bq + TM.S_v @ bv, TM.SD_q @ bq + TM.SD_v @ bv, key, buf), rec

        thq = jnp.tile(d.qpos[7:][None], (K, 1))
        u0 = jnp.tile(d.qpos[7:][None, None], (delay + 1, n_sub, 1))
        _, recs = jax.lax.scan(es, (d, thq, jnp.zeros((K, 12)), jax.random.PRNGKey(1000 + seed), u0), jnp.arange(a.T))
        return recs

    t = time.time()
    recs = jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(episode))(jnp.arange(a.seed0, a.seed0 + a.episodes)))
    summ = C.summarise(recs, recs["ctrl"])
    tr = {k[3:]: recs.pop(k) for k in ("tr_x", "tr_u", "tr_x1", "tr_h1")}
    tr["fell"] = np.asarray(recs["fallen"]).astype(bool)
    if a.record_cand > 0:                                 # keep candidates from states up to the first fall
        cd = {k[3:]: recs.pop(k) for k in [k for k in recs if k.startswith("cd_")]}
        fl = tr["fell"]
        ff = np.where(fl.any(1), fl.argmax(1), fl.shape[1] - 1)
        keep = np.arange(fl.shape[1])[None, :] <= ff[:, None]
        np.savez_compressed(a.out + "_cand.npz", src="wm-mppi-true-cand",
                            **{"cand_" + k: v[keep].astype(np.float16) for k, v in cd.items()})
        print(f"recorded true candidates at {int(keep.sum())} pre-fall states", flush=True)
    np.savez_compressed(a.out + ".npz", **recs)
    np.savez_compressed(a.out + "_trans.npz", src="wm-mppi", **tr)     # adaptation data (same layout as gen)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summ.items()}
    js.update(cond=a.cond, wm=a.wm, wall_s=time.time() - t, args=vars(a))
    json.dump(js, open(a.out + ".json", "w"), indent=1)
    print(f"wm-mppi {'ORACLE' if a.oracle else os.path.basename(a.wm)} {a.cond}: J={np.mean(summ['J']):.3f} fell={np.mean(summ['fell']):.2f} "
          f"vx={np.nanmean(summ['vx_mean']):.3f} wall={time.time() - t:.0f}s", flush=True)


def feat_cost(x, h, pose_xy_end, target_x, wq, f0, run_scale):
    """Feature-based E0 cost of one 45-step sequence (x: (J, XD) after each step, h: (J, HD))."""
    tilt2 = jnp.arccos(jnp.clip(-x[:, 3], -1.0, 1.0)) ** 2
    yaw = jnp.cumsum(h[:, 22])
    c = TM.W["h"] * jnp.abs(x[:, 0] - Z_DES) + TM.W["orient"] * (tilt2 + yaw ** 2)
    c += wq * jnp.sum(x[:, 4:16] ** 2, -1)
    con = h[:, 0:4]
    c += TM.W["cvel"] * jnp.sum(jnp.abs(h[:, 8:20].reshape(-1, 4, 3)) * con[..., None], (1, 2))
    c += TM.W["cforce"] * f0 * jnp.sum(jnp.abs(h[:, 4:8] - 1.0), -1)
    return run_scale * c.sum() + TM.W["H"] * (jnp.abs(pose_xy_end[0] - target_x) + jnp.abs(pose_xy_end[1]))


def integrate_xy(h):
    yaw = jnp.concatenate([jnp.zeros(1), jnp.cumsum(h[:, 22])[:-1]])
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array([jnp.sum(c * h[:, 20] - s * h[:, 21]), jnp.sum(s * h[:, 20] + c * h[:, 21])])


def rank(a):
    """Candidate ranking vs true sim at anchor states (genmpc file with --cand K on a fixed condition)."""
    env = C.load_env()
    F = C.Feat(env)
    f0 = F.mass * 9.81 / 4.0
    z = np.load(a.anchors)
    cx, cu, cx1, ch1 = (z["cand_" + k].astype(np.float32) for k in ("x", "u", "x1", "h1"))
    E, T, K = cx.shape[:3]
    sl = slice(100, T, a.every)                       # anchors after the 2 s start
    cx, cu, cx1, ch1 = (v[:, sl].reshape(-1, K, *v.shape[3:]) for v in (cx, cu, cx1, ch1))
    with open(a.wm, "rb") as f:
        wm = pickle.load(f)
    step = wm_step_fn(wm)
    target = 0.6 * TM.H_S

    def true_cost(x1, h1):
        return feat_cost(x1, h1, integrate_xy(h1), target, a.wq, f0, a.run_scale)

    def pred_cost(x0, u):
        def body(x, uu):
            x1, h = step(x, uu)
            return x1, (x1, h)
        _, (xs, hs) = jax.lax.scan(body, x0, u)
        return feat_cost(xs, hs, integrate_xy(hs), target, a.wq, f0, a.run_scale), xs

    tc = jax.jit(jax.vmap(jax.vmap(true_cost)))(cx1, ch1)
    pc, xs = jax.jit(jax.vmap(jax.vmap(pred_cost)))(cx[:, :, 0], cu)
    tc, pc, xs = np.asarray(tc), np.asarray(pc), np.asarray(xs)

    def spearman(u, v):
        ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
        return np.corrcoef(ru, rv)[0, 1]

    rho = np.array([spearman(tc[i], pc[i]) for i in range(len(tc))])
    top_hit = np.mean(np.argmin(pc, 1) == np.argmin(tc, 1))
    regret = np.mean(tc[np.arange(len(tc)), np.argmin(pc, 1)] - tc.min(1))
    err = np.mean(((xs - cx1) / (cx1.reshape(-1, XD).std(0) + 1e-3)) ** 2)
    out = dict(wm=a.wm, anchors=a.anchors, n_anchor=int(len(tc)), K=int(K), rho_mean=float(np.nanmean(rho)),
               rho_median=float(np.nanmedian(rho)), top1_hit=float(top_hit), regret=float(regret),
               regret_rel=float(regret / np.mean(tc.max(1) - tc.min(1))), open_loop_mse=float(err))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "genmpc", "train", "evalcl", "rank"))
    ap.add_argument("--src", default="ppo", choices=("ppo", "ppo_noise", "rand"))
    ap.add_argument("--params", default="/mnt/newssd/cjkim/toy/runs/ppo_s0/params.pkl")
    ap.add_argument("--noise", type=float, default=0.0, help="max per-episode action-noise SD (action units)")
    ap.add_argument("--n_ep", type=int, default=64)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand", type=int, default=4)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--sigma_q", type=float, default=0.2)
    ap.add_argument("--sigma_v", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--wq", type=float, default=0.3)
    ap.add_argument("--run_scale", type=float, default=5.0)
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--which", default="exec,cand")
    ap.add_argument("--max_trans", type=int, default=400000)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wm", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--replay_frac", type=float, default=0.5)
    ap.add_argument("--replay_which", default="exec,cand")
    ap.add_argument("--anchors", default="")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--cond", default="nominal", help="evalcl: condition; genmpc: dr (default use) or a condition")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--drive", default="mpc", choices=("mpc", "ppo", "ppo_noise"), help="design 41 genmpc: who acts")
    ap.add_argument("--trunc_fall", action="store_true", help="design 41 train: drop windows after the first fall")
    ap.add_argument("--oracle", action="store_true", help="design 41 evalcl: plan with the true condition model")
    ap.add_argument("--record_cand", type=int, default=0, help="design 43 evalcl: re-simulate K last-iteration samples per step in the true model")
    ap.add_argument("--box_hip", type=float, default=0.0, help="v2 joint box (rad) around default pose; 0 = off")
    ap.add_argument("--box_thigh", type=float, default=0.0)
    ap.add_argument("--box_knee", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "genmpc" and a.cond == "nominal":
        a.cond = "dr"
    {"gen": gen_policy_data, "genmpc": gen_mpc_data, "train": train, "evalcl": eval_closed_loop, "rank": rank}[a.mode](a)


if __name__ == "__main__":
    main()
