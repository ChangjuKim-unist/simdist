"""SimDist planner with every candidate rolled out in the simulator (exact dynamics), and its consensus with the sim model.

Registered as design 14 of the TD-MPC2 research line (tdmpc2 repo, research/docs/design14_go1_exact_dynamics_ko.md).
Question: on Go1 with the SimDist world model (prediction length 25), is a planner with exact dynamics and the frozen
reward / value heads better or worse than the un-adapted planner, and what does the consensus of the two do?

oracle.arm
  model      the usual planner (BatchedMppiController) with model.checkpoint; no candidate rollouts. With oracle.restore=true
             every robot is restored to its own snapshot before the executed step (measures the artefact of restoring).
  exact      the same MPPI loop (same sampling, elites, temperature, warm start), but every candidate is executed open loop
             for the prediction length in the simulator from a copy of the robot's state; the frozen reward head and value
             head score the encoded TRUE observations (reward of the step from the latent before it, value at the last latent).
             After a termination inside a rollout the remaining rewards and the value count as zero.
  consensus  score = min(relative model score, relative exact score); relative = minus the mean over the robot's candidates.

Layout: R robots (oracle.speeds x oracle.seeds), each with C = num_samples + policy-mixture samples simulator copies
(env index r * C + c; copy 0 is the robot itself). One control step = snapshot the R robots, per MPPI iteration restore
all copies and step the prediction length, then restore once more and execute the chosen action in every copy.

    python scripts/sim2sim/oracle_plan.py oracle.arm=exact model.checkpoint=go1_paper sim.friction=0.15 sim.terrain=plane \
        sim.total_steps=500 sim.episode_length_s=100 "oracle.speeds=[0.6]" "oracle.seeds=[11]" oracle.out=/x/exact --headless
    Smoke: control.num_samples=40 control.iterations=2 sim.total_steps=60.
Output: <oracle.out>.npz (per-step records of the R robots) and <oracle.out>.json (summary).
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".30")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import simulate_batch as sb  # noqa: E402  (launches the Isaac Sim app, strips --headless)

import functools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import hydra  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flax import nnx  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from simdist.control.mppi_batch import BatchedMppiController, BatchedMppiState  # noqa: E402
from simdist.utils import model as model_utils, paths  # noqa: E402


def spearman(a, b):
    """Row-wise Spearman correlation of two (R, C) arrays."""
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64); rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)


class Scorer:
    """Jitted pieces of BatchedMppiController._mppi_step, split so that the candidates can be executed in the simulator
    between sampling and the elite update."""

    def __init__(self, pl: BatchedMppiController):
        self.pl, self.model, self.cfg = pl, pl.model, pl.ctrl_cfg
        self.R, self.S, self.B, self.T, self.A = pl.N, pl.S, pl.B, pl.T, pl.act_dim
        self.C = self.S + self.B
        assert pl.init_mode == "shift", "oracle_plan.py reproduces control.init_mode=shift only"
        assert not pl.noise_schedule.get("enabled", False) and pl.noise_corr == 0.0 and pl.calib is None and pl.gen is None
        assert float(self.cfg["momentum"]) == 0.0

    @functools.partial(nnx.jit, static_argnames=["self"])
    def start(self, mean, key_data, base_actions):
        """Warm start and the policy-mixture samples of this control step (as at the top of _mppi_step)."""
        keys = jax.random.wrap_key_data(key_data)
        k = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        keys, k_base = k[:, 0], k[:, 1]
        base_noise = jax.vmap(lambda k: jax.random.normal(k, (self.B, self.T, self.A)))(k_base) * self.cfg["base_act_std"]
        noised_base = base_actions[:, None] + base_noise
        mean = jnp.roll(mean, shift=self.pl.shift, axis=1)
        mean = mean.at[:, self.pl.shift:].set(0.0)
        std = jnp.full((self.R, self.T, self.A), self.cfg["init_std"])
        return mean, std, jax.random.key_data(keys), noised_base

    @functools.partial(nnx.jit, static_argnames=["self"])
    def sample(self, mean, std, key_data, noised_base):
        keys = jax.random.wrap_key_data(key_data)
        k = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        keys, k_act = k[:, 0], k[:, 1]
        noise = jax.vmap(lambda k: jax.random.normal(k, (self.S, self.T, self.A)))(k_act) * std[:, None]
        acts = mean[:, None] + noise
        if self.B > 0:
            acts = jnp.concatenate([acts, noised_base], axis=1)
        return acts, jax.random.key_data(keys)

    def _rep(self, x, acts):
        xx = jax.tree.map(lambda a: jnp.repeat(a, self.C, axis=0), x)
        xx = dict(xx); xx["fut_acts"] = acts.reshape(self.R * self.C, self.T, self.A)
        return xx

    @functools.partial(nnx.jit, static_argnames=["self"])
    def score_model(self, x, acts):
        y = self.model.inference(self._rep(x, acts))
        ret = jnp.sum(y["rewards"] * self.pl.discounts, axis=-1) + y["values"][:, -1] * self.pl.final_discount
        return ret.reshape(self.R, self.C)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def score_true(self, x, acts, P, E, alive):
        """P (R*C, T, proprio), E (R*C, T, extero): the observations after each executed step; alive (R*C, T): 1 while the
        copy has not terminated up to and including that step."""
        xs = self.model.scaler.scale(self._rep(x, acts))
        s = self.model.scaler.scale({"proprio_obs_hist": P, "extero_obs": E})
        z = jax.vmap(lambda p, e: self.model.encode_latent(p, e, deterministic=True), in_axes=1, out_axes=1)(s["proprio_obs_hist"], s["extero_obs"])
        r_s = self.model.reward_from_latents(xs, z, deterministic=True)
        v_s = self.model.value_from_latents(z, xs["fut_cmds"][:, 1:], deterministic=True)
        y = self.model.scaler.unscale({"rewards": r_s.reshape(-1, self.T), "values": v_s.reshape(-1, self.T)})
        alive_before = jnp.concatenate([jnp.ones_like(alive[:, :1]), alive[:, :-1]], axis=1)   # reward j is predicted from the latent before step j
        rew = jnp.sum(y["rewards"] * self.pl.discounts * alive_before, axis=-1)
        ret = rew + y["values"][:, -1] * self.pl.final_discount * alive[:, -1]
        return ret.reshape(self.R, self.C), rew.reshape(self.R, self.C)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def update(self, acts, returns):
        elite_returns, elite_idxs = jax.lax.top_k(returns, self.cfg["num_elites"])
        elite_acts = jnp.take_along_axis(acts, elite_idxs[:, :, None, None], axis=1)
        score = jnp.exp(self.cfg["temperature"] * (elite_returns - elite_returns.max(axis=1, keepdims=True)))
        w = score / score.sum(axis=1, keepdims=True)
        mean = jnp.sum(w[:, :, None, None] * elite_acts, axis=1)
        std = jnp.sqrt(jnp.sum(w[:, :, None, None] * (elite_acts - mean[:, None]) ** 2, axis=1))
        return mean, jnp.clip(std, min=self.cfg["min_std"])


class OracleSim(sb.BatchSim):
    def __init__(self, cfg):
        o = cfg["oracle"]; self.o = o
        self.arm = str(o["arm"]); assert self.arm in ("model", "exact", "consensus")
        self.rpairs = [(float(v), int(s)) for v in o["speeds"] for s in o["seeds"]]
        self.R = len(self.rpairs)
        S = int(cfg["control"]["num_samples"]); self.C = 1 if self.arm == "model" else S + int(S * cfg["control"]["mixture_coef"])
        self.N_env = self.R * self.C
        self.do_restore = self.arm != "model" or bool(o.get("restore", True))
        cfg["batch"]["speeds"] = [0.0]; cfg["batch"]["seeds"] = list(range(self.N_env))
        pcfg = dict(cfg["control"]); pcfg["prior_only"] = False
        cfg["control"] = dict(cfg["control"]); cfg["control"]["prior_only"] = True   # BatchSim's own controller over all copies is not used to plan
        assert not cfg["sim"].get("act_delay_steps"), "the action-delay line is not snapshotted here"
        super().__init__(cfg)
        assert self.N == self.N_env
        dev = self.env.device
        self.group = torch.arange(self.N_env, device=dev) // self.C
        self.L = np.arange(self.R) * self.C; self.Lt = torch.as_tensor(self.L, device=dev)
        self.fv[:] = torch.as_tensor(np.repeat([v for v, _ in self.rpairs], self.C).astype(np.float32), device=dev)
        ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), cfg["model"]["checkpoint"])
        _, model_cfg, _ = model_utils.load_model_from_ckpt(ckpt_dir, cfg["model"]["step"])
        self.pl = BatchedMppiController(self.controller.model, model_cfg, pcfg, self.R, [s for _, s in self.rpairs])
        self.I = int(pcfg["iterations"]); self.T = self.pl.T
        proprio, extero = self.get_obs()
        self.zero_R = np.zeros((self.R, 12), np.float32)
        self.pl.reset(range(self.R), proprio[self.L], extero[self.L], self.zero_R, np.zeros((self.R, 3)))
        self.sc = Scorer(self.pl) if self.arm != "model" else None
        if self.arm == "model":
            print(f"Compiling the planner ({self.R} robots)...", flush=True); self.pl.run_control(); print("ready.", flush=True)
            self.pl.reset(range(self.R), proprio[self.L], extero[self.L], self.zero_R, np.zeros((self.R, 3)))
        self.last_act = self.zero_R.copy()
        self.since = np.zeros(self.R, np.int64)
        self.step_i = 0
        robot = self.env.scene["robot"]
        self.x_prev = robot.data.root_pos_w[self.Lt, 0].cpu().numpy().copy()
        keys = ("reward", "fall", "reset", "resetting", "dx", "act", "torque_sq", "angvel", "j_model", "j_exact", "binds", "rho_ms", "rho_rew", "cand_falls")
        self.rec = {k: [] for k in keys}
        self.t0 = time.time(); self.t_roll = 0.0; self.n_roll = 0
        print(f"oracle_plan: arm {self.arm}, {self.R} robots x {self.C} copies = {self.N_env} simulator robots, restore {self.do_restore}", flush=True)

    # ---- per-robot forward speed kept in self.fv (as in branch_eval.BranchSim) ------------------------------------
    def _patch_per_env_speed(self):
        import types as pytypes
        from isaaclab.utils import math as math_utils

        term = self.cmd_manager.get_term("base_velocity")
        self.fv = torch.zeros(self.N, device=self.env.device); fv = self.fv

        def _compute_errors(self, env_ids):
            env_ids = torch.as_tensor(list(env_ids) if not torch.is_tensor(env_ids) else env_ids, device=fv.device)
            curr_pos = self.robot.data.root_pos_w[env_ids]; curr_heading = self.robot.data.heading_w[env_ids]
            curr_vx = self.robot.data.root_lin_vel_b[env_ids, 0]
            delta_pos = curr_pos - self.init_pos[env_ids]
            heading_error = math_utils.wrap_to_pi(curr_heading - self.init_heading[env_ids])
            cos_h = torch.cos(-self.init_heading[env_ids]); sin_h = torch.sin(-self.init_heading[env_ids])
            dx = cos_h * delta_pos[:, 0] - sin_h * delta_pos[:, 1]; dy = sin_h * delta_pos[:, 0] + cos_h * delta_pos[:, 1]
            return {"x_error": self.x_des[env_ids] - dx, "y_error": -dy, "yaw_error": -heading_error, "vx_error": fv[env_ids] - curr_vx}

        def _update_command(self):
            self.x_des += fv * self._env.step_dt
            errors = self._compute_errors(range(self.num_envs))
            self.vel_command_b[:, 0] = self.cfg.kx * errors["x_error"]; self.vel_command_b[:, 1] = self.cfg.ky * errors["y_error"]
            self.vel_command_b[:, 2] = self.cfg.kyaw * errors["yaw_error"]

        term._compute_errors = pytypes.MethodType(_compute_errors, term); term._update_command = pytypes.MethodType(_update_command, term)

    # ---- snapshot of the R robots / restore into all copies (fields as in branch_eval) ------------------------------
    def snapshot(self):
        L = self.Lt
        st = self.env.scene.get_state(is_relative=True)["articulation"]["robot"]
        term = self.cmd_manager.get_term("base_velocity"); am = self.env.action_manager; jt = am.get_term("joint_pos")
        cs = self.env.scene.sensors["contact_forces"]; _ = cs.data
        c = lambda t: t[L].clone()  # noqa: E731
        return dict(root_pose=c(st["root_pose"]), root_vel=c(st["root_velocity"]), jpos=c(st["joint_position"]), jvel=c(st["joint_velocity"]),
                    ep_len=c(self.env.episode_length_buf), init_pos=c(term.init_pos - self.env.scene.env_origins), init_heading=c(term.init_heading),
                    x_des=c(term.x_des), vel_cmd=c(term.vel_command_b), act=c(am._action), prev_act=c(am._prev_action),
                    raw_act=c(jt._raw_actions), proc_act=c(jt._processed_actions),
                    air_cur=c(cs._data.current_air_time), air_last=c(cs._data.last_air_time), con_cur=c(cs._data.current_contact_time),
                    con_last=c(cs._data.last_contact_time), forces_hist=c(cs._data.net_forces_w_history))

    def restore(self, s):
        g = self.group; T = lambda k: s[k][g]  # noqa: E731
        state = {"articulation": {"robot": {"root_pose": T("root_pose"), "root_velocity": T("root_vel"), "joint_position": T("jpos"), "joint_velocity": T("jvel")}},
                 "deformable_object": {}, "rigid_object": {}}
        self.env.reset_to(state, torch.arange(self.N_env, device=self.env.device), is_relative=True)
        robot = self.env.scene["robot"]   # reset_to sets velocity targets to the restored joint velocity (see branch_eval.restore)
        robot.set_joint_velocity_target(torch.zeros_like(state["articulation"]["robot"]["joint_velocity"]))
        robot.set_joint_effort_target(torch.zeros_like(state["articulation"]["robot"]["joint_velocity"]))
        self.env.episode_length_buf[:] = T("ep_len")
        term = self.cmd_manager.get_term("base_velocity")
        term.init_pos[:] = T("init_pos") + self.env.scene.env_origins; term.init_heading[:] = T("init_heading")
        term.x_des[:] = T("x_des"); term.vel_command_b[:] = T("vel_cmd")
        am = self.env.action_manager; jt = am.get_term("joint_pos")
        am._action[:] = T("act"); am._prev_action[:] = T("prev_act"); jt._raw_actions[:] = T("raw_act"); jt._processed_actions = T("proc_act").clone()
        cs = self.env.scene.sensors["contact_forces"]; _ = cs.data
        cs._data.current_air_time[:] = T("air_cur"); cs._data.last_air_time[:] = T("air_last")
        cs._data.current_contact_time[:] = T("con_cur"); cs._data.last_contact_time[:] = T("con_last"); cs._data.net_forces_w_history[:] = T("forces_hist")

    def env_step(self, action_np):
        out = self.env.step(self.cond.before_step(torch.from_numpy(action_np).to(self.env.device)))
        self.obs_dict, reward, terminated, truncated = out[0:4]
        self.cond.after_step(terminated | truncated)
        return reward, terminated | truncated

    def rollout(self, snap, acts):
        """acts (N_env, T, A) executed open loop from the snapshot. Returns the observations after every step and the alive mask."""
        tic = time.time()
        self.restore(snap)
        P = None; done = torch.zeros(self.N_env, dtype=torch.bool, device=self.env.device)
        alive = np.zeros((self.N_env, self.T), np.float32); R = np.zeros((self.N_env, self.T), np.float32)
        for t in range(self.T):
            reward, reset = self.env_step(acts[:, t])
            done |= reset
            p, e = self.get_obs()
            if P is None:
                P = np.zeros((self.N_env, self.T, p.shape[1]), np.float32); E = np.zeros((self.N_env, self.T, e.shape[1]), np.float32)
            P[:, t], E[:, t] = p, e
            alive[:, t] = (~done).float().cpu().numpy(); R[:, t] = reward.cpu().numpy()
        self.t_roll += time.time() - tic; self.n_roll += 1
        return P, E, alive, R

    def plan(self, snap):
        sc, pl = self.sc, self.pl
        x = pl._model_inputs()
        base = pl._base_policy_actions(x); pl.last_base_actions = base
        mean, std, key_data, noised_base = sc.start(pl.state.mean, pl.state.key_data, base)
        d = {}
        for i in range(self.I):
            acts, key_data = sc.sample(mean, std, key_data, noised_base)
            P, E, alive, R = self.rollout(snap, np.asarray(acts, np.float32).reshape(self.N_env, self.T, -1))
            jx, rew_x = sc.score_true(x, acts, jnp.asarray(P), jnp.asarray(E), jnp.asarray(alive))
            js = sc.score_model(x, acts)
            if self.arm == "consensus":
                a_s = js - js.mean(axis=1, keepdims=True); a_x = jx - jx.mean(axis=1, keepdims=True)
                ret = jnp.minimum(a_s, a_x)
            else:
                ret = jx
            mean, std = sc.update(acts, ret)
        # diagnostics of the last iteration
        js_n, jx_n, ret_n = np.asarray(js), np.asarray(jx), np.asarray(ret)
        best = ret_n.argmax(1); rr = np.arange(self.R)
        d["j_model"] = js_n[rr, best]; d["j_exact"] = jx_n[rr, best]
        d["binds"] = ((js_n - js_n.mean(1, keepdims=True))[rr, best] < (jx_n - jx_n.mean(1, keepdims=True))[rr, best]).astype(np.float32)
        d["rho_ms"] = spearman(js_n, jx_n)
        env_ret = (R * np.asarray(pl.discounts)[None]).sum(1).reshape(self.R, self.C)
        d["rho_rew"] = spearman(np.asarray(rew_x), env_ret)     # reward-head sum on true latents vs the simulator's own reward over the same steps
        d["cand_falls"] = 1.0 - alive[:, -1].reshape(self.R, self.C).mean(1)
        pl.state = BatchedMppiState(mean.astype(jnp.float32), std.astype(jnp.float32), key_data)
        return np.asarray(mean), d

    def step(self):
        L, Lt = self.L, self.Lt
        resetting = self.since < self.reset_steps
        cmd = self.cmd_manager.get_command("base_velocity")[Lt].cpu().numpy().copy()
        cmd[resetting] = 0.0
        self.pl.set_fut_cmd(cmd)
        proprio, extero = self.get_obs()
        self.pl.update(proprio[L], extero[L], self.last_act)
        nan = np.full(self.R, np.nan, np.float32)
        d = dict(j_model=nan, j_exact=nan, binds=nan, rho_ms=nan, rho_rew=nan, cand_falls=nan)
        snap = self.snapshot() if self.do_restore else None
        if self.arm == "model":
            plan = self.pl.run_control()
        elif resetting.all():
            plan = np.zeros((self.R, self.T, 12), np.float32)    # nothing is executed during the post-reset phase
        else:
            plan, d = self.plan(snap)
        action = np.where(resetting[:, None], 0.0, plan[:, 0]).astype(np.float32)
        self.last_act = action
        if self.do_restore:
            self.restore(snap)
        reward, reset = self.env_step(np.repeat(action, self.C, axis=0))
        reward = reward[Lt].cpu().numpy(); reset = reset[Lt].cpu().numpy()
        tm = self.env.termination_manager
        fall = np.zeros(self.R, bool)
        for t in self.fall_terms:
            fall |= tm.get_term(t)[Lt].cpu().numpy()
        robot = self.env.scene["robot"]
        x_now = robot.data.root_pos_w[Lt, 0].cpu().numpy().copy()
        dx = np.where(reset, 0.0, x_now - self.x_prev); self.x_prev = x_now
        r = self.rec
        r["reward"].append(reward.astype(np.float32)); r["fall"].append(fall); r["reset"].append(reset); r["resetting"].append(resetting.copy())
        r["dx"].append(dx.astype(np.float32)); r["act"].append(action)
        r["torque_sq"].append((robot.data.applied_torque[Lt] ** 2).mean(1).cpu().numpy().astype(np.float32))
        r["angvel"].append(robot.data.root_ang_vel_b[Lt].norm(dim=1).cpu().numpy().astype(np.float32))
        for k in ("j_model", "j_exact", "binds", "rho_ms", "rho_rew", "cand_falls"):
            r[k].append(np.asarray(d[k], np.float32))
        self.since += 1; self.step_i += 1
        if reset.any():
            ids = np.where(reset)[0]
            self.since[ids] = 0
            proprio, extero = self.get_obs()
            pr = np.zeros((self.R, proprio.shape[1]), np.float32); ex = np.zeros((self.R, extero.shape[1]), np.float32)
            pr[:], ex[:] = proprio[L], extero[L]
            self.pl.reset(ids, pr, ex, self.zero_R, np.zeros((self.R, 3)))
            self.last_act[ids] = 0.0
        if self.step_i % 25 == 0:
            el = time.time() - self.t0
            print(f"step {self.step_i}/{self.total_steps} {el / self.step_i:.2f} s/step, falls so far {int(np.sum(r['fall']))}, "
                  f"rollouts {self.n_roll} ({self.t_roll / max(self.n_roll, 1):.2f} s each)", flush=True)

    def run(self):
        while sb.simulation_app.is_running() and self.step_i < self.total_steps:
            self.step()
        self.report()
        self.env.close(); sb.simulation_app.close()

    def report(self):
        r = {k: np.stack(v) for k, v in self.rec.items()}                    # (steps, R)
        active = ~r["resetting"]
        n_act = active.sum(0)
        a = r["act"]; d2 = a[2:] - 2 * a[1:-1] + a[:-2]
        ok2 = active[2:] & active[1:-1] & active[:-2] & ~r["reset"][1:-1] & ~r["reset"][:-2]
        per = []
        for i, (v, s) in enumerate(self.rpairs):
            m = active[:, i]
            per.append(dict(speed=v, seed=s, steps=int(m.sum()), falls=int(r["fall"][:, i].sum()), falls_per_1000=float(r["fall"][:, i].sum() / max(m.sum(), 1) * 1000),
                            reward_per_step=float(r["reward"][m, i].mean()), forward_m=float(r["dx"][m, i].sum()),
                            forward_speed=float(r["dx"][m, i].sum() / (m.sum() * float(self.env.step_dt))),
                            smooth=float((d2[ok2[:, i], i] ** 2).sum(-1).mean()) if ok2[:, i].any() else float("nan"),
                            effort=float(r["torque_sq"][m, i].mean()), shake=float(r["angvel"][m, i].mean()),
                            rho_model_exact=float(np.nanmean(r["rho_ms"][:, i])), rho_reward_head_env=float(np.nanmean(r["rho_rew"][:, i])),
                            model_binds=float(np.nanmean(r["binds"][:, i])), cand_fall_frac=float(np.nanmean(r["cand_falls"][:, i]))))
        out = dict(arm=self.arm, checkpoint=self.cfg["model"]["checkpoint"], restore=self.do_restore, robots=per, copies=self.C,
                   sim={k: self.cfg["sim"][k] for k in ("friction", "add_mass", "motor_strength", "terrain", "total_steps")},
                   control={k: self.cfg["control"][k] for k in ("num_samples", "iterations", "num_elites", "temperature", "discount", "value_weight")},
                   seconds=time.time() - self.t0, seconds_per_step=(time.time() - self.t0) / max(self.step_i, 1),
                   rollout_seconds=self.t_roll / max(self.n_roll, 1),
                   mean=dict(falls_per_1000=float(np.mean([p["falls_per_1000"] for p in per])), reward_per_step=float(np.mean([p["reward_per_step"] for p in per])),
                             forward_speed=float(np.mean([p["forward_speed"] for p in per]))))
        base = str(self.o["out"])
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        np.savez_compressed(base + ".npz", **r)
        with open(base + ".json", "w") as f:
            json.dump(out, f, indent=1)
        print("RESULT", json.dumps(out["mean"]), f"{out['seconds_per_step']:.2f} s/step", flush=True)
        for p in per:
            print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}, flush=True)


@hydra.main(config_path=paths.get_config_dir(), config_name="oracle_plan", version_base=None)
def main(cfg: DictConfig):
    OracleSim(OmegaConf.to_container(cfg, resolve=True)).run()


if __name__ == "__main__":
    main()
