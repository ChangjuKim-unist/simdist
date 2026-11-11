"""Design 18 gate 0 (docs/design_branch_headroom_ko.md): counterfactual branching in simulation.

mode=anchor   The base controller (the model's policy head, control.prior_only=true) drives the usual batch of
              robots (scripts/simulate_batch.py).  A shadow SimDist planner runs on the same inputs every step
              without acting.  Every branch.every steps the complete state of every eligible robot is snapshotted
              (sim state, manager buffers, controller history, the shadow plan) together with the robot's own
              realised return over the next branch.horizon steps.  Output: branch.out (npz).
mode=branch   W branch points x (candidates x J clones) robots.  Each round restores the snapshots with
              env.reset_to, runs one candidate per robot for branch.chunk steps, continues with the base
              controller to branch.horizon steps and appends the realised return / fall to branch.csv.

    python scripts/sim2sim/branch_eval.py mode=anchor model.checkpoint=go1_paper control.prior_only=true \
        sim.friction=0.8 sim.add_mass=0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 \
        sim.episode_length_s=100 branch.out=/x/anchor.npz --headless
    python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=go1_paper control.prior_only=true <same physics> \
        branch.anchor=/x/anchor.npz branch.csv=/x/branch.csv --headless
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".30")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import simulate_batch as sb  # noqa: E402  (launches the Isaac Sim app, strips --headless)

import csv  # noqa: E402
import time  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from simdist.control.mppi_batch import BatchedMppiController  # noqa: E402
from simdist.utils import model as model_utils, paths  # noqa: E402

CMD_VARIANTS = [("vx0.5", 0, 0.5, "mul"), ("vx0.75", 0, 0.75, "mul"), ("yaw+0.3", 2, 0.3, "add"),
                ("yaw-0.3", 2, -0.3, "add"), ("vy+0.15", 1, 0.15, "add"), ("vy-0.15", 1, -0.15, "add")]
GAUSS_STDS = [2.0] * 4 + [0.5] * 4
N_GEN = 8
PERTURB = {"joint_pos": 0.002, "joint_vel": 0.02, "lin_vel": 0.005, "ang_vel": 0.005}


def gen_decode(g, p_hist, a_hist, cmds, u):
    """Decode generator chunks (gen_chunks.py CVAE decoder) in numpy: u (n, du) -> (n, T, A) raw actions."""
    cond = np.concatenate([((p_hist - g["p_mean"]) / g["p_std"]).reshape(-1), ((a_hist - g["a_mean"]) / g["a_std"]).reshape(-1),
                           ((cmds - g["c_mean"]) / g["c_std"]).reshape(-1)]).astype(np.float32)
    h = np.maximum(np.concatenate([u, np.tile(cond[None], (len(u), 1))], -1) @ g["d1"] + g["db1"], 0.0)
    h = np.maximum(h @ g["d2"] + g["db2"], 0.0)
    m = h @ g["d3"] + g["db3"]
    T = int(g["T"]); A = g["a_mean"].shape[0]
    return (m * np.tile(g["a_std"], T) + np.tile(g["a_mean"], T)).reshape(len(u), T, A).astype(np.float32)


class AnchorSim(sb.BatchSim):
    def __init__(self, cfg):
        assert cfg["control"].get("prior_only", False), "anchor mode drives with the policy head: control.prior_only=true"
        super().__init__(cfg)
        b = cfg["branch"]; self.b = b
        self.every, self.warmup, self.horizon = int(b["every"]), int(b["warmup"]), int(b["horizon"])
        self.max_ep = int(self.env.max_episode_length)
        ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), cfg["model"]["checkpoint"])
        _, model_cfg, _ = model_utils.load_model_from_ckpt(ckpt_dir, cfg["model"]["step"])
        shadow_cfg = dict(cfg["control"]); shadow_cfg["prior_only"] = False
        self.shadow = BatchedMppiController(self.controller.model, model_cfg, shadow_cfg, self.N, [s for _, s in self.pairs])
        self.shadow._model_inputs = self.controller._model_inputs  # same history / commands as the acting controller
        proprio, extero = self.get_obs()
        self.shadow.reset(range(self.N), proprio, extero, self.zero_action, np.zeros((self.N, 3)))
        print("Compiling the shadow planner...", flush=True); self.shadow.run_control(); print("Shadow planner ready.", flush=True)
        self._run_control = self.controller.run_control; self.controller.run_control = self._hooked_run_control
        self._ctrl_reset = self.controller.reset; self.controller.reset = self._hooked_reset
        self._env_step = self.env.step; self.env.step = self._hooked_env_step
        self.rew_log = np.zeros((self.total_steps, self.N), np.float32)
        self.fall_log = np.zeros((self.total_steps, self.N), bool)
        self.reset_log = np.zeros((self.total_steps, self.N), bool)
        self.snaps = []

    def _hooked_reset(self, env_ids, *a):
        self._ctrl_reset(env_ids, *a); self.shadow.reset(env_ids, *a)

    def _hooked_env_step(self, action):
        out = self._env_step(action)
        reward, terminated, truncated = out[1], out[2], out[3]
        tm = self.env.termination_manager
        fall = np.zeros(self.N, bool)
        for t in self.fall_terms:
            fall |= tm.get_term(t).cpu().numpy()
        self.rew_log[self.total] = reward.cpu().numpy(); self.fall_log[self.total] = fall
        self.reset_log[self.total] = (terminated | truncated).cpu().numpy()
        return out

    def _hooked_run_control(self):
        plan = self._run_control()
        F = self.shadow.run_control()  # every step: keeps the shift warm start alive
        if self.total % self.every == 0:
            ep = self.env.episode_length_buf.cpu().numpy()
            ok = (self.steps_since_reset >= self.warmup) & (ep + self.horizon < self.max_ep) & (self.total + self.horizon <= self.total_steps)
            if ok.any():
                self._snapshot(np.where(ok)[0], F)
        return plan

    def _snapshot(self, ids, F):
        st = self.env.scene.get_state(is_relative=True)["articulation"]["robot"]
        term = self.cmd_manager.get_term("base_velocity"); am = self.env.action_manager; jt = am.get_term("joint_pos")
        cs = self.env.scene.sensors["contact_forces"]; _ = cs.data  # refresh
        origins = self.env.scene.env_origins
        ep = self.env.episode_length_buf.cpu().numpy()
        g = lambda t: t.detach().cpu().numpy()  # noqa: E731
        for i in ids:
            v, s = self.pairs[i]
            self.snaps.append({
                "env": i, "speed": v, "seed": s, "step": self.total, "since_reset": int(self.steps_since_reset[i]), "ep_len": int(ep[i]),
                "root_pose": g(st["root_pose"][i]), "root_vel": g(st["root_velocity"][i]),
                "joint_pos": g(st["joint_position"][i]), "joint_vel": g(st["joint_velocity"][i]),
                "cmd_init_pos": g(term.init_pos[i] - origins[i]), "cmd_init_heading": float(term.init_heading[i]),
                "cmd_x_des": float(term.x_des[i]), "cmd_vel": g(term.vel_command_b[i]),
                "act": g(am._action[i]), "prev_act": g(am._prev_action[i]), "raw_act": g(jt._raw_actions[i]), "proc_act": g(jt._processed_actions[i]),
                "air_cur": g(cs._data.current_air_time[i]), "air_last": g(cs._data.last_air_time[i]),
                "con_cur": g(cs._data.current_contact_time[i]), "con_last": g(cs._data.last_contact_time[i]),
                "forces_hist": g(cs._data.net_forces_w_history[i]),
                "hist_p": self.controller.proprio_hist[i].copy(), "hist_a": self.controller.acts_hist[i].copy(), "fut_cmds": self.controller.fut_cmds[i].copy(),
                "F": np.asarray(F[i], np.float32),
            })

    def report(self):
        super().report()
        n = len(self.snaps)
        if n == 0:
            print("no branch points recorded"); return
        out = {k: np.stack([s[k] for s in self.snaps]) for k in self.snaps[0]}
        ret = np.zeros(n, np.float32); fell = np.zeros(n, bool); length = np.zeros(n, np.int64)
        for q, s in enumerate(self.snaps):
            i, t = s["env"], s["step"]
            r = self.reset_log[t:t + self.horizon, i]
            k = int(np.argmax(r)) + 1 if r.any() else self.horizon
            ret[q] = self.rew_log[t:t + k, i].sum(); fell[q] = self.fall_log[t:t + k, i].any(); length[q] = k
        out.update(anchor_ret=ret, anchor_fell=fell, anchor_len=length,
                   model=np.array((self.cfg.get("output", {}) or {}).get("model_name") or self.cfg["model"]["checkpoint"]),
                   condition=np.array((self.cfg.get("output", {}) or {}).get("condition") or ""), horizon=np.array(self.horizon))
        os.makedirs(os.path.dirname(os.path.abspath(self.b["out"])), exist_ok=True)
        np.savez(self.b["out"], **out)
        print(f"{n} branch points written to {self.b['out']} (anchor return mean {ret.mean():.3f}, fell {fell.mean():.3f})", flush=True)


class BranchSim(sb.BatchSim):
    def __init__(self, cfg):
        assert cfg["control"].get("prior_only", False), "branch mode continues with the policy head: control.prior_only=true"
        b = cfg["branch"]; self.b = b
        self.A = dict(np.load(b["anchor"]))
        self.chunk, self.horizon, self.J, self.W = int(b["chunk"]), int(b["horizon"]), int(b["J"]), int(b["W"])
        sel = np.ones(len(self.A["step"]), bool)
        if b.get("speeds"):
            sel &= np.isin(self.A["speed"], [float(v) for v in b["speeds"]])
        if b.get("seeds"):
            sel &= np.isin(self.A["seed"], [int(s) for s in b["seeds"]])
        idx = np.where(sel)[0]
        if b.get("max_points"):
            idx = idx[: int(b["max_points"])]
        done = set()
        if b["csv"] and os.path.exists(b["csv"]):
            with open(b["csv"]) as f:
                for row in csv.DictReader(f):
                    done.add((float(row["speed"]), int(row["seed"]), int(row["step"])))
        self.idx = np.array([q for q in idx if (float(self.A["speed"][q]), int(self.A["seed"][q]), int(self.A["step"][q])) not in done], np.int64)
        print(f"branch points: {len(idx)} selected, {len(idx) - len(self.idx)} already done, {len(self.idx)} to run", flush=True)
        groups = list(b["groups"]); self.experts_cfg = list(b["experts"]) if "E" in groups else []
        self.cands = []  # (group, cand id, label)
        for gname in groups:
            if gname in ("A0", "A1", "F"):
                self.cands.append((gname, 0, gname))
            elif gname == "B":
                self.cands += [("B", k, f"B{k}_std{s}") for k, s in enumerate(GAUSS_STDS)]
            elif gname == "C":
                self.cands += [("C", k, f"C{k}") for k in range(N_GEN)]
            elif gname == "D":
                self.cands += [("D", k, f"D_{v[0]}") for k, v in enumerate(CMD_VARIANTS)]
            elif gname == "E":
                self.cands += [("E", k, f"E_{r}") for k, r in enumerate(self.experts_cfg)]
            else:
                raise ValueError(gname)
        self.K = len(self.cands); self.N_env = self.W * self.K * self.J
        self.e_slot = np.repeat(np.arange(self.W), self.K * self.J)
        self.e_cand = np.tile(np.repeat(np.arange(self.K), self.J), self.W)
        self.e_clone = np.tile(np.arange(self.J), self.W * self.K)
        cfg["batch"]["speeds"] = [0.0]; cfg["batch"]["seeds"] = list(range(self.N_env))
        super().__init__(cfg)
        assert self.N == self.N_env
        self.experts = []
        for r in self.experts_cfg:
            from simdist.utils.torch import get_actor_critic_from_iteration
            self.experts.append(get_actor_critic_from_iteration(r, int(b["expert_iteration"]), "cuda:0")[0])
        self.gen = None
        if "C" in groups:
            w = np.load(str(b["gen_weights"]))
            self.gen = {k: np.asarray(w[k], np.float32) for k in ("d1", "db1", "d2", "db2", "d3", "db3", "p_mean", "p_std", "a_mean", "a_std", "c_mean", "c_std")}
            self.gen["du"] = int(w["du"]); self.gen["T"] = int(w["T"])
        self.rows_written = 0

    def _patch_per_env_speed(self):
        """Per-robot forward speed as in BatchSim, but kept in self.fv so that it can be set per round."""
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

    # ---- restore ---------------------------------------------------------------------------------------------
    def restore(self, pts):
        A = self.A; dev = self.env.device
        bp = pts[np.minimum(self.e_slot, len(pts) - 1)]  # branch index per env (unused slots repeat the last point)
        T = lambda k: torch.as_tensor(np.asarray(A[k][bp], np.float32), device=dev)  # noqa: E731
        root_pose, root_vel = T("root_pose"), T("root_vel"); jpos, jvel = T("joint_pos"), T("joint_vel")
        if self.b.get("perturb", True):
            rng = np.random.default_rng(int(self.b["seed"]) + 7919 * int(pts[0]))
            m = torch.as_tensor((self.e_clone >= 1).astype(np.float32), device=dev)[:, None]
            nz = lambda shape, s: m * torch.as_tensor(rng.normal(0.0, s, shape).astype(np.float32), device=dev)  # noqa: E731
            jpos = jpos + nz(jpos.shape, PERTURB["joint_pos"]); jvel = jvel + nz(jvel.shape, PERTURB["joint_vel"])
            root_vel = torch.cat([root_vel[:, :3] + nz((self.N, 3), PERTURB["lin_vel"]), root_vel[:, 3:] + nz((self.N, 3), PERTURB["ang_vel"])], 1)
        state = {"articulation": {"robot": {"root_pose": root_pose, "root_velocity": root_vel, "joint_position": jpos, "joint_velocity": jvel}},
                 "deformable_object": {}, "rigid_object": {}}
        env_ids = torch.arange(self.N, device=dev)
        self.env.reset_to(state, env_ids, is_relative=True)
        # reset_to sets the joint velocity target to the restored joint velocity; the PD actuator uses it (Kd * (qd_target - qd)),
        # while the env's position action never writes velocity targets (they are zero in the anchor run): zero them here
        robot = self.env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(jvel)); robot.set_joint_effort_target(torch.zeros_like(jvel))
        self.env.episode_length_buf[:] = torch.as_tensor(A["ep_len"][bp], device=dev, dtype=torch.long)
        term = self.cmd_manager.get_term("base_velocity")
        term.init_pos[:] = T("cmd_init_pos") + self.env.scene.env_origins; term.init_heading[:] = T("cmd_init_heading")
        term.x_des[:] = T("cmd_x_des"); term.vel_command_b[:] = T("cmd_vel"); self.fv[:] = T("speed")
        am = self.env.action_manager; jt = am.get_term("joint_pos")
        am._action[:] = T("act"); am._prev_action[:] = T("prev_act"); jt._raw_actions[:] = T("raw_act"); jt._processed_actions = T("proc_act")
        cs = self.env.scene.sensors["contact_forces"]; _ = cs.data
        cs._data.current_air_time[:] = T("air_cur"); cs._data.last_air_time[:] = T("air_last")
        cs._data.current_contact_time[:] = T("con_cur"); cs._data.last_contact_time[:] = T("con_last"); cs._data.net_forces_w_history[:] = T("forces_hist")
        self.obs_dict = self.env.observation_manager.compute()
        # controller history: rolled back by one so that the first update() re-appends the current measurement
        self.controller.proprio_hist[:] = np.roll(A["hist_p"][bp], 1, axis=1); self.controller.acts_hist[:] = np.roll(A["hist_a"][bp], 1, axis=1)
        self.last_action = np.asarray(A["hist_a"][bp][:, -1], np.float32).copy()
        self.steps_since_reset[:] = self.reset_steps + 1
        return bp

    # ---- one round -------------------------------------------------------------------------------------------
    def run_round(self, pts):
        bp = self.restore(pts); A = self.A; n_valid = len(pts) * self.K * self.J
        valid = np.arange(self.N) < n_valid
        grp = np.array([self.cands[k][0] for k in self.e_cand]); cid = np.array([self.cands[k][1] for k in self.e_cand])
        openloop = np.isin(grp, ["A1", "B", "C", "F"]); is_D = grp == "D"; is_E = grp == "E"
        chunks = np.zeros((self.N, self.chunk, 12), np.float32)
        ret = np.zeros(self.N); alive = np.ones(self.N, bool); fell = np.zeros(self.N, bool); fall_step = np.full(self.N, -1)
        for t in range(self.horizon):
            cmd = self.cmd_manager.get_command("base_velocity").cpu().numpy().copy()
            if t < self.chunk and is_D.any():
                for k, (_, dim, val, how) in enumerate(CMD_VARIANTS):
                    e = is_D & (cid == k)
                    if how == "mul": cmd[e, dim] *= val
                    else: cmd[e, dim] += val
            self.controller.set_fut_cmd(cmd)
            proprio, extero = self.get_obs()
            self.controller.update(proprio, extero, self.last_action)
            plan = self.controller.run_control()  # (N, T, A): the policy head's chunk for every robot
            if t == 0:
                for w in range(len(pts)):
                    q = pts[w]; rng = np.random.default_rng(int(self.b["seed"]) + 1000003 * int(A["seed"][q]) + int(A["step"][q]) + 7 * int(round(10 * float(A["speed"][q]))))
                    sl = self.e_slot == w
                    def env_of(g, k=0, j=0):
                        e = np.where(sl & (grp == g) & (cid == k) & (self.e_clone == j))[0]
                        return int(e[0]) if len(e) else None
                    e1 = env_of("A1"); base = plan[e1] if e1 is not None else plan[np.where(sl)[0][0]]
                    gen = None
                    if self.gen is not None and env_of("C") is not None:
                        e = env_of("C"); u = np.clip(rng.normal(size=(N_GEN, self.gen["du"])), -2.5, 2.5).astype(np.float32)
                        gen = gen_decode(self.gen, self.controller.proprio_hist[e], self.controller.acts_hist[e], self.controller.fut_cmds[e], u)
                    noise = {k: rng.normal(0.0, s, base.shape).astype(np.float32) for k, s in enumerate(GAUSS_STDS)}
                    for e in np.where(sl)[0]:
                        g, k = grp[e], cid[e]
                        if g == "A1": chunks[e] = base
                        elif g == "B": chunks[e] = base + noise[k]
                        elif g == "C": chunks[e] = gen[k]
                        elif g == "F": chunks[e] = A["F"][q]
            action = np.asarray(plan[:, 0], np.float32).copy()
            if t < self.chunk:
                action[openloop] = chunks[openloop, t]
                for k, pol in enumerate(self.experts):
                    e = np.where(is_E & (cid == k))[0]
                    if len(e):
                        with torch.no_grad():
                            action[e] = pol(self.obs_dict["policy"][torch.as_tensor(e, device=self.env.device)].to("cuda:0")).cpu().numpy()
            self.last_action = action
            self.obs_dict, reward, terminated, truncated = self.env.step(torch.from_numpy(action).to(self.env.device))[0:4]
            reward = reward.cpu().numpy(); reset = (terminated | truncated).cpu().numpy()
            tm = self.env.termination_manager; fall = np.zeros(self.N, bool)
            for tn in self.fall_terms:
                fall |= tm.get_term(tn).cpu().numpy()
            ret[alive] += reward[alive]
            new = alive & reset; fell[new] = fall[new]; fall_step[new] = t; alive &= ~reset
        rows = []
        out = self.cfg.get("output", {}) or {}
        model_name = out.get("model_name") or self.cfg["model"]["checkpoint"]; cond = out.get("condition") or ""
        for e in np.where(valid)[0]:
            q = bp[e]
            rows.append({"model": model_name, "condition": cond, "speed": float(A["speed"][q]), "seed": int(A["seed"][q]), "step": int(A["step"][q]),
                         "group": grp[e], "cand": int(cid[e]), "label": self.cands[self.e_cand[e]][2], "clone": int(self.e_clone[e]),
                         "ret": float(ret[e]), "fell": int(fell[e]), "fall_step": int(fall_step[e]), "alive_steps": int(self.horizon if fall_step[e] < 0 else fall_step[e] + 1),
                         "anchor_ret": float(A["anchor_ret"][q]), "anchor_fell": int(A["anchor_fell"][q]), "anchor_len": int(A["anchor_len"][q])})
        return rows

    def run(self):
        path = self.b["csv"]; os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        t0 = time.time(); n_rounds = int(np.ceil(len(self.idx) / self.W))
        for r in range(n_rounds):
            if not sb.simulation_app.is_running():
                break
            pts = self.idx[r * self.W:(r + 1) * self.W]
            rows = self.run_round(pts)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(rows)
            self.rows_written += len(rows)
            el = time.time() - t0
            print(f"round {r + 1}/{n_rounds}: {len(pts)} points, {len(rows)} rows, {el / (r + 1):.1f} s/round, eta {el / (r + 1) * (n_rounds - r - 1) / 60:.1f} min", flush=True)
        print(f"RESULT branch rows written: {self.rows_written} -> {path}", flush=True)
        self.env.close(); sb.simulation_app.close()


@hydra.main(config_path=paths.get_config_dir(), config_name="branch_eval", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg["mode"] == "anchor":
        AnchorSim(cfg).run()
    elif cfg["mode"] == "branch":
        BranchSim(cfg).run()
    else:
        raise ValueError(cfg["mode"])


if __name__ == "__main__":
    main()
