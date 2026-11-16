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
PLAN_GROUPS = ("Fcl", "Pcalfpol", "Ponlycalf")  # candidates that act with the separate replanning planner
CALF = np.array([2, 5, 8, 11])  # calf joints in the action order (FR, FL, RR, RL x hip, thigh, calf)
NONCALF = np.array([j for j in range(12) if j not in CALF])
N_GEN = 8
PERTURB = {"joint_pos": 0.002, "joint_vel": 0.02, "lin_vel": 0.005, "ang_vel": 0.005}


def gen_decode(g, p_hist, a_hist, cmds, u):
    """Decode generator chunks (gen_chunks.py CVAE decoder) in numpy: one history, u (n, du) -> (n, T, A) raw actions."""
    n = len(u)
    return gen_decode_b(g, np.repeat(p_hist[None], n, 0), np.repeat(a_hist[None], n, 0), np.repeat(cmds[None], n, 0), u)


def gen_decode_b(g, P, Ah, C, U):
    """Batched decode: histories P (n, H+1, p), Ah (n, H, A), commands C (n, T+1, 3), latents U (n, du) -> (n, T, A)."""
    n = len(U)
    cond = np.concatenate([((P - g["p_mean"]) / g["p_std"]).reshape(n, -1), ((Ah - g["a_mean"]) / g["a_std"]).reshape(n, -1),
                           ((C - g["c_mean"]) / g["c_std"]).reshape(n, -1)], -1).astype(np.float32)
    h = np.maximum(np.concatenate([U.astype(np.float32), cond], -1) @ g["d1"] + g["db1"], 0.0)
    h = np.maximum(h @ g["d2"] + g["db2"], 0.0)
    m = h @ g["d3"] + g["db3"]
    T = int(g["T"]); A = g["a_mean"].shape[0]
    return (m * np.tile(g["a_std"], T) + np.tile(g["a_mean"], T)).reshape(n, T, A).astype(np.float32)


def load_gen(path):
    w = np.load(str(path))
    g = {k: np.asarray(w[k], np.float32) for k in ("d1", "db1", "d2", "db2", "d3", "db3", "p_mean", "p_std", "a_mean", "a_std", "c_mean", "c_std")}
    g["du"] = int(w["du"]); g["T"] = int(w["T"])
    return g


class GenBase:
    """Design 18 section 8.3: the design-16 generator's mean (u = 0) decoded every step on the controller's own history
    buffers and executed receding-horizon = design 17's "generator mean only, iterations 0" arm, in numpy."""

    def __init__(self, g, ctrl):
        self.g, self.ctrl = g, ctrl

    def plan(self, ids=None):
        c = self.ctrl
        P, Ah, C = (c.proprio_hist, c.acts_hist, c.fut_cmds) if ids is None else (c.proprio_hist[ids], c.acts_hist[ids], c.fut_cmds[ids])
        return gen_decode_b(self.g, P, Ah, C, np.zeros((len(P), self.g["du"]), np.float32))


def foot_indices(env):
    """Foot body ids in the robot and in the contact sensor, both in the robot's foot order (design doc 26/27)."""
    robot = env.scene["robot"]; cs = env.scene.sensors["contact_forces"]
    ids_r, names_r = robot.find_bodies(".*_foot"); ids_c, names_c = cs.find_bodies(".*_foot")
    order = [names_c.index(n) for n in names_r]
    return list(ids_r), [ids_c[k] for k in order], list(names_r)


def contact_slip(env, feet, feet_cs):
    """(N,) mean horizontal speed of the feet in contact (contact force > 1 N; 0 if none), (N, 4) contact flags."""
    robot = env.scene["robot"].data; cs = env.scene.sensors["contact_forces"].data
    fs = torch.linalg.norm(robot.body_lin_vel_w[:, feet, :2], dim=-1)
    c = (torch.linalg.norm(cs.net_forces_w[:, feet_cs], dim=-1) > 1.0).float()
    slip = (fs * c).sum(-1) / torch.clamp(c.sum(-1), min=1.0)
    return slip.cpu().numpy(), c.cpu().numpy().astype(bool)


class AnchorSim(sb.BatchSim):
    def __init__(self, cfg):
        b = cfg["branch"]
        # design doc 27: the anchor run can also be driven by the SimDist planner (its own plan is then the warm start F)
        self.drive = str(b.get("drive", "policy")); assert self.drive in ("policy", "planner")
        if self.drive == "policy":
            assert cfg["control"].get("prior_only", False), "drive=policy acts with the policy head: control.prior_only=true"
        else:
            assert not cfg["control"].get("prior_only", False), "drive=planner acts with the SimDist planner: control.prior_only=false"
            b["shadow"] = False
        super().__init__(cfg)
        self.b = b
        self.every, self.warmup, self.horizon = int(b["every"]), int(b["warmup"]), int(b["horizon"])
        # design doc 27: trigger=slip snapshots at the detection of a slip event (10-step trailing mean of the contact-foot
        # slip above slip_theta for 5 consecutive released steps, run start >= min_since steps after the reset; the same
        # definition as design doc 26 §5) plus quiet control points every ctrl_every steps (mean slip <= theta for ctrl_quiet steps)
        self.trigger = str(b.get("trigger", "every")); assert self.trigger in ("every", "slip")
        if self.trigger == "slip":
            self.theta = float(b["slip_theta"]); self.min_since = int(b.get("min_since", 300))
            self.ctrl_every, self.ctrl_quiet = int(b.get("ctrl_every", 100)), int(b.get("ctrl_quiet", 50))
            self.feet, self.feet_cs, _ = foot_indices(self.env)
            self.slip_buf = np.zeros((self.N, 10)); self.slip_n = np.zeros(self.N, np.int64)
            self.over = np.zeros(self.N, np.int64); self.quiet = np.zeros(self.N, np.int64)
            self.n_events = 0; self.n_ctrl = 0
        self.max_ep = int(self.env.max_episode_length)
        ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), cfg["model"]["checkpoint"])
        _, model_cfg, _ = model_utils.load_model_from_ckpt(ckpt_dir, cfg["model"]["step"])
        self.base = str(b.get("base", "policy")); assert self.base in ("policy", "gen")
        self.gen_base = GenBase(load_gen(b["gen_weights"]), self.controller) if self.base == "gen" else None
        self.shadow = None
        if b.get("shadow", True):
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
        self._ctrl_reset(env_ids, *a)
        if self.shadow is not None:
            self.shadow.reset(env_ids, *a)

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
        plan = self.gen_base.plan() if self.gen_base is not None else self._run_control()
        if self.drive == "planner":
            F = plan  # the acting planner's own plan
        else:
            F = self.shadow.run_control() if self.shadow is not None else np.zeros_like(plan)  # every step: keeps the shift warm start alive
        ep = self.env.episode_length_buf.cpu().numpy()
        room = (ep + self.horizon < self.max_ep) & (self.total + self.horizon <= self.total_steps)
        if self.trigger == "every":
            if self.total % self.every == 0:
                ok = (self.steps_since_reset >= self.warmup) & room
                if ok.any():
                    self._snapshot(np.where(ok)[0], F)
        else:
            self._slip_trigger(F, room)
        return plan

    def _slip_trigger(self, F, room):
        released = self.steps_since_reset >= self.reset_steps
        slip, _ = contact_slip(self.env, self.feet, self.feet_cs)
        self.slip_n[~released] = 0; self.over[~released] = 0; self.quiet[~released] = 0
        self.slip_buf[released] = np.roll(self.slip_buf[released], 1, axis=1); self.slip_buf[released, 0] = slip[released]
        self.slip_n[released] += 1
        k = np.clip(self.slip_n, 1, 10)
        ma = np.array([self.slip_buf[i, :k[i]].mean() for i in range(self.N)])
        hi = released & (ma > self.theta)
        self.over = np.where(hi, self.over + 1, 0); self.quiet = np.where(released & ~hi, self.quiet + 1, 0)
        ev = hi & (self.over == 5) & (self.steps_since_reset - 4 >= self.min_since) & room
        if ev.any():
            self._snapshot(np.where(ev)[0], F, is_event=1, ma=ma); self.n_events += int(ev.sum())
        if self.total % self.ctrl_every == 0:
            ct = released & (self.quiet >= self.ctrl_quiet) & (self.steps_since_reset >= self.min_since) & room & ~ev
            if ct.any():
                self._snapshot(np.where(ct)[0], F, is_event=0, ma=ma); self.n_ctrl += int(ct.sum())

    def _snapshot(self, ids, F, is_event=-1, ma=None):
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
                "is_event": int(is_event), "slip_ma": float(ma[i]) if ma is not None else np.nan,
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
        if self.trigger == "slip":
            print(f"slip trigger: {self.n_events} events, {self.n_ctrl} control points", flush=True)


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
        if "is_event" in self.A and (b.get("max_events") or b.get("max_ctrl")):
            # design doc 27: seeded random subsets of the slip events / quiet control points (the same subset on a restart)
            rng = np.random.default_rng(int(b["seed"]) + 555)
            ev = idx[self.A["is_event"][idx] == 1]; ct = idx[self.A["is_event"][idx] == 0]
            if b.get("max_events") and len(ev) > int(b["max_events"]):
                ev = np.sort(rng.choice(ev, int(b["max_events"]), replace=False))
            if b.get("max_ctrl") and len(ct) > int(b["max_ctrl"]):
                ct = np.sort(rng.choice(ct, int(b["max_ctrl"]), replace=False))
            print(f"slip events {int((self.A['is_event'][idx] == 1).sum())} -> {len(ev)}, control points {int((self.A['is_event'][idx] == 0).sum())} -> {len(ct)}", flush=True)
            idx = np.sort(np.concatenate([ev, ct]))
            if b.get("only_events"):  # design doc 27 stage 2 K: the same capped events, no control points
                idx = np.sort(ev)
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
            elif gname == "G":   # section 8.3: generator latent samples re-decoded every step (closed loop), same u as C
                self.cands += [("G", k, f"G{k}") for k in range(N_GEN)]
            elif gname == "Fcl":  # section 8.2: SimDist planner replanning every step, warm-started from the shadow plan F
                self.cands.append(("Fcl", 0, "Fcl"))
            elif gname == "Ccl":  # section 8.2: generator mean (u = 0) re-decoded every step (closed loop)
                self.cands.append(("Ccl", 0, "Ccl"))
            elif gname in ("Ncl", "Nrep", "Pcalfpol", "Ponlycalf"):
                # design doc 27: Ncl = policy head + AR(1) noise matched to the planner's deviation (branch.noise_json);
                # Pcalfpol = planner with the four calf joints from the policy head; Ponlycalf = policy head with the calves from the planner
                self.cands.append((gname, 0, gname))
            else:
                raise ValueError(gname)
        self.K = len(self.cands); self.N_env = self.W * self.K * self.J
        self.e_slot = np.repeat(np.arange(self.W), self.K * self.J)
        self.e_cand = np.tile(np.repeat(np.arange(self.K), self.J), self.W)
        self.e_clone = np.tile(np.arange(self.J), self.W * self.K)
        cfg["batch"]["speeds"] = [0.0]; cfg["batch"]["seeds"] = list(range(self.N_env))
        super().__init__(cfg)
        assert self.N == self.N_env
        ok = self.A["ep_len"][self.idx] + self.horizon < int(self.env.max_episode_length)  # section 9.7: long horizons
        if (~ok).any():
            print(f"dropping {int((~ok).sum())} branch points whose episode would time out within the horizon", flush=True); self.idx = self.idx[ok]
        if self.chunk != 25:
            assert not ({"A1", "B", "C", "F"} & set(groups)), "open-loop chunk candidates are 25 steps long: branch.chunk must be 25 with them"
        self.experts = []
        for r in self.experts_cfg:
            from simdist.utils.torch import get_actor_critic_from_iteration
            self.experts.append(get_actor_critic_from_iteration(r, int(b["expert_iteration"]), "cuda:0")[0])
        self.base = str(b.get("base", "policy")); assert self.base in ("policy", "gen")
        self.gen = load_gen(b["gen_weights"]) if (set(groups) & {"C", "G", "Ccl"} or self.base == "gen") else None
        self.gen_base = GenBase(self.gen, self.controller) if self.base == "gen" else None
        # section 8.2 F_cl: a separate planner over the F_cl robots only (W x J), same config as the shadow planner
        self.fcl = None; self.fcl_ids = np.array([e for e in range(self.N) if self.cands[self.e_cand[e]][0] in PLAN_GROUPS], np.int64)
        if len(self.fcl_ids):
            ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), cfg["model"]["checkpoint"])
            _, model_cfg, _ = model_utils.load_model_from_ckpt(ckpt_dir, cfg["model"]["step"])
            fcfg = dict(cfg["control"]); fcfg["prior_only"] = False
            self.fcl = BatchedMppiController(self.controller.model, model_cfg, fcfg, len(self.fcl_ids), [int(b["seed"]) + 100 + k for k in range(len(self.fcl_ids))])
            proprio, extero = self.get_obs(); z = np.zeros((len(self.fcl_ids), 12), np.float32)
            self.fcl.reset(range(len(self.fcl_ids)), proprio[self.fcl_ids], extero[self.fcl_ids], z, np.zeros((len(self.fcl_ids), 3)))
            print(f"Compiling the F_cl planner ({len(self.fcl_ids)} robots)...", flush=True); self.fcl.run_control(); print("F_cl planner ready.", flush=True)
        self.log_terms = bool(b.get("terms", False))
        self.term_names = list(self.env.reward_manager.active_terms)
        self.rows_written = 0
        # design doc 27: AR(1) noise for Ncl, per joint (action order), from the planner's measured deviation after slips
        self.noise_sigma = self.noise_rho = None
        if "Ncl" in groups:
            import json
            nj = json.load(open(b["noise_json"]))[str(b["noise_key"])]
            self.noise_sigma = np.asarray(nj["sigma"], np.float32); self.noise_rho = np.asarray(nj["rho"], np.float32)
            print(f"Ncl noise ({b['noise_key']}): sigma {np.round(self.noise_sigma, 3)}, rho {np.round(self.noise_rho, 3)}", flush=True)
        # design doc 27: Nrep = policy head + a replayed stretch of the planner's own deviation (a - a_pol) recorded after other
        # slip events (branch.dev_bank npz, key branch.noise_key): same size, tails and persistence, no link to the current state
        self.dev_bank = None
        if "Nrep" in groups:
            self.dev_bank = np.asarray(np.load(b["dev_bank"])[str(b["noise_key"])], np.float32)
            print(f"Nrep deviation bank ({b['noise_key']}): {len(self.dev_bank)} steps, mean |delta| {np.linalg.norm(self.dev_bank, axis=-1).mean():.3f}", flush=True)
        # design doc 27: per-step record of every branch (one npz per round in branch.steps_dir)
        self.steps_dir = b.get("steps_dir"); self.round_idx = 0
        if self.steps_dir:
            os.makedirs(self.steps_dir, exist_ok=True)
            self.feet, self.feet_cs, _ = foot_indices(self.env)

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
        if self.fcl is not None:
            ids = self.fcl_ids; f = self.fcl
            f.proprio_hist[:] = self.controller.proprio_hist[ids]; f.acts_hist[:] = self.controller.acts_hist[ids]
            f.fut_cmds[:] = self.controller.fut_cmds[ids]
            f.reset_counts += 1
            key_data = np.stack([f._env_key_data(i) for i in range(f.N)])
            import jax.numpy as jnp
            f.state = f.state.replace(mean=jnp.asarray(np.asarray(A["F"][bp[ids]], np.float32)),
                                      std=jnp.full((f.N, f.T, f.act_dim), float(f.ctrl_cfg["init_std"]), jnp.float32), key_data=jnp.asarray(key_data))
        return bp

    # ---- one round -------------------------------------------------------------------------------------------
    def run_round(self, pts):
        bp = self.restore(pts); A = self.A; n_valid = len(pts) * self.K * self.J
        valid = np.arange(self.N) < n_valid
        grp = np.array([self.cands[k][0] for k in self.e_cand]); cid = np.array([self.cands[k][1] for k in self.e_cand])
        openloop = np.isin(grp, ["A1", "B", "C", "F"]); is_D = grp == "D"; is_E = grp == "E"
        is_G = grp == "G"; is_Ccl = grp == "Ccl"; gen_cl = np.where(is_G | is_Ccl)[0]
        chunks = np.zeros((self.N, self.chunk, 12), np.float32); u_cl = np.zeros((self.N, self.gen["du"] if self.gen else 1), np.float32)
        ret = np.zeros(self.N); alive = np.ones(self.N, bool); fell = np.zeros(self.N, bool); fall_step = np.full(self.N, -1)
        nt = len(self.term_names); term_sum = np.zeros((self.N, nt)); term_chunk = np.zeros((self.N, nt))
        # design doc 27 candidates
        ncl = np.where(grp == "Ncl")[0]; pcp = np.where(grp == "Pcalfpol")[0]; poc = np.where(grp == "Ponlycalf")[0]
        nstate = np.zeros((self.N, 12), np.float32)
        rngn = np.random.default_rng(int(self.b["seed"]) + 104729 * int(pts[0]) + 31)
        nrep = np.where(grp == "Nrep")[0]
        rep_start = rngn.integers(0, len(self.dev_bank) - self.horizon, len(nrep)) if len(nrep) else None
        rec = None; prev_pred = None
        if self.steps_dir:
            H = self.horizon
            rec = {"tilt": np.zeros((H, self.N), np.float32), "slip": np.zeros((H, self.N), np.float32), "contact": np.zeros((H, self.N, 4), bool),
                   "lin_vel": np.zeros((H, self.N, 3), np.float16), "ang_vel": np.zeros((H, self.N, 3), np.float16), "height": np.zeros((H, self.N), np.float16),
                   "act": np.zeros((H, self.N, 12), np.float16), "act_pol": np.zeros((H, self.N, 12), np.float16), "cmdvx": np.zeros((H, self.N), np.float16),
                   "lat_err": np.full((H, self.N), np.nan, np.float32), "tau2": np.zeros((H, self.N), np.float32), "power": np.zeros((H, self.N), np.float32),
                   "alive": np.zeros((H, self.N), bool)}
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
            plan = self.gen_base.plan() if self.gen_base is not None else self.controller.run_control()  # (N, T, A): the base controller's chunk
            if self.fcl is not None and t < self.chunk:
                ids = self.fcl_ids
                self.fcl.set_fut_cmd(cmd[ids]); self.fcl.update(proprio[ids], extero[ids], self.last_action[ids])
            if t == 0:
                for w in range(len(pts)):
                    q = pts[w]; rng = np.random.default_rng(int(self.b["seed"]) + 1000003 * int(A["seed"][q]) + int(A["step"][q]) + 7 * int(round(10 * float(A["speed"][q]))))
                    sl = self.e_slot == w
                    def env_of(g, k=0, j=0):
                        e = np.where(sl & (grp == g) & (cid == k) & (self.e_clone == j))[0]
                        return int(e[0]) if len(e) else None
                    e1 = env_of("A1"); base = plan[e1] if e1 is not None else plan[np.where(sl)[0][0]]
                    gen = None
                    if self.gen is not None and (env_of("C") is not None or env_of("G") is not None):
                        e = env_of("C") if env_of("C") is not None else env_of("G")
                        u = np.clip(rng.normal(size=(N_GEN, self.gen["du"])), -2.5, 2.5).astype(np.float32)
                        gen = gen_decode(self.gen, self.controller.proprio_hist[e], self.controller.acts_hist[e], self.controller.fut_cmds[e], u)
                    noise = {k: rng.normal(0.0, s, base.shape).astype(np.float32) for k, s in enumerate(GAUSS_STDS)}
                    for e in np.where(sl)[0]:
                        g, k = grp[e], cid[e]
                        if g == "A1": chunks[e] = base
                        elif g == "B": chunks[e] = base + noise[k]
                        elif g == "C": chunks[e] = gen[k]
                        elif g == "G": u_cl[e] = u[k]       # same latent as C_k, re-decoded every step
                        elif g == "F": chunks[e] = A["F"][q]
            action = np.asarray(plan[:, 0], np.float32).copy()
            if t < self.chunk:
                action[openloop] = chunks[openloop, t]
                if len(gen_cl):
                    c = self.controller
                    action[gen_cl] = gen_decode_b(self.gen, c.proprio_hist[gen_cl], c.acts_hist[gen_cl], c.fut_cmds[gen_cl], u_cl[gen_cl])[:, 0]
                if self.fcl is not None:
                    # t = 0 executes the shadow plan's first action (the planner already planned at this state); then replan
                    action[self.fcl_ids] = A["F"][bp[self.fcl_ids]][:, 0] if t == 0 else self.fcl.run_control()[:, 0]
                    pol0 = np.asarray(plan[:, 0], np.float32)
                    if len(pcp):
                        action[pcp[:, None], CALF[None, :]] = pol0[pcp][:, CALF]
                    if len(poc):
                        action[poc[:, None], NONCALF[None, :]] = pol0[poc][:, NONCALF]
                if len(ncl):
                    eps = rngn.normal(size=(len(ncl), 12)).astype(np.float32)
                    if t == 0:
                        nstate[ncl] = self.noise_sigma * eps  # stationary start
                    else:
                        nstate[ncl] = self.noise_rho * nstate[ncl] + np.sqrt(1.0 - self.noise_rho ** 2) * self.noise_sigma * eps
                    action[ncl] = action[ncl] + nstate[ncl]
                if len(nrep):
                    action[nrep] = action[nrep] + self.dev_bank[rep_start + t]
                for k, pol in enumerate(self.experts):
                    e = np.where(is_E & (cid == k))[0]
                    if len(e):
                        with torch.no_grad():
                            action[e] = pol(self.obs_dict["policy"][torch.as_tensor(e, device=self.env.device)].to("cuda:0")).cpu().numpy()
            self.last_action = action
            if rec is not None:
                rd = self.env.scene["robot"].data; gg = lambda x: x.detach().cpu().numpy()  # noqa: E731
                slip, con = contact_slip(self.env, self.feet, self.feet_cs)
                pe = np.asarray(plan, np.float32).copy(); pe[:, 0] = action
                z, pl, _, _ = self.controller.trace(pe)
                z = np.asarray(z, np.float32)
                if prev_pred is not None:
                    rec["lat_err"][t] = np.linalg.norm(z - prev_pred, axis=-1)
                prev_pred = np.asarray(pl, np.float32)[:, 0].copy()
                tau, qd = gg(rd.applied_torque), gg(rd.joint_vel)
                rec["tilt"][t] = np.linalg.norm(gg(rd.projected_gravity_b)[:, :2], axis=-1); rec["slip"][t] = slip; rec["contact"][t] = con
                rec["lin_vel"][t] = gg(rd.root_lin_vel_b); rec["ang_vel"][t] = gg(rd.root_ang_vel_b); rec["height"][t] = gg(rd.root_pos_w[:, 2])
                rec["act"][t] = action; rec["act_pol"][t] = np.asarray(plan[:, 0], np.float32); rec["cmdvx"][t] = cmd[:, 0]
                rec["tau2"][t] = (tau ** 2).sum(-1); rec["power"][t] = np.abs(tau * qd).sum(-1); rec["alive"][t] = alive
            self.obs_dict, reward, terminated, truncated = self.env.step(torch.from_numpy(action).to(self.env.device))[0:4]
            reward = reward.cpu().numpy(); reset = (terminated | truncated).cpu().numpy()
            tm = self.env.termination_manager; fall = np.zeros(self.N, bool)
            for tn in self.fall_terms:
                fall |= tm.get_term(tn).cpu().numpy()
            ret[alive] += reward[alive]
            if self.log_terms:
                tr = self.env.reward_manager._step_reward.cpu().numpy() * float(self.env.step_dt)
                term_sum[alive] += tr[alive]
                if t < self.chunk:
                    term_chunk[alive] += tr[alive]
            new = alive & reset; fell[new] = fall[new]; fall_step[new] = t; alive &= ~reset
        rows = []
        out = self.cfg.get("output", {}) or {}
        model_name = out.get("model_name") or self.cfg["model"]["checkpoint"]; cond = out.get("condition") or ""
        for e in np.where(valid)[0]:
            q = bp[e]
            rows.append({"model": model_name, "condition": cond, "speed": float(A["speed"][q]), "seed": int(A["seed"][q]), "step": int(A["step"][q]),
                         "group": grp[e], "cand": int(cid[e]), "label": self.cands[self.e_cand[e]][2], "clone": int(self.e_clone[e]),
                         "ret": float(ret[e]), "fell": int(fell[e]), "fall_step": int(fall_step[e]), "alive_steps": int(self.horizon if fall_step[e] < 0 else fall_step[e] + 1),
                         "anchor_ret": float(A["anchor_ret"][q]), "anchor_fell": int(A["anchor_fell"][q]), "anchor_len": int(A["anchor_len"][q]),
                         "is_event": int(A["is_event"][q]) if "is_event" in A else -1})
            if self.log_terms:
                rows[-1].update({f"t_{n}": f"{term_sum[e, k]:.4g}" for k, n in enumerate(self.term_names)})
                rows[-1].update({f"c_{n}": f"{term_chunk[e, k]:.4g}" for k, n in enumerate(self.term_names)})
        if rec is not None:
            v = np.where(valid)[0]
            meta = {"speed": np.asarray(A["speed"][bp[v]], np.float32), "seed": np.asarray(A["seed"][bp[v]]), "step": np.asarray(A["step"][bp[v]]),
                    "group": grp[v], "cand": cid[v], "clone": self.e_clone[v], "fell": fell[v], "fall_step": fall_step[v],
                    "is_event": np.asarray(A["is_event"][bp[v]]) if "is_event" in A else np.full(len(v), -1)}
            arrs = {k: np.swapaxes(x[:, v], 0, 1) for k, x in rec.items()}  # (rows, H, ...)
            np.savez_compressed(os.path.join(self.steps_dir, f"pts_{int(pts[0]):06d}.npz"), **arrs, **meta)
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


class FeatureSim(BranchSim):
    """Section 9.2 (mode=features): restore every branch point through the same path as branch mode (restore + first
    update), re-create the candidates' t = 0 chunks (A0 = generator mean, D_k = mean under the variant command,
    G_k = latent sample u_k with the branch run's rng rule) and record the world model's imagination for each chunk
    (rewards, values, latents, J-hat as the planner computes it) plus the branch-point history. No env steps.
    Output: branch.out npz."""

    def __init__(self, cfg):
        cfg["branch"]["groups"] = ["A0"]; cfg["branch"]["J"] = 1; cfg["branch"]["perturb"] = False; cfg["branch"]["csv"] = None
        super().__init__(cfg)
        assert self.gen is not None, "features mode needs branch.base=gen"
        self.cand_defs = [("A0", 0, "A0")] + [("D", k, f"D_{v[0]}") for k, v in enumerate(CMD_VARIANTS)] + [("G", k, f"G{k}") for k in range(N_GEN)]
        self.parts = []

    def run_round(self, pts):
        import jax.numpy as jnp
        bp = self.restore(pts); A = self.A; n = len(pts); c = self.controller
        cmd = self.cmd_manager.get_command("base_velocity").cpu().numpy().copy()
        c.set_fut_cmd(cmd)
        proprio, extero = self.get_obs(); c.update(proprio, extero, self.last_action)
        x = c._model_inputs()
        T, du, K = int(self.gen["T"]), int(self.gen["du"]), len(self.cand_defs)
        U = np.zeros((self.N, N_GEN, du), np.float32)
        for w in range(n):
            q = pts[w]; rng = np.random.default_rng(int(self.b["seed"]) + 1000003 * int(A["seed"][q]) + int(A["step"][q]) + 7 * int(round(10 * float(A["speed"][q]))))
            U[w] = np.clip(rng.normal(size=(N_GEN, du)), -2.5, 2.5)
        disc = np.asarray(c.discounts, np.float64); fdisc = float(c.final_discount)
        chunks = np.zeros((n, K, T, 12), np.float32); rew = np.zeros((n, K, T), np.float32); val = np.zeros((n, K, T), np.float32)
        jhat = np.zeros((n, K), np.float32); cand_cmd = np.zeros((n, K, 3), np.float32); cand_u = np.zeros((n, K, du), np.float32)
        zT = zmean = znow = None
        for k, (g, i, label) in enumerate(self.cand_defs):
            cm = cmd.copy()
            if g == "D":
                _, dim, v, how = CMD_VARIANTS[i]
                if how == "mul": cm[:, dim] *= v
                else: cm[:, dim] += v
            C = np.repeat(cm[:, None, :], T + 1, 1).astype(np.float32)
            u = U[:, i] if g == "G" else np.zeros((self.N, du), np.float32)
            chunk = gen_decode_b(self.gen, c.proprio_hist, c.acts_hist, C, u)
            xx = dict(x); xx["fut_cmds"] = jnp.asarray(C)
            z, lat, r, v_ = c._trace(xx, jnp.asarray(chunk, jnp.float32))
            z, lat, r, v_ = np.asarray(z), np.asarray(lat), np.asarray(r).reshape(self.N, T), np.asarray(v_).reshape(self.N, T)
            if zT is None:
                D = z.shape[-1]; znow = z[:n].copy(); zT = np.zeros((n, K, D), np.float32); zmean = np.zeros((n, K, D), np.float32)
            lat = lat.reshape(self.N, T, -1)
            chunks[:, k] = chunk[:n]; rew[:, k] = r[:n]; val[:, k] = v_[:n]; zT[:, k] = lat[:n, -1]; zmean[:, k] = lat[:n].mean(axis=1)
            jhat[:, k] = (r[:n] * disc).sum(-1) + v_[:n, -1] * fdisc
            cand_cmd[:, k] = cm[:n]; cand_u[:, k] = u[:n]
        self.parts.append({"speed": A["speed"][bp[:n]], "seed": A["seed"][bp[:n]], "step": A["step"][bp[:n]],
                           "anchor_ret": A["anchor_ret"][bp[:n]], "anchor_fell": A["anchor_fell"][bp[:n]],
                           "hist_p": c.proprio_hist[:n].copy(), "hist_a": c.acts_hist[:n].copy(), "cmd": cmd[:n].copy(), "z_now": znow,
                           "chunk": chunks, "rew": rew, "val": val, "jhat": jhat, "cand_cmd": cand_cmd, "cand_u": cand_u, "z_T": zT, "z_mean": zmean})
        return n

    def run(self):
        t0 = time.time(); n_rounds = int(np.ceil(len(self.idx) / self.W)); done = 0
        for r in range(n_rounds):
            if not sb.simulation_app.is_running():
                break
            done += self.run_round(self.idx[r * self.W:(r + 1) * self.W])
            el = time.time() - t0
            print(f"round {r + 1}/{n_rounds}: {done} points, {el / (r + 1):.1f} s/round", flush=True)
        out = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        out["labels"] = np.array([d[2] for d in self.cand_defs]); out["groups"] = np.array([d[0] for d in self.cand_defs])
        out["model"] = np.array(self.cfg["model"]["checkpoint"]); out["condition"] = np.array((self.cfg.get("output", {}) or {}).get("condition") or "")
        os.makedirs(os.path.dirname(os.path.abspath(self.b["out"])), exist_ok=True)
        np.savez(self.b["out"], **out)
        print(f"RESULT features: {len(out['step'])} branch points x {len(self.cand_defs)} candidates -> {self.b['out']}", flush=True)
        self.env.close(); sb.simulation_app.close()


@hydra.main(config_path=paths.get_config_dir(), config_name="branch_eval", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg["mode"] == "anchor":
        AnchorSim(cfg).run()
    elif cfg["mode"] == "branch":
        BranchSim(cfg).run()
    elif cfg["mode"] == "features":
        FeatureSim(cfg).run()
    else:
        raise ValueError(cfg["mode"])


if __name__ == "__main__":
    main()
