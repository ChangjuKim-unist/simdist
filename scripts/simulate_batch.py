"""Batched simulation evaluation: several robots (speed x seed) in one Isaac Sim instance.

    python scripts/simulate_batch.py model.checkpoint=go1_paper sim.terrain=plane \
        sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6 sim.total_steps=5000 \
        sim.episode_length_s=100 batch.speeds=[0.3,0.6,0.9] batch.seeds=[11,12,13] \
        output.csv=results/x.csv output.condition=combo_hard --headless

Per robot it reports the whole-run reward per step, the number of resets (any cause),
the number of falls (base/thigh contact or bad orientation) and time-outs separately,
and the first episode's length / reward / forward progress. Falls per 1000 steps =
num_falls / (total_steps / 1000). The ground slab is 100 m wide so no robot reaches an edge. The planner is the batched MPPI
(simdist/control/mppi_batch.py); with policy.rl_run set, the exported RL expert acts.
"""

import argparse
import os
import sys
import types as pytypes

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args


args = parse_args()
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".40")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import csv
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import torch
from tqdm import tqdm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from simdist.utils import paths, model as model_utils, config
from simdist.control.mppi_batch import BatchedMppiController
from simdist.rl.envs import get_sim_env_cfg
from simdist.rl import conditions


class BatchSim:
    def __init__(self, cfg: dict):
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        self.speeds = [float(v) for v in cfg["batch"]["speeds"]]
        self.seeds = [int(s) for s in cfg["batch"]["seeds"]]
        self.pairs = [(v, s) for v in self.speeds for s in self.seeds]
        self.N = len(self.pairs)
        self.total_steps = cfg["sim"]["total_steps"]
        self.reset_steps = cfg["sim"]["reset_steps"]

        ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), cfg["model"]["checkpoint"])
        model, model_cfg, _ = model_utils.load_model_from_ckpt(ckpt_dir, cfg["model"]["step"])
        # design doc 23: weight interpolation theta = (1 - alpha) * theta_base + alpha * theta_model over all parameters
        # (only the dynamics differ after a SimDist adaptation; scalers are identical)
        if cfg["model"].get("interp_from"):
            alpha = float(cfg["model"]["interp_alpha"])
            base_m, _, _ = model_utils.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), cfg["model"]["interp_from"]))
            s_m, s_b = nnx.state(model, nnx.Param), nnx.state(base_m, nnx.Param)
            nnx.update(model, jax.tree.map(lambda a, b: (1.0 - alpha) * b + alpha * a, s_m, s_b))
            print(f"Interpolated parameters: {1.0 - alpha:.2f} x {cfg['model']['interp_from']} + {alpha:.2f} x {cfg['model']['checkpoint']}", flush=True)
        # design doc 22 §12: take the reward and/or value head of another checkpoint (attribution of a head refit)
        _head_mods = {"reward": ("reward_emb", "reward", "reward_dec"), "value": ("value_emb", "value", "value_dec")}
        for part, src in ((cfg["model"].get("heads_from") or {}) or {}).items():
            if not src:
                continue
            assert part in _head_mods, f"model.heads_from.{part}: only reward / value"
            other, _, _ = model_utils.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), src))
            for name in _head_mods[part]:
                nnx.update(getattr(model, name), nnx.state(getattr(other, name)))
            print(f"{part} head taken from {src}", flush=True)
        self.proprio_obs_names = config.proprio_obs_names_from_sys_config(model_cfg["system"])
        self.controller = BatchedMppiController(
            model, model_cfg, cfg["control"], self.N, [s for _, s in self.pairs]
        )
        # design doc 29 G1: extra models scored together with this one during the first seconds after every reset
        rb = dict(cfg["control"].get("robust", {}) or {})
        self.robust_hold = int(rb.get("hold", 50)); self.robust_decay = int(rb.get("decay", 50))
        if rb.get("enabled", False):
            for name in list(rb.get("models", []) or []):
                m_extra, _, _ = model_utils.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), str(name)))
                self.controller.robust_models.append(m_extra)
                print(f"robust scoring: + {name} (hold {self.robust_hold}, decay {self.robust_decay})", flush=True)
        # design doc 33: policy start and online one-step error switch (config/control/mppi.yaml)
        self.ps_steps = int(cfg["control"].get("policy_start_steps", 0) or 0)
        sw = dict(cfg["control"].get("err_switch", {}) or {})
        self.sw_on = bool(sw.get("enabled", False)); self.sw_window = int(sw.get("window", 50)); self.sw_fallback = str(sw.get("fallback", "model"))
        self.sw_theta = float(sw["theta"]) if self.sw_on else float("inf")
        if self.sw_on and self.sw_fallback == "model":
            assert self.controller.robust_models and rb.get("mode", "min") == "select", \
                "control.err_switch.fallback=model needs control.robust.enabled=true control.robust.mode=select"
        self.err_buf = np.zeros((self.N, self.sw_window), np.float32); self.err_cnt = np.zeros(self.N, np.int64)
        self.prev_pred = None
        self.mode_steps = np.zeros(4, np.int64)  # controlled steps as 0 planner, 1 policy start, 2 fallback model, 3 fallback policy
        if self.ps_steps or self.sw_on:
            print(f"design 33: policy start {self.ps_steps} steps, error switch {'on' if self.sw_on else 'off'} "
                  f"(theta {self.sw_theta:.4f}, window {self.sw_window}, fallback {self.sw_fallback})", flush=True)
        # design doc 13: oracle context (psi[:3] := standardized true physics of this run) and
        # logging of the model's own online physics estimate (output.context_csv, every 25 steps)
        self.ctx_rows = []
        self.ctx_csv = (cfg.get("output", {}) or {}).get("context_csv")
        # design doc 14: per-step trace of the model's prediction for the executed plan vs. what happened
        # (output.trace_npz): z_t, predicted latents/rewards/values, realized reward, fall, reset flags.
        self.trace_path = (cfg.get("output", {}) or {}).get("trace_npz")
        self.trace = {k: [] for k in ("z", "pred_latents", "pred_rewards", "pred_values", "action", "reward", "fall", "reset", "resetting")} if self.trace_path else None
        if self.trace is not None and ((cfg.get("policy", {}) or {}).get("rl_run") or (cfg.get("prior", {}) or {}).get("student")):
            print("WARNING: output.trace_npz is only recorded on the planner/prior path (not rl_run / student_act)", flush=True)
        if cfg["sim"].get("context_oracle", False):
            assert getattr(model, "context", None) is not None, "sim.context_oracle needs a model with a context slot"
            xi = [float(cfg["sim"].get("motor_strength", 1.0)), float(cfg["sim"]["friction"]), float(cfg["sim"]["add_mass"])]
            xi_s = np.asarray(model.scale_xi(np.asarray(xi, np.float32)), np.float32)
            self.controller.extra_inputs["context_override"] = jnp.asarray(np.tile(xi_s[None], (self.N, 1)))
            print(f"Oracle context: xi={xi} -> standardized {xi_s.tolist()}")
        # design doc 21 gate 0: a fixed, fitted context (standardized psi[:3]) instead of the inferred one
        if cfg["sim"].get("context_override_xi") is not None:
            assert getattr(model, "context", None) is not None, "sim.context_override_xi needs a model with a context slot"
            assert not cfg["sim"].get("context_oracle", False), "context_oracle and context_override_xi are exclusive"
            xi_s = np.asarray(list(cfg["sim"]["context_override_xi"]), np.float32)
            assert xi_s.shape == (3,), "sim.context_override_xi must be 3 standardized values"
            self.controller.extra_inputs["context_override"] = jnp.asarray(np.tile(xi_s[None], (self.N, 1)))
            print(f"Fitted context override (standardized): {xi_s.tolist()}")
        # design doc 24: a fixed context xi for the FiLM-modulated dynamics (JSON with "xi_fit"; blank = xi 0)
        if cfg["sim"].get("xi_json"):
            import json as _json
            assert getattr(model, "film_proj", None) is not None, "sim.xi_json needs a model with a FiLM context"
            xi = np.asarray(_json.load(open(cfg["sim"]["xi_json"]))["xi_fit"], np.float32)
            self.controller.extra_inputs["xi"] = jnp.asarray(np.tile(xi[None], (self.N, 1)))
            print(f"FiLM context xi from {cfg['sim']['xi_json']}: {np.round(xi, 3).tolist()}", flush=True)
        # optional student prior (design doc 4): replaces the policy-head prior samples, or
        # acts alone when policy.use_as == "student_act"
        self.student = None
        st_cfg = cfg.get("prior", {}) or {}
        if st_cfg.get("student"):
            from simdist.control.student_prior import StudentPrior

            self.student = StudentPrior(st_cfg["student"])
            self.controller.external_prior = self.student
            print(f"Prior samples from student {st_cfg['student']}")
        if st_cfg.get("head_from"):  # policy head of another checkpoint as the prior (design doc 8, broad head on adapted dynamics)
            head_model, _, _ = model_utils.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), st_cfg["head_from"]))
            self.controller.external_prior = lambda x, _m=head_model: _m.inference(x)["actions"]
            print(f"Prior samples from the policy head of {st_cfg['head_from']}")
        self.rl_policy = None
        pol_cfg = cfg.get("policy", {}) or {}
        # use_as: "act" = the expert acts instead of the planner; "init" = the expert's
        # current action (repeated over the horizon) initializes the MPPI mean each step
        # (oracle test M4 of design doc 3; needs control.init_mode=external)
        self.expert_use = str(pol_cfg.get("use_as", "act"))
        if pol_cfg.get("rl_run"):
            from simdist.utils.torch import get_actor_critic_from_iteration

            self.rl_policy, _ = get_actor_critic_from_iteration(pol_cfg["rl_run"], int(pol_cfg["iteration"]), "cuda:0")
            print(f"Acting with RL policy {pol_cfg['rl_run']} iteration {pol_cfg['iteration']}")

        env_cfg = get_sim_env_cfg(model_cfg["system"]["name"])
        env_cfg.scene.num_envs = self.N
        env_cfg.commands.base_velocity.forward_vel = self.speeds[0]
        env_cfg.commands.base_velocity.kx = cfg["task"]["kx"]
        env_cfg.commands.base_velocity.ky = cfg["task"]["ky"]
        env_cfg.commands.base_velocity.k_heading = cfg["task"]["k_heading"]
        terr = cfg["sim"]["terrain"]; diff = cfg["sim"]["terrain_difficulty"]
        env_cfg.scene.terrain.terrain_generator.seed = cfg["sim"]["seed"]
        if cfg["sim"].get("episode_length_s") is not None:
            env_cfg.episode_length_s = float(cfg["sim"]["episode_length_s"])
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.size = (15.0, 15.0)
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (diff, diff)
        terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {terr: terrains[terr]}
        # Ground must outlast the run: with the default 20 m border the robot walked off the slab at x = 27.5 m
        # (first episode always ended at ~26 m) and that edge reset was counted as a fall (code review 2026-11-03).
        env_cfg.scene.terrain.terrain_generator.border_width = 100.0
        # deterministic physics: seed the env (gain/joint-friction draws) and switch those per-reset randomizations off,
        # so "nominal" means the nominal Kp/Kd/joint friction the condition is defined with
        env_cfg.seed = int(cfg["sim"]["seed"])
        for ev in ("joint_stiffness_and_damping", "joint_friction"):
            if hasattr(env_cfg.events, ev):
                setattr(env_cfg.events, ev, None)
        env_cfg.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        fric = cfg["sim"]["friction"]
        env_cfg.events.physics_material.params["static_friction_range"] = (fric, fric)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (fric, fric)
        rest = cfg["sim"]["restitution"]
        env_cfg.events.physics_material.params["restitution_range"] = (rest, rest)
        mass = cfg["sim"]["add_mass"]
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (mass, mass)
        # Weak motor = scaled torque cap (effort_limit only). This is the same mechanism as the training DR
        # (simdist/rl/go1.py randomize_motor_strength), the replay identification and the hardware bridge cap
        # (|tau| <= scale * 23.7). Scaling saturation_effort as well (old code) made eval "0.4" harsher than DR 0.4
        # whenever the joint moved (torque-speed curve shrunk too).
        motor_strength = cfg["sim"].get("motor_strength", 1.0)
        if motor_strength != 1.0:
            act_cfg = env_cfg.scene.robot.actuators["base_legs"]
            act_cfg.effort_limit = act_cfg.effort_limit * motor_strength
        conditions.apply_to_cfg(env_cfg, cfg["sim"])  # design doc 34

        self.env = ManagerBasedRLEnv(env_cfg)
        self.cond = conditions.Conditions(self.env, cfg["sim"])  # design doc 34: leg strength, action delay, tow force
        # design doc 13 P4: switch every robot's torque cap at sim.motor_switch_step to
        # sim.motor_switch_strength x nominal (online tracking of the context estimate)
        self.motor_switch_step = cfg["sim"].get("motor_switch_step")
        if self.motor_switch_step is not None:
            act = self.env.scene["robot"].actuators["base_legs"]
            self._effort_nominal = act.effort_limit.clone() / float(motor_strength)
        self.cmd_manager = self.env.command_manager
        self._patch_per_env_speed()
        # design doc 25: sim.cmd_ramp_steps = R (blank = off). The command's target position stops during the
        # post-reset phase and its speed then rises linearly from 0 to the robot's speed over R steps.
        self.cmd_ramp_steps = cfg["sim"].get("cmd_ramp_steps")
        self.cmd_term = self.cmd_manager.get_term("base_velocity")
        self.obs_dict, _ = self.env.reset()
        self._init_obs_mask(pol_cfg)

        self.zero_action = np.zeros((self.N, 12), np.float32)
        proprio, extero = self.get_obs()
        self.controller.reset(range(self.N), proprio, extero, self.zero_action, np.zeros((self.N, 3)))
        print("Compiling the batched planner...", flush=True)
        self.controller.run_control()
        print("Planner ready.", flush=True)

        self.last_action = self.zero_action.copy()
        self.steps_since_reset = np.zeros(self.N, np.int64)
        # design doc 35: integral term on the model-side command (sim.cmd_ki, blank = off)
        self.cmd_ki = float(cfg["sim"].get("cmd_ki") or 0.0)
        self.cmd_i_max = float(cfg["sim"].get("cmd_i_max") or 0.5)
        self.cmd_I = np.zeros((self.N, 3), np.float32)
        self.cmd_box_lo = np.array([-0.5, -0.5, -1.0], np.float32); self.cmd_box_hi = np.array([1.5, 0.5, 1.0], np.float32)
        if self.cmd_ki:
            print(f"integral command term on: ki {self.cmd_ki}, |I| <= {self.cmd_i_max}", flush=True)
        self.run_reward = np.zeros(self.N)
        self.num_resets = np.zeros(self.N, np.int64)
        self.num_falls = np.zeros(self.N, np.int64)     # base/thigh contact or bad orientation
        self.num_timeouts = np.zeros(self.N, np.int64)  # episode time limit
        tm = self.env.termination_manager
        self.fall_terms = [t for t in ("base_contact", "bad_orientation") if t in tm.active_terms]
        self.first_done = np.zeros(self.N, bool)
        self.first_len = np.zeros(self.N, np.int64)
        self.first_reward = np.zeros(self.N)
        self.start_x = np.full(self.N, np.nan)
        self.end_x = np.full(self.N, np.nan)
        self.total = 0
        # output.terms_csv: per-robot mean of every reward term over the run (design 16 supplement: reward decomposition)
        self.terms_csv = (self.cfg.get("output", {}) or {}).get("terms_csv")
        self.term_names = list(self.env.reward_manager.active_terms)
        self.term_sums = np.zeros((self.N, len(self.term_names)))
        # output.falls_csv (design doc 25): one row per fall with the number of steps since the last reset
        self.falls_csv = (self.cfg.get("output", {}) or {}).get("falls_csv")
        self.fall_events = []
        # output.diag_npz (design doc 26): per-step diagnostics. Body state, foot slip, executed action vs the policy head's
        # action on the same inputs, the model's score of both plans, one-step latent prediction error, realized reward.
        self.diag_path = (self.cfg.get("output", {}) or {}).get("diag_npz")
        self.diag = None
        if self.diag_path:
            robot = self.env.scene["robot"]; cs = self.env.scene.sensors["contact_forces"]
            ids_r, names_r = robot.find_bodies(".*_foot"); ids_c, names_c = cs.find_bodies(".*_foot")
            order = [names_c.index(n) for n in names_r]  # sensor feet in the robot's foot order
            self.foot_names = list(names_r); self.foot_ids = list(ids_r); self.foot_ids_cs = [ids_c[k] for k in order]
            self.diag = {k: [] for k in ("since", "resetting", "cmd", "lin_vel", "ang_vel", "grav", "height", "foot_speed", "contact",
                                         "act", "act_pol", "R_exec", "V_exec", "R_pol", "V_pol", "r0_exec", "lat_err",
                                         # design doc 36: elite ESS fraction and policy-sample weight share (first / last iteration)
                                         "ess_first", "ess_last", "pol_first", "pol_last",
                                         "reward", "fall", "reset", "terms",
                                         # design doc 27: joint velocity, applied torque (articulation joint order), foot force
                                         "joint_vel", "torque", "foot_force",
                                         # design doc 33: who acted (0 planner, 1 policy start, 2 fallback model, 3 fallback policy)
                                         # and the window mean of the one-step error that the switch saw at this step
                                         "mode", "err_mean",
                                         # design doc 35: the command handed to the model (outer-loop command + integral term)
                                         "cmd_model")}
            self.joint_names = list(robot.joint_names)
            print(f"diagnostics on: feet {self.foot_names}", flush=True)

    def _patch_per_env_speed(self):
        """Give every robot its own forward speed (the command term has one scalar)."""
        term = self.cmd_manager.get_term("base_velocity")
        fv = torch.tensor(np.array([v for v, _ in self.pairs], np.float32), device=self.env.device)

        def _compute_errors(self, env_ids):
            env_ids = torch.as_tensor(list(env_ids) if not torch.is_tensor(env_ids) else env_ids, device=fv.device)
            curr_pos = self.robot.data.root_pos_w[env_ids]
            curr_heading = self.robot.data.heading_w[env_ids]
            curr_vx = self.robot.data.root_lin_vel_b[env_ids, 0]
            delta_pos = curr_pos - self.init_pos[env_ids]
            heading_error = math_utils.wrap_to_pi(curr_heading - self.init_heading[env_ids])
            cos_h = torch.cos(-self.init_heading[env_ids]); sin_h = torch.sin(-self.init_heading[env_ids])
            dx = cos_h * delta_pos[:, 0] - sin_h * delta_pos[:, 1]
            dy = sin_h * delta_pos[:, 0] + cos_h * delta_pos[:, 1]
            return {"x_error": self.x_des[env_ids] - dx, "y_error": -dy, "yaw_error": -heading_error,
                    "vx_error": fv[env_ids] - curr_vx}

        def _update_command(self):
            # design doc 25: per-robot speed scale for a gentle start (None = original behaviour)
            scale = getattr(self, "speed_scale", None)
            self.x_des += (fv if scale is None else fv * scale) * self._env.step_dt
            errors = self._compute_errors(range(self.num_envs))
            self.vel_command_b[:, 0] = self.cfg.kx * errors["x_error"]
            if vx_max is not None:  # design doc 26 stage 2: cap the forward command (no catch-up surge above the speed)
                self.vel_command_b[:, 0] = torch.clamp(self.vel_command_b[:, 0], max=vx_max)
            self.vel_command_b[:, 1] = self.cfg.ky * errors["y_error"]
            self.vel_command_b[:, 2] = self.cfg.kyaw * errors["yaw_error"]

        vx_max = self.cfg["sim"].get("cmd_vx_max")
        vx_max = None if vx_max is None else float(vx_max)
        term._compute_errors = pytypes.MethodType(_compute_errors, term)
        term._update_command = pytypes.MethodType(_update_command, term)

    def _diag_pre(self, plan, resetting, cmd, tr, lat_err, mode, err_mean):
        """Design doc 26: record the state at step t and the model's view of the executed plan and of the policy head's plan."""
        c = self.controller; d = self.diag
        plan_exec = np.where(resetting[:, None, None], 0.0, plan).astype(np.float32)
        disc = np.asarray(c.discounts, np.float32); fd = float(c.final_discount)
        if tr is None:  # design doc 35: the RL expert acted; no model view of the plan
            base = plan_exec
            pr = br = np.full((self.N, len(disc)), np.nan, np.float32); pv = bv = np.full((self.N, 1), np.nan, np.float32)
        else:
            base = np.asarray(c.last_base_actions, np.float32)
            z, pl, pr, pv = tr  # model.checkpoint's view of the executed plan (computed in step)
            _, _, br, bv = c.trace(base)
        d["cmd_model"].append(np.asarray(getattr(self, "cmd_model", cmd), np.float32).astype(np.float16))
        d["mode"].append(mode.copy()); d["err_mean"].append(err_mean.copy())
        robot = self.env.scene["robot"].data; cs = self.env.scene.sensors["contact_forces"].data
        g = lambda t: t.detach().cpu().numpy()  # noqa: E731
        foot_speed = np.linalg.norm(g(robot.body_lin_vel_w[:, self.foot_ids, :2]), axis=-1)
        foot_force = np.linalg.norm(g(cs.net_forces_w[:, self.foot_ids_cs]), axis=-1)
        contact = foot_force > 1.0
        d["joint_vel"].append(g(robot.joint_vel).astype(np.float16)); d["torque"].append(g(robot.applied_torque).astype(np.float16))
        d["foot_force"].append(foot_force.astype(np.float16))
        d["since"].append(self.steps_since_reset.astype(np.int16).copy()); d["resetting"].append(resetting.copy())
        d["cmd"].append(cmd.astype(np.float16)); d["lin_vel"].append(g(robot.root_lin_vel_b).astype(np.float16))
        d["ang_vel"].append(g(robot.root_ang_vel_b).astype(np.float16)); d["grav"].append(g(robot.projected_gravity_b).astype(np.float16))
        d["height"].append(g(robot.root_pos_w[:, 2]).astype(np.float16))
        d["foot_speed"].append(foot_speed.astype(np.float16)); d["contact"].append(contact)
        d["act"].append(plan_exec[:, 0].astype(np.float16)); d["act_pol"].append(base[:, 0].astype(np.float16))
        d["R_exec"].append((pr @ disc).astype(np.float32)); d["V_exec"].append((pv[:, -1] * fd).astype(np.float32))
        d["R_pol"].append((br @ disc).astype(np.float32)); d["V_pol"].append((bv[:, -1] * fd).astype(np.float32))
        d["r0_exec"].append(pr[:, 0].astype(np.float32)); d["lat_err"].append(lat_err)
        md = getattr(c, "last_mppi_diag", None)  # design doc 36 (None when the planner did not run: prior_only / RL expert)
        for k, src, i in (("ess_first", "ess", 0), ("ess_last", "ess", -1), ("pol_first", "pol", 0), ("pol_last", "pol", -1)):
            v = np.full(self.N, np.nan, np.float32) if (md is None or tr is None) else np.asarray(md[src][i], np.float32)
            d[k].append(np.where(resetting, np.nan, v).astype(np.float16))

    def _init_obs_mask(self, pol_cfg):
        """Design doc 38: hide privileged observation terms of the RL expert.  policy.mask_terms = names of terms in the
        'policy' observation group; each is replaced by its nominal reference (mid-walk mean per commanded speed, from a run
        with policy.mask_record=<npz>, loaded with policy.mask_ref=<npz>).  History entries of a term are replaced too."""
        self.mask_terms = list(pol_cfg.get("mask_terms") or [])
        self.mask_record = pol_cfg.get("mask_record")
        self.mask_idx = None
        if self.rl_policy is None or not (self.mask_terms or self.mask_record):
            return
        om = self.env.observation_manager
        names = list(om.active_terms["policy"])
        dims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
        off = np.concatenate([[0], np.cumsum(dims)]).astype(int)
        self.obs_slices = {n: (int(off[i]), int(off[i + 1])) for i, n in enumerate(names)}
        D = int(off[-1]); assert self.obs_dict["policy"].shape[-1] == D, (self.obs_dict["policy"].shape, D)
        print("design 38 expert obs terms (dim):", {n: b - a for n, (a, b) in self.obs_slices.items()}, flush=True)
        unknown = [t for t in self.mask_terms if t not in self.obs_slices]
        assert not unknown, f"unknown policy obs terms {unknown}; available {names}"
        if self.mask_record:
            self.mask_sum = np.zeros((self.N, D)); self.mask_cnt = np.zeros(self.N)
        if self.mask_terms:
            z = np.load(str(pol_cfg["mask_ref"]))
            assert list(z["names"]) == names, "mask reference was recorded with a different observation layout"
            ref_sp, ref_mean = z["speeds"].astype(np.float64), z["mean"].astype(np.float32)
            rows = [int(np.argmin(np.abs(ref_sp - v))) for v, _ in self.pairs]
            assert all(abs(ref_sp[r] - v) < 1e-6 for r, (v, _) in zip(rows, self.pairs)), "speed missing in the mask reference"
            idx = np.concatenate([np.arange(*self.obs_slices[t]) for t in self.mask_terms])
            self.mask_idx = torch.tensor(idx, device="cuda:0")
            self.mask_val = torch.tensor(ref_mean[rows][:, idx], device="cuda:0")
            print(f"design 38: masking {self.mask_terms} ({len(idx)} of {D} expert inputs) with the nominal reference", flush=True)

    def _expert_obs(self, resetting):
        obs = self.obs_dict["policy"].to("cuda:0")
        if self.mask_record:
            mid = (self.steps_since_reset >= 300) & ~resetting  # same mid-walk definition as the task score S
            if mid.any():
                o = obs.cpu().numpy().astype(np.float64)
                self.mask_sum[mid] += o[mid]; self.mask_cnt[mid] += 1
        if self.mask_idx is not None:
            obs = obs.clone(); obs[:, self.mask_idx] = self.mask_val
        return obs

    def _save_mask_record(self):
        sp = np.array([v for v, _ in self.pairs], np.float64); us = np.unique(sp)
        mean = np.stack([self.mask_sum[sp == u].sum(0) / max(1.0, self.mask_cnt[sp == u].sum()) for u in us])
        names = list(self.obs_slices)
        os.makedirs(os.path.dirname(os.path.abspath(str(self.mask_record))), exist_ok=True)
        np.savez(str(self.mask_record), speeds=us, mean=mean, names=np.array(names),
                 slices=np.array([self.obs_slices[n] for n in names]), counts=np.array([self.mask_cnt[sp == u].sum() for u in us]))
        print(f"design 38 mask reference written to {self.mask_record}: speeds {us.tolist()}, steps {[int(self.mask_cnt[sp == u].sum()) for u in us]}", flush=True)

    def get_obs(self):
        proprio =np.concatenate([self.obs_dict["obs"][n].cpu().numpy() for n in self.proprio_obs_names], axis=-1)
        extero = self.obs_dict["obs"]["height_scan"].cpu().numpy()
        return proprio.astype(np.float32), extero.astype(np.float32)

    def root_x(self):
        return self.env.scene["robot"].data.root_pos_w[:, 0].cpu().numpy().copy()

    def run(self):
        pbar = tqdm(total=self.total_steps, desc="Batched simulation", unit="step")
        while simulation_app.is_running() and self.total < self.total_steps:
            self.step()
            pbar.update(1)
        self.report()
        self.env.close()
        simulation_app.close()

    def step(self):
        if self.motor_switch_step is not None and self.total == int(self.motor_switch_step):
            act = self.env.scene["robot"].actuators["base_legs"]
            act.effort_limit[:] = self._effort_nominal * float(self.cfg["sim"]["motor_switch_strength"])
            print(f"motor strength switched to {self.cfg['sim']['motor_switch_strength']} at step {self.total}", flush=True)
        resetting = self.steps_since_reset < self.reset_steps
        cmd = self.cmd_manager.get_command("base_velocity").cpu().numpy().copy()
        cmd[resetting] = 0.0
        cmd_model = cmd
        if self.cmd_ki:
            self.cmd_I[resetting] = 0.0
            cmd_model = np.clip(cmd + self.cmd_I, self.cmd_box_lo, self.cmd_box_hi).astype(np.float32)
            cmd_model[resetting] = 0.0
        self.cmd_model = cmd_model
        self.controller.set_fut_cmd(cmd_model)
        proprio, extero = self.get_obs()
        self.controller.update(proprio, extero, self.last_action)
        if self.ctx_csv and self.total % 25 == 0:
            est = self.controller.context_estimate()
            if est is not None:
                out = self.cfg.get("output", {}) or {}
                for i, (v, s) in enumerate(self.pairs):
                    self.ctx_rows.append({
                        "model": out.get("model_name") or self.cfg["model"]["checkpoint"], "condition": out.get("condition") or "",
                        "speed": v, "seed": s, "step": self.total, "resetting": int(resetting[i]),
                        "motor_hat": float(est[i, 0]), "friction_hat": float(est[i, 1]), "mass_hat": float(est[i, 2]),
                    })
        if self.rl_policy is not None and self.expert_use == "act":
            with torch.no_grad():
                action = self.rl_policy(self._expert_obs(resetting)).cpu().numpy()
            if self.diag is not None:  # design doc 35: body state, reward terms and falls of the expert (no model fields)
                plan = np.repeat(np.asarray(action, np.float32)[:, None, :], self.controller.T, axis=1)
                self._diag_pre(plan, resetting, cmd, None, np.full(self.N, np.nan, np.float32),
                               np.zeros(self.N, np.int8), np.full(self.N, np.nan, np.float32))
        elif self.student is not None and self.expert_use == "student_act":
            # the student alone: first step of its action sequence, receding horizon
            action = np.asarray(self.student(self.controller._model_inputs()))[:, 0]
        else:
            if self.rl_policy is not None and self.expert_use == "init":
                with torch.no_grad():
                    a_exp = self.rl_policy(self.obs_dict["policy"].to("cuda:0")).cpu().numpy().astype(np.float32)
                mean = np.repeat(a_exp[:, None, :], self.controller.T, axis=1)
                self.controller.state = self.controller.state.replace(mean=jnp.asarray(mean))
            # design doc 33: policy start (first ps_steps controlled steps) and the online one-step error switch
            since_ctrl = self.steps_since_reset - self.reset_steps  # 0 at the first step after the post-reset phase
            pol_phase = (since_ctrl < self.ps_steps) & (~resetting)
            err_mean = np.where(self.err_cnt >= self.sw_window, self.err_buf.sum(1) / self.sw_window, np.nan).astype(np.float32)
            alarm = np.zeros(self.N, bool)
            if self.sw_on:
                alarm = (np.nan_to_num(err_mean, nan=-np.inf) > self.sw_theta) & (~pol_phase) & (~resetting)
                if self.sw_fallback == "model":
                    self.controller.robust_gamma = alarm.astype(np.float32)
            elif self.controller.robust_models:
                k = (self.steps_since_reset - self.reset_steps - self.robust_hold).astype(np.float32)  # < 0 inside the hold
                self.controller.robust_gamma = np.clip(1.0 - np.maximum(k, 0.0) / max(self.robust_decay, 1), 0.0, 1.0).astype(np.float32)
            plan = self.controller.run_control()
            use_pol = pol_phase | (alarm & (self.sw_fallback == "policy"))
            if use_pol.any():
                # execute the policy head's plan and restart the planner's mean from it
                plan = np.where(use_pol[:, None, None], np.asarray(self.controller.last_base_actions, np.float32), plan).astype(np.float32)
                self.controller.state = self.controller.state.replace(mean=jnp.asarray(plan))
            mode = np.where(pol_phase, 1, np.where(alarm, 2 if self.sw_fallback == "model" else 3, 0)).astype(np.int8)
            self.mode_steps += np.bincount(mode[~resetting], minlength=4)[:4]
            action = plan[:, 0]
            tr = None; lat_err = np.full(self.N, np.nan, np.float32)
            if self.diag is not None or self.sw_on:
                # one-step latent error of model.checkpoint for the action that was really executed at the previous step
                tr = self.controller.trace(np.where(resetting[:, None, None], 0.0, plan).astype(np.float32))
                if self.prev_pred is not None:
                    lat_err = np.linalg.norm(tr[0].astype(np.float32) - self.prev_pred, axis=-1).astype(np.float32)
                self.prev_pred = tr[1][:, 0].astype(np.float32).copy()  # predicts the latent of the next step
                ok = np.isfinite(lat_err) & (~resetting)
                self.err_buf[ok, :-1] = self.err_buf[ok, 1:]; self.err_buf[ok, -1] = lat_err[ok]; self.err_cnt[ok] += 1
            if self.trace is not None:
                # the plan that is really executed: zeros during the post-reset phase (code review 2026-11-06)
                plan_exec = np.where(resetting[:, None, None], 0.0, plan).astype(np.float32)
                z, pl, pr, pv = self.controller.trace(plan_exec)
                self.trace["z"].append(z.astype(np.float16)); self.trace["pred_latents"].append(pl.astype(np.float16))
                self.trace["pred_rewards"].append(pr.astype(np.float32)); self.trace["pred_values"].append(pv.astype(np.float32))
                self.trace["action"].append(plan_exec[:, 0].astype(np.float16))
            if self.diag is not None:
                self._diag_pre(plan, resetting, cmd, tr, lat_err, mode, err_mean)
        action = np.where(resetting[:, None], 0.0, action).astype(np.float32)
        self.last_action = action

        first_start = (~self.first_done) & np.isnan(self.start_x) & (~resetting)
        if first_start.any():
            self.start_x[first_start] = self.root_x()[first_start]

        if self.cmd_ramp_steps:
            k = self.steps_since_reset - self.reset_steps + 1  # 1 at the first step after the post-reset phase
            scale = np.clip(k / float(self.cmd_ramp_steps), 0.0, 1.0).astype(np.float32)
            self.cmd_term.speed_scale = torch.from_numpy(scale).to(self.env.device)

        self.obs_dict, reward, terminated, truncated = self.env.step(self.cond.before_step(torch.from_numpy(action).to(self.env.device)))[0:4]
        self.cond.after_step(terminated | truncated)
        reward = reward.cpu().numpy(); reset = (terminated | truncated).cpu().numpy()
        if self.cmd_ki:  # design doc 35: integrate the gap between the outer-loop command and the measured motion
            rd = self.env.scene["robot"].data
            meas = np.concatenate([rd.root_lin_vel_b[:, :2].cpu().numpy(), rd.root_ang_vel_b[:, 2:3].cpu().numpy()], axis=1)
            upd = (~resetting) & (~reset)
            self.cmd_I[upd] += self.cmd_ki * float(self.env.step_dt) * (cmd[upd] - meas[upd])
            np.clip(self.cmd_I, -self.cmd_i_max, self.cmd_i_max, out=self.cmd_I)
            self.cmd_I[reset] = 0.0
        tm = self.env.termination_manager
        fall = np.zeros(self.N, bool)
        for t in self.fall_terms:
            fall |= tm.get_term(t).cpu().numpy()
        timeout = tm.get_term("time_out").cpu().numpy() if "time_out" in tm.active_terms else np.zeros(self.N, bool)
        self.num_falls += fall; self.num_timeouts += timeout & ~fall
        if self.trace is not None and len(self.trace["z"]) > len(self.trace["reward"]):
            self.trace["reward"].append(reward.astype(np.float32)); self.trace["fall"].append(fall.copy())
            self.trace["reset"].append(reset.copy()); self.trace["resetting"].append(resetting.copy())
        if self.diag is not None and len(self.diag["act"]) > len(self.diag["reward"]):
            d = self.diag
            d["reward"].append(reward.astype(np.float32)); d["fall"].append(fall.copy()); d["reset"].append(reset.copy())
            d["terms"].append((self.env.reward_manager._step_reward.cpu().numpy() * float(self.env.step_dt)).astype(np.float16))
        if reset.any():
            if self.prev_pred is not None:
                self.prev_pred[reset] = np.nan  # no one-step prediction across a reset
            self.err_cnt[reset] = 0  # the switch's error window starts again in the new episode
        self.total += 1
        self.steps_since_reset += 1
        if self.falls_csv and fall.any():
            for i in np.where(fall)[0]:
                self.fall_events.append((int(i), self.total, int(self.steps_since_reset[i])))
        active = ~resetting
        self.run_reward[active] += reward[active]
        if self.terms_csv:
            self.term_sums[active] += self.env.reward_manager._step_reward.cpu().numpy()[active] * float(self.env.step_dt)
        fe = ~self.first_done
        self.first_reward[fe] += reward[fe]
        self.first_len[fe] += 1
        x = self.root_x()
        keep = fe & (~resetting) & (~reset)
        self.end_x[keep] = x[keep]

        if reset.any():
            ids = np.where(reset)[0]
            self.num_resets[ids] += 1
            self.first_done[ids] = True
            self.steps_since_reset[ids] = 0
            proprio, extero = self.get_obs()
            self.controller.reset(ids, proprio, extero, self.zero_action, np.zeros((self.N, 3)))

    def report(self):
        rows = []
        out = self.cfg.get("output", {}) or {}
        model_name = out.get("model_name") or self.cfg["model"]["checkpoint"]
        cond = out.get("condition") or ""
        for i, (v, s) in enumerate(self.pairs):
            prog = float(self.end_x[i] - self.start_x[i]) if not (np.isnan(self.end_x[i]) or np.isnan(self.start_x[i])) else 0.0
            row = {
                "model": model_name, "condition": cond, "speed": v, "seed": s,
                "run_reward_per_step": float(self.run_reward[i] / max(self.total, 1)),
                "num_resets": int(self.num_resets[i]),
                "first_episode_length": int(self.first_len[i]),
                "first_episode_reward": float(self.first_reward[i]),
                "forward_progress": prog,
                "num_falls": int(self.num_falls[i]),
                "num_timeouts": int(self.num_timeouts[i]),
            }
            rows.append(row)
            print("RESULT", row, flush=True)
        if getattr(self.controller, "gate_steps", 0):
            print(f"consistency gate rate: {self.controller.gated_steps / self.controller.gate_steps:.4f} (design doc 7)")
        if self.ps_steps or self.sw_on:
            ms = self.mode_steps / max(1, int(self.mode_steps.sum()))
            print(f"design 33 modes (share of controlled steps): planner {ms[0]:.3f}, policy start {ms[1]:.3f}, "
                  f"fallback model {ms[2]:.3f}, fallback policy {ms[3]:.3f}", flush=True)
        if self.trace is not None and self.trace["z"]:
            n = min(len(v) for v in self.trace.values())
            arrs = {k: np.stack(v[:n]) for k, v in self.trace.items()}
            arrs.update(speeds=np.array([v for v, _ in self.pairs], np.float32), seeds=np.array([s for _, s in self.pairs], np.int64),
                        model=np.array(model_name), condition=np.array(cond), motor_switch_step=np.array(self.motor_switch_step if self.motor_switch_step is not None else -1))
            os.makedirs(os.path.dirname(os.path.abspath(self.trace_path)), exist_ok=True)
            np.savez(self.trace_path, **arrs)
            print(f"trace written to {self.trace_path}: {n} steps, pred_latents {arrs['pred_latents'].shape}")
        if getattr(self, "mask_record", None):
            self._save_mask_record()
        if self.diag is not None and self.diag["act"]:
            n = min(len(v) for v in self.diag.values())
            arrs = {k: np.stack(v[:n]) for k, v in self.diag.items()}
            arrs.update(speeds=np.array([v for v, _ in self.pairs], np.float32), seeds=np.array([s for _, s in self.pairs], np.int64),
                        model=np.array(model_name), condition=np.array(cond), term_names=np.array(self.term_names),
                        foot_names=np.array(self.foot_names), joint_names=np.array(self.joint_names), reset_steps=np.array(self.reset_steps),
                        discount=np.array(float(self.controller.ctrl_cfg["discount"])), T=np.array(self.controller.T))
            try:  # design doc 27: total robot mass per robot (cost of transport)
                arrs["robot_mass"] = self.env.scene["robot"].root_physx_view.get_masses().cpu().numpy().sum(-1).astype(np.float32)
            except Exception as e:  # noqa: BLE001
                print(f"robot mass not recorded: {e}", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(self.diag_path)), exist_ok=True)
            np.savez_compressed(self.diag_path, **arrs)
            print(f"diagnostics written to {self.diag_path}: {n} steps", flush=True)
        if self.terms_csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.terms_csv)), exist_ok=True)
            new = not os.path.exists(self.terms_csv)
            with open(self.terms_csv, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["model", "condition", "speed", "seed"] + self.term_names)
                for i, (v, s) in enumerate(self.pairs):
                    w.writerow([model_name, cond, v, s] + [float(x) for x in self.term_sums[i] / max(self.total, 1)])
        if self.falls_csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.falls_csv)), exist_ok=True)
            new = not os.path.exists(self.falls_csv)
            with open(self.falls_csv, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["model", "condition", "speed", "seed", "step", "steps_since_reset"])
                for i, step, since in self.fall_events:
                    v, s = self.pairs[i]
                    w.writerow([model_name, cond, v, s, step, since])
            print(f"falls written to {self.falls_csv} ({len(self.fall_events)} rows)")
        if out.get("csv"):
            path = out["csv"]
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(rows)
        if self.ctx_csv and self.ctx_rows:
            os.makedirs(os.path.dirname(os.path.abspath(self.ctx_csv)), exist_ok=True)
            new = not os.path.exists(self.ctx_csv)
            with open(self.ctx_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.ctx_rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(self.ctx_rows)
            print(f"context estimates written to {self.ctx_csv} ({len(self.ctx_rows)} rows)")


@hydra.main(config_path=paths.get_config_dir(), config_name="simulate_batch", version_base=None)
def main(cfg: DictConfig):
    BatchSim(OmegaConf.to_container(cfg, resolve=True)).run()


if __name__ == "__main__":
    main()
