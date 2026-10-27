"""Batched simulation evaluation: several robots (speed x seed) in one Isaac Sim instance.

    python scripts/simulate_batch.py model.checkpoint=go1_paper sim.terrain=plane \
        sim.friction=0.3 sim.add_mass=3.0 sim.motor_strength=0.6 sim.total_steps=5000 \
        sim.episode_length_s=100 batch.speeds=[0.3,0.6,0.9] batch.seeds=[11,12,13] \
        output.csv=results/x.csv output.condition=combo_hard --headless

Per robot it reports the whole-run reward per step, the number of resets (falls and
time-outs), and the first episode's length / reward / forward progress, exactly as
scripts/simulate_go2.py does for one robot. The planner is the batched MPPI
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
import jax.numpy as jnp
import torch
from tqdm import tqdm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from simdist.utils import paths, model as model_utils, config
from simdist.control.mppi_batch import BatchedMppiController
from simdist.rl.envs import get_sim_env_cfg


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
        self.proprio_obs_names = config.proprio_obs_names_from_sys_config(model_cfg["system"])
        self.controller = BatchedMppiController(
            model, model_cfg, cfg["control"], self.N, [s for _, s in self.pairs]
        )
        # optional student prior (design doc 4): replaces the policy-head prior samples, or
        # acts alone when policy.use_as == "student_act"
        self.student = None
        st_cfg = cfg.get("prior", {}) or {}
        if st_cfg.get("student"):
            from simdist.control.student_prior import StudentPrior

            self.student = StudentPrior(st_cfg["student"])
            self.controller.external_prior = self.student
            print(f"Prior samples from student {st_cfg['student']}")
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
        env_cfg.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        fric = cfg["sim"]["friction"]
        env_cfg.events.physics_material.params["static_friction_range"] = (fric, fric)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (fric, fric)
        rest = cfg["sim"]["restitution"]
        env_cfg.events.physics_material.params["restitution_range"] = (rest, rest)
        mass = cfg["sim"]["add_mass"]
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (mass, mass)
        motor_strength = cfg["sim"].get("motor_strength", 1.0)
        if motor_strength != 1.0:
            act_cfg = env_cfg.scene.robot.actuators["base_legs"]
            act_cfg.effort_limit = act_cfg.effort_limit * motor_strength
            if getattr(act_cfg, "saturation_effort", None) is not None:
                act_cfg.saturation_effort = act_cfg.saturation_effort * motor_strength

        self.env = ManagerBasedRLEnv(env_cfg)
        self.cmd_manager = self.env.command_manager
        self._patch_per_env_speed()
        self.obs_dict, _ = self.env.reset()

        self.zero_action = np.zeros((self.N, 12), np.float32)
        proprio, extero = self.get_obs()
        self.controller.reset(range(self.N), proprio, extero, self.zero_action, np.zeros((self.N, 3)))
        print("Compiling the batched planner...", flush=True)
        self.controller.run_control()
        print("Planner ready.", flush=True)

        self.last_action = self.zero_action.copy()
        self.steps_since_reset = np.zeros(self.N, np.int64)
        self.run_reward = np.zeros(self.N)
        self.num_resets = np.zeros(self.N, np.int64)
        self.first_done = np.zeros(self.N, bool)
        self.first_len = np.zeros(self.N, np.int64)
        self.first_reward = np.zeros(self.N)
        self.start_x = np.full(self.N, np.nan)
        self.end_x = np.full(self.N, np.nan)
        self.total = 0

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
            self.x_des += fv * self._env.step_dt
            errors = self._compute_errors(range(self.num_envs))
            self.vel_command_b[:, 0] = self.cfg.kx * errors["x_error"]
            self.vel_command_b[:, 1] = self.cfg.ky * errors["y_error"]
            self.vel_command_b[:, 2] = self.cfg.kyaw * errors["yaw_error"]

        term._compute_errors = pytypes.MethodType(_compute_errors, term)
        term._update_command = pytypes.MethodType(_update_command, term)

    def get_obs(self):
        proprio = np.concatenate([self.obs_dict["obs"][n].cpu().numpy() for n in self.proprio_obs_names], axis=-1)
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
        resetting = self.steps_since_reset < self.reset_steps
        cmd = self.cmd_manager.get_command("base_velocity").cpu().numpy().copy()
        cmd[resetting] = 0.0
        self.controller.set_fut_cmd(cmd)
        proprio, extero = self.get_obs()
        self.controller.update(proprio, extero, self.last_action)
        if self.rl_policy is not None and self.expert_use == "act":
            with torch.no_grad():
                action = self.rl_policy(self.obs_dict["policy"].to("cuda:0")).cpu().numpy()
        elif self.student is not None and self.expert_use == "student_act":
            # the student alone: first step of its action sequence, receding horizon
            action = np.asarray(self.student(self.controller._model_inputs()))[:, 0]
        else:
            if self.rl_policy is not None and self.expert_use == "init":
                with torch.no_grad():
                    a_exp = self.rl_policy(self.obs_dict["policy"].to("cuda:0")).cpu().numpy().astype(np.float32)
                mean = np.repeat(a_exp[:, None, :], self.controller.T, axis=1)
                self.controller.state = self.controller.state.replace(mean=jnp.asarray(mean))
            action = self.controller.run_control()[:, 0]
        action = np.where(resetting[:, None], 0.0, action).astype(np.float32)
        self.last_action = action

        first_start = (~self.first_done) & np.isnan(self.start_x) & (~resetting)
        if first_start.any():
            self.start_x[first_start] = self.root_x()[first_start]

        self.obs_dict, reward, terminated, truncated = self.env.step(torch.from_numpy(action).to(self.env.device))[0:4]
        reward = reward.cpu().numpy(); reset = (terminated | truncated).cpu().numpy()
        self.total += 1
        self.steps_since_reset += 1
        active = ~resetting
        self.run_reward[active] += reward[active]
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
            }
            rows.append(row)
            print("RESULT", row, flush=True)
        if getattr(self.controller, "gate_steps", 0):
            print(f"consistency gate rate: {self.controller.gated_steps / self.controller.gate_steps:.4f} (design doc 7)")
        if out.get("csv"):
            path = out["csv"]
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(rows)


@hydra.main(config_path=paths.get_config_dir(), config_name="simulate_batch", version_base=None)
def main(cfg: DictConfig):
    BatchSim(OmegaConf.to_container(cfg, resolve=True)).run()


if __name__ == "__main__":
    main()
