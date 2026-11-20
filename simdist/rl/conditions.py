"""Design doc 34 (docs/design34_condition_screen_ko.md): extra test conditions for the closed-loop simulators
(scripts/simulate_go2.py and scripts/simulate_batch.py). All default to off.

  sim.leg_strength    [FL, FR, RL, RR] scale of the torque cap of each leg's three joints (on top of sim.motor_strength)
  sim.act_delay_steps the executed joint targets lag the commanded ones by this many control steps (20 ms each);
                      the controller, the model history and the logs keep the commanded action (the delay is unknown)
  sim.tow_force       horizontal force [N] on the trunk against the heading, a dragged load: it rises linearly with the
                      forward speed and saturates at 0.1 m/s (no pull while standing or moving backwards)
  sim.slope_deg       uphill slope along +x (robots start with yaw 0) emulated by tilting gravity on the flat ground;
                      the projected gravity shows the tilt, the height scan stays flat
"""
import math

import torch

LEGS = ("FL", "FR", "RL", "RR")
TOW_SPEED_SAT = 0.1  # m/s


def apply_to_cfg(env_cfg, sim_cfg) -> None:
    """Conditions that must be set before the environment is created."""
    slope = float(sim_cfg.get("slope_deg") or 0.0)
    if slope != 0.0:
        g = 9.81
        env_cfg.sim.gravity = (-g * math.sin(math.radians(slope)), 0.0, -g * math.cos(math.radians(slope)))
        print(f"condition: slope {slope} deg uphill along +x (gravity {env_cfg.sim.gravity})", flush=True)


class Conditions:
    """Conditions applied on the created environment and at every control step."""

    def __init__(self, env, sim_cfg):
        self.env = env
        self.robot = env.scene["robot"]
        n = env.num_envs
        leg = sim_cfg.get("leg_strength")
        if leg is not None:
            leg = [float(v) for v in leg]
            assert len(leg) == 4, "sim.leg_strength = [FL, FR, RL, RR]"
            act = self.robot.actuators["base_legs"]
            for prefix, scale in zip(LEGS, leg):
                idx = [i for i, name in enumerate(act.joint_names) if name.startswith(prefix + "_")]
                assert len(idx) == 3, f"expected three joints for leg {prefix}, got {idx} from {act.joint_names}"
                act.effort_limit[:, idx] = act.effort_limit[:, idx] * scale
            print(f"condition: leg torque caps scaled by {dict(zip(LEGS, leg))} -> {act.effort_limit[0].tolist()}", flush=True)
        self.delay = int(sim_cfg.get("act_delay_steps") or 0)
        self.buf = [torch.zeros((n, 12), device=env.device) for _ in range(self.delay)]
        if self.delay:
            print(f"condition: executed action delayed by {self.delay} control step(s)", flush=True)
        self.tow = float(sim_cfg.get("tow_force") or 0.0)
        if self.tow:
            from simdist.rl.go1 import BASE_BODY  # noqa: F401  (Go1 trunk; falls back to the root body)
            ids, names = self.robot.find_bodies(BASE_BODY)
            self.body_ids = ids if ids else [0]
            print(f"condition: tow force {self.tow} N on {names if ids else 'root body'}", flush=True)

    def before_step(self, action: torch.Tensor) -> torch.Tensor:
        """Sets the external force for this control step and returns the action that is really executed."""
        if self.tow:
            from isaaclab.utils import math as math_utils
            d = self.robot.data
            yaw = d.heading_w
            v_fwd = d.root_lin_vel_b[:, 0]
            mag = self.tow * torch.clamp(v_fwd / TOW_SPEED_SAT, 0.0, 1.0)
            f_w = torch.stack((-mag * torch.cos(yaw), -mag * torch.sin(yaw), torch.zeros_like(mag)), dim=-1)
            f_b = math_utils.quat_rotate_inverse(d.root_quat_w, f_w)  # the wrench is given in the body frame
            self.robot.set_external_force_and_torque(f_b[:, None, :], torch.zeros_like(f_b)[:, None, :], body_ids=self.body_ids)
        if not self.delay:
            return action
        self.buf.append(action.clone())
        return self.buf.pop(0)

    def after_step(self, reset) -> None:
        """reset: boolean per environment (tensor or array); a reset robot starts with an empty delay line."""
        if self.delay:
            mask = torch.as_tensor(reset, device=self.env.device, dtype=torch.bool)
            for b in self.buf:
                b[mask] = 0.0


# ---- design doc 35: the same conditions inside RL training (specialist experts) ----------------------------------------
def from_env():
    """SIMDIST_COND = JSON dict with the sim.* keys above plus motor_strength and friction (blank = no condition)."""
    import json
    import os

    v = os.environ.get("SIMDIST_COND")
    return json.loads(v) if v else {}


def apply_train_cfg(env_cfg, cond: dict) -> None:
    """Before gym.make: uniform torque cap (as sim.motor_strength in the evaluators), fixed foot friction, slope."""
    ms = float(cond.get("motor_strength") or 1.0)
    if ms != 1.0:
        act_cfg = env_cfg.scene.robot.actuators["base_legs"]
        act_cfg.effort_limit = act_cfg.effort_limit * ms
        print(f"condition: motor strength {ms}", flush=True)
    if cond.get("friction") is not None:
        f = float(cond["friction"])
        env_cfg.events.physics_material.params["static_friction_range"] = (f, f)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (f, f)
        print(f"condition: foot friction fixed at {f}", flush=True)
    apply_to_cfg(env_cfg, cond)


try:
    import gymnasium as _gym

    class ConditionEnvWrapper(_gym.Wrapper):
        """Wraps the Isaac Lab env (before RslRlVecEnvWrapper) so that leg caps, tow force and delay act during training."""

        def __init__(self, env, cond: dict):
            super().__init__(env)
            self.cond = Conditions(env.unwrapped, cond)

        def step(self, action):
            out = self.env.step(self.cond.before_step(action))
            self.cond.after_step(out[2] | out[3])
            return out
except ImportError:  # the closed-loop simulators do not need it
    pass
