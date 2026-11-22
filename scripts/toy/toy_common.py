"""Design 40 toy: shared pieces (Playground Go1, conditions, state features, metrics).

Conditions only change actuator torque limits (forcerange), like design 34's
'torque limit only' weak-leg option. Actuator order: FR, FL, RR, RL x (hip, thigh, calf).
"""
import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry

ENV = "Go1JoystickFlatTerrain"
LEG = {"FR": 0, "FL": 1, "RR": 2, "RL": 3}
CONDITIONS = ("nominal", "legR50", "mot50", "kneeRR0", "kneeFR0")
V_CMD = 0.6          # forward command (m/s), as designs 34-38
RAMP_S = 1.0         # cautious start: command ramps 0 -> V_CMD over 1 s
EVAL_FROM = 100      # score steps 100..T-1 (after 2 s)
FALL_UPZ = 0.5       # torso up-vector z < cos(60 deg) = fall (SimDist bad_orientation)
FALL_Z = 0.12        # or torso height below 12 cm
TRACK_SIGMA = 0.25   # Playground tracking kernel


def load_env():
    env = registry.load(ENV)
    return env


def condition_forcerange(env, cond):
    fr = np.array(env.mj_model.actuator_forcerange, dtype=np.float32).copy()
    if cond == "nominal":
        pass
    elif cond == "mot50":
        fr *= 0.5
    elif cond == "legR50":          # rear-left leg, three joints, 50 % torque limit
        fr[3 * LEG["RL"]:3 * LEG["RL"] + 3] *= 0.5
    elif cond == "kneeRR0":         # rear-right knee (calf) motor gives no torque
        fr[3 * LEG["RR"] + 2] = 0.0
    elif cond == "kneeFR0":         # front-right knee motor gives no torque
        fr[3 * LEG["FR"] + 2] = 0.0
    else:
        raise ValueError(cond)
    return fr


def condition_model(env, cond):
    """mjx model of the true (condition) dynamics."""
    fr = condition_forcerange(env, cond)
    return env.mjx_model.replace(actuator_forcerange=jnp.asarray(fr))


def v_des(t):
    return V_CMD * jnp.clip(t / RAMP_S, 0.0, 1.0)


class Feat:
    """State features read from mjx data (no noise)."""

    def __init__(self, env):
        m = env.mj_model
        self.torso = env._torso_body_id
        self.feet_site = jnp.asarray(env._feet_site_id)
        self.feet_geom = jnp.asarray(env._feet_geom_id)
        self.foot_vel_adr = jnp.asarray(env._foot_linvel_sensor_adr)
        self.found_adr = jnp.asarray([m.sensor_adr[s] for s in env._feet_floor_found_sensor])
        self.env = env
        self.mass = float(env._torso_mass)
        self.q_lo = jnp.asarray(env._soft_lowers)
        self.q_hi = jnp.asarray(env._soft_uppers)
        self.kp = float(m.actuator_gainprm[0, 0])
        self.kd = float(m.dof_damping[6])
        self.default_q = jnp.asarray(env._default_pose)

    def upz(self, d):
        return d.xmat[self.torso][2, 2]

    def yaw(self, d):
        w, x, y, z = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def rollpitch(self, d):
        w, x, y, z = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1, 1))
        return roll, pitch

    def local_linvel(self, d):
        return self.env.get_local_linvel(d)

    def contact(self, d):
        return d.sensordata[self.found_adr] > 0

    def feet_pos(self, d):
        return d.site_xpos[self.feet_site]

    def feet_vel(self, d):
        return d.sensordata[self.foot_vel_adr]

    def feet_fn(self, d):
        """Normal contact force per foot (pyramidal cone: sum of the pyramid rows)."""
        c = d._impl.contact
        addr = c.efc_address
        dim = c.dim
        nrow = jnp.where(dim == 1, 1, 2 * (dim - 1))
        j = jnp.arange(8)
        idx = jnp.clip(addr[:, None] + j[None, :], 0, d._impl.efc_force.shape[0] - 1)
        f = jnp.take(d._impl.efc_force, idx) * (j[None, :] < nrow[:, None]) * (addr[:, None] >= 0)
        fn = f.sum(-1)                                     # per contact
        g = c.geom                                         # (ncon, 2)
        hit = (g[:, :1] == self.feet_geom[None, :]) | (g[:, 1:2] == self.feet_geom[None, :])
        return (hit * fn[:, None]).sum(0)                  # per foot

    def fallen(self, d):
        return (self.upz(d) < FALL_UPZ) | (d.qpos[2] < FALL_Z)


def step_log(F, d):
    """Per-step record from the true environment."""
    v = F.local_linvel(d)
    roll, pitch = F.rollpitch(d)
    fp = F.feet_pos(d)
    fv = F.feet_vel(d)
    con = F.contact(d)
    return dict(
        vx=v[0], vy=v[1], wz=d.qvel[5], z=d.qpos[2], roll=roll, pitch=pitch, yaw=F.yaw(d),
        upz=F.upz(d), contact=con, foot_h=fp[:, 2], slip=jnp.linalg.norm(fv[:, :2], axis=-1) * con,
        tau=d.actuator_force, q=d.qpos[7:], qd=d.qvel[6:], fallen=F.fallen(d),
    )


def summarise(log, ctrl=None):
    """log: dict of arrays (E, T, ...). Returns per-episode metrics."""
    T = log["vx"].shape[1]
    fallen = np.maximum.accumulate(np.asarray(log["fallen"]).astype(bool), axis=1)
    alive = ~fallen
    sl = slice(EVAL_FROM, T)
    track = np.exp(-((log["vx"] - V_CMD) ** 2 + log["vy"] ** 2) / TRACK_SIGMA) * alive
    out = {
        "J": track[:, sl].mean(1),                                   # 0..1, 1 = perfect speed tracking
        "fell": fallen[:, -1].astype(float),
        "falls_per_1000": fallen[:, -1] / T * 1000,
        "vx_mean": np.where(alive[:, sl], log["vx"][:, sl], np.nan).mean(1) if alive[:, sl].any() else np.full(len(alive), np.nan),
        "yaw_drift_deg": np.degrees(np.abs(log["yaw"][:, -1] - log["yaw"][:, 0])),
        "roll_sd_deg": np.degrees(np.nanstd(np.where(alive[:, sl], log["roll"][:, sl], np.nan), 1)),
        "pitch_sd_deg": np.degrees(np.nanstd(np.where(alive[:, sl], log["pitch"][:, sl], np.nan), 1)),
        "tau2": np.nanmean(np.where(alive[:, sl, None], log["tau"][:, sl] ** 2, np.nan), (1, 2)),
        "slip": np.nanmean(np.where(alive[:, sl, None], log["slip"][:, sl], np.nan), (1, 2)),
        "duty": np.nanmean(np.where(alive[:, sl, None], log["contact"][:, sl], np.nan), 1),  # (E, 4)
    }
    if ctrl is not None:
        jerk = np.diff(np.asarray(ctrl), n=2, axis=1)[:, EVAL_FROM:]
        out["ctrl_jerk"] = np.abs(jerk).mean((1, 2))
    return out


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
