"""Unitree Go1 variants of the Go2 environments.

The Go1 and Go2 share the same 12 joint names and leg layout, so the Go2
environment configuration is reused as-is and only the robot-specific parts
are swapped out: the robot asset, the base body name (``trunk`` instead of
``base``) and the actuator limits.
"""

from isaaclab.actuators import DCMotorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

from simdist.rl.go2 import (
    Go2EnvCfg,
    Go2PlayEnvCfg,
    Go2RecordEnvCfg,
    Go2SimEnvCfg,
    ISAAC_DEFAULT_JOINT_ANGLES,
    JOINT_DAMPING,
    JOINT_NAMES,
    JOINT_STIFFNESS,
)

BASE_BODY = "trunk"
# IMU position in the trunk frame, from the imu_joint of go1_description's URDF
IMU_OFFSET = (-0.01592, -0.06659, -0.00617)
# Go1 motor limits taken from the spec sheet (same values Isaac Lab uses for
# the Go1 actuator net). The Go2 environment's joint stiffness and damping
# randomization requires a PD-based motor model, so a DC motor model is used
# instead of the actuator net.
EFFORT_LIMIT = 23.7
VELOCITY_LIMIT = 30.0


def _apply_go1(cfg: Go2EnvCfg):
    """Replace the Go2-specific parts of an environment configuration with Go1 ones."""
    # robot asset with the same joint ordering, default pose, and PD gains as the Go2
    init_state = UNITREE_GO1_CFG.init_state.replace(joint_pos=ISAAC_DEFAULT_JOINT_ANGLES)
    dc_motor_cfg = DCMotorCfg(
        joint_names_expr=JOINT_NAMES,
        effort_limit=EFFORT_LIMIT,
        saturation_effort=EFFORT_LIMIT,
        velocity_limit=VELOCITY_LIMIT,
        stiffness=JOINT_STIFFNESS,
        damping=JOINT_DAMPING,
        friction=0.0,
    )
    cfg.scene.robot = UNITREE_GO1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=init_state,
        actuators={"base_legs": dc_motor_cfg},
    )

    # everything attached to or referring to the base body
    cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY}"
    cfg.scene.imu_body.prim_path = f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY}"
    cfg.scene.imu_body.offset.pos = IMU_OFFSET
    cfg.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY
    cfg.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY
    if hasattr(cfg.observations.policy, "base_mass"):
        cfg.observations.policy.base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=BASE_BODY, preserve_order=True
        )
    # the Go1 has no head bodies
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = [
        BASE_BODY,
        ".*_thigh",
    ]


class Go1EnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_go1(self)


class Go1PlayEnvCfg(Go2PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_go1(self)


class Go1RecordEnvCfg(Go2RecordEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_go1(self)


class Go1SimEnvCfg(Go2SimEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_go1(self)
