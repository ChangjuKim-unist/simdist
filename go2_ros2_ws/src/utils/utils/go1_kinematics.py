"""Leg kinematics of the Unitree Go1.

Joint order follows the Unitree convention used throughout the workspace:
FR, FL, RR, RL legs, each (hip, thigh, calf). Link lengths and offsets are
from go1_description/xacro/const.xacro.
"""

import numpy as np

# Hip joint positions in the trunk frame (x forward, y left, z up).
LEG_OFFSET_X = 0.1881
LEG_OFFSET_Y = 0.04675
# Lateral offset from the hip joint to the thigh joint (thigh_offset).
HIP_LENGTH = 0.08
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

LEG_NAMES = ["FR", "FL", "RR", "RL"]
# (x sign, y sign) of each hip joint; the lateral offsets are mirrored on the right side.
_LEG_SIGNS = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]], dtype=np.float64)


def foot_position(leg: int, q: np.ndarray) -> np.ndarray:
    """Position of the foot of ``leg`` in the trunk frame for joint angles ``q`` (hip, thigh, calf)."""
    sx, sy = _LEG_SIGNS[leg]
    q_hip, q_thigh, q_calf = q
    # thigh and calf hang below the thigh joint in the leg's sagittal plane
    x = -THIGH_LENGTH * np.sin(q_thigh) - CALF_LENGTH * np.sin(q_thigh + q_calf)
    z = -THIGH_LENGTH * np.cos(q_thigh) - CALF_LENGTH * np.cos(q_thigh + q_calf)
    y = sy * HIP_LENGTH
    # hip roll about x
    c, s = np.cos(q_hip), np.sin(q_hip)
    y, z = c * y - s * z, s * y + c * z
    return np.array([sx * LEG_OFFSET_X + x, sy * LEG_OFFSET_Y + y, z])


def foot_jacobian(leg: int, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """3x3 Jacobian of ``foot_position`` with respect to the leg's joint angles."""
    J = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = eps
        J[:, j] = (foot_position(leg, q + dq) - foot_position(leg, q - dq)) / (2 * eps)
    return J


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (body -> world) from a (w, x, y, z) quaternion, as reported by the Unitree IMU."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
