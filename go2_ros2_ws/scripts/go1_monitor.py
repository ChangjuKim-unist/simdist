#!/usr/bin/env python3
"""Live summary of the Go1 low-level state for bench tests.

Prints joint angles, motor temperatures, foot forces, IMU orientation, battery
and the wireless remote at 2 Hz. Run inside the container after
``source scripts/setup.sh``, with the bridge running:

    python3 scripts/go1_monitor.py
"""

import math

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, WirelessController

LEGS = ["FR", "FL", "RR", "RL"]


def quat_to_rpy_deg(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


class Monitor(Node):
    def __init__(self):
        super().__init__("go1_monitor")
        self.state = None
        self.remote = None
        self.count = 0
        self.create_subscription(LowState, "/lowstate", self.on_state, 1)
        self.create_subscription(WirelessController, "/wirelesscontroller", self.on_remote, 1)
        self.create_timer(0.5, self.show)

    def on_state(self, msg):
        self.state = msg
        self.count += 1

    def on_remote(self, msg):
        self.remote = msg

    def show(self):
        s = self.state
        if s is None:
            print("waiting for /lowstate ...")
            return
        print(f"\n--- /lowstate msgs: {self.count}  battery: {s.bms_state.soc}%  "
              f"tick: {s.tick}")
        print("leg   hip     thigh   calf   | temp hip/thigh/calf | foot_force")
        for i, leg in enumerate(LEGS):
            m = [s.motor_state[3 * i + j] for j in range(3)]
            print(f"{leg}  {m[0].q:7.3f} {m[1].q:7.3f} {m[2].q:7.3f} | "
                  f"{m[0].temperature:3d} / {m[1].temperature:3d} / {m[2].temperature:3d} C   | "
                  f"{s.foot_force[i]:5d}")
        q = s.imu_state.quaternion
        r, p, y = quat_to_rpy_deg(*q)
        g = s.imu_state.gyroscope
        print(f"imu roll {r:6.1f}  pitch {p:6.1f}  yaw {y:6.1f} deg   "
              f"gyro {g[0]:5.2f} {g[1]:5.2f} {g[2]:5.2f} rad/s")
        if self.remote is not None:
            rm = self.remote
            print(f"remote lx {rm.lx:5.2f} ly {rm.ly:5.2f} rx {rm.rx:5.2f} ry {rm.ry:5.2f} keys {rm.keys:#06x}")
        self.count = 0


def main():
    rclpy.init()
    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
