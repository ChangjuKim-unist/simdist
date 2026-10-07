#!/usr/bin/env python3
"""Elevation vector for flat ground without a lidar.

The elevation mapping stack publishes, for each cell of the height-scan grid,
the terrain height relative to the body origin in a gravity-aligned frame. On
flat ground every cell is simply ``-body_height``. The body height is measured
from the legs: for each foot in contact, the foot position from leg kinematics
rotated into the gravity-aligned frame gives the ground height below the body.
"""

import threading

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowState

from utils import go1_kinematics as kin


class FlatElevationNode(Node):
    def __init__(self):
        super().__init__("flat_elevation")
        self.declare_parameter("topics.lowstate", "/lowstate")
        self.declare_parameter("topics.elevation_vec", "/elevation_vec")
        self.declare_parameter("flat_elevation.rate", 50.0)
        # height scan grid of the simulation: size [2.0, 1.4] m at 0.1 m resolution
        self.declare_parameter("flat_elevation.num_cells", 21 * 15)
        self.declare_parameter("flat_elevation.contact_force_threshold", 20.0)
        # body height used until a foot touches the ground
        self.declare_parameter("flat_elevation.default_body_height", 0.30)
        self.declare_parameter("flat_elevation.alpha", 0.2)

        lowstate_topic = self.get_parameter("topics.lowstate").value
        elevation_topic = self.get_parameter("topics.elevation_vec").value
        self.rate = self.get_parameter("flat_elevation.rate").value
        self.num_cells = self.get_parameter("flat_elevation.num_cells").value
        self.contact_threshold = self.get_parameter(
            "flat_elevation.contact_force_threshold"
        ).value
        self.body_height = self.get_parameter("flat_elevation.default_body_height").value
        self.alpha = self.get_parameter("flat_elevation.alpha").value

        self.lock = threading.Lock()
        self.latest_state = None

        self.elevation_pub = self.create_publisher(Float32MultiArray, elevation_topic, 10)
        self.create_subscription(
            LowState,
            lowstate_topic,
            self.lowstate_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_timer(
            1.0 / self.rate, self.publish, callback_group=MutuallyExclusiveCallbackGroup()
        )

    def lowstate_callback(self, msg: LowState):
        with self.lock:
            self.latest_state = msg

    def publish(self):
        with self.lock:
            msg = self.latest_state
        if msg is not None:
            q = np.array([msg.motor_state[i].q for i in range(12)])
            R = kin.quat_wxyz_to_matrix(np.array(msg.imu_state.quaternion))
            heights = []
            for leg in range(4):
                if msg.foot_force[leg] < self.contact_threshold:
                    continue
                p_world = R @ kin.foot_position(leg, q[3 * leg : 3 * leg + 3])
                heights.append(-p_world[2])
            if heights:
                self.body_height = (
                    self.alpha * float(np.mean(heights)) + (1 - self.alpha) * self.body_height
                )

        out = Float32MultiArray()
        out.data = [-self.body_height] * self.num_cells
        self.elevation_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FlatElevationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
