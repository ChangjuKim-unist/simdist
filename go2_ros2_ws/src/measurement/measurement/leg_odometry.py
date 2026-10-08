#!/usr/bin/env python3
"""Base state estimate from leg kinematics, IMU and foot contact (no lidar or mocap).

The observer needs the base linear velocity, angular velocity and orientation
from an odometry message. The Go2 setup gets these from Point-LIO or motion
capture. This node produces the same message for a robot on flat ground from
the low-level state alone: for every foot in contact the body velocity is
``v = -(J dq + w x p)`` with ``p`` the foot position in the body frame, ``J``
the leg Jacobian, ``dq`` the joint velocities and ``w`` the gyro rate. The
estimate is averaged over contact feet and low-pass filtered. Position is the
integral of the velocity and drifts, which only affects visualization.
"""

import threading

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from unitree_go.msg import LowState

from utils import go1_kinematics as kin


class LegOdometryNode(Node):
    def __init__(self):
        super().__init__("leg_odometry")
        self.declare_parameter("topics.lowstate", "/lowstate")
        self.declare_parameter(
            "topics.state_transform_and_filtered", "/state_transform_and_filtered"
        )
        self.declare_parameter("frames.world", "world")
        self.declare_parameter("frames.body", "body")
        self.declare_parameter("leg_odometry.rate", 100.0)
        # foot_force (raw sensor units) above which a foot counts as in contact
        self.declare_parameter("leg_odometry.contact_force_threshold", 20.0)
        # feet whose force sensor is used for contact detection (FR, FL, RR, RL); set
        # an entry to false for a broken sensor
        self.declare_parameter("leg_odometry.use_foot", [True, True, True, True])
        # raw foot_force reading of each unloaded foot (sensor zero offset), subtracted
        # before comparing with the threshold
        self.declare_parameter("leg_odometry.foot_force_offset", [0.0, 0.0, 0.0, 0.0])
        # low-pass factor on the velocity estimate (1.0 = no filtering)
        self.declare_parameter("leg_odometry.alpha", 0.3)

        lowstate_topic = self.get_parameter("topics.lowstate").value
        odom_topic = self.get_parameter("topics.state_transform_and_filtered").value
        self.world_frame = self.get_parameter("frames.world").value
        self.body_frame = self.get_parameter("frames.body").value
        self.rate = self.get_parameter("leg_odometry.rate").value
        self.contact_threshold = self.get_parameter(
            "leg_odometry.contact_force_threshold"
        ).value
        self.use_foot = list(self.get_parameter("leg_odometry.use_foot").value)
        self.foot_force_offset = np.array(
            self.get_parameter("leg_odometry.foot_force_offset").value, dtype=float
        )
        self.alpha = self.get_parameter("leg_odometry.alpha").value

        self.lock = threading.Lock()
        self.latest_state = None
        self.lin_vel = np.zeros(3)
        self.position = np.zeros(3)
        self.last_time = None

        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
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
        if msg is None:
            return

        q = np.array([msg.motor_state[i].q for i in range(12)])
        dq = np.array([msg.motor_state[i].dq for i in range(12)])
        quat = np.array(msg.imu_state.quaternion)  # (w, x, y, z)
        gyro = np.array(msg.imu_state.gyroscope)
        foot_force = np.array(msg.foot_force) - self.foot_force_offset

        velocities = []
        for leg in range(4):
            if not self.use_foot[leg] or foot_force[leg] < self.contact_threshold:
                continue
            q_leg = q[3 * leg : 3 * leg + 3]
            dq_leg = dq[3 * leg : 3 * leg + 3]
            p = kin.foot_position(leg, q_leg)
            J = kin.foot_jacobian(leg, q_leg)
            # a stance foot is stationary in the world, so the body moves opposite to it
            velocities.append(-(J @ dq_leg + np.cross(gyro, p)))

        if velocities:
            self.lin_vel = self.alpha * np.mean(velocities, axis=0) + (1 - self.alpha) * self.lin_vel
        else:
            # airborne: no information, decay the estimate
            self.lin_vel *= 1 - self.alpha

        now = self.get_clock().now()
        R = kin.quat_wxyz_to_matrix(quat)
        if self.last_time is not None:
            dt = (now - self.last_time).nanoseconds * 1e-9
            if 0.0 < dt < 1.0:
                self.position += R @ self.lin_vel * dt
        self.last_time = now

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.body_frame
        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = float(self.position[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        # twist in the body frame, matching the mocap and LIO odometry used for the Go2
        odom.twist.twist.linear.x = float(self.lin_vel[0])
        odom.twist.twist.linear.y = float(self.lin_vel[1])
        odom.twist.twist.linear.z = float(self.lin_vel[2])
        odom.twist.twist.angular.x = float(gyro[0])
        odom.twist.twist.angular.y = float(gyro[1])
        odom.twist.twist.angular.z = float(gyro[2])
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = LegOdometryNode()
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
