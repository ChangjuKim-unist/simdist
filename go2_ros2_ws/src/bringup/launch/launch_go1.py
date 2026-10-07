"""Bring up the sensing side of the stack for a Unitree Go1 on flat ground.

Replaces launch_go2.py. The Go1 is driven through go1_bridge (unitree_legged_sdk)
and, having no lidar, gets its base velocity from leg odometry and a flat
elevation map from the leg kinematics instead of Point-LIO and elevation mapping.
Start the state machine and controller as for the Go2:

    ros2 launch bringup launch_state_machine.py
    ros2 launch bringup launch_control.py controller_config:=simdist_controller_go1.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node

config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
measurement_config = os.path.join(config_dir, "config", "measurement.yaml")
go1_config = os.path.join(config_dir, "config", "go1.yaml")
go1_urdf = os.path.join(get_package_share_directory("go1_description"), "urdf", "go1.urdf")


def generate_launch_description():
    tf_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory("bringup"), "launch", "launch_tfs.py")
        )
    )

    go1_bridge = Node(
        package="go1_bridge",
        executable="go1_bridge",
        name="go1_bridge",
        output="screen",
        parameters=[common_config, go1_config],
    )
    repub_body_imu = Node(
        package="measurement",
        executable="repub_body_imu",
        name="repub_body_imu",
        output="screen",
        parameters=[common_config, measurement_config, go1_config],
    )
    leg_odometry = Node(
        package="measurement",
        executable="leg_odometry.py",
        name="leg_odometry",
        output="screen",
        parameters=[common_config, measurement_config, go1_config],
    )
    flat_elevation = Node(
        package="measurement",
        executable="flat_elevation.py",
        name="flat_elevation",
        output="screen",
        parameters=[common_config, measurement_config, go1_config],
    )
    observer = Node(
        package="measurement",
        executable="observer.py",
        name="observer",
        output="screen",
        parameters=[common_config, measurement_config, go1_config],
    )
    joint_state_pub = Node(
        package="visualization",
        executable="joint_state_pub.py",
        name="joint_state_pub",
        output="screen",
        parameters=[common_config, go1_config],
    )
    with open(go1_urdf, "r") as f:
        robot_description = f.read()
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[common_config, {"robot_description": robot_description}],
        output="screen",
    )

    return LaunchDescription(
        [
            tf_launch,
            go1_bridge,
            repub_body_imu,
            leg_odometry,
            flat_elevation,
            observer,
            joint_state_pub,
            robot_state_publisher,
        ]
    )
