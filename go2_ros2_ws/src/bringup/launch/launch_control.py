import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# Configs
config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
control_config = os.path.join(config_dir, "config", "control.yaml")


def generate_launch_description():
    cmd_vel_pub = Node(
        package="control",
        executable="cmd_vel_pub",
        name="cmd_vel_pub",
        output="screen",
        parameters=[
            common_config,
            control_config,
            PathJoinSubstitution([config_dir, "config", LaunchConfiguration("robot_config")]),
        ],
    )

    controller = Node(
        package="simdist_controller",
        executable="simdist_controller_node.py",
        name="simdist_controller_node",
        output="screen",
        parameters=[
            common_config,
            {"controller_config": LaunchConfiguration("controller_config")},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config",
                default_value="control.yaml",
                description="Extra params file in the config package for cmd_vel_pub, e.g. go1.yaml",
            ),
            DeclareLaunchArgument(
                "controller_config",
                default_value="simdist_controller.yaml",
                description="Controller config file in the config package (e.g. simdist_controller_go1.yaml)",
            ),
            cmd_vel_pub,
            controller,
        ]
    )
