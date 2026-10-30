# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.
"""OpenVIO on the topics of an EuRoC MAV or TUM-VI bag.

  ros2 launch openvio_ros2 openvio.launch.py dataset:=euroc stereo:=true
  ros2 launch openvio_ros2 openvio.launch.py config:=/path/to/my_camera.yaml imu:=/my/imu image0:=/my/image

dataset: euroc | tum_vi (config/<dataset>.yaml, or <dataset>_stereo.yaml with stereo:=true); config overrides it.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def nodes(context):
    share = get_package_share_directory("openvio_ros2")
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    stereo = value("stereo").lower() in ("true", "1")
    config = value("config") or os.path.join(share, "config", value("dataset") + ("_stereo" if stereo else "") + ".yaml")
    actions = [Node(
        package="openvio_ros2",
        executable="openvio_node",
        name="openvio",
        output="screen",
        parameters=[{"config_path": config, "output_path": value("output_path")}],
        remappings=[("imu", value("imu")), ("image0", value("image0")), ("image1", value("image1"))],
    )]
    if value("rviz").lower() in ("true", "1"):
        actions.append(Node(package="rviz2", executable="rviz2", name="rviz2",
                            arguments=["-d", os.path.join(share, "rviz", "openvio.rviz")]))
    return actions


def generate_launch_description():
    arguments = [
        ("dataset", "euroc", "euroc or tum_vi"),
        ("stereo", "false", "use the stereo config of the dataset"),
        ("config", "", "config file (overrides dataset and stereo)"),
        ("output_path", "", "TUM trajectory file written by the node (empty: none)"),
        ("imu", "/imu0", "IMU topic"),
        ("image0", "/cam0/image_raw", "camera-0 image topic"),
        ("image1", "/cam1/image_raw", "camera-1 image topic (stereo)"),
        ("rviz", "true", "start RViz"),
    ]
    return LaunchDescription([DeclareLaunchArgument(name, default_value=default, description=text)
                              for name, default, text in arguments] + [OpaqueFunction(function=nodes)])
