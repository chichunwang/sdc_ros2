from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('localization')
    rviz_config = os.path.join(
        pkg_share,
        'config',
        'visualization.rviz'
    )

    visualization_node = Node(
        package='localization',
        executable='visualization',
        name='visualization_node',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='display_result',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        visualization_node,
        rviz_node
    ])