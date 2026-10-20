import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    fusion_dir = get_package_share_directory('fusion')

    result_dir_arg = DeclareLaunchArgument(
        'result_dir',
        default_value='/ros_ws/hw4/result',
        description='Directory for the host-mounted result.png'
    )

    rviz_config_path = os.path.join(fusion_dir, 'rviz_config', 'rviz_config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    #  Fusion Node
    fusion_node = Node(
        package='fusion',
        executable='fusion.py',
        name='EKF_fusion',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'result_dir': LaunchConfiguration('result_dir'),
            'fixed_frame': LaunchConfiguration('fixed_frame'),
        }]
    )

    # return Launch description
    return LaunchDescription([
        result_dir_arg,
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('fixed_frame', default_value='origin'),
        rviz_node,
        fusion_node
    ])
