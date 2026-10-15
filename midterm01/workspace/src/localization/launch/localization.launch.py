import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('localization')
    arguments = [
        DeclareLaunchArgument('data_path', default_value='/root/data/Track1', description='Track directory containing map.pcd and init_guess.txt'),
        DeclareLaunchArgument("bag_path", default_value="", description="Optional .db3 file or bag directory for GNSS and IMU messages"),
        DeclareLaunchArgument('ply_path', default_value=PathJoinSubstitution([LaunchConfiguration('data_path'), 'raw_ply'])),
        DeclareLaunchArgument('init_guess_file', default_value=PathJoinSubstitution([LaunchConfiguration('data_path'), 'init_guess.txt'])),

        DeclareLaunchArgument('save_path', default_value='/root/midterm/src/localization/result/'),
        DeclareLaunchArgument('save_file', default_value='Track1_result.txt'),
        
        DeclareLaunchArgument('log_level', default_value='info'),
    ]
    parameter_types = {
        'map_voxel_leaf_size': float, 'input_voxel_leaf_size': float,
        'min_input_points': int, 'submap_half_size': float, 'submap_half_height': float,
        'ndt_resolution': float, 'ndt_step_size': float,
        'ndt_transformation_epsilon': float, 'ndt_max_iterations': int,
        'ply_scale': float, 'publish_global_map': bool,
        'max_frames': int, 'keep_alive': bool,
    }
    arguments += [
        DeclareLaunchArgument(name, default_value='', description='Optional override; omitted uses localization.h default')
        for name in parameter_types
    ]

    def create_nodes(context):
        # Pass only explicit terminal overrides; let the C++ node own defaults.
        overrides = {
            name: ParameterValue(LaunchConfiguration(name), value_type=value_type)
            for name, value_type in parameter_types.items()
            if LaunchConfiguration(name).perform(context).strip()
        }
        localization_node = Node(
            package='localization', executable='localization',
            name='image_point_cloud_localizer', output='screen',
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            parameters=[{
                'use_sim_time': False,
                'map_path': LaunchConfiguration('data_path'),
                'map_file': 'map.pcd',
                'map_cache_file': 'map_downsampled_1m.pcd',
                'bag_path': LaunchConfiguration('bag_path'),
                'ply_path': LaunchConfiguration('ply_path'),
                'init_guess_file': LaunchConfiguration('init_guess_file'),
                
                'save_path': LaunchConfiguration('save_path'),
                'save_file': LaunchConfiguration('save_file'),

                **overrides,
            }],
        )
        rviz_node = Node(
            package='rviz2', executable='rviz2', name='display_result', output='screen',
            arguments=['-d', os.path.join(pkg_share, 'config', 'localization.rviz')],
            parameters=[{'use_sim_time': False}],
        )
        return [localization_node, rviz_node]

    return LaunchDescription(arguments + [OpaqueFunction(function=create_nodes)])