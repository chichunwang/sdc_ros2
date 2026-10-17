import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
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

        DeclareLaunchArgument('save_path', default_value='/root/outputs/runs/launch_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f')),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('map_cache_file', default_value='', description='Empty disables cache; use a unique absolute outputs path for explicit caches'),
        DeclareLaunchArgument('save_file', default_value='Track1_result.txt'),
        
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('registration_backend', default_value='cuda_ndt', description='cuda_ndt, pcl_ndt, or cuda_robust_icp (requires sensor_guided profile)'),
        DeclareLaunchArgument('cuda_ndt_mode', default_value='D2D'),
        DeclareLaunchArgument('localization_profile', default_value='legacy'),
        DeclareLaunchArgument('sensor_prior_file', default_value=''),
        DeclareLaunchArgument('geometry_mode', default_value='off'),
    ]
    parameter_types = {
        'map_voxel_leaf_size': float, 'input_voxel_leaf_size': float,
        'min_input_points': int, 'submap_half_size': float, 'submap_half_height': float,
        'ndt_resolution': float, 'ndt_step_size': float,
        'ndt_transformation_epsilon': float, 'ndt_max_iterations': int,
        'ply_scale': float, 'publish_global_map': bool,
        'max_frames': int, 'keep_alive': bool,
        'cuda_translation_epsilon': float, 'cuda_rotation_epsilon': float,
        'sensor_max_xy_correction': float, 'sensor_max_yaw_correction_deg': float,
        'sensor_smoothing_alpha': float,
        'geometry_normal_gain': float, 'geometry_tangent_weight': float,
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
                'map_cache_file': ParameterValue(LaunchConfiguration('map_cache_file'), value_type=str),
                'bag_path': LaunchConfiguration('bag_path'),
                'ply_path': LaunchConfiguration('ply_path'),
                'init_guess_file': LaunchConfiguration('init_guess_file'),
                
                'save_path': LaunchConfiguration('save_path'),
                'save_file': LaunchConfiguration('save_file'),
                'registration_backend': LaunchConfiguration('registration_backend'),
                'cuda_ndt_mode': LaunchConfiguration('cuda_ndt_mode'),
                'localization_profile': LaunchConfiguration('localization_profile'),
                'sensor_prior_file': ParameterValue(LaunchConfiguration('sensor_prior_file'), value_type=str),
                'geometry_mode': ParameterValue(LaunchConfiguration('geometry_mode'), value_type=str),

                **overrides,
            }],
        )
        rviz_node = Node(
            package='rviz2', executable='rviz2', name='display_result', output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            arguments=['-d', os.path.join(pkg_share, 'config', 'localization.rviz')],
            parameters=[{'use_sim_time': False}],
        )
        return [localization_node, rviz_node]

    return LaunchDescription(arguments + [OpaqueFunction(function=create_nodes)])
