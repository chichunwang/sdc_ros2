from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('data_path', default_value='/root/data/Track1', description='Track directory used for default input/output paths'),
        DeclareLaunchArgument('input_file', default_value=PathJoinSubstitution([LaunchConfiguration('data_path'), 'map.pcd']), description='Source PCD map'),
        DeclareLaunchArgument('output_file', default_value=PathJoinSubstitution([LaunchConfiguration('data_path'), 'map_downsampled_1m.pcd']), 
                              description='Binary PCD output, compatible with localization map_cache_file'),
        DeclareLaunchArgument('map_voxel_leaf_size', default_value='1.0', description='Voxel size in meters; 0 disables downsampling'),
        Node(
            package='localization', executable='map_modified',
            name='map_modifier', output='screen',
            parameters=[{
                'input_file': LaunchConfiguration('input_file'),
                'output_file': LaunchConfiguration('output_file'),
                'map_voxel_leaf_size': ParameterValue(LaunchConfiguration('map_voxel_leaf_size'), value_type=float),
            }],
        ),
    ])
