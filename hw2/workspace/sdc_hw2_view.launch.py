"""Starts rviz2 plus the camera decoders needed for Task 3.

rviz2's Image display only accepts raw sensor_msgs/Image, but the bag stores
JPEGs (sensor_msgs/CompressedImage) to keep the download small.  The two
`image_transport republish` nodes below decode them into raw topics that
rviz2 can show.

    ros2 launch /ros_ws/workspace/sdc_hw2_view.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node

RVIZ_CONFIG = '/ros_ws/workspace/sdc_hw2.rviz'

# compressed topic (without the /compressed suffix) -> raw topic for rviz2
CAMERAS = {
    '/lucid_cameras_x00/gige_100_f_hdr': '/front_camera_100',
    '/lucid_cameras_x00/gige_30_f_hdr': '/front_camera_30',
}


def generate_launch_description() -> LaunchDescription:
    actions = [
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', RVIZ_CONFIG],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
    ]
    for i, (src, dst) in enumerate(CAMERAS.items()):
        actions.append(Node(
            package='image_transport',
            executable='republish',
            name=f'camera_decoder_{i}',
            arguments=['compressed', 'raw'],
            remappings=[('in/compressed', f'{src}/compressed'), ('out', dst)],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ))
    return LaunchDescription(actions)
