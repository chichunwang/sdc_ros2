#!/bin/bash
set -e

source "/opt/ros/$ROS_DISTRO/setup.bash"

# Keep every student's ROS 2 traffic inside their own machine.  Without this,
# everyone on the same lab subnet discovers everyone else's topics and rviz2
# shows the wrong point clouds.
export ROS_LOCALHOST_ONLY=1

exec "$@"
