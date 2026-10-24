#!/bin/bash
set -e

# Source ROS 2 Humble
source "/opt/ros/humble/setup.bash"

# Source workspace if it has already been built
if [ -f "/root/midterm/install/setup.bash" ]; then
    source "/root/midterm/install/setup.bash"
fi

exec "$@"
