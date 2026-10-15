#!/usr/bin/env bash
# Run inside the midterm container, normally via dev.sh build-workspace.
set -eo pipefail

ros_setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [[ ! -f "$ros_setup" ]]; then
    printf 'ROS setup is missing: %s. Run this in the configured ROS container.\n' "$ros_setup" >&2
    exit 1
fi
# ROS setup scripts may reference unset variables, so enable nounset afterwards.
source "$ros_setup"
set -u

cd /root/midterm
if ! command -v colcon >/dev/null 2>&1; then
    printf 'colcon is missing in the current ROS environment.\n' >&2
    exit 1
fi
packages="$(colcon list --base-paths src --names-only)"
if [[ -z "$packages" ]]; then
    printf 'No ROS packages found. Import the course localization package into workspace/src/.\n' >&2
    exit 1
fi
printf 'Building ROS packages:\n%s\n' "$packages"
colcon build --symlink-install "$@"

