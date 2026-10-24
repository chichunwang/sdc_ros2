#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
export PATH="/opt/track3/bin:$PATH"
mkdir -p "$HOME"
exec "$@"
