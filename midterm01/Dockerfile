# scripts/dev.sh builds the existing HW2 image first if it is missing.
FROM sdc_hw2_ros2:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libeigen3-dev \
        libpcl-dev \
        python3-colcon-common-extensions \
        ros-humble-ament-cmake \
        ros-humble-pcl-conversions \
        ros-humble-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# The HW2 image supplies /ros_entrypoint.sh and sources ROS 2 Humble.
WORKDIR /root/midterm
CMD ["sleep", "infinity"]

