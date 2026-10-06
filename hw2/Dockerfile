FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Basic tooling plus the ROS 2 packages this homework needs.
#   sensor-msgs-py  : point_cloud2 helpers (read_points / create_cloud)
#   rosbag2-storage-mcap : lets students open .mcap bags as well as .db3
#   tf2-tools       : view_frames, for inspecting the TF tree
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        ros-humble-sensor-msgs-py \
        ros-humble-rosbag2-storage-mcap \
        ros-humble-tf2-tools \
        ros-humble-tf-transformations \
        ros-humble-image-transport-plugins \
        vim \
        tmux \
        less \
    && rm -rf /var/lib/apt/lists/*

# rosbags lets you convert further ROS 1 bags from inside the container.
RUN pip3 install --no-cache-dir rosbags

WORKDIR /ros_ws

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Convenience: source ROS 2 in every interactive shell.
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "export ROS_LOCALHOST_ONLY=1" >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
