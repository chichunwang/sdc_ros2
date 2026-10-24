FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV TORCH_CUDA_ARCH_LIST="10.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-rviz2 \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-cpp \
    libpcl-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy==1.23.5 \
    pandas<2.1 \
    tqdm \
    gnupg \
    pycryptodomex \
    rosbags

WORKDIR /root/midterm

# Automate environment sourcing for all new terminals
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
