# Reuse the existing ROS 2 Humble/PCL image without changing its tag or container.
ARG ROS_BASE_IMAGE=sdc_midterm01_ros2:latest
FROM ${ROS_BASE_IMAGE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=120

# Install the compiler, runtime headers and CCCL, not a host NVIDIA driver.
# CUDA 12.8 supports the RTX 5060's native sm_120 architecture.
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 \
        cuda-cudart-dev-12-8 \
        cuda-cccl-12-8 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV CUDACXX=/usr/local/cuda-12.8/bin/nvcc
ENV CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64

COPY scripts/cuda_probe.cu /opt/midterm01_gpu/cuda_probe.cu
RUN nvcc -O3 -arch=sm_${CUDA_ARCHITECTURES} /opt/midterm01_gpu/cuda_probe.cu \
        -o /usr/local/bin/midterm-cuda-probe

WORKDIR /root/midterm
CMD ["sleep", "infinity"]

