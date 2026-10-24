# 既有本機 CUDA12.8/Humble 映像的可重現重建入口見 Dockerfile.base。
ARG BASE_IMAGE=sdc_midterm01_gpu_ros2:cuda12.8
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/track3
COPY requirements.txt /opt/track3-requirements.txt
RUN /opt/track3/bin/pip install --no-cache-dir -r /opt/track3-requirements.txt
ENV PATH=/opt/track3/bin:/usr/local/cuda-12.8/bin:$PATH \
    PYTHONPATH=/workspace/midterm01_gpu \
    PYTHONUNBUFFERED=1 \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    CUPY_CACHE_DIR=/tmp/cupy-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp/track3-home
WORKDIR /workspace/midterm01_gpu
COPY ros_entrypoint.sh /track3_entrypoint.sh
ENTRYPOINT ["/track3_entrypoint.sh"]
CMD ["sleep", "infinity"]
