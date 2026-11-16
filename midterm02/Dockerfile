# Midterm02 Visual SLAM 執行環境（GPU；官方 starter 指定 Python 3.9）
# GPU 由主機的 NVIDIA Container Toolkit 注入驅動；CUDA runtime 函式庫來自 pip 的 nvidia-*-cu12 wheel，
# 所以不需要 nvidia/cuda 基底映像，也不需要 nvcc（CuPy RawKernel 用 NVRTC 即時編譯）。
FROM python:3.9-slim-bookworm

# open3d / opencv 執行期需要的系統函式庫
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 選用：學習式特徵（SuperPoint／ALIKED + LightGlue，官方規則允許的公開預訓練模型）。
# 建置時加 --build-arg INSTALL_LEARNED=1 才安裝。
ARG INSTALL_LEARNED=0
COPY requirements-learned.txt /tmp/requirements-learned.txt
RUN if [ "$INSTALL_LEARNED" = "1" ]; then \
        pip install --no-cache-dir -r /tmp/requirements-learned.txt \
            --extra-index-url https://download.pytorch.org/whl/cu128; \
    fi \
    && pip freeze > /opt/pip_freeze.txt

ENV NVIDIA_PY=/usr/local/lib/python3.9/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVIDIA_PY}/cuda_nvrtc/lib:${NVIDIA_PY}/cuda_runtime/lib:${NVIDIA_PY}/cublas/lib:${NVIDIA_PY}/cusolver/lib:${NVIDIA_PY}/cusparse/lib:${NVIDIA_PY}/nvjitlink/lib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUPY_CACHE_DIR=/tmp/cupy-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    HOME=/tmp/vslam-home

WORKDIR /workspace/midterm02
CMD ["sleep", "infinity"]
