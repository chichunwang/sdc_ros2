# DROID-SLAM 對照組的執行環境（與主 SLAM 容器分開；需要 nvcc 編譯 CUDA 擴充，Blackwell sm_120）
# 建置：docker build -t midterm02_droid:cu128 docker/droid
# DROID-SLAM：https://github.com/princeton-vl/DROID-SLAM（BSD-3-Clause），固定 commit 2dfd39f
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir wheel tqdm scipy opencv-python-headless pyyaml gdown numpy==1.26.4 evo==1.31.1
ARG DROID_COMMIT=2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99
RUN git clone https://github.com/princeton-vl/DROID-SLAM.git /opt/DROID-SLAM \
    && cd /opt/DROID-SLAM && git checkout ${DROID_COMMIT} && git submodule update --init --recursive
ENV TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=12
# lietorch 的 setup.py 寫死 sm_60–75：改成只編 sm_120（Blackwell）
RUN cd /opt/DROID-SLAM/thirdparty/lietorch && python3 - <<'PY'
import re
s = open('setup.py').read()
s = re.sub(r"'-gencode=arch=compute_\d+,code=(sm|compute)_\d+',\s*", '', s)
s = s.replace("'nvcc': ['-O2',", "'nvcc': ['-O2', '-gencode=arch=compute_120,code=sm_120',")
open('setup.py', 'w').write(s)
PY
RUN cd /opt/DROID-SLAM && grep -n "gencode\|nvcc" thirdparty/lietorch/setup.py \
    && pip3 install --no-cache-dir --no-build-isolation thirdparty/lietorch \
    && pip3 install --no-cache-dir --no-build-isolation thirdparty/pytorch_scatter \
    && pip3 install --no-cache-dir --no-build-isolation -e . \
    && pip3 freeze > /opt/pip_freeze.txt
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp/droid-home
WORKDIR /workspace/midterm02
CMD ["sleep", "infinity"]
