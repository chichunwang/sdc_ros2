# Midterm02 Visual SLAM 執行環境（CPU；官方 starter 指定 Python 3.9）
FROM python:3.9-slim-bookworm

# open3d / opencv 執行期需要的系統函式庫
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip freeze > /opt/pip_freeze.txt

WORKDIR /workspace/midterm02
CMD ["sleep", "infinity"]
