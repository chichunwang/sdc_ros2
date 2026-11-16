# 單張影像深度模型的評估環境（沿用 DROID 映像的 CUDA 12.8、torch 2.7.1）
# 建置：docker build -t midterm02_depth:cu128 docker/depth
# 模型（執行時下載到 outputs/cache/hf）：
#   MoGe-2（microsoft/MoGe，MIT）：Ruicheng/moge-2-vitl-normal
#   Depth Anything V2 Metric Outdoor Large（transformers）：depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
FROM midterm02_droid:cu128
RUN pip3 install --no-cache-dir --upgrade pip "setuptools==70.0.0" wheel
RUN pip3 install --no-cache-dir "transformers==4.46.3" "accelerate==1.1.1" "huggingface-hub>=0.26,<1.0" safetensors
RUN pip3 install --no-cache-dir git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a \
    && pip3 freeze > /opt/pip_freeze_depth.txt
CMD ["sleep", "infinity"]
