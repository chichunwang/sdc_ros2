# VGGT（Visual Geometry Grounded Transformer，Meta 2025）：一次看多張影像，直接輸出每張的相機位姿與深度
# 建置：docker build -t midterm02_vggt:cu128 docker/vggt
# 權重（執行時下載到 outputs/cache/hf）：facebook/VGGT-1B
# 沿用深度模型映像的 CUDA 12.8、torch 2.7.1；--no-deps 避免重裝 torch
FROM midterm02_depth:cu128
RUN pip3 install --no-cache-dir einops \
    && pip3 install --no-cache-dir --no-deps git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce \
    && python3 -c "import vggt, einops" \
    && pip3 freeze > /opt/pip_freeze_vggt.txt
CMD ["sleep", "infinity"]
