# MASt3R-SLAM 第三個獨立估計器（沿用 DROID 映像的 CUDA 12.8 devel、torch 2.7.1、lietorch）
# 不安裝 thirdparty/in3d（視覺化，scripts/run_mast3r.py 不需要）
# 建置：docker build -t midterm02_mast3r:cu128 docker/mast3r
# MASt3R-SLAM：https://github.com/rmurai0610/MASt3R-SLAM（CC BY-NC-SA 4.0），固定 commit e6f4e3d
# MASt3R 權重：naver（CC BY-NC-SA 4.0，見 thirdparty/mast3r/CHECKPOINTS_NOTICE），執行時放 outputs/cache/mast3r/checkpoints
FROM midterm02_droid:cu128
RUN apt-get update && apt-get install -y --no-install-recommends wget libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
ARG MS_COMMIT=e6f4e3d474fad0e11f561482012be864ba8c3f17
RUN git clone https://github.com/rmurai0610/MASt3R-SLAM.git /opt/MASt3R-SLAM \
    && cd /opt/MASt3R-SLAM && git checkout ${MS_COMMIT} && git submodule update --init --recursive
ENV TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=12
RUN cd /opt/MASt3R-SLAM && python3 - <<'PY'
import re
s = open('setup.py').read()
s = s.replace('has_cuda = torch.cuda.is_available()', 'has_cuda = True   # docker build 沒有 GPU，強制編 CUDA 後端')
s = re.sub(r'"-gencode=arch=compute_\d+,code=sm_\d+",\s*', '', s)
s = s.replace('"-O3",\n', '"-O3",\n        "-gencode=arch=compute_120,code=sm_120",\n', 1)
open('setup.py', 'w').write(s)
PY
RUN pip3 install --no-cache-dir --upgrade pip "setuptools==70.0.0" wheel
RUN cd /opt/MASt3R-SLAM && grep -n gencode setup.py \
    && pip3 install --no-cache-dir roma einops trimesh natsort plyfile "huggingface-hub[torch]>=0.22" matplotlib tensorboard \
       pyglet\<2 imageio moderngl moderngl-window glfw pyyaml \
    && sed -i 's#Path(__file__).parent#Path(__file__).resolve().parent#' thirdparty/mast3r/setup.py \
    && pip3 install --no-cache-dir scikit-learn faiss-cpu cython \
    && sed -i 's/tokens\.type()/tokens.scalar_type()/g' thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu \
    && pip3 install --no-cache-dir --no-build-isolation thirdparty/mast3r/dust3r/croco/models/curope \
    && pip3 install --no-cache-dir --no-build-isolation thirdparty/mast3r/asmk \
    && pip3 install --no-cache-dir --no-deps -e thirdparty/mast3r \
    && sed -i -E 's/torch::linalg::linalg_norm\(([A-Za-z_]+), std::optional<c10::Scalar>\(\), \{\}, false, \{\}\)/\1.norm()/g' mast3r_slam/backend/src/gn_kernels.cu \
    && sed -i 's/D11\.type()/D11.scalar_type()/g' mast3r_slam/backend/src/matching_kernels.cu \
    && grep -n 'linalg_norm\|\.norm()' mast3r_slam/backend/src/gn_kernels.cu \
    && pip3 install --no-cache-dir --no-deps --no-build-isolation -e . \
    && pip3 freeze > /opt/pip_freeze_mast3r.txt
WORKDIR /opt/MASt3R-SLAM
CMD ["sleep", "infinity"]
