# Track3 rewrite native package

此 package 實作 `track3_cpp_cuda_observable_icp_v1` 的原生 C++／CUDA 核心。

- `src/main.cpp`：native runner entry 與逐幀處理協調。
- `src/registration.cpp`：CPU 端 registration orchestration 與小型求解器連接。
- `src/geometry.cu`：CUDA voxel、bounded nearest-neighbor、PCA、殘差與 normal-equation 統計。
- `include/track3_rewrite/core.hpp`：核心資料型別與介面。
- `tests/mechanisms.cpp`：合成機制測試。

完整數學模型、資料／尺度契約、狀態、驗證結果與手動命令見 [唯一 Track3 架構文件](../../../doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。此 README 僅作檔案導航；測試紀錄以 `outputs/run/track03_test_2610240045/` 為準。
