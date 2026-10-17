# CUDA 後端：最近鄰與歷史 NDT

此 package 目前同時提供本專案的 `src/nearest_neighbor.cu`／`include/cuda_ndt_backend/nearest_neighbor.h` 與上游 CUDA NDT。主要 `sensor_guided`／`cuda_robust_icp` 使用自寫 CUDA 最近鄰，CPU 做 robust 四自由度細修，不執行 NDT；見 [新版實測](../../../SENSOR_GUIDED_20261017.md)。下面上游來源與 LM 說明只適用 `legacy`／`cuda_ndt`。

`nearest_neighbor_test` 驗證自寫 GPU 搜尋與 CPU 完整搜尋一致；`cuda_ndt_smoke` 檢查上游 NDT 的已知變換。主 workspace 以 [GPU README](../../../README.md) 的 Release 指令建置。

使用 [SMRT-AIST／koide3 fast_gicp](https://github.com/koide3/fast_gicp)，固定 commit `0e7ec1441c99f7be453db2ea216d5de029387417`，BSD-3-Clause。CMake 下載固定 SHA256 的原始碼，只編譯 CUDA NDT、voxelmap、correspondence、covariance 與 LSQ optimizer 所需來源，不啟用未使用的 VGICP、cuBLAS、Python 或 ROS1 元件。

上游 NDT 演算法與 upstream 相同；自寫最近鄰核心另在本 package，兩者勿混為同一演算法。針對新版 CCCL，將 `ndt_cuda.cuh` 原先 `namespace thrust` 的 forward declaration 改用 `THRUST_NAMESPACE_BEGIN/END`，避免新版 Thrust ABI namespace 衝突。本專案另新增現代 CMake／CUDA 12.8／`sm_120` 建置、ROS 2 package 包裝、已知變換檢查與 localization 整合。安裝時保存 upstream 的完整授權檔。首次建置需網路；原始碼下載到本 workspace 的 build 目錄，不放在資料目錄。

P2D 的 voxel 統計／correspondence／cost、gradient、Hessian 大量點運算使用 CUDA；6×6 LM 解算、PCL fitness、輸入讀檔、裁切與 ROS 發布仍在 CPU。此實作的 LM／收斂門檻與 PCL NDT 的 More-Thuente line search 不同，`ndt_step_size` 不適用於 CUDA，不能假設相同參數即相同軌跡。
