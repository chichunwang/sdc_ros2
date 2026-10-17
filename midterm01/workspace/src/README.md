# 原 CPU ROS 2 工作區（保留參考）

2026-10-17 起，後續編輯與建置統一使用 [GPU 工作區](../../../midterm01_gpu/README.md)。本目錄保留原 `localization/` CPU PCL NDT 參考 package，不與 GPU 程式快照或 build／install 自動同步。

```text
workspace/src/localization/
├── CMakeLists.txt
├── package.xml
├── include/localization/localization.h
├── src/{localization,map_modified,visualization}.cpp
├── launch/{localization,map_modified,visualization}.launch.py
├── config/{localization,visualization}.rviz
└── processing_codes/txt_to_idx_x_y_yaw.py
```

原 CPU 已有逐幀 PLY／timestamp 索引、尺度與濾波、submap、NDT、選用 bag 讀取、ROS 發布與 TXT，不能把它當空架構。CPU 的 bag 未融合到 NDT；`visualization.cpp` 是舊 Radar／PCD 介面，不能直接回放本次 PLY。

CPU Ubuntu 建置與兩條單幀／前 50 幀已通過，見 [首跑](../../docs/UBUNTU_FIRST_RUN_20261017.md)；指定 Track1 完整輸出見 [完成紀錄](../../docs/CPU_TRACK1_COMPLETED_20261017.md)。launch 已有 `use_rviz=false`、空字串停用 cache、outputs 下唯一預設路徑；轉檔已移除 pandas／NumPy／SciPy 未使用 import。舊「尚未建置／固定開 RViz」敘述已過時。

CPU 容器程式在 `/root/midterm/src/localization`，主機為 `/home/ricky/sdc_ros2/midterm01/workspace/src/localization`；GPU 同名容器路徑對應 `midterm01_gpu/workspace/src/localization`。要修改目前新版請依 [GPU 操作手冊](../../../midterm01_gpu/doc/MANUAL_WORKFLOW.md)，不要修改 CPU build 或複製其他環境建置產物。
