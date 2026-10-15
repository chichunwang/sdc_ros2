# 目前的 ROS 2 工作區

`localization/` 已放入完整 package，目前包含基礎 NDT，無須再次下載或多套一層 `src/`。

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

`localization.cpp` 已實作逐幀 PLY 載入、尺度／濾波、局部地圖、基礎 NDT、選用 bag 讀取、點雲／TF／path 與結果寫入；TODO 是接續設計自己的定位改善，不表示目前沒有 scan matching。`visualization.cpp` 仍是舊 Radar／PCD 介面，需另行調整才能回放本次點雲。

建置與無 GUI 首跑見 [專案 README](../../README.md)，目前參數預設在標頭，launch 只傳入明確覆寫。launch 固定啟動 RViz、map cache 指向唯讀資料目錄；直接使用 README 的 `ros2 run` 可先停用這兩項。

容器內程式位於 `/root/midterm/src/localization`，Ubuntu 主機位於 `~/sdc_ros2/midterm01/workspace/src/localization`。修改 C++ 後重新建置；不要將其他環境的 `build/`、`install/`、`log/` 複製進來。目前尚無本次 Ubuntu 編譯／實測成功紀錄。
