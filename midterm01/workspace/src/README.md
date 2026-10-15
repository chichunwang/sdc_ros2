# ROS 2 package 放置位置

將老師提供的完整 `localization` package 放在這裡：

```text
workspace/src/localization/
├── package.xml
├── CMakeLists.txt
├── include/
├── src/
├── launch/
└── 其他模板原有檔案
```

老師的壓縮檔可能包含外層工作區。找出同時包含 `package.xml` 與 `CMakeLists.txt` 的 `localization/` package，再整個複製至此，不要多套一層 `src/`，也不要將範例的 `build/`、`install/`、`log/` 複製進來。

目前先保留模板接入位置。取得模板後先核對 TODO、讀取格式、座標轉換與結果輸出，再實作 ICP。`build_workspace.sh` 發現沒有 ROS package 時會清楚報錯，不會把空工作區當成建置完成。

