# Midterm01 GPU 工作區

文件已集中到 **[doc/](doc/28_DOCUMENTATION_INDEX.md)**，以兩位數編號排列。

- **第一次讀： [01｜程式架構圖與完整演算法流程](doc/01_PROGRAM_FLOW.md)**。從主機／container、Python 入口、C++ solver 一路讀到 CSV，也包含三相機分支。
- 日常指令：[08｜操作手冊](doc/08_MANUAL_WORKFLOW.md)。目前已評分流程明確加 `--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional`；省略選項的程式預設仍是 legacy／fixed／off。
- 專案現況與原 README 完整內容：[27｜專案總覽](doc/27_PROJECT_OVERVIEW.md)。所有歷史文件、舊名對照與官方／CPU 文件索引：[28｜文件目錄](doc/28_DOCUMENTATION_INDEX.md)。

最新三相機完整成績：Track1 **1.15346**、Track2 **0.31064**，均未超過原前相機最佳 1.15275／0.30576。兩條輸出與 24 組消融已核對，原因與改進順序見 [29｜三相機完整成績分析](doc/29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。

後續開發使用本 GPU workspace；原始資料唯讀共用 `../midterm01/data`。完整 Track 由使用者手動執行，協作約定見 [AGENTS.md](AGENTS.md)。
