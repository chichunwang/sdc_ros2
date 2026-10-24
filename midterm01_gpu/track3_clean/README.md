# Track3 clean package

此為獨立 Track3 前向定位 package，method=`track3_clean_local_sim3_v1`。架構與模型說明集中於 [doc/51](../doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，手動命令見 [doc/50](../doc/50_COMMAND_REFERENCE.md)。本 README 僅作模組導航與驗證狀態索引。

## 模組導航

- `cli.py`：命令列參數、輸入盤點、dry-run 與定位入口。
- `contracts.py`、`config.py`：資料／pose 契約與工程設定。
- `data.py`、`map_cache.py`：PLY 輸入及 map XYZ cache 的讀取與來源核對。
- `geometry.py`、`cuda_grid.py`：幾何運算及 provenance 記錄的 CUDA 數值 primitives。
- `registration.py`、`initialization.py`、`sampling.py`、`visibility.py`、`quality.py`：初始化、配準、取樣、視線檢查及候選品質裁決。
- `engine.py`、`state.py`：逐幀流程與追蹤狀態轉移。
- `output.py`：run 目錄、固定輸出序列化及輸出完整性檢查。
- `runtime.py`：來源快照、CUDA runtime 設定、在跑定位程序檢查與 full 後圖表流程。
- `provenance.json`：抽取數值 primitives 的來源檔與 SHA256。

## 驗證範圍

30 項 CPU 數值／契約／I/O／CLI／故障檢查通過；CPU 合成 3／5 幀 engine 序列的 TXT／CSV 前綴逐byte相同。真實3142幀dry-run、首／中／尾PLY解析取樣及map/cache SHA唯讀盤點通過；未初始化CUDA或執行真實定位。證據在 [`track03_test_2610240007`](../outputs/run/track03_test_2610240007/README.md)。新入口的GPU與正式精度尚未驗證；原full程序與來源保持。
