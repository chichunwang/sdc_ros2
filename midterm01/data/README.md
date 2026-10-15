# Midterm01 原始資料位置

此處可存放官方 Track1／Track2 資料。原始資料被 Git 忽略，這份說明會保留。

```text
data/
├── Track1/
│   ├── map.pcd
│   ├── init_guess.txt
│   ├── raw_ply/
│   ├── sample_submission.csv    # 自 Kaggle 另行取得，建議放置位置
│   └── 其他官方資料
└── Track2/
    └── 同上
```

下載連結及必要／選用項目見 `../docs/DOWNLOADS.md`。`sample_submission.csv` 是本架構建議的儲存名稱，不保證原始下載檔名相同；保留官方內容。

若資料已放在遠端 Ubuntu 的其他位置，修改 `midterm01/.env` 的 `MIDTERM_DATA_DIR`，指向同時包含 `Track1/` 與 `Track2/` 的主機目錄即可。若實際 Track 資料夾名稱不同，修改 `config/tracks.json` 的 `directory`。資料會唯讀掛載至容器的 `/root/data/`。

HW2 的 `guanfu_road_clipped_ros2` bag 與本次競賽資料不同，請使用對應 Midterm01 Track 的地圖與點雲。

