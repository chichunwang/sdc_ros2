# 官方資源與查漏補缺

更新：2026-10-16。**模板與 Ubuntu 資料已建立，先檢查既有內容，不重新下載整包。** 使用者確認資料位於 `~/sdc_ros2/midterm01/data`；實際資料樹與檢查方式見 [data/README](../data/README.md)，建置與首跑見 [專案 README](../README.md)。

## 已取得與待核對

| 項目 | 目前狀態 |
| --- | --- |
| 官方題目 | 位於本目錄 [PDF](<SDC 2026 Midterm01 Competition -  Localization.pdf>)，不再使用 localreference 舊路徑 |
| Localization package | 已在 `workspace/src/localization/`，含基礎 NDT 與 `processing_codes/txt_to_idx_x_y_yaw.py` |
| Track1、Track2 | Ubuntu 截圖各有 map、init_guess、raw_ply、影像、bag 及 metadata；數量／內容待執行檢查 |
| MapAnything 點雲 | Ubuntu 有兩組 `kaggle_track0N/TrackN_Mapanything_ply/{ply,index.csv,report.json}` |
| 校正資料 camera_info | 截圖未顯示，先搜尋現有資料與下載包，不直接判定缺少 |
| 官方 sample submission | 截圖未顯示，先搜尋；不可用 MapAnything index.csv 替代 |

官方欄位順序為 `idx,x,y,yaw`，兩條資料列數為 3011／3299，均不含表頭；使用者提供的原文與 topics 見 [Kaggle 規格](KAGGLE_DATASET_SPEC.md)。完整 ID、座標與 yaw 單位仍需核對正式範例／公告。

## 官方來源索引

連結取自 PDF p.9、11；本次未重新登入 Kaggle／Drive 查驗網站最新內容或重新下載。

| 資源 | 官方來源 | 用途 |
| --- | --- | --- |
| Localization | [Google Drive](https://drive.google.com/file/d/1dAnE_-QwlmxVQIdNu13cNBjqEZSL6vCp/view?usp=sharing) | 缺少原始模板時查對，不覆寫目前 package |
| Track01 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=Track1) | 核對缺檔、取得該 Track sample |
| Track02 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=Track2) | 同上，使用對應地圖／sample |
| Track01 三相機 | [Kaggle 3cameras_ply](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=3cameras_ply) | 核對已取得的 MapAnything 來源與結構 |
| Track02 三相機 | [Kaggle 3cameras_ply](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=3cameras_ply) | 同上 |
| LoGeR 原始碼（選用） | [Google Drive](https://drive.google.com/file/d/1ZHeWxVrf86GDtHtFY7JyXY7w1vge3Fsa/view?usp=sharing) | 需要修改影像重建流程時再研究 |

加入邀請在 PDF p.16，隊名要求見 p.18。保留原始下載內容及來源紀錄；sample 的實際檔名不預設為 `sample_submission.csv`。官方 PDF 對後處理目錄有 process／processing_codes 的差異，本 package 實際使用 `processing_codes/`。

## 使用既有資料

Ubuntu 專案 `midterm01/.env` 可沿用：

```dotenv
MIDTERM_DATA_DIR=./data
```

若改用其他資料位置，填同時含 `Track1/` 與 `Track2/` 的真實 Ubuntu 根目錄。它會唯讀掛載到容器 `/root/data`。`config/tracks.json` 只影響資料檢查工具；執行定位另核對 `map_path`／`data_path`、`ply_path`、`init_guess_file`。保留 HW2 的資料與工作區。

舊的 `UBUNTU_DATA_DOWNLOAD.md` 與 `scripts/download_kaggle.py` 已移除。歷史上 Kaggle CLI 2.2.4 的 ACCESS_TOKEN／單檔與清單成功，Track01 整包與網頁 ZIP 曾 404；這不再是當前「尚無資料」的證據，也不構成重設 token 或重下載的理由。若實際缺檔，依現在可用的 Kaggle 網頁／CLI 補取；不強制使用自訂腳本。

## 接手檢查

- [x] 模板與官方 PDF 在目前 Windows checkout 可直接讀取。
- [x] 使用者確認 Ubuntu 的 Track1／Track2 與兩組 MapAnything 已建立。
- [ ] Ubuntu `check_data.py --strict-count` 通過並保存 JSON。
- [ ] 可解析 PLY 時間數量與唯一性、MapAnything index／report 完成核對。
- [ ] 找到校正資料與兩條 official sample，確認完整 ID、座標與單位。
- [ ] `.env`、實際容器掛載與程式輸入路徑一致。

被 Git 忽略的大型資料不會同步到 Windows 或其他 checkout；新 PDF 與文件若尚未追蹤，也需明確納入程式／文件的交接。
