# 21｜三相機資料核對與接入評估

日期：**2026-10-18（Asia/Taipei）**。本文保存使用者表示「如果更好的話可以加入三相機的 data」後的**初始只讀評估**，以下未實作／規畫等描述屬當時範圍。後續使用者已要求開始實作，完成尺度／同步工具與短段對照，見 [三相機實作與確認方法](22_THREE_CAMERA_IMPLEMENTATION_20261018.md)；尚無三相機 Kaggle 分數，完整 Track 仍由使用者手動執行。

## 為什麼值得優先研究

目前前相機＋heading_rts＋directional 的地圖資訊大多集中在高度，XY／yaw 仍主要跟隨感測器先驗。三相機若能增加可靠、不同朝向的側面幾何，就有機會補足水平位置與轉向的約束；這比單純增加同方向地面點更符合現有問題。現行成績與平面 Jacobian 說明見 [算法與精度分析](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)。

本機抽樣顯示三相機點雲的視野分布確實較廣，但**視野較廣不等於有更多可靠牆面，也不代表精度已改善**。還需核對尺度、重建品質與時間，再做配準對照。

## 已有資料，不需要重新下載

| 項目 | Track1 | Track2 |
| --- | --- | --- |
| 目前前相機輸入 | `../midterm01/data/Track1/raw_ply/` | `../midterm01/data/Track2/raw_ply/` |
| 現成三相機輸入 | `../midterm01/data/kaggle_track01/Track1_Mapanything_ply/ply/` | `../midterm01/data/kaggle_track02/Track2_Mapanything_ply/ply/` |
| 三相機 group／PLY 數 | 3011 | 3299 |
| 每路原始影像數，前／左前／右前 | 3011／3011／3011 | 3299／3299／3299 |
| index 前相機時間與目前 raw_ply 逐筆對應 | 通過 | 通過 |
| 以 index＋整數時間核對三路影像 | 缺檔 0 | 缺檔 0 |
| binary header／點數／payload 長度檢查 | 異常 0 | 異常 0 |
| 每幀合併點數中位數，未下採樣 | 327712 | 244796 |

表中路徑相對 GPU 根目錄；實際仍只有原 `midterm01/data` 一份，GPU 唯讀使用。這兩個 MapAnything 目錄各只有 `ply/`、`index.csv`、`report.json`；不是三路各自獨立的點雲目錄。

`index.csv` 保存 `f/fl/fr_image`、`f/fl/fr_ts` 與合併 PLY 路徑。index 影像名含 `_f/_fl/_fr`，本機同一相機子目錄的影像名則省略該相機標籤；本次以 **index＋整數 timestamp** 對應，沒有改名。原始三路影像及相機內外參均在 `TrackN/TrackN_images/`，校正檔為 `camera_info/lucid_camera_extrinsics.yaml`。

## 重要座標與融合證據

兩份 `report.json` 均宣告：

```text
coordinate_frame = model-predicted front-camera frame
fusion_uses_calibrated_tf = false
```

本次巡查全部 6310 份三相機 PLY header，均含：

```text
comment coordinate_frame=front_camera_FLU_x_front_y_left_z_up
```

因此，**這是已用模型預測相對姿態合併、並宣告為前相機 FLU 的點雲**。`fusion_uses_calibrated_tf=false` 並不表示三路尚未合併；不能再把整份 merged PLY 套一次左／右相機外參。header 是輸出宣告，仍需以地圖／影像幾何檢查實際是否符合。

MapAnything 上游 API 使用 OpenCV camera pose convention，並可提供 per-view pointmap、pose、confidence 與 metric scaling；但本機 PLY 已明確另行宣告 FLU，不能照上游原生 API 再做一次 optical-to-FLU 軸轉換。[上游官方輸出與介面](https://github.com/facebookresearch/map-anything)

抽樣的 PLY 只有 XYZRGB，沒有 `camera_id` 或 confidence；`index.csv.source_cache` 指向原製作者的外部路徑，本目錄沒有那些 per-view cache。僅靠現有 merged PLY，無法可靠分拆三路後重做相機外參／每路時間補償，或恢復原始 confidence。若證據指出預測的跨相機融合品質有問題，需取得對應原始 per-view 輸出，或另行從已有三路影像與校正重建；那是另一個開發階段，本次沒有啟動重建。

## 視野與時間抽查結果

每條 Track 取 idx `0、49、中間、最後`，共 8 組；每組讀目前前相機與三相機 PLY。這 16 份 PLY 的全部 XYZ 都有限，點數與 header 一致。用 `|atan2(y,x)| > 45°` 表示落在前向 ±45° 之外的比例：

| 抽樣幀範圍統計，非全序列分布 | Track1 | Track2 |
| --- | ---: | ---: |
| 目前前相機：前向 ±45° 之外 | 6.85%–24.34% | 11.96%–13.60% |
| 三相機：前向 ±45° 之外 | **65.87%–67.77%** | **62.98%–69.55%** |

這個比例也包含後向點，不是牆面比例、獨立觀測數或幾何正確率；只能支持「合併點雲涵蓋更多不同方位」的判斷。來源重建器、點密度與保留區域也不同，不能由點數或距離中位數直接估尺度倍率。

由完整 index 直接相減的初始數字如下。**後續 bag 核對已確認：前相機欄位是 bag 時間，左右是 header 時間，因此這張歷史表混合了不同時間來源，不能當作跨相機拍攝差。** 正確的同 header 時基比較見 [實作紀錄](22_THREE_CAMERA_IMPLEMENTATION_20261018.md#同步怎麼核對找到混合時間來源)。

| index 時間診斷 | Track1 | Track2 |
| --- | ---: | ---: |
| 左前 − 前，中位數 | −1.927 ms | −9.770 ms |
| 右前 − 前，中位數 | +12.315 ms | +13.316 ms |
| 同組最早至最晚，最大跨度 | 15.489 ms | 25.797 ms |

上表的最大 index 跨度 25.797 ms 不能再作為實際拍攝跨度或運動誤差的依據；後續已從相同 header 時基與原始感測器重算敏感度。仍不清楚既有輸出是否已處理時間差，不能直接重複補償。

## 建議如何接入，才有辦法判斷是否更好

1. **先核對 source 契約與尺度。** 保持原前相機版本作對照；以多個直行／轉彎與幾何豐富片段，檢查三相機 FLU 方向、地面／牆面距離與 map 的一致性。現行 `ply_scale=3.0` 對三相機沒有標定依據，也不能僅因模型名稱含 metric 就斷言必須用 1。只在可識別且有幾何支持的方向估計，保留其他片段／另一條 Track 交叉驗證；不逐 Track 或逐幀依排行榜猜倍率。
2. **契約通過後才新增可記錄的來源選擇。** 目前 benchmark 寫死前相機路徑與尺度，尚沒有三相機選項；不能只改 `config/tracks.json`。接入時需記錄 source 類型、index／report／輸入 hash、source→既有 pose frame 轉換及尺度，原始 PLY 不改動。沒有驗證前不提供看似可直接執行的三相機命令。
3. **先比較「目前前相機」與「現成三相機＋相同 directional」**，使用逐筆相同的感測器先驗、map、接受保護與評估片段。先在真實短段看可靠側向支持、不同 target 與空間跨度，以及修正穩定性／耗時；不只看殘差或回退數。此比較同時改變重建器與視野，不能把改善單獨歸功於多兩台相機；同模型 front-only 對照需另有對應輸出。
4. 觀察到有效幾何後，再獨立研究法向／空間均衡與重複 target 貢獻限制，避免大量相關地面點稀釋少量側面資訊。保留每幀總地圖項的合理正規化，不因點多就假設資訊成比例增加。多尺度地圖另做一項消融。
5. 完整 3011／3299 幀由使用者手動執行與提交。兩條正式分數與失敗片段均保留，再判斷是否採用；未得到完整結果前，現行最佳仍為前相機 heading_rts＋directional 的 **1.15275／0.30576**。

三相機是有資料支持的優先候選，仍需上述驗證才能判斷是否更準。本次沒有用三相機更動既有最佳 CSV，也沒有沿用其分數宣稱三相機成果。

## 本次證據與重現

有效報告：[完整 index／header 與抽樣 XYZ audit](../outputs/runs/three_camera_audit_261018_62wbvytl/report.json)；[獨立 audit 程式快照](../outputs/runs/three_camera_audit_261018_62wbvytl/audit.py)。報告包含來源 commit、index／report／校正 SHA256、抽樣 PLY SHA256、完整時間差及原始座標分位數。audit 不讀 map、不估 pose、不讀真值，也不啟動 localization。

最初 audit 因誤把 index 的相機標籤要求套到本機影像名，出現假缺檔；該目錄已加入 `AUDIT_INVALID.txt`，保留失敗紀錄，未引用為資料缺失證據。上方為修正成 index＋整數時間對應後重新完成的有效報告，缺檔為 0。

outputs 被 Git 忽略，交接需另行保存報告；本文保留關鍵數字與範圍，不能把 header／有限值核對當成完整幾何或定位精度驗證。
