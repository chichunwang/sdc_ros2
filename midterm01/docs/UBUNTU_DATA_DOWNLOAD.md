# 在 Ubuntu 下載 Kaggle 資料

以下全部在 **Ubuntu 主機的 Bash 終端機** 執行。可以使用目前 VS Code 的 SSH 終端機，也可以直接登入同一台 Ubuntu 操作；SSH 是連線方式，不是下載或執行 ROS 的必要條件。

Kaggle 網頁的「Download via」選擇 **Kaggle CLI**。此流程在 Ubuntu 主機保存原始資料，後續再由 Docker 掛載使用。

目前已確認 Ubuntu 使用 Kaggle CLI **2.2.4**，認證方式為 **ACCESS_TOKEN**，且 Track01 的測試 CSV 已下載成功。現有認證可用，不需要因整包下載的 404 重設 token。Track01 整包 `DownloadDataFiles` 請求仍回傳 404，使用者手動點網頁 ZIP 也顯示找不到頁面；同時回報 Track02 網頁下載正常（是否已完整下載、解壓待確認）。這讓問題較集中在 Track01 的整包下載，不能僅憑截圖確定後端原因。此版也不支援 `--unzip`。

**為什麼逐檔方式仍可行？** 不加 `-f` 的官方 CLI 取得整包 ZIP，使用 `DownloadDataFiles`；第 4 節工具則先列清單，再為每一檔呼叫官方 CLI 的 `-f`，使用單檔 `DownloadDataFile`。單檔服務可用不代表 ZIP 服務也可用；腳本沿用相同認證，沒有額外權限或取得隱藏連結。目前只驗證「測試 CSV 下載成功」與「舊版腳本預覽成功」，尚未確認腳本已下載全部 12,053 個檔案。依據：[官方 CLI 2.2.4 下載實作](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/src/kaggle/api/kaggle_api_extended.py)。

使用者希望了解不使用自訂腳本的方式：目前可先處理回報正常的 Track02，Track01 整包則需由助教／Kaggle 確認 ZIP 供應狀態或提供其他正式下載位置。第 4 節逐檔工具保留為備選，不視為使用者必須採用的流程。

使用者已在 Ubuntu 完成 Track01 全清單預覽：`Track1/` 有 **12,053 個檔案、約 9.95 GiB**；獨立的 `3cameras_ply/` 有 **3,013 個檔案、約 13.74 GiB**，目前未選取下載。清單中的 bag 與影像路徑有 `Track1/Track1/` 雙層目錄。舊版只在競賽根目錄辨識提交範例，顯示未找到不代表子目錄中也沒有；新版會顯示子目錄候選檔與其他 CSV。實際 `map.pcd`／`init_guess.txt` 所在目錄仍依完整清單辨識。

## 1. 安裝獨立的 Kaggle CLI

目前 Kaggle CLI 文件要求 Python 3.11 以上，並建議用隔離工具環境。以下透過 uv 指定 Python 3.12；CLI 的 Python 環境與 ROS 分開管理。來源：[Kaggle 安裝文件](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#installation)、[uv 安裝](https://docs.astral.sh/uv/getting-started/installation/)、[uv 指定工具的 Python](https://docs.astral.sh/uv/guides/tools/#requesting-python-versions)。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install --python 3.12 kaggle
kaggle --version
```

若已經有這套 CLI，可先確認版本及 `kaggle competitions download --help`。若 `curl` 不存在，先在 Ubuntu 安裝 `curl`。以一般使用者執行 uv 與 Kaggle，讓登入認證與資料留在自己的帳號下。

## 2. 加入兩場競賽並設定認證

用自己的 Kaggle 帳號加入兩場課程競賽並接受規則。接著到 [Kaggle API 設定](https://www.kaggle.com/settings/api)，在 API 區塊使用 **Generate New Token**。

先單獨執行下面這一行，貼上 token 並按 Enter；輸入不會顯示在畫面上：

```bash
read -rsp "Kaggle API Token: " KAGGLE_API_TOKEN
```

完成輸入後再執行：

```bash
printf '\n'
export KAGGLE_API_TOKEN
```

token 只貼在自己的終端機，不放進專案或聊天。上述環境變數只對目前 shell 及其後啟動的命令有效；重新開終端機時須再設定。若要使用持久登入，可依 [官方認證文件](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication) 將 token 存在使用者自己的 `~/.kaggle/access_token`，而非專案資料夾。

## 3. 準備下載位置並查看檔案清單

```bash
cd ~/sdc_ros2/midterm01
bash scripts/dev.sh init
df -h .

kaggle competitions files sdc-2026-midterm-01-track-01 --page-size 200
kaggle competitions files sdc-2026-localization-track-02 --page-size 200
```

清單每頁最多 200 筆；若回傳下一頁 token，還有其他檔案。先查看官方檔名與大小，確認可用空間。第一版需要每條 Track 的地圖、初始姿態、前相機點雲、校正資訊，以及該競賽的 sample submission。

## 4. 備選：逐檔下載並保留目錄

先將本專案的 `scripts/download_kaggle.py` 同步到 Ubuntu。它使用 Python 標準函式庫呼叫目前已能成功下載的 `kaggle` CLI，不需要將 Kaggle 套件安裝到 ROS 的 Python。

先預覽 Track01：

```bash
cd ~/sdc_ros2/midterm01
python3 scripts/download_kaggle.py --track track01
```

工具會自動翻完所有檔案清單頁面，顯示官方頂層目錄、大小、選取範圍與尚需下載容量。**預設僅預覽，不下載資料檔案。** 確認後執行：

```bash
python3 scripts/download_kaggle.py --track track01 --download
```

Track02 先確認帳號也已加入並接受規則，再依相同方式預覽、下載：

```bash
python3 scripts/download_kaggle.py --track track02
python3 scripts/download_kaggle.py --track track02 --download
```

下載範圍與保存方式：

- 預設選取官方 `Track1/` 或 `Track2/` 內的全部檔案，包含其中的影像／bag（若有），而非僅下載前相機 PLY。不選取另一個頂層目錄 `3cameras_ply/`。
- 以同層 `map.pcd` 與 `init_guess.txt` 辨識實際資料目錄，將該目錄內容放入本地 `data/Track1/` 或 `data/Track2/`。例如辨識為 `Track1/Track1/` 時，`Track1/Track1/raw_ply/a.ply` 會存為 `data/Track1/raw_ply/a.ply`。內部檔名與子目錄保留；不單憑兩個資料夾同名就裁切路徑。
- 下載前會顯示來源與本地路徑對應、地圖、初始姿態檔與 `raw_ply/` 的 PLY 數量。若找不到上述成對檔案，保留原本前綴並提示未找到必要檔案；此時先確認官方結構，不能宣稱資料已適用現有定位設定。發現多組候選資料目錄時停止，須用 `--prefix` 明確選取。
- 所選 Track 子目錄中的提交範例會一併下載與顯示。競賽根目錄、或自訂前綴外但位於同 Track 的 sample/submission CSV，存入該 Track 的本地目錄。依檔名辨識只是候選，仍需核對正式欄位與 ID；若未辨識到，會列出所選範圍的其他 CSV（最多 10 個）。
- 辨識成功且資料內部結構符合現有設定時，可沿用 `.env.example` 與 `config/tracks.json`。來源資料目錄外的附屬檔案仍保留原始路徑；若整理後有同名目的路徑衝突會停止，不互相覆寫。
- 如清單沒有預期的 Track 目錄，工具會列出實際頂層目錄並停止；確認後再用 `--prefix '實際目錄/'` 指定，不會改下載整場競賽。
- 已存在且大小符合清單的檔案會跳過；下載先進入個別暫存目錄，核對大小完成後才放到正式路徑。中斷後重跑會保留已完成的檔案，當前未完成檔案重新下載。
- 單檔若以 ZIP 包裝，工具只取出對應檔案並核對大小；清單本身列出的 ZIP 則保留原檔，是否需額外解壓依實際內容確認。
- 大小相同不等於內容校驗成功；目前未比對官方 checksum，也未驗證點雲／CSV 語意。逐檔下載包含多次 API 請求，會比整包連續下載慢。

若資料要放在其他磁碟，兩次執行使用相同的資料根目錄：

```bash
python3 scripts/download_kaggle.py --track track01 --data-root /實際資料根目錄
python3 scripts/download_kaggle.py --track track01 --data-root /實際資料根目錄 --download
```

此時 `.env` 的 `MIDTERM_DATA_DIR` 也須指向該根目錄。

工具已通過 16 項模擬測試，包含分頁、目錄保留、同名檔案、重跑跳過、失敗重試、大小不符時保留原檔、單檔 ZIP、雙層 Track 目錄、子目錄 CSV 與多組資料目錄衝突。Ubuntu 已驗證舊版清單預覽成功；新版整理邏輯與完整競賽下載仍待 Ubuntu 實測。

### 整包下載：目前失敗，留作診斷參考

Ubuntu 已實測下列 Track01 指令收到 404，目前不以它作為主要下載方式：

```bash
kaggle competitions download sdc-2026-midterm-01-track-01 -p data/kaggle_track01
```

已確認存在的檔案可使用 `-f` 單獨下載；CLI 2.2.4 可能省略上層路徑，因此正式批次下載使用上述工具來還原目錄。以下只是格式，需替換成實際檔名：

```bash
kaggle competitions download sdc-2026-midterm-01-track-01 \
  -f '清單中的完整檔名' -p data/kaggle_track01
```

目前已有使用者回報的 Track01 全清單預覽摘要、前五筆路徑與測試檔下載結果；助理尚未直接登入競賽或取得完整檔名列表。資料內容與容量以工具在 Ubuntu 讀取的正式清單為準。

## 5. 確認下載結果並接入專案

下載完成後，在 Ubuntu 執行：

```bash
cd ~/sdc_ros2/midterm01
find data/Track1 data/Track2 -type f \
  \( -name map.pcd -o -name init_guess.txt -o -iname '*submission*.csv' \)
```

逐檔工具的預設輸出為 `data/Track1/`、`data/Track2/`；若實際內部結構符合 `map.pcd`、`init_guess.txt`、`raw_ply/`，可直接使用目前的檢查設定。若資料是經由其他方式下載，先保留官方結構，再依實際結果選擇接入方式：

- 若 `map.pcd` 位於 `data/kaggle_track01/Track1/`，檢查設定的 Track01 `directory` 應為 `kaggle_track01/Track1`；Track02 依相同原則填入實際路徑。
- 也可以在確認來源與目標後，整理成原架構的 `data/Track1/`、`data/Track2/`，保留兩場各自的 sample submission。
- `.env` 的 `MIDTERM_DATA_DIR=./data` 對應上述資料根目錄；`config/tracks.json` 的 `directory` 是相對於該根目錄的 Track 路徑。
- `config/tracks.json` 目前只供資料檢查工具使用。啟動模板時的 `data_path`／`ply_path` 也須指定相對應的容器路徑，例如 `/root/data/kaggle_track01/Track1`。

完成路徑對應後，可先在 Ubuntu 主機以標準 Python 檢查結構，再處理容器建置：

```bash
python3 scripts/check_data.py --strict-count
```

這項檢查不驗證官方 ID、校正、點雲尺度或 pose；下載完成不等於定位程式已驗證。

## 常見狀況

### DownloadDataFiles 回傳 404

如果指令已通過參數解析，卻在 `competitions.CompetitionApiService/DownloadDataFiles` 收到 404，先檢查競賽存取及實際 CLI，不反覆重試下載或直接改裝其他版本。僅憑這個狀態碼，還不能判定是競賽代號、帳號存取或下載服務問題。

在 Ubuntu 執行下列唯讀檢查：

```bash
type -a kaggle
kaggle --version
kaggle competitions files -c sdc-2026-midterm-01-track-01
```

- 如果能列出檔案，表示該命令可讀取這場競賽的檔案清單；接著依實際清單選一個小檔測試單檔下載，再定位整包下載是否有問題。
- 如果列清單也失敗，先確認瀏覽器實際競賽網址、帳號已加入並接受規則，以及 CLI token 所屬帳號和瀏覽器帳號一致。
- `type -a` 與版本輸出用來確認是否執行了預期安裝的 CLI；不要從另一份文件的選項猜測本機版本。
- 分享上述輸出及可開啟的競賽網址即可，不需分享 token 或認證檔。

### 已能列清單：確認帳號與單檔下載

使用者已回報 CLI 2.2.4 成功列出 Track01 清單，且已下載 `3cameras_ply/Track1_Mapanything_ply/index.csv`（約 1.3 MB）。另已確認有效帳號與認證方式為 ACCESS_TOKEN；目前認證可用，整包下載 404 的確切原因仍未知。以下保留當時的診斷步驟。

此版本的認證流程會依序嘗試 access token、legacy key 與 OAuth，因此成功列檔也不代表一定採用了剛設定的環境變數。先只輸出有效帳號與認證方式：

```bash
kaggle config view | grep -E '^- (username|auth_method):'
```

比對 `username` 是否為瀏覽器上已加入競賽、接受規則的帳號。`KAGGLE_API_TOKEN` 環境變數若沒有設定，仍可能使用已儲存的其他認證，不能僅從變數是否存在就判定登入失敗。依據：[Kaggle CLI 2.2.4 認證實作](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/src/kaggle/api/kaggle_api_extended.py#L1141)。

接著只下載已確認存在的小檔測試：

```bash
cd ~/sdc_ros2/midterm01
mkdir -p data/kaggle_probe
kaggle competitions download -c sdc-2026-midterm-01-track-01 \
  -f '3cameras_ply/Track1_Mapanything_ply/index.csv' \
  -p data/kaggle_probe
```

此檔僅用來診斷下載服務，先與正式 Track 資料分開。CLI 2.2.4 的單檔與整包下載呼叫不同 API；單檔成功後可改走逐檔方式，並另行保留目錄結構。若單檔也失敗，仍需結合錯誤與帳號狀態判斷，不能只憑 404 就要求重設 token。依據：[2.2.4 單檔與整包下載實作](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/src/kaggle/api/kaggle_api_extended.py#L2054)。

使用者後續已實測 Track01 網頁 Download as zip 也顯示找不到頁面，Track02 則回報正常。暫不反覆要求重試 Track01 的相同整包指令；若不採逐檔下載，先使用正常的 Track02，並請助教確認 Track01 整包下載狀態。Windows 瀏覽器下載的檔案仍會存到 Windows，需要另行傳至 Ubuntu。

### 其他狀況

- `unrecognized arguments: --unzip`：移除該選項，使用本文件的獨立解壓步驟；這是參數解析錯誤，不能據此判斷認證是否成功。

- 401／認證失敗：確認目前終端機已設定自己的有效 token。
- 403／下載被拒絕：確認該帳號已加入對應競賽並接受規則，以及 token 的存取權限。
- SSH 中斷：下載程序是否繼續取決於終端機工作階段；長時間下載可在 Ubuntu 的 tmux 工作階段內執行。
- 直接使用 Ubuntu 桌面時，可用瀏覽器下載回報正常的 Track02 ZIP；Track01 網頁整包目前也失敗。成功下載後仍需依實際解壓結構接入相同資料根目錄。

本文件提供在使用者 Ubuntu 上操作的步驟，本次沒有讀取 token、登入 Kaggle 或代為下載資料。
