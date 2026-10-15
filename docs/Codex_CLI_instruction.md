你可以在 **Ubuntu 主機的專案根目錄啟動 Codex CLI**，再讓它透過 Docker 執行 ROS 建置與測試。

1. **先同步剛才更新的文件。**

   上一輪修改在 Windows，尚未提交 Git。請透過 Git 或直接複製，把 `AGENTS.md`、`midterm01/` 的更新文件與官方 PDF 同步到 Ubuntu；資料保留在 Ubuntu 原位。

2. **在 Ubuntu 終端機啟動：**

   ```bash
   cd ~/sdc_ros2
   codex
   ```

   第一次啟動依畫面登入。Codex 會在啟動時載入專案的 `AGENTS.md`。[官方啟動說明](https://learn.chatgpt.com/docs/codex/cli)、[AGENTS.md 說明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

3. **進入 Codex 後，直接貼上這段交接提示：**

   ```text
   請接手這個 Midterm01 定位專案，使用繁體中文。

   先遵守根目錄 AGENTS.md，閱讀：
   - midterm01/README.md
   - midterm01/docs/README.md
   - midterm01/docs/KAGGLE_DATASET_SPEC.md
   - midterm01/data/README.md
   - midterm01/docs/ 裡的官方比賽 PDF

   資料已建立在 ~/sdc_ros2/midterm01/data，
   包含 Track1、Track2，以及兩組 MapAnything 點雲。

   本次請依序完成：
   1. 檢查 Git 狀態、既有環境與資料完整性，保存檢查報告。
   2. 核對 PLY 時間、初始矩陣、尺度、座標與官方 frame 對應。
   3. 建置 Midterm01 Docker 與 ROS package。
   4. 處理阻礙首跑的依賴、唯讀地圖快取與 GUI 問題。
   5. 使用現有前相機 NDT，先完成單幀，再完成 50 幀測試。
   6. 保存命令、參數、輸出與耗時，更新文件中的實際進度。

   保留既有修改、原始資料與舊結果。
   Docker 管理在 Ubuntu 主機執行，ROS 與 colcon 在容器內執行。
   請直接執行與驗證，遇到需要我提供資訊或操作的阻塞時再說明。
   ```

使用過程中常用這幾個指令：

| 在 Codex 裡輸入 | 用途 |
|---|---|
| `/status` | 查看目前設定 |
| `/permissions` | 查看或調整操作權限 |
| `/diff` | 查看修改 |
| `/quit` | 離開 |

以上是 [官方 CLI 指令](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。

下次回到 **Ubuntu 終端機**，可接續最近一次對話：

```bash
cd ~/sdc_ros2
codex resume --last
```

這會接續該目錄最近的 CLI 對話。[官方接續說明](https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-resume)
---
# 從桌面版轉到 Codex CLI

繁體中文使用指南｜Windows / PowerShell 優先｜核對日期：2026-10-13

## 01｜先用這一頁開始

Codex CLI 讓你在終端機用自然語言交代工作，讀取專案檔案、修改內容、執行工具並檢查結果。你仍然可以用繁體中文對話；需要適應的是「先選工作資料夾，再開始聊天」。官方入門見 [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)。

### 你的電腦已經有 CLI

本次只讀檢查找到 `codex.exe`，版本為 `0.162.0-alpha.17.2`，位於桌面程式管理的安裝目錄。本次沒有替你安裝、更新、登入或修改設定。在你自己新開的 PowerShell 中，再確認一次是否找得到它。

```powershell
codex --version
codex login status
```

若尚未登入，執行 `codex login`，用原本的 ChatGPT 帳號完成瀏覽器登入。接著，把以下範例路徑換成你真正要工作的資料夾：

```powershell
Set-Location -LiteralPath 'C:\Projects\my-project'
codex --sandbox workspace-write --ask-for-approval on-request
```

進入 Codex 後，輸入：

```text
請用繁體中文說明這個資料夾的結構、用途與重要檔案。
先讀取並說明，不要修改檔案。
```

### 桌面操作與 CLI 的對應

| 桌面版習慣 | CLI 做法 |
| --- | --- |
| 選擇本機專案資料夾 | 先 `Set-Location`，或用 `codex -C '路徑'` |
| 開新聊天 | 在終端機執行 `codex`；聊天內可用 `/new` |
| 選模型、查看權限 | 聊天內用 `/model`、`/permissions`、`/status` |
| 查看修改內容 | 聊天內 `/diff`，或離開後 `git diff` |
| 回到舊工作 | 終端機用 `codex resume` |

CLI 的資料夾工作方式見 [Projects and chats](https://learn.chatgpt.com/docs/projects)。指令詳解見 [Developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。

閱讀順序：第 2 節安裝與登入 → 第 3 至 5 節日常操作 → 第 6 至 8 節設定與轉移 → 第 9 至 10 節腳本及排錯。

<!-- pagebreak -->

## 02｜安裝、更新與登入

### Windows 主路徑：官方獨立安裝程式

若新開的 PowerShell 找不到 `codex`，使用官方安裝指令。這種方式不必先安裝 Node.js；Windows 原生執行也不必先安裝 WSL。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

完成後重新開啟 PowerShell，執行 `codex --version`。官方也提供同一指令作更新用途。若你使用的是桌面版附帶的 CLI，先確認實際執行位置，避免裝了兩份後更新到另一份。

```powershell
Get-Command codex -All | Select-Object Name, Source
where.exe codex
```

另一條路徑是 npm：先具備可用的 Node.js / npm，再執行 `npm install -g @openai/codex`；更新也可使用同一指令。支援自行更新的版本另有 `codex update`。選一個安裝來源長期維護即可。見 [CLI 安裝方式](https://learn.chatgpt.com/docs/codex/cli)及 [更新指令](https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-update)。

### 延續原本帳號：優先選 ChatGPT 登入

```powershell
codex login
codex login status
```

瀏覽器登入時確認帳號與工作區。ChatGPT 登入使用帳號方案提供的 Codex 存取；實際可用功能、額度及模型仍依方案與工作區政策。

API key 登入則使用 OpenAI Platform 的 API 計費，不是沿用 ChatGPT 方案內含額度。若只是從桌面版改用 CLI，不必為此先建立 API key。見 [Authentication](https://learn.chatgpt.com/docs/auth)。

若需要 API key，而且已由自己的安全流程設定好環境變數：

```powershell
$env:OPENAI_API_KEY | codex login --with-api-key
```

不要把金鑰直接寫進範例腳本、聊天或 Git。認證檔 `auth.json` 也不要分享。需要更換登入方式時才使用 `codex logout`，再重新登入。

其他系統：macOS / Linux 可用官方獨立安裝程式；macOS 也可用 `brew install --cask codex`。完整指令請查閱上方官方安裝頁。

<!-- pagebreak -->

## 03｜一次完整的日常工作

### 1. 進資料夾，確認目前狀態

```powershell
Set-Location -LiteralPath 'C:\Projects\my-project'
git status
codex --sandbox workspace-write --ask-for-approval on-request
```

`git status` 只適用於 Git 專案。沒有 Git 的一般文件資料夾也能用互動式 Codex；先備份重要檔案即可。若已有尚未提交的修改，先了解哪些是你原本的工作。

### 2. 交代目標、範圍與完成條件

```text
請修正登入後仍跳回登入頁的問題。
先找出原因，只修改與這個問題有關的檔案。
遵守 AGENTS.md，執行相關測試。
完成後說明修改原因、驗證結果及未解決事項。
```

沒有寫程式也能這樣使用：

```text
請讀取 notes 資料夾中的 Markdown，整理成一份繁體中文摘要。
保留重要日期、名詞及待辦事項，輸出到 outputs/summary.md。
若內容互相矛盾，列出矛盾，不要自行補造資訊。
```

大型工作可先輸入 `/plan`，討論步驟與完成條件，再切回執行模式。這裡的範例是建議的工作寫法，可依你的任務調整。

### 3. 在同一聊天中追問或修正方向

```text
請把剛才的改動縮小到登入流程，保留既有 API 介面。
```

### 4. 檢查結果，再保存

聊天內使用 `/diff` 查看修改，`/review` 請 Codex 審查。測試是否真的通過，要看執行結果；檢查報告也要留意它說「未執行」的項目。

用 `/quit` 離開後，在終端機執行：

```powershell
git status
git diff
```

確認內容後再提交你要保存的檔案。Git 是版本保存工具；聊天紀錄本身不能取代檔案備份。官方工作流程見 [CLI 入門](https://learn.chatgpt.com/docs/codex/cli)及 [命令參考](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。

<!-- pagebreak -->

## 04｜兩種指令，分清楚在哪裡輸入

### A. 在 PowerShell 提示符輸入

看到 `PS C:\...>` 時，輸入的是終端機指令。

| 指令 | 用途 |
| --- | --- |
| `codex` | 開互動式聊天 |
| `codex -C 'C:\Projects\my-project'` | 指定工作資料夾後開聊天 |
| `codex '請說明這個專案'` | 開聊天，並送出第一個要求 |
| `codex resume` | 選擇以前的本機聊天 |
| `codex resume --last` | 延續目前資料夾最近的聊天 |
| `codex resume --all` | 在選單中查看跨資料夾的聊天 |
| `codex fork --last` | 複製最近聊天，開另一條工作路線 |
| `codex review --uncommitted` | 審查尚未提交的改動 |
| `codex --help` | 查這個版本支援的指令 |

### B. 在 Codex 的聊天輸入框輸入

| 指令 | 用途 |
| --- | --- |
| `/status` | 查看聊天狀態、上下文使用量與額度資訊 |
| `/model` | 選可用模型；可用選項依帳號及版本 |
| `/permissions` | 調整目前聊天的存取與核准方式 |
| `/plan` | 切換規劃模式 |
| `/init` | 產生專案用 `AGENTS.md` 草稿，需自行檢查 |
| `/diff`、`/review` | 查看修改、審查改動 |
| `/compact` | 整理長對話，釋放上下文空間 |
| `/skills`、`/plugins`、`/mcp` | 查看技能、外掛或 MCP 狀態 |
| `/new`、`/rename` | 開新聊天、重新命名目前聊天 |
| `/quit` 或 `/exit` | 離開 CLI |

不必一次記完。先記住 `codex`、`codex resume`、`/status`、`/diff`、`/quit`。

在聊天輸入 `/` 可查看目前版本的清單；空白輸入框中按 `?` 可查看快捷鍵。長提示詞可用 `Ctrl+G` 開啟 `VISUAL` 或 `EDITOR` 指定的編輯器。見 [Developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)與 [CLI customization](https://learn.chatgpt.com/docs/cli-customization)。

<!-- pagebreak -->

## 05｜權限：先懂兩個設定

**Sandbox 決定指令能存取什麼；approval 決定何時請你核准。** 它們是不同設定。只說「不要問我」不等於解除檔案或網路限制。

| 設定 | 意義 |
| --- | --- |
| `--sandbox read-only` | 指令以唯讀方式工作，適合理解與分析 |
| `--sandbox workspace-write` | 可在工作區內寫入；額外路徑與網路仍受政策限制 |
| `--ask-for-approval on-request` | 由 Codex 判斷何時提出核准要求 |
| `--ask-for-approval never` | 不向人詢問；受限操作可能直接失敗 |

### 只想理解檔案

```powershell
codex --sandbox read-only --ask-for-approval on-request
```

### 要讓它修改目前資料夾

```powershell
codex --sandbox workspace-write --ask-for-approval on-request
```

如果需要寫入另一個工作資料夾，可以在啟動時明確加入：

```powershell
codex -C 'C:\Projects\my-project' --add-dir 'C:\Projects\shared'
```

`--add-dir` 擴充可寫入範圍；請同時檢查 `/permissions` 與 `/status` 的實際設定。公司電腦的管理政策可能限制你能選的權限。

### Windows 的 Sandbox

目前 Windows 支援原生 Sandbox。官方優先採用裝置及政策支援的 MXC，並保留 `elevated`、`unelevated` 等 fallback；CLI 初次使用時依畫面提示完成設定。若程式要求舊式 elevated setup，可能需要管理員核准。

WSL 是另一個開發環境選項。先用 PowerShell 即可；若改在 WSL 執行，安裝位置、路徑、工具與設定環境都要重新確認。

本指南使用明確的權限參數。舊教學常見的 `--full-auto` 已是相容性參數；新腳本請使用明確的 sandbox 設定。不要為了省去提示而直接使用跳過核准及 sandbox 的參數。

來源：[Agent approvals & security](https://learn.chatgpt.com/docs/agent-approvals-security)、[Windows sandbox](https://learn.chatgpt.com/docs/windows/windows-sandbox)、[Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)。

<!-- pagebreak -->

## 06｜把偏好寫成設定與 AGENTS.md

### config.toml：控制執行方式

個人設定預設在 `~/.codex/config.toml`。Windows 通常對應 `C:\Users\你的使用者名稱\.codex\config.toml`；若設定過 `CODEX_HOME`，位置會不同。修改前先備份，將所需欄位合併進現有設定，不要整份覆蓋。

```toml
approval_policy = "on-request"
sandbox_mode = "workspace-write"
web_search = "cached"
model_reasoning_effort = "medium"
```

初期可先保留預設模型，用 `/model` 查看帳號可用選項。思考強度也要由模型支援；遇到不支援的設定，就移除該欄位或在選單重選。

若要永久指定模型，可以加入 `model = "實際模型ID"`，但請換成選單或官方文件確認的真實 ID。

啟動參數可覆蓋設定檔，例如 `codex --search` 使用即時網頁搜尋。個人設定之外，已信任專案的 `.codex/config.toml` 也可提供專案設定；管理政策仍有約束力。見 [Config basics](https://learn.chatgpt.com/docs/config-file/config-basic)。

### AGENTS.md：控制工作習慣

適合放「用什麼語言回答、輸出到哪裡、如何驗證」等約定。全域檔可放 `~/.codex/AGENTS.md`；專案檔放在專案根目錄。較接近目前工作目錄的指示可覆蓋較上層指示，`AGENTS.override.md` 另有優先規則。

可用 `/init` 產生草稿，或自行建立以下內容：

```markdown
# 工作約定

- 使用繁體中文回答，保留必要的英文技術名詞。
- 開始前先讀取專案說明與相關檔案。
- 暫存資料放 work/，完成品放 outputs/。
- 修改後執行與本次變更相關的驗證。
- 完成時列出修改檔案、驗證結果及未完成事項。
```

這是可調整的範例。若專案已有檔案，先閱讀並合併。改好後開新的 CLI 聊天，請它「列出已載入的指示來源與工作約定」，確認生效。見 [Custom instructions with AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

<!-- pagebreak -->

## 07｜Skills、Plugins、MCP 與本機工具

### 先理解各自用途

| 項目 | 作用 | 例子 |
| --- | --- | --- |
| Skill | 保存某種工作的流程與規格 | 複習講義、報告格式、PDF 排版 |
| Plugin | 封裝技能、工具或服務整合 | 文件工具、GitHub、Google Drive |
| MCP | 把外部工具提供給 Codex 的協定 | 連接本機或遠端工具服務 |
| 本機程式 | 真正處理檔案或跑指令的環境 | Python、Git、Node.js、PDF 轉檔工具 |

在 Codex 聊天內使用 `/skills` 查看技能，或輸入 `$` 選技能；`/plugins` 查看外掛。終端機可用 `codex plugin list`、`codex mcp list` 查看工具設定。見 [Skills & Plugins](https://learn.chatgpt.com/docs/skills-and-plugins)與 [命令參考](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。

### 轉移你自己的技能

目前官方文件列出的個人技能位置是 `~/.agents/skills`，專案技能位置是 `.agents/skills`。一個技能是一個資料夾，裡面有 `SKILL.md`，可另附腳本、範本或參考檔。

```text
my-project/
  AGENTS.md
  .agents/
    skills/
      my-report/
        SKILL.md
        references/
```

你目前桌面環境也有 `.codex/skills` 下的自訂技能。這是本次環境觀察，不代表每個獨立 CLI 版本都會以同樣方式載入。先用 `/skills` 確認；若缺少自己的技能，再把整個技能資料夾複製到目前文件支援的位置，保留原本檔案。見 [Build skills](https://learn.chatgpt.com/docs/build-skills)。

### 有技能不代表依賴都已安裝

若你會產生 PDF、Word、簡報或試算表，CLI 還需要對應外掛或可用的處理工具。桌面版提供的工具、執行環境與預覽功能是否在獨立 CLI 可用，要逐項確認。

第一個驗證任務可以是：「請使用我的報告技能，產生一頁測試文件到 outputs，並確認中文字體與排版。」若工具缺少，先確認缺的是技能、外掛連線或本機程式，再補上。

新的外掛或技能加入後，開一個新聊天確認清單與工具；外部服務若需要登入，依其連線流程完成授權。

<!-- pagebreak -->

## 08｜聊天紀錄、檔案與圖片怎麼銜接

### 先辨識原本是哪種聊天

桌面版中的本機 Codex 聊天，與一般 ChatGPT / Work 對話可能屬於不同紀錄系統。不要只因為用了相同帳號，就假設所有對話都能在 CLI 延續。官方也區分 Work 與 Codex 的歷史紀錄。見 [Projects and chats](https://learn.chatgpt.com/docs/projects)。

先在原本工作的資料夾中嘗試：

```powershell
codex resume
codex resume --all
```

若找到原本聊天，就選取並確認工作目錄。你這台電腦的版本另外提供 `codex agents`，可瀏覽共用本機 app-server 上的工作；這是本次 `--help` 核對到的版本能力，其他安裝版本請先查 `codex agents --help`。

若找不到，建議在桌面聊天請它整理「目標、已完成工作、重要檔案路徑、決定與下一步」，存成 `handoff.md`。在 CLI 請它閱讀這份檔案並繼續；不需要手動搬動內部聊天資料庫。

反向從 CLI 回桌面版，支援的版本可在聊天內輸入 `/app`，開啟同一個已保存聊天。見 [Developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。

### 文件：讓 Codex 讀到正確的實體檔案

把資料放進工作資料夾，交代檔名、範圍及輸出位置。例如：

```text
請讀取 materials/lecture.md 和 work/handoff.md。
依這兩份內容繼續整理，輸出 outputs/review.md。
```

PDF / DOCX 等格式可能需要解析工具。跨電腦繼續聊天時，也要確認新環境真的能存取那些檔案；對話同步不等於整個本機資料夾已同步。

### 圖片：啟動時附上

```powershell
codex -i '.\screenshots\error.png' '請分析錯誤並提出修正步驟'
```

圖片貼上支援會受終端機與系統影響，檔案參數較容易確認。CLI 支援圖片輸入，但不能因此推定桌面版所有預覽、瀏覽器、語音或電腦操作能力都已具備。見 [Image inputs](https://learn.chatgpt.com/docs/image-inputs)。

<!-- pagebreak -->

## 09｜進階：用 codex exec 做一次性工作

`codex` 是持續聊天；`codex exec` 適合一次交代、跑完後取得結果，方便腳本與批次處理。初期先熟悉互動模式，再使用這一節。

### 只分析，保存最後答覆

先確定 `outputs` 資料夾存在；以下範例在 Git 專案根目錄執行。

```powershell
codex exec --sandbox read-only -o '.\outputs\summary.md' '用繁體中文摘要這個專案'
```

`-o` 保存 Codex 最後一則答覆，由 CLI 寫出；它不代表任務要求的所有附件都會自動保存。若你要程式實際產生其他檔案，需給予適當寫入權限並指定路徑。

### 允許在工作區產生檔案

```powershell
codex exec --sandbox workspace-write '整理 notes 並建立 outputs/notes-summary.md'
```

### 把長提示詞存成檔案再執行

```powershell
Get-Content -Raw -Encoding utf8 '.\work\prompt.txt' | codex exec -
```

### 記錄機器可讀的執行事件

```powershell
codex exec --json '摘要這個專案' | Set-Content -Encoding utf8 '.\work\events.jsonl'
```

`--json` 產生 JSONL：每一行是一個事件，並不是只有最終回答的一份 JSON。若只要最後答覆，使用 `-o`。執行後可查看 `$LASTEXITCODE`，仍需檢查產出內容與任務完成情況。

一般執行時，進度會送到 stderr，最後答覆送到 stdout。需要續跑時，可用 `codex exec resume --last '繼續上一項工作'`。見 [Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)。

`exec` 預設會檢查 Git 儲存庫。若是在你確認可操作的一般文件資料夾，可加 `--skip-git-repo-check`。這只是略過 Git 檢查，不會替你備份或解除 sandbox。

腳本執行可能沿用現有設定；範例明確寫出 sandbox，是為了讓該次工作的權限容易辨識。若會跑測試或產生 build 檔，通常需要 `workspace-write`。

<!-- pagebreak -->

## 10｜常見問題與轉移完成清單

| 遇到的情況 | 先做這件事 |
| --- | --- |
| 找不到 `codex` | 重開 PowerShell；用 `Get-Command codex -All` 查路徑；仍找不到再安裝 |
| 更新後版本沒變 | 查是否有多份 CLI，確認目前 PATH 選到哪一份 |
| 找不到 `npm` | 若走官方獨立安裝，可不使用 npm；若選 npm 路徑才補其環境 |
| 登入未成功 | 用 `codex login status`；必要時再用 `codex login` 完成登入 |
| 模型不可用或額度耗盡 | 查看 `/model`、`/status`，確認帳號、工作區與登入方式 |
| 看不到原本聊天 | 用 `codex resume --all`；仍缺少就以 handoff.md 銜接 |
| 無法寫入或跑測試 | 查看 `/permissions`；確認工作資料夾、sandbox 與允許路徑 |
| Sandbox 啟動失敗 | 重啟 CLI；依 Windows 官方排錯頁確認裝置支援與設定 |
| 中文或終端畫面異常 | 改用支援中文的終端字體；可嘗試 `codex --no-alt-screen` |
| 舊教學的指令不能用 | 用 `codex --help` 或 `codex 子命令 --help` 核對目前版本 |

支援的版本可用 `codex doctor` 產生診斷報告。分享前先確認內容沒有你不想公開的路徑、專案資訊或認證資料。診斷用途見 [Developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-doctor)，Windows 問題見 [Windows sandbox](https://learn.chatgpt.com/docs/windows/windows-sandbox)。

### 完成這七項，就能開始日常使用

- [ ] 在自己新開的 PowerShell 能執行 `codex --version`。
- [ ] 確認目前是 ChatGPT 登入或 API key 登入。
- [ ] 能指定工作資料夾，請它正確說明檔案內容。
- [ ] 能在小任務中修改檔案，查看差異並驗證結果。
- [ ] 知道如何用 `codex resume` 延續工作。
- [ ] 在新聊天確認 AGENTS.md、技能及必要工具有載入。
- [ ] 重要檔案有備份或版本紀錄。

### 版本與資料來源說明

本文依 2026-10-13 取得的 OpenAI 官方文件整理，並以這台電腦的 CLI `--help` 核對常用參數。alpha 版本、獨立安裝版本與桌面版附帶版本可能不同；可用模型、功能與權限以你的帳號和當前版本為準。

文中的 `C:\Projects\my-project`、資料夾及檔名都是範例，須換成實際路徑。部分處理工具與技能需要另外安裝或連線。這份指南提供操作方法，沒有替你更動電腦設定。

如需查最新操作，從 [Codex CLI 官方入門](https://learn.chatgpt.com/docs/codex/cli)進入；完整設定查 [Config reference](https://learn.chatgpt.com/docs/config-file/config-reference)。
