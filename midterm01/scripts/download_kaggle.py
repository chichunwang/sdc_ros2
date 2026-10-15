#!/usr/bin/env python3
"""以現有 Kaggle CLI 逐檔下載 Track，保留官方目錄；預設只預覽。"""

import argparse
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


TRACKS = {
    "track01": ("sdc-2026-midterm-01-track-01", "Track1/"),
    "track02": ("sdc-2026-localization-track-02", "Track2/"),
}


def run_kaggle(arguments):
    result = subprocess.run(
        ["kaggle", *arguments], capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Kaggle 指令失敗（{result.returncode}）：{detail}")
    return result.stdout


def parse_page(output):
    # 2.2.4 即使用 --format json，仍在 JSON 前輸出下一頁 token。
    lines = output.strip().splitlines()
    next_token = None
    if lines and lines[0].startswith("Next Page Token = "):
        next_token = lines.pop(0).split(" = ", 1)[1].strip()
    payload = "\n".join(lines).strip()
    records = [] if payload == "No files found" else json.loads(payload)
    if not isinstance(records, list):
        raise ValueError("Kaggle 檔案清單不是 JSON 陣列。")
    files = []
    for record in records:
        name, size = record["name"], record["size"]
        if not isinstance(name, str) or not name:
            raise ValueError("Kaggle 回傳無效的檔名。")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Kaggle 回傳無效的檔案大小：{name}")
        files.append((name, size))
    return files, next_token


def list_files(competition):
    files = {}
    token = None
    seen_tokens = set()
    page = 0
    while True:
        command = ["competitions", "files", competition, "--page-size", "200", "--format", "json"]
        if token:
            command.extend(["--page-token", token])
        records, token = parse_page(run_kaggle(command))
        for name, size in records:
            if name in files and files[name] != size:
                raise ValueError(f"分頁中的檔案大小不一致：{name}，請重新讀取清單。")
            files[name] = size
        page += 1
        if page == 1 or page % 10 == 0:
            print(f"已讀取 {page} 頁，共 {len(files)} 個檔案。", flush=True)
        if not token:
            return files
        if token in seen_tokens:
            raise ValueError("Kaggle 回傳重複的下一頁 token，停止以免無限讀取。")
        seen_tokens.add(token)


def destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"不接受這個遠端檔案路徑：{name}")
    target = root.joinpath(*relative.parts)
    target.resolve().relative_to(root.resolve())
    return target


def is_sample(name):
    path = PurePosixPath(name)
    return path.suffix.lower() == ".csv" and (
        "sample" in path.name.lower() or "submission" in path.name.lower()
    ) and len(path.parts) == 1


def download_file(competition, name, size, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # 每檔使用獨立暫存目錄：2.2.4 可能只保存 URL 的 basename。
    # 驗證大小完成才移到正式路徑，不把中斷檔當作已完成檔案。
    with tempfile.TemporaryDirectory(prefix=".kaggle-", dir=target.parent) as stage:
        run_kaggle(["competitions", "download", competition, "-f", name, "-p", stage])
        artifacts = [p for p in Path(stage).rglob("*") if p.is_file()]
        if len(artifacts) != 1:
            raise ValueError(f"單檔下載產生 {len(artifacts)} 個檔案，無法確認內容：{name}")
        artifact = artifacts[0]
        if artifact.stat().st_size != size and zipfile.is_zipfile(artifact):
            # 若服務以 ZIP 包裝單檔，只讀取對應項目，不直接展開任意路徑。
            with zipfile.ZipFile(artifact) as archive:
                candidates = [
                    member for member in archive.infolist()
                    if not member.is_dir() and member.filename in (name, PurePosixPath(name).name)
                ]
                if len(candidates) != 1 or candidates[0].file_size != size:
                    raise ValueError(f"單檔 ZIP 內容與清單不符：{name}")
                unpacked = Path(stage) / ".unpacked-payload"
                with archive.open(candidates[0]) as source, unpacked.open("wb") as output:
                    shutil.copyfileobj(source, output)
                artifact = unpacked
        if artifact.stat().st_size != size:
            raise ValueError(f"下載大小與清單不符：{name}（預期 {size} bytes）")
        artifact.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--prefix", help="以清單中的實際目錄前綴取代預設 Track1/ 或 Track2/。")
    parser.add_argument("--download", action="store_true", help="實際下載；未指定時只列預覽。")
    args = parser.parse_args()
    if not shutil.which("kaggle"):
        parser.error("找不到 kaggle；請使用已能成功下載測試檔的 Ubuntu 終端機。")

    competition, default_prefix = TRACKS[args.track]
    prefix = (args.prefix or default_prefix).rstrip("/") + "/"
    root = args.data_root.expanduser().resolve()
    try:
        files = list_files(competition)
        groups = defaultdict(lambda: [0, 0])
        for name, size in files.items():
            group = name.split("/", 1)[0] + "/" if "/" in name else "（根目錄檔案）"
            groups[group][0] += 1
            groups[group][1] += size
        print("\n官方清單（只讀取檔名與大小，尚未下載資料）：")
        for group, (count, size) in sorted(groups.items()):
            print(f"  {group}: {count} 個檔案，{size / 1024**3:.2f} GiB")
        if not any(name.startswith(prefix) for name in files):
            raise ValueError(f"清單中沒有 {prefix}；請依上方實際目錄使用 --prefix，不猜測檔名。")

        selected = []
        for name, size in sorted(files.items()):
            if name.startswith(prefix):
                target = destination(root, name)
            elif is_sample(name):
                # 兩場根目錄同名 CSV 各放在自己的 Track，避免互相覆寫。
                target = destination(root, prefix + name)
            else:
                continue
            selected.append((name, size, target))
        targets = [target for _, _, target in selected]
        if len(set(targets)) != len(targets):
            raise ValueError("根目錄 CSV 與 Track 內檔案的目的路徑重複，請先確認官方結構。")
        pending = [(name, size, target) for name, size, target in selected
                   if not (target.is_file() and target.stat().st_size == size)]
        needed = sum(size for _, size, _ in pending)
        print(f"\n競賽：{competition}\n下載範圍：{prefix} 與根目錄 sample/submission CSV")
        print(f"資料根目錄：{root}")
        print(f"選取 {len(selected)} 個檔案；已完成 {len(selected) - len(pending)} 個。")
        print(f"尚需下載 {len(pending)} 個，約 {needed / 1024**3:.2f} GiB。")
        for name, _, target in selected[:5]:
            print(f"  {name} -> {target.relative_to(root)}")
        samples = [name for name, _, _ in selected if is_sample(name)]
        print("根目錄提交範例：" + (", ".join(samples) or "未找到；請另核對 Kaggle Data 頁的正式範例。"))
        if not args.download:
            print("\n目前只預覽。確認範圍與容量後，加上 --download 才會下載。")
            return 0
        if not pending:
            print("已選取檔案的大小均符合清單，無須下載。")
            return 0
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < needed:
            raise ValueError("磁碟可用空間小於尚需下載的資料量；請先騰出空間或改用 --data-root。")
        for index, (name, size, target) in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {name}", flush=True)
            for attempt in range(3):
                try:
                    download_file(competition, name, size, target)
                    break
                except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
                    if attempt == 2:
                        raise
                    print("下載未完成，稍後重試。", flush=True)
                    time.sleep(2 ** attempt)
        print("下載完成。已核對檔案大小；未驗證點雲內容或官方校驗碼。")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"錯誤：{error}\n可在排除問題後重跑；已完成且大小相符的檔案會跳過。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中止；重跑會跳過已完成的檔案，當前檔案需重新下載。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
