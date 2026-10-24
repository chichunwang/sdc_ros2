from pathlib import Path
import numpy as np

path = Path(
    "../midterm01/data/Track3/raw_ply/"
    "000001_ts_1727239790269957473.ply"
)

with path.open(encoding="ascii") as stream:
    # 跳過描述檔案格式與欄位的表頭。
    for line in stream:
        if line.strip() == "end_header":
            break
    else:
        raise ValueError("找不到 PLY 表頭結尾")

    # 這份檔案前三欄依序是 x、y、z，後三欄是顏色。
    points = np.loadtxt(
        stream,
        usecols=(0, 1, 2),
        dtype=np.float64,
    )

print("點雲形狀:", points.shape)
print("前三個點:")
print(points[:6])