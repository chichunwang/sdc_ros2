"""gslam：Midterm02 自行撰寫的 GPU 視覺－幾何 SLAM 模組（不含 midterm01／midterm01_gpu 的程式）。

座標約定（全專案一致）：
- 相機座標為 OpenCV 軸（x 右、y 下、z 前）。
- pose 一律是 camera-to-world：T_wc，p_world = R_wc @ p_cam + t_wc。
- 配準的 source 是目前幀（相機座標），target 是世界座標的局部子圖；配準結果就是 T_wc。
"""
