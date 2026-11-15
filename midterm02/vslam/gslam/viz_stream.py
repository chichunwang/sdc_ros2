"""把 SLAM 執行中的狀態串流給 RViz bridge（scripts/rviz_bridge.py）。

SLAM 容器是 Python 3.9、沒有 ROS；RViz 在另一個 ROS 2 Humble 容器。兩者用 TCP 傳「JSON 標頭＋原始陣列」：
    [4 bytes 標頭長度][JSON 標頭][陣列 1 bytes][陣列 2 bytes]...
視覺化只讀取 SLAM 狀態，不影響任何 pose；連線中斷時印出警告並停止串流，SLAM 繼續執行。
"""
import json
import socket
import struct
import time

import cv2
import numpy as np


class VizStream:
    def __init__(self, address, K, size, log=print, retries=30):
        host, port = address.rsplit(':', 1)
        self.log, self.sock = log, None
        for _ in range(retries):   # bridge 可能還在啟動
            try:
                self.sock = socket.create_connection((host, int(port)), timeout=2.0)
                self.sock.settimeout(10.0)
                break
            except OSError:
                time.sleep(1.0)
        if self.sock is None:
            raise ConnectionError('連不上 RViz bridge %s（先執行 scripts/dev.sh rviz）' % address)
        self.send('reset', dict(K=np.asarray(K, dtype=np.float64).tolist(), width=int(size[0]), height=int(size[1])))

    def send(self, kind, meta=None, **arrays):
        if self.sock is None:
            return
        header = dict(type=kind, meta=meta or {}, arrays=[])
        blobs = []
        for name, value in arrays.items():
            a = np.ascontiguousarray(value)
            header['arrays'].append(dict(name=name, dtype=str(a.dtype), shape=list(a.shape)))
            blobs.append(a.tobytes())
        head = json.dumps(header).encode()
        try:
            self.sock.sendall(struct.pack('>I', len(head)) + head + b''.join(blobs))
        except OSError as exc:
            self.log('警告：RViz 串流中斷（%s），SLAM 繼續執行但不再更新畫面' % exc)
            self.sock = None

    # ------------------------------------------------------------ 各種訊息
    @staticmethod
    def color_cloud(xyz, rgb, max_depth, step):
        """相機座標的彩色點雲（依深度過濾後等間隔取樣），供顯示用。"""
        keep = np.flatnonzero(np.all(np.isfinite(xyz), axis=1) & (xyz[:, 2] > 0.3) & (xyz[:, 2] < max_depth))[::step]
        return xyz[keep].astype(np.float32), np.clip(rgb[keep] * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def annotate(image, feats, prev_feats, matches, mask, lines):
        """半解析度影像：黃點＝特徵點，綠線＝通過幾何驗證的匹配（指向上一幀位置），紅線＝被剔除的匹配，左上角為狀態文字。"""
        out = cv2.resize(image, None, fx=0.5, fy=0.5)
        if feats is not None:
            for p in feats.kpts[::2]:
                cv2.circle(out, (int(p[0] / 2), int(p[1] / 2)), 1, (0, 255, 255), -1)
            if prev_feats is not None and matches is not None and len(matches):
                good = np.zeros(len(matches), dtype=bool) if mask is None else mask
                for k in range(0, len(matches), max(1, len(matches) // 500)):
                    a = prev_feats.kpts[matches[k, 0]] / 2
                    b = feats.kpts[matches[k, 1]] / 2
                    cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             (0, 220, 0) if good[k] else (0, 0, 255), 1, cv2.LINE_AA)
        for n, text in enumerate(lines):
            cv2.putText(out, text, (8, 20 + 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (8, 20 + 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def frame(self, rec, T, odom_T, image, annotated, xyz, rgb):
        arrays = dict(T=T, odom_T=odom_T, cloud_xyz=xyz, cloud_rgb=rgb)
        if image is not None:
            arrays.update(image=cv2.resize(image, None, fx=0.5, fy=0.5), annotated=annotated)
        self.send('frame', dict(idx=int(rec['idx']), status=rec['tracking_status']), **arrays)

    def keyframe(self, kf_id, xyz, rgb):
        self.send('keyframe', dict(kf_id=int(kf_id)), xyz=xyz, rgb=rgb)

    def graph(self, kf_poses, loops, submap=None):
        arrays = dict(kf_poses=np.asarray(kf_poses, dtype=np.float64),
                      loops=np.asarray(loops, dtype=np.int32).reshape(-1, 2))
        if submap is not None:
            arrays['submap'] = submap.astype(np.float32)
        self.send('graph', None, **arrays)

    def final(self, odom, optimized):
        self.send('final', None, odom=np.asarray(odom, dtype=np.float64), optimized=np.asarray(optimized, dtype=np.float64))
