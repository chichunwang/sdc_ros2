#!/usr/bin/env python3
"""Render the Track3 v8.1 architecture as editable SVG and PNG, without CUDA.

Run on the Ubuntu host: python3 midterm01_gpu/doc/analysis/render_v8_1_architecture.py
Requires the host's python3-gi, GdkPixbuf SVG loader and Pango introspection packages.
"""
from html import escape
from pathlib import Path
import json
import os
# VS Code's Snap loader cache points at a different GLib/libc. Use host loaders
# in this renderer process only; do not change the user's desktop environment.
for key in ('GDK_PIXBUF_MODULE_FILE', 'GIO_MODULE_DIR', 'GTK_PATH'):
    os.environ.pop(key, None)
import gi

gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import GdkPixbuf, Pango, PangoCairo

DOC = Path(__file__).resolve().parents[1]
STEM = '63_V8_1_PROGRAM_ARCHITECTURE'
WIDTH, HEIGHT = 2400, 2770
FONT = 'Noto Sans CJK TC'
COLORS = {
    'program': ('#eff6ff', '#93b4ef', '#2563eb'),
    'data': ('#ffffff', '#cbd5e1', '#64748b'),
    'gpu': ('#fff7ed', '#fdba74', '#ea580c'),
    'output': ('#ecfdf5', '#6ee7b7', '#059669'),
    'note': ('#f1f5f9', '#cbd5e1', '#64748b'),
}
EDGE = {'call': '#2563eb', 'read': '#64748b', 'write': '#059669'}


def build_svg():
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>Midterm01 Track3 v8.1｜程式呼叫與資料讀寫架構</title>
<desc>依2026-10-26原始碼核對：因果影像與gyro預測，靜止gyro-bias Kalman，當幀Monte Carlo搜尋，CUDA NDT與ICP，通過閘門才提交pose和scale。完整定位由使用者手動啟動。</desc>
<defs>''']
    for name, color in EDGE.items():
        parts.append(f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="{color}" stroke-width="1.5"/></marker>')
    parts.append(f'''</defs><style>
text{{font-family:'{FONT}',sans-serif;fill:#172b4d}}
.title{{font-size:38px;font-weight:700}} .subtitle{{font-size:23px;fill:#52647b}}
.tag{{font-size:19px;font-weight:700}} .name{{font-size:24px;font-weight:700}}
.body{{font-size:22px}} .small{{font-size:20px;fill:#52647b}}
.edge{{font-size:20px;fill:#334155;paint-order:stroke;stroke:#f8fafc;stroke-width:8px;stroke-linejoin:round}}
path{{stroke-linecap:round;stroke-linejoin:round}}
</style><rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>''')
    measurements = []
    context = PangoCairo.FontMap.get_default().create_context()

    def text(x, y, value, cls='body', anchor='start', max_width=None):
        size = {'title': 38, 'subtitle': 23, 'tag': 19, 'name': 24,
                'body': 22, 'small': 20, 'edge': 20}[cls]
        layout = Pango.Layout.new(context)
        desc = Pango.FontDescription(FONT)
        desc.set_absolute_size(size * Pango.SCALE)
        if cls in ('title', 'name', 'tag'):
            desc.set_weight(Pango.Weight.BOLD)
        layout.set_font_description(desc)
        layout.set_text(value, -1)
        _, extent = layout.get_pixel_extents()
        if max_width is not None:
            measurements.append(dict(text=value, width=extent.width, available=max_width))
            if extent.width > max_width:
                raise ValueError(f'Text overflows box: {value}: {extent.width} > {max_width}')
        parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>')

    def node(x, y, w, h, kind, tag, names, body):
        fill, stroke, accent = COLORS[kind]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y+18}" width="6" height="{h-36}" rx="3" fill="{accent}"/>')
        text(x+26, y+32, tag, 'tag', max_width=w-52)
        baseline = y + 72
        for value in names:
            text(x+26, baseline, value, 'name', max_width=w-52)
            baseline += 34
        for value in body:
            text(x+26, baseline+2, value, 'body', max_width=w-52)
            baseline += 33
        if baseline > y+h+5:
            raise ValueError(f'Too many lines in {tag}')

    def edge(path, kind='call', label=None, pos=None, anchor='middle'):
        color = EDGE[kind]
        dash = ' stroke-dasharray="8 7"' if kind == 'read' else ''
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8"{dash} marker-end="url(#{kind})"/>')
        if label:
            for i, value in enumerate(label.split('\n')):
                text(pos[0], pos[1]+i*28, value, 'edge', anchor)

    text(60, 67, 'Midterm01 Track3 v8.1｜程式呼叫與資料讀寫架構', 'title')
    text(60, 111, '因果預測 → 當幀局部 Monte Carlo → CUDA NDT／ICP → 提交或回復　｜　依本機原始碼核對 · 2026-10-26', 'subtitle')
    for x, kind, label in [(60, 'call', '呼叫／回傳'), (380, 'read', '讀取'), (630, 'write', '寫入')]:
        edge(f'M{x} 147 H{x+62}', kind)
        text(x+78, 154, label, 'small')
    text(920, 154, '藍：Python 程式　灰：資料　橘：CUDA　綠：輸出', 'small')

    # Host entry. Source snapshots are separate from mutable live sources.
    node(60, 210, 620, 180, 'program', 'UBUNTU 主機｜環境', ['scripts/dev.sh up'],
         ['docker-compose.yml → midterm01_track3', '共用資料唯讀；只有 outputs 可寫'])
    node(890, 210, 620, 180, 'program', 'UBUNTU 主機｜使用者手動啟動', ['scripts/run.sh --full'],
         ['flock → 建立 run、記錄環境、啟動 RViz', '執行 run/source/scripts/run_track3.py'])
    node(1720, 210, 620, 180, 'data', 'create_run.py｜來源凍結', ['outputs/run/track03_full_*/'],
         ['source/、source_manifest.json', 'git_status.txt、uncommitted.patch'])
    edge('M680 300 H890', label='先啟動環境', pos=(785, 281))
    edge('M1510 300 H1720', 'write', '建立＋凍結', (1615, 281))
    edge('M2030 390 V430 H1310 V470', 'call', '由 run 內快照執行', (1670, 452))

    # The central column is the runner's causal per-frame control flow.
    node(60, 470, 620, 240, 'data', '唯讀 Track3 資料', ['raw_ply/*.ply'],
         ['map.pcd：只取 XYZ', 'init_guess.txt：初始 T_map_ply', 'io.py：讀取、排序、檢查與 SHA256', '原始幀序 0–3141；整數 timestamp_ns'])
    node(890, 470, 620, 240, 'program', '01｜runner 載入與檢查', ['scripts/run_track3.py'],
         ['io.frames / load_ply / load_map', '核對幀數、初始矩陣與版本設定', 'map 以 CUDA voxel_mean 下採樣', '建立 MonteCarloState 與 Registration'])
    node(1720, 470, 620, 240, 'data', '唯讀 ROS BAG', ['Track3.db3'],
         ['/imu/data：gyro angular_velocity', '前相機訊息：header 擷取時間', '保留 bag arrival 與感測器 header', 'IMU orientation、GNSS 不進入定位'])
    edge('M680 586 H890', 'read', 'read XYZ／T₀', (785, 565))
    edge('M1200 710 V820', label='逐幀處理', pos=(1320, 772))
    edge('M2030 710 V820', 'read', '唯讀 SQLite', (2160, 772))

    node(60, 820, 620, 270, 'data', '前視影像與相機模型', ['Track3_images/Track3_f_images/'],
         ['*.jpg：目前與過去前視影像', 'camera_info：標定 YAML 與 ego mask', 'CameraMask：去畸變／FOV／車身遮罩', '3D 像素對應在 CUDA；PnP 在 CPU'])
    node(890, 820, 620, 270, 'program', '02｜目前與過去資料形成運動觀測', ['motion_constraints.py'],
         ['MotionConstraints.observe', '光流＋raw gyro＋持續時間＋3D 支持', '輸出 stationary、因果尺度與 PnP 位移', '移動時尺度交給 map Sim(3) 估計'])
    node(1720, 820, 620, 270, 'program', '時間對齊｜time_alignment.py', ['CameraClock / AcquisitionImuReader'],
         ['bag arrival ≤ 當幀 PLY arrival', 'IMU header ≤ 當幀 image header', '較晚 header 暫存；遲到樣本不回修', '以影像擷取時間推進狀態與 Δt'])
    edge('M680 954 H890', 'read', '影像＋遮罩', (785, 933))
    edge('M1720 954 H1510', 'call', '可用 IMU', (1615, 933))
    edge('M1200 1090 V1200', label='運動／靜止證據', pos=(1320, 1151))

    node(60, 1200, 620, 270, 'gpu', 'predict 內呼叫｜gyro_bias.py', ['StationaryGyroBias.step'],
         ['三軸 gyro bias／variance Kalman', '只在確認靜止且收到新 IMU 時量測', 'raw-rate 與 NIS gate；CuPy CUDA', '更新後 bias 供後續角速度積分'])
    node(890, 1200, 620, 270, 'program', '03｜因果預測與新候選', ['MonteCarloState.predict'],
         ['旋轉：積分 gyro − 已知 bias', '平移：PnP 位移，否則上一 metric velocity', '靜止：保留上一輸出 pose', '每幀重設 256 候選，包含未擾動 prior'])
    node(1720, 1200, 620, 270, 'gpu', '當幀局部幾何搜尋｜monte_carlo.py', ['MonteCarloState.observe_cloud'],
         ['共同點雲／地圖 → GpuNearest.query', 'robust cost＋coverage＋尺度懲罰', '選中 pose／scale 只作 registration seed', 'ESS 僅描述分數集中；不做重採樣'])
    edge('M890 1290 H680', label='step()', pos=(785, 1268))
    edge('M680 1410 H890', label='bias／診斷', pos=(785, 1388))
    edge('M1510 1290 H1720', label='observe_cloud()', pos=(1615, 1268))
    edge('M1720 1410 H1640 V1530 H1370 V1580', label='候選 seed＋scale', pos=(1575, 1558))
    edge('M1200 1470 V1580', label='固定 prior／scale_reference', pos=(1050, 1527))

    node(60, 1580, 620, 270, 'gpu', '首筆 MAP ANCHOR｜initialization.py', ['GroundInitializer.align / score_pose'],
         ['地面／高度與非地面幾何產生四候選', '每個候選獨立做完整 NDT／ICP', '相同非地面點集評分與品質 gate', '只從合格候選選第一個 map anchor'])
    node(890, 1580, 620, 270, 'gpu', '04｜CUDA 點雲配準', ['gpu.py / Registration.align'],
         ['preprocessing.py：深度正規化 SOR', 'NDT 8／4／2 m → point ICP 1 m', '移動：估 pose＋scale；靜止：固定 scale', '採樣與尺度懲罰使用因果 scale_reference'])
    node(1720, 1580, 620, 270, 'gpu', '函式呼叫｜同一 Python 程序內的 CUDA', ['gpu.py / search.py / nearest.py'],
         ['voxel_mean、VoxelModel／PointModel', 'linearize、normal_information 與 CUDA 求解', 'cuda_math.py：旋轉、位姿與尺度增量', 'line_search.py：backtrack／收斂檢查'])
    edge('M890 1670 H680', label='首筆 anchor', pos=(785, 1648))
    edge('M680 1790 H890', label='回傳最佳精修', pos=(785, 1768))
    edge('M1510 1670 H1720', label='CUDA 計算', pos=(1615, 1648))
    edge('M1720 1790 H1510', label='pose／scale／品質', pos=(1615, 1768))
    edge('M1200 1850 V1960', label='配準候選＋品質診斷', pos=(1365, 1908))

    node(60, 1960, 620, 240, 'note', '參數入口｜直接修改 Python', ['config/track3.json'],
         ['monte_carlo_parameters.py：候選數／seed', 'imu_parameters.py：外參／gyro noise／bias', 'motion_parameters.py：靜止與視覺門檻', '有效值凍結於本次 parameters.json'])
    node(890, 1960, 620, 240, 'program', '05｜提交、回復與靜止 HOLD', ['MonteCarloState.commit'],
         ['品質：收斂、coverage、可觀測性、有效值', '非 bootstrap 再檢查 50 m/s、2 rad/s', '通過：提交配準 pose／scale', '拒絕：回復本幀 prior；靜止：pose hold'])
    node(1720, 1960, 620, 240, 'program', '下一幀錨點｜因果狀態傳遞', ['最後真正輸出的 pose／scale'],
         ['接受與拒絕都形成一致的狀態', '下一幀從這份輸出重新提出候選', '不跨幀保留 importance-weighted posterior', '不使用未來影格回改舊 pose'])
    edge('M1510 2080 H1720', label='更新錨點', pos=(1615, 2058))
    edge('M2340 2070 H2375 V1150 H1420 V1200', label='下一幀：只帶最後輸出，重新局部搜尋', pos=(1900, 1133))
    edge('M1200 2200 V2310', 'write', 'runner 直接輸出', (1335, 2258))

    node(60, 2310, 620, 260, 'program', '即時顯示｜不是另一個配準 NODE', ['visualization.py / RvizPublisher'],
         ['/track3/map、cloud、pose、path、status', 'map → track3_camera TF', 'ACCEPTED／PRIOR ONLY／STATIONARY', '只呈現估計；RViz 不能證明定位精度'])
    node(890, 2310, 620, 260, 'output', '06｜本次 RUN 的直接輸出', ['poses.txt / result_3.csv'],
         ['timings.csv、frame_manifest.csv', 'observations.jsonl、pose_trace.jsonl', 'gpu_memory.jsonl、test_info.json', '來源 SHA、參數、耗時與失敗資訊'])
    node(1720, 2310, 620, 260, 'output', '成功 3142 幀 FULL 後｜唯讀分析', ['analyze_run.py'],
         ['analysis/：軌跡、pose 變化、耗時與顯存', '標準圖成功才跑 analyze_monte_carlo.py', 'analysis/monte_carlo/：ESS／bias 等圖', 'PNG／PDF／SHA；畫圖失敗保留定位輸出'])
    edge('M890 2435 H680', label='publish_frame()', pos=(785, 2413))
    edge('M1510 2435 H1720', 'read', 'read-only', (1615, 2413))

    text(60, 2640, '輸出契約：result_3.csv = idx,x,y,yaw（rad）；poses.txt = row-major 3×4（12 值）；timestamp_ns 保持整數。', 'subtitle')
    text(60, 2687, '解讀限制：靜止 Kalman 只估 gyro bias；不融合 map pose、不積分 accelerometer。分數、ESS、接受率與輸出完整不等於精度。', 'subtitle')
    text(60, 2734, '此圖依原始碼整理，未啟動定位或 GPU 工作；來源 SHA 與精確程式版本見同目錄 .sources.json。', 'small')
    parts.append('</svg>')
    return '\n'.join(parts)+'\n', measurements


def main():
    svg, measurements = build_svg()
    svg_path = DOC / f'{STEM}.svg'
    png_path = DOC / f'{STEM}.png'
    svg_path.write_text(svg, encoding='utf-8')
    pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(svg_path))
    pixbuf.savev(str(png_path), 'png', [], [])
    print(json.dumps(dict(svg=str(svg_path), png=str(png_path), width=WIDTH, height=HEIGHT,
                          checked_text_lines=len(measurements),
                          minimum_text_margin_px=min(r['available']-r['width'] for r in measurements)),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
