#!/usr/bin/env python3
"""讀 submission_settings.py，製作報告；證據齊全才輸出四檔正式 ZIP。"""
import hashlib
import argparse
import json
from pathlib import Path
import re
import runpy
from urllib.parse import urlparse
import zipfile
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    Preformatted, KeepTogether, Flowable,
)
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(settings):
    """只讀實測 run，不執行濾波、不製造測量，也不修改原有輸出。"""
    evidence = dict(missing=[], run=None, summary={}, analysis={}, covariance={}, sources={})
    run_value = settings.get('RUN_DIR', '')
    if run_value:
        evidence['run'] = resolve(run_value)
        for name, key in (('summary.json', 'summary'), ('analysis.json', 'analysis'),
                          ('covariance_statistics.json', 'covariance')):
            path = evidence['run'] / name
            if path.is_file():
                evidence[key] = json.loads(path.read_text(encoding='utf-8'))
            else:
                evidence['missing'].append(name)
        if evidence['summary'].get('data_status') != 'has_sensor_data':
            evidence['missing'].append('具有 radar/GPS/GT/estimate 與 GPS update 的實測結果')
        if not evidence['analysis'].get('sensors', {}).get('estimate', {}).get('samples', 0):
            evidence['missing'].append('有重疊 GT 的位置誤差分析')
        if not all(name in evidence['covariance'] for name in ('gps', 'radar')):
            evidence['missing'].append('GPS 與 radar 的實際 covariance 紀錄')
        if not (evidence['run'] / 'result.png').is_file():
            evidence['missing'].append('本輪 result.png')
        manifest_path = evidence['run'] / 'source_manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for name in ('fusion.py', 'EKF.py'):
                source = evidence['run'] / 'source' / name
                expected = manifest.get('sources', {}).get(name, {}).get('sha256')
                if source.is_file() and digest(source) == expected:
                    evidence['sources'][name] = source
                else:
                    evidence['missing'].append(f'與來源 manifest 相符的 {name}')
        else:
            evidence['missing'].append('本輪 source_manifest.json')
    else:
        evidence['missing'].append('Ubuntu 完整播放 bag 的 RUN_DIR 與實測輸出')
    screenshot = settings.get('TERMINAL_SCREENSHOT', '')
    evidence['screenshot'] = resolve(screenshot) if screenshot else None
    if not evidence['screenshot'] or not evidence['screenshot'].is_file():
        evidence['missing'].append('顯示本人學號的執行中終端機截圖')
        evidence['screenshot'] = None
    video = settings.get('VIDEO_URL', '').strip()
    host = (urlparse(video).hostname or '').lower()
    evidence['video'] = video if urlparse(video).scheme == 'https' and host in (
        'youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com') else ''
    if not evidence['video']:
        evidence['missing'].append('RViz2 螢幕錄影的 YouTube HTTPS 連結')
    return evidence


def register_font():
    # 使用者同意採外觀相近字型：中文楷體、英文 Times 風格襯線體。
    candidates = (
        ('C:/Windows/Fonts/kaiu.ttf', 0),
        ('/usr/share/fonts/truetype/arphic/ukai.ttc', 2),  # AR PL UKai TW
    )
    for value, index in candidates:
        if Path(value).is_file():
            pdfmetrics.registerFont(TTFont('HW4Chinese', value, subfontIndex=index))
            break
    else:
        raise FileNotFoundError('需要中文楷體字型；Ubuntu 請安裝 fonts-arphic-ukai。')
    for path in (Path('C:/Windows/Fonts/times.ttf'),
                 Path('/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'),
                 Path('/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf')):
        if path.is_file():
            pdfmetrics.registerFont(TTFont('HW4English', str(path)))
            break
    else:
        raise FileNotFoundError('需要英文 Times 風格字型；Ubuntu 請安裝 fonts-liberation。')
    return 'HW4Chinese'


def build_pdf(destination, student_id, evidence):
    font = register_font()
    styles = getSampleStyleSheet()
    for key in ('Normal', 'BodyText', 'Heading1', 'Heading2', 'Title'):
        styles[key].fontName = font
        styles[key].wordWrap = 'CJK'
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 16
    styles['BodyText'].spaceAfter = 7
    styles['Heading1'].fontSize = 17
    styles['Heading1'].leading = 23
    styles['Heading1'].textColor = colors.HexColor('#16324f')
    styles['Heading2'].fontSize = 12
    styles['Heading2'].leading = 18
    styles['Title'].fontSize = 24
    styles['Title'].leading = 31
    styles.add(ParagraphStyle('SmallNote', parent=styles['BodyText'], fontSize=8.5, leading=13,
                              textColor=colors.HexColor('#526071')))
    styles.add(ParagraphStyle('CenterInfo', parent=styles['BodyText'], alignment=TA_CENTER))
    story = []

    def mixed_fonts(value):
        # 保留 Paragraph 的 link 等 markup，只對文字片段分配中英字型。
        chunks = []
        for part in re.split(r'(<[^>]+>)', value):
            if part.startswith('<'):
                chunks.append(part)
                continue
            for match in re.finditer(
                    r'[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+|'
                    r'[^\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+', part):
                content = match.group()
                name = font if '\u2e80' <= content[0] <= '\uffef' else 'HW4English'
                chunks.append(f'<font name="{name}">{content}</font>')
        return ''.join(chunks)

    def text(value, style='BodyText'):
        story.append(Paragraph(mixed_fonts(value), styles[style]))

    def heading(value):
        text(value, 'Heading2')

    def formula(value):
        story.append(Preformatted(value, ParagraphStyle('Equation', fontName='HW4English', fontSize=10,
                                                       leading=14, spaceBefore=4, spaceAfter=9)))

    def picture(path, max_height=100*mm):
        image = Image(str(path))
        ratio = min(172*mm/image.imageWidth, max_height/image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth*ratio, image.imageHeight*ratio
        story.append(image)

    def screenshot_detail(path, box, width=172*mm):
        """以 PDF clipping 放大原始截圖區域，不改寫證據影像。box 為左上座標。"""
        class ScreenshotDetail(Flowable):
            def __init__(self):
                super().__init__()
                self.reader = ImageReader(str(path))
                self.image_width, self.image_height = self.reader.getSize()
                left, top, right, bottom = box
                self.scale = width / (right-left)
                self.width, self.height = width, (bottom-top)*self.scale

            def draw(self):
                left, top, right, bottom = box
                canvas = self.canv
                canvas.saveState()
                clip = canvas.beginPath()
                clip.rect(0, 0, self.width, self.height)
                canvas.clipPath(clip, stroke=0, fill=0)
                canvas.drawImage(self.reader, -left*self.scale,
                                 -(self.image_height-bottom)*self.scale,
                                 width=self.image_width*self.scale,
                                 height=self.image_height*self.scale)
                canvas.restoreState()

        story.append(ScreenshotDetail())
        story.append(Spacer(1, 3*mm))

    def table(rows, widths):
        cells = [[Paragraph(mixed_fonts(escape(str(value))), styles['SmallNote'])
                  for value in row] for row in rows]
        result = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
        result.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eaf0f6')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0, 0), (-1, -1), .5, colors.HexColor('#b7c5d3')),
            ('INNERGRID', (0, 0), (-1, -1), .25, colors.HexColor('#d4dde6')),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(result)
        story.append(Spacer(1, 4*mm))

    draft = bool(evidence['missing'])
    text('SDC Assignment 4', 'Title')
    text('Application of EKF', 'Heading1')
    text(f'學號：{student_id}　｜　個人完成，貢獻 100%')
    text('Radar odometry 與 GPS 的平面擴展卡爾曼濾波')
    if draft:
        text('報告草稿 - 尚缺實測與交付證據，請勿直接繳交', 'Heading2')
        for missing in evidence['missing']:
            text('待補：' + escape(missing), 'SmallNote')
    heading('作業目標與感測器角色')
    text('Radar odometry 提供相鄰姿態的運動增量，但累積誤差會造成漂移；GPS 提供全域 x、y 位置，'
         '可以校正漂移。EKF 依兩種輸入的不確定性整合資訊。/gt_odom 是驗證參考，完全不參與初始化、預測或更新。')
    table([['Topic', '用途'], ['/radar_odometry', '相鄰 pose 轉為局部位移與 yaw 增量'],
           ['/gps', '使用 x、y 及其 2×2 covariance 校正'],
           ['/gt_odom', '只保存與事後評估，不回寫濾波狀態']], [57*mm, 115*mm])
    heading('執行環境')
    text('沿用 Assignment 2 的 ROS 2 Humble Docker 環境；colcon_ws 掛載於 /ros_ws/hw4。'
         '啟動 fusion 與 RViz2 後播放老師提供的 bag。bag 已有 /clock，播放時不另加 --clock。')
    if evidence['run'] and (evidence['run'] / 'result.png').is_file():
        heading('本輪軌跡')
        picture(evidence['run'] / 'result.png', 90*mm)
        text('圖 1　同輪實測 result.png。灰色為 GT、綠色為 radar、橙點為 GPS、藍色為 EKF。'
             'Radar 出現累積漂移；EKF 在 GPS 位置更新後沿參考路徑前進。', 'SmallNote')
    else:
        text('目前尚未取得真實 bag 的 result.png，因此未以示意圖代替實測軌跡。', 'SmallNote')

    story.append(PageBreak())
    text('Discussion 1｜EKF 如何設計', 'Heading1')
    heading('1. 狀態與控制')
    text('狀態是地圖座標下的 x、y 與 yaw，單位分別為 m、m、rad。先用上一個 radar yaw，'
         '把兩個 radar pose 的全域位置差旋轉回上一個 radar 局部座標；yaw 差包回 ±π。')
    formula('mu = [x, y, yaw]^T\nu_xy = Rot(-yaw_radar_previous) @ (p_radar_now - p_radar_previous)\nu_yaw = wrap(yaw_radar_now - yaw_radar_previous)')
    text('例如車頭朝北時，全域位移 (0, 1) m 會轉成局部向前 (1, 0) m。之後使用 EKF 自己估計的 yaw，'
         '將局部控制轉回地圖。這讓 GPS 修正過的方向能影響往後的位置預測。')
    text('第一筆 radar pose 初始化 x、y、yaw，其 pose covariance 初始化 S。'
         'GPS 若先到會暫存，不以 GT 或 GPS orientation 初始化方向。', 'SmallNote')
    formula('x_next   = x + cos(yaw)*u_x - sin(yaw)*u_y\ny_next   = y + sin(yaw)*u_x + cos(yaw)*u_y\nyaw_next = wrap(yaw + u_yaw)')
    heading('2. Jacobian 與 covariance 預測')
    formula('A = [[1, 0, -sin(yaw)*u_x - cos(yaw)*u_y],\n     [0, 1,  cos(yaw)*u_x - sin(yaw)*u_y],\n     [0, 0, 1]]\nB = [[cos(yaw), -sin(yaw), 0],\n     [sin(yaw),  cos(yaw), 0],\n     [0, 0, 1]]\nS_prior = A @ S @ A.T + B @ R @ B.T')
    heading('3. GPS 觀測與更新')
    text('GPS 只量測位置，觀測向量是二維，觀測矩陣 C 是 2×3；GPS orientation 不被當作 yaw 量測。')
    formula('z = [gps_x, gps_y]^T\nC = [[1, 0, 0], [0, 1, 0]]\nK = S_prior @ C.T @ inv(C @ S_prior @ C.T + Q)\nmu_post = mu_prior + K @ (z - C @ mu_prior)\nS_post = (I-K@C) @ S_prior @ (I-K@C).T + K @ Q @ K.T')
    text('程式用線性方程求解取代顯式 inverse，並用 Joseph form 更新 covariance。yaw 會包回 ±π。'
         'GPS 雖不直接量 yaw，運動後的位置與 yaw cross covariance 仍可間接校正方向；靜止時方向資訊有限。', 'SmallNote')

    story.append(PageBreak())
    text('Discussion 2｜Covariance 與意義', 'Heading1')
    text('本作業沿用模板命名：S 是狀態 covariance，R 是 radar 控制噪聲，Q 是 GPS 量測噪聲。'
         '其他課本的 Q、R 名稱可能相反，應以定義為準。')
    table([['矩陣', '形狀與意義'], ['S', '3×3，x、y、yaw 狀態的不確定性'],
           ['Q', '2×2，正常時取 GPS pose covariance 的 [x,y] 子矩陣；全零另用 fallback'],
           ['Radar pose covariance', '3×3，原始 6×6 covariance 的 [x,y,yaw] 子矩陣'],
           ['R', '3×3，由相鄰 radar pose covariance 經 Jacobian 傳遞的局部增量近似']],
          [57*mm, 115*mm])
    text('ROS covariance 的索引順序是 [x,y,z,roll,pitch,yaw]。對角線是 variance：位置為 m²、yaw 為 rad²；'
         'x-yaw 交叉項為 m·rad。標準差需取 variance 的平方根。例如 variance=9 m² 對應 std=3 m；此數字只作解釋。')
    heading('本輪實際量測 covariance')
    if evidence['covariance']:
        for sensor, label in (('gps', 'GPS 原始位置 covariance'), ('radar', 'Radar absolute pose covariance')):
            item = evidence['covariance'].get(sensor)
            if item:
                text(label, 'SmallNote')
                matrix = item['first_selected_covariance']
                formula('\n'.join('[' + ', '.join(f'{v:.6g}' for v in row) + ']' for row in matrix))
                text(f"有效 covariance 紀錄 {item['messages']} 筆，其中 fallback {item['fallback_messages']} 筆。"
                     '矩陣取自首筆通過檢查訊息；完整 6×6 原值與對角線範圍另存於 covariance_statistics.json。', 'SmallNote')
                if sensor == 'gps':
                    text('本輪 GPS 的 x、y variance 均為 3 m²，對應單軸標準差約 1.732 m；'
                         '此為 bag 提供的量測不確定性，不是 GT 評估 RMSE，也不是 std=3 m 的 fallback。', 'SmallNote')
                if item['first_selected_covariance'] != item['first_effective_pose_covariance']:
                    effective = '; '.join(', '.join(f'{v:.6g}' for v in row)
                                          for row in item['first_effective_pose_covariance'])
                    text('首筆含 fallback／數值修正後實際使用的 pose covariance（分號分列）：['
                         + effective + ']。', 'SmallNote')
    else:
        text('尚未讀到本輪 sensor 訊息，故此處沒有填寫任何實測矩陣。完成播放後會由 covariance_statistics.json 自動帶入。')
    heading('Radar R 的推導與限制')
    formula('R ~= J_previous @ Sigma_previous @ J_previous.T\n   + J_now @ Sigma_now @ J_now.T')
    text('J 是局部相對控制對兩端 absolute radar pose 的 Jacobian。因 bag 沒有兩端的 cross covariance，'
         '此式假設兩端誤差獨立；累積里程計通常存在相關性，因此這是近似，不代表經過精確校準。')
    heading('Covariance 如何影響融合')
    text('同樣的先驗下，GPS Q 越小，Kalman 更新越信任 GPS；R 越大，預測不確定性增長越快。'
         '只有選取的 covariance 全為零時才使用有警告的 fallback：GPS std=3 m、radar 位置 std=0.1 m、yaw std=1°。'
         '這些是起始假設，並非老師提供的實際 covariance；報告須揭露 fallback 次數。')

    story.append(PageBreak())
    text('驗證、時間處理與結果', 'Heading1')
    heading('不同頻率與角度處理')
    text('GPS 先暫存；下一筆 radar 到達後，在兩筆 radar 之間依 header timestamp 插值位置與最短角差，'
         '先推到 GPS 時刻更新，再推到 radar 時刻。此法最多延遲一個 radar interval，並非零延遲融合。'
         '舊時間 GPS 不回溯更新；重複或倒序 radar 拒絕並計數。時間先保存整數 ns，相減後才轉秒。')
    heading('時間對齊的實測位置誤差')
    metrics = evidence['analysis'].get('sensors', {})
    rows = [['來源', '有效樣本', 'XY RMSE (m)', 'P95 (m)']]
    for sensor in ('radar', 'gps', 'estimate'):
        item = metrics.get(sensor, {})
        if item.get('samples'):
            rows.append([sensor, item['samples'], f"{item['rmse_xy_m']:.4f}", f"{item['p95_xy_m']:.4f}"])
    if len(rows) > 1:
        table(rows, [43*mm, 35*mm, 47*mm, 47*mm])
        text('GT 在各 sensor 自己的 timestamp 上插值，只用重疊範圍且不外插。'
             '三種統計的時間樣本不完全相同，因此這不是逐筆配對比較；不能只看表格就將差異歸因於單一參數。', 'SmallNote')
    else:
        text('尚無真實資料，不填寫 RMSE，也不宣稱融合較準。合成測試只檢查數學、時序與輸出機制。')
    error_plot = evidence['run'] / '02_position_error.png' if evidence['run'] else None
    if error_plot and error_plot.is_file():
        picture(error_plot, 60*mm)
        text('圖 2　各來源在自身時間戳對齊 GT 的 XY 距離誤差。圖例配色依原始分析圖，'
             '與 RViz 的軌跡配色不同。', 'SmallNote')
    summary = evidence['summary']
    if summary:
        heading('執行診斷')
        samples = summary.get('samples', {})
        text('本輪 ' + '、'.join(f'{name} {samples.get(name, 0)} 筆' for name in (
            'radar', 'gps', 'estimate', 'gt')) + '。', 'SmallNote')
        text('；'.join(f'{key}={summary.get(key, "未記錄")}' for key in (
            'gps_updates', 'late_gps', 'nonincreasing_radar', 'invalid_messages')) + '。', 'SmallNote')
        if evidence['run'] and evidence['run'].name == 'run_261029_232220_oYEV':
            text('原 bag 有 radar 973、GPS 64、GT 7714 筆；本輪分別少收 12、1、1066 筆。'
                 '使用 best effort subscription，錄影播放時有訊息未收到，因此不能宣稱所有訊息均已處理。'
                 '雷達首末 timestamp 均與 bag 邊界吻合，涵蓋約 79.616 秒；上述統計僅適用本輪實收資料。', 'SmallNote')
        estimate = metrics.get('estimate', {})
        if estimate:
            text(f"GT 同一 timestamp 保留最後到達樣本；本輪去重後有 {estimate.get('gt_unique_timestamps', 0)} 筆。"
                 f"EKF 評估使用的最大 GT 插值跨距為 {estimate.get('max_interpolated_gt_gap_s', 0):.5f} s。"
                 'GPS 在自己量測時刻的誤差小於 EKF 所列誤差，本實驗不宣稱融合優於 GPS。', 'SmallNote')
    heading('方法限制')
    text('此模型只估計平面 x、y、yaw，不做隱式 frame 轉換。Radar 相鄰誤差獨立及區間分段運動／噪聲均為近似；'
         'GPS 大誤差或時間／frame 錯誤仍可能使融合失效。GT 插值跨距也會影響誤差統計。')
    if evidence['screenshot']:
        story.append(PageBreak())
        text('執行識別與錄影', 'Heading1')
    else:
        heading('執行識別與錄影')
    if evidence['screenshot']:
        if evidence['screenshot'].name == '06_terminal_recording_identity.png':
            heading('Ubuntu 終端身分與同輪 EKF 啟動')
            screenshot_detail(evidence['screenshot'], (65, 74, 1115, 129))
            text('圖 3a　原始截圖的身分區域放大：主機使用者 ricky、學號 112611054。', 'SmallNote')
            screenshot_detail(evidence['screenshot'], (65, 207, 1760, 327))
            text('圖 3b　同一張原始截圖的 run 路徑與 EKF ready 訊息放大。僅在 PDF 中裁切顯示區域。', 'SmallNote')
            picture(evidence['screenshot'], 78*mm)
            text('圖 3c　原始完整桌面截圖，拍攝時 fusion 正在執行且等待 bag；'
                 '此圖與影片、實測數據對應同一輪 run。', 'SmallNote')
        else:
            picture(evidence['screenshot'], 78*mm)
            text('本人執行畫面；身分與實際播放情形由繳交者核對。', 'SmallNote')
    else:
        text('本人終端機識別截圖：尚未提供。', 'SmallNote')
    if evidence['video']:
        url = escape(evidence['video'], {'"': '&quot;'})
        heading('RViz2 操作錄影')
        text(f'<link href="{url}" color="#165b94">{url}</link>')
        text('影片由本人錄製並提供連結。RViz 圖例：綠色 radar、紅色 GT、藍色 EKF、白色 LiDAR。'
             'GPS 在圖 1 另外疊圖。', 'SmallNote')
    else:
        text('RViz2 YouTube 錄影：尚未錄製。', 'SmallNote')
    heading('資料與程式來源')
    if evidence['run']:
        text('實測日期：2026/10/29；run：' + escape(evidence['run'].name) + '。', 'SmallNote')
    text('題目與原始程式／資料連結：<link href="https://hackmd.io/muS7dwNbQjemnah98PLNRA" color="#165b94">'
         'SDC Assignment 4 - Application of EKF</link>。', 'SmallNote')

    def footer(canvas, doc):
        canvas.setFont('HW4English', 8)
        canvas.setFillColor(colors.HexColor('#526071'))
        canvas.drawRightString(191*mm, 12*mm, str(doc.page))

    SimpleDocTemplate(str(destination), pagesize=(210*mm, 297*mm),
                      leftMargin=19*mm, rightMargin=19*mm,
                      topMargin=17*mm, bottomMargin=21*mm,
                      title='SDC HW4 - Application of EKF', author=student_id).build(
                          story, onFirstPage=footer, onLaterPages=footer)


def main(argv=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-only', action='store_true',
                        help='只產生 PDF 供審閱，不建立或更新交付 ZIP')
    args = parser.parse_args(argv)
    settings = runpy.run_path(str(ROOT / 'submission_settings.py'))
    student_id = settings['STUDENT_ID']
    if not re.fullmatch(r'[0-9A-Za-z_-]+', student_id):
        raise ValueError('STUDENT_ID may only contain letters, digits, underscores and hyphens')
    evidence = collect(settings)
    output = ROOT / 'output'
    pdf_dir = output / 'pdf'
    pdf_dir.mkdir(parents=True, exist_ok=True)
    draft = bool(evidence['missing'])
    pdf_path = pdf_dir / f'{student_id}_hw4{"_DRAFT" if draft else ""}.pdf'
    build_pdf(pdf_path, student_id, evidence)
    print(f'PDF: {pdf_path}')
    if draft:
        print('DRAFT; no submission ZIP created. Missing evidence:')
        for missing in evidence['missing']:
            print(' - ' + missing)
        return
    if args.pdf_only:
        print('PDF review only; no submission ZIP created or updated.')
        return
    package = output / f'{student_id}_hw4.zip'
    files = {pdf_path.name: pdf_path, 'result.png': evidence['run'] / 'result.png',
             **evidence['sources']}
    with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, arcname=name)
    manifest = dict(student_id=student_id, run_dir=str(evidence['run'].resolve()),
                    package_sha256=digest(package),
                    files={name: digest(path) for name, path in files.items()})
    (output / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Submission ZIP (4 files): {package}')


if __name__ == '__main__':
    main(None)
