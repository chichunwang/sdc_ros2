#!/usr/bin/env python3
"""檢查 Midterm02 提交 CSV（只用標準函式庫，相容 Python 3.9）。

檢查：表頭、timestamp 與 sample_submission 完全一致（不多、不少、不重複）、
全部為有限數值、旋轉正交且 det=+1、最後一列為 0,0,0,1。
通過只代表格式合法，不代表精度。
"""
import argparse
import csv
import math
import sys

HEADER = ['timestamp'] + ['m%d%d' % (r, c) for r in range(4) for c in range(4)]


def read_timestamps(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows or rows[0][:1] != ['timestamp']:
        raise ValueError('%s: 第一欄表頭必須是 timestamp' % path)
    return [r[0].strip() for r in rows[1:] if r]


def check_pose(values, tol):
    m = [values[i * 4:(i + 1) * 4] for i in range(4)]
    if any(abs(a - b) > tol for a, b in zip(m[3], (0.0, 0.0, 0.0, 1.0))):
        return '最後一列不是 0,0,0,1'
    r = [row[:3] for row in m[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > tol:
                return '旋轉不正交 (R^T R[%d][%d]=%.6g)' % (i, j, dot)
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if abs(det - 1.0) > tol:
        return 'det(R)=%.6g 不是 +1' % det
    return None


def check(submission, sample, tol=1e-4):
    errors = []
    with open(submission, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows or [h.strip() for h in rows[0]] != HEADER:
        return ['表頭必須是：' + ','.join(HEADER)]
    seen = []
    for line, row in enumerate(rows[1:], start=2):
        if not row:
            errors.append('第 %d 列為空' % line)
            continue
        if len(row) != 17:
            errors.append('第 %d 列欄位數 %d≠17' % (line, len(row)))
            continue
        ts = row[0].strip()
        if not ts.isdigit():
            errors.append('第 %d 列 timestamp 不是整數：%r' % (line, ts))
        seen.append(ts)
        try:
            values = [float(v) for v in row[1:]]
        except ValueError:
            errors.append('第 %d 列含非數值' % line)
            continue
        if not all(math.isfinite(v) for v in values):
            errors.append('第 %d 列含 NaN/Inf' % line)
            continue
        bad = check_pose(values, tol)
        if bad:
            errors.append('第 %d 列（%s）%s' % (line, ts, bad))
    expected = read_timestamps(sample)
    dup = len(seen) - len(set(seen))
    if dup:
        errors.append('重複 timestamp %d 筆' % dup)
    missing = set(expected) - set(seen)
    extra = set(seen) - set(expected)
    if missing:
        errors.append('缺少 %d 筆 timestamp，例如 %s' % (len(missing), sorted(missing)[:3]))
    if extra:
        errors.append('多出 %d 筆 timestamp，例如 %s' % (len(extra), sorted(extra)[:3]))
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('submission')
    p.add_argument('sample', help='官方 sample_submission.csv')
    p.add_argument('--tol', type=float, default=1e-4)
    a = p.parse_args()
    errors = check(a.submission, a.sample, a.tol)
    if errors:
        print('FAIL：%d 項問題' % len(errors))
        for e in errors[:50]:
            print(' -', e)
        sys.exit(1)
    print('OK：格式合法，%d 筆 timestamp 與 sample 一致' % len(read_timestamps(a.sample)))


if __name__ == '__main__':
    main()
