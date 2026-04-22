#!/usr/bin/env python3
"""
在多个 /dev/video* 中挑出更像「真彩色 RGB」的一路（RealSense UVC 上常见：video2=红外/灰度复刻成 BGR，video4=彩色）。
不依赖 pyrealsense2；原理：灰度三路数值几乎相同 -> color_score≈0；彩色三路有差异 -> score 较大。
"""
from __future__ import annotations

import glob
import os
import re

import cv2
import numpy as np


def _video_paths():
    paths = []
    for p in glob.glob('/dev/video*'):
        m = re.match(r'^/dev/video(\d+)$', p)
        if m:
            paths.append((int(m.group(1)), p))
    return [p for _, p in sorted(paths)]


def color_score_bgr(frame: np.ndarray) -> float:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return -1.0
    b = frame[..., 0].astype(np.float32)
    g = frame[..., 1].astype(np.float32)
    r = frame[..., 2].astype(np.float32)
    return float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)))


def probe_one(path: str):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return {'path': path, 'opened': False}
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return {'path': path, 'opened': True, 'read': False}
    return {
        'path': path,
        'opened': True,
        'read': True,
        'shape': tuple(fr.shape),
        'color_score': color_score_bgr(fr),
    }


def probe_all():
    return [probe_one(p) for p in _video_paths()]


def pick_best_color_path(min_score: float = 0.25):
    """返回 (path 或 None, 最佳 score)。score 低于 min_score 则视为没有可靠彩色流。"""
    best_path, best_s = None, -1.0
    for p in _video_paths():
        r = probe_one(p)
        if not r.get('read'):
            continue
        s = float(r.get('color_score', -1.0))
        if s > best_s:
            best_s, best_path = s, p
    if best_path is not None and best_s >= min_score:
        return best_path, best_s
    return None, best_s


def main():
    rows = probe_all()
    for r in rows:
        print(r)
    bp, sc = pick_best_color_path()
    print('---')
    print('推荐彩色设备:', bp, 'color_score=', round(sc, 4))
    if bp:
        print('示例: DEVICE_PATH=%s python3 run_aruco_camera.py' % bp)


if __name__ == '__main__':
    main()
