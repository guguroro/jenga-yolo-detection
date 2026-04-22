#!/usr/bin/env python3
"""
从本机摄像头连续保存画面，用于 Roboflow / Label Studio 标注后「微调」Jenga 模型。

  python3 yolo_aruco/capture_real_frames.py --out yolo_aruco/real_captures --n 80 --interval 0.5

按 q 提前结束。标注后把图片并入 train/images 并重新 train_jenga.py。
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from v4l_color_probe import pick_best_color_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='yolo_aruco/real_captures')
    ap.add_argument('--n', type=int, default=60, help='保存张数')
    ap.add_argument('--interval', type=float, default=0.4, help='间隔秒')
    ap.add_argument('--device-path', default=None)
    ap.add_argument('--source', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cap = None
    desc = ''
    if args.device_path:
        cap = cv2.VideoCapture(args.device_path, cv2.CAP_V4L2)
        desc = args.device_path
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(args.source, cv2.CAP_V4L2)
        desc = 'index %s' % args.source
    if not cap.isOpened():
        best, _ = pick_best_color_path()
        if best:
            cap = cv2.VideoCapture(best, cv2.CAP_V4L2)
            desc = best
    if not cap.isOpened():
        print('无法打开摄像头')
        sys.exit(1)

    print('源:', desc, '→', args.out, '共', args.n, '张，按 q 退出')
    t0 = time.time()
    i = 0
    try:
        while i < args.n:
            ok, fr = cap.read()
            if not ok:
                break
            path = os.path.join(args.out, 'cap_%05d.jpg' % i)
            cv2.imwrite(path, fr)
            i += 1
            print('saved', path)
            cv2.imshow('capture', cv2.resize(fr, (960, 540)))
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
            time.sleep(max(0.0, args.interval))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print('完成', i, '张，用时 %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
