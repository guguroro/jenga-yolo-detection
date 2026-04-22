#!/usr/bin/env python3
"""
Jenga 积木实时检测（YOLOv8-seg + ROS2 摄像头）。

用法:
  python3 yolo_aruco/detect_jenga.py --model yolo_aruco/jenga_best.pt

低光、噪点大时默认：CLAHE + 轨迹平滑 + 同类框合并（减轻双框、略扩大并集框）。
  --no-enhance / --no-stabilize
  --merge-strength 0.08   合并更激进（两个碎框更容易合成一个）
  --max-per-class 1       每类只保留面积最大的一个（单积木场景）
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v4l_color_probe import pick_best_color_path

CLASS_COLORS = [
    (0,   200, 100),
    (255, 140,   0),
    (30,  144, 255),
    (220,  20,  60),
    (148,   0, 211),
]
# 运行时从模型动态覆盖；此处仅为类型声明占位
CLASS_NAMES: list[str] = ['Jenga', 'Jenga_Top', 'Jenga_side']


def _bbox_area(xyxy: np.ndarray) -> float:
    return float(max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]))


def _merge_strength(xyxy_a: np.ndarray, xyxy_b: np.ndarray) -> float:
    """同类框是否视为同一物体：max(IoU, 交集/较小框面积) —— 适合两个贴在一起的小误检。"""
    iou = _bbox_iou(xyxy_a, xyxy_b)
    x1 = max(xyxy_a[0], xyxy_b[0])
    y1 = max(xyxy_a[1], xyxy_b[1])
    x2 = min(xyxy_a[2], xyxy_b[2])
    y2 = min(xyxy_a[3], xyxy_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = _bbox_area(xyxy_a)
    bb = _bbox_area(xyxy_b)
    m = min(aa, bb)
    iomin = float(inter / m) if m > 1e-6 else 0.0
    return max(iou, iomin)


def merge_overlapping_tracks(tracks: list[dict], merge_thresh: float) -> list[dict]:
    """并查集合并同类、位置强相关的轨迹；框取并集，mask 取凸包。"""
    n = len(tracks)
    if n <= 1:
        return tracks

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i in range(n):
        for j in range(i + 1, n):
            if int(tracks[i]['cls']) != int(tracks[j]['cls']):
                continue
            if _merge_strength(tracks[i]['xyxy'], tracks[j]['xyxy']) >= merge_thresh:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    out: list[dict] = []
    for _root, idxs in groups.items():
        if len(idxs) == 1:
            t = tracks[idxs[0]]
            out.append({
                'cls': int(t['cls']),
                'conf': float(t['conf']),
                'xyxy': t['xyxy'].astype(np.float32).copy(),
                'mask_xy': t.get('mask_xy'),
            })
            continue

        cls = int(tracks[idxs[0]]['cls'])
        conf = max(float(tracks[i]['conf']) for i in idxs)
        x1 = min(float(tracks[i]['xyxy'][0]) for i in idxs)
        y1 = min(float(tracks[i]['xyxy'][1]) for i in idxs)
        x2 = max(float(tracks[i]['xyxy'][2]) for i in idxs)
        y2 = max(float(tracks[i]['xyxy'][3]) for i in idxs)
        xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

        pts_list = []
        for i in idxs:
            m = tracks[i].get('mask_xy')
            if m is not None and len(m) >= 3:
                pts_list.append(np.asarray(m, dtype=np.float32))
        mask_xy = None
        if pts_list:
            allp = np.vstack(pts_list)
            hull = cv2.convexHull(allp.astype(np.float32))
            mask_xy = hull.reshape(-1, 2)

        out.append({'cls': cls, 'conf': conf, 'xyxy': xyxy, 'mask_xy': mask_xy})

    return out


def limit_per_class(tracks: list[dict], max_per: int) -> list[dict]:
    """每类最多保留 max_per 个，按框面积从大到小（单积木场景可设 1）。"""
    if max_per <= 0:
        return tracks
    from collections import defaultdict

    byc: dict[int, list[dict]] = defaultdict(list)
    for t in tracks:
        byc[int(t['cls'])].append(t)
    out: list[dict] = []
    for _c, lst in byc.items():
        lst.sort(key=lambda t: -_bbox_area(t['xyxy']))
        out.extend(lst[:max_per])
    return out


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """a,b: xyxy float"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return float(inter / u) if u > 1e-6 else 0.0


def enhance_low_light_bgr(frame: np.ndarray) -> np.ndarray:
    """CLAHE 提亮暗部、压噪点，利于低光 RealSense。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)
    return out


def _predict(model, frame, conf: float, imgsz: int, augment: bool, max_det: int):
    return model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        augment=augment,
        max_det=max_det,
        verbose=False,
        iou=0.45,
    )


class TrackStabilizer:
    """
    多目标 IoU 匹配 + 框 EMA，抑制低置信度下框乱跳、背景误检。
    掩码仅在 IoU 与上一帧足够接近时更新，减少 mask 闪烁。
    """

    def __init__(
        self,
        alpha: float = 0.55,
        iou_match: float = 0.22,
        max_missed: int = 5,
        min_area_ratio: float = 0.0015,
        new_track_conf: float = 0.22,
    ):
        self.alpha = alpha
        self.iou_match = iou_match
        self.max_missed = max_missed
        self.min_area_ratio = min_area_ratio
        self.new_track_conf = new_track_conf
        self.tracks: list[dict] = []

    def _area_ratio(self, xyxy: np.ndarray, hw: tuple[int, int]) -> float:
        h, w = hw
        return float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / max(w * h, 1))

    def update(self, dets: list[dict], hw: tuple[int, int]) -> list[dict]:
        """dets: {cls, conf, xyxy, mask_xy}；返回用于绘制的稳定轨迹列表。"""
        H, W = hw
        dets = [
            d for d in dets
            if self._area_ratio(d['xyxy'], (H, W)) >= self.min_area_ratio
            and d['conf'] >= self.new_track_conf * 0.45
        ]
        dets.sort(key=lambda x: -x['conf'])

        for t in self.tracks:
            t['missed'] = t.get('missed', 0) + 1

        used_track_idx: set[int] = set()
        for d in dets:
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(self.tracks):
                if i in used_track_idx:
                    continue
                if int(t['cls']) != int(d['cls']):
                    continue
                iou = _bbox_iou(t['xyxy'], d['xyxy'])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_i >= 0 and best_iou >= self.iou_match:
                t = self.tracks[best_i]
                a = self.alpha
                t['xyxy'] = a * d['xyxy'] + (1.0 - a) * t['xyxy']
                t['conf_e'] = 0.4 * d['conf'] + 0.6 * t.get('conf_e', d['conf'])
                t['missed'] = 0
                if d.get('mask_xy') is not None and best_iou > 0.28:
                    t['mask_xy'] = d['mask_xy']
                used_track_idx.add(best_i)
            elif d['conf'] >= self.new_track_conf:
                self.tracks.append({
                    'cls': int(d['cls']),
                    'xyxy': d['xyxy'].copy(),
                    'conf_e': float(d['conf']),
                    'mask_xy': d.get('mask_xy'),
                    'missed': 0,
                })

        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]
        out = []
        for t in self.tracks:
            if t['missed'] > 2 and t['conf_e'] < 0.2:
                continue
            out.append({
                'cls': t['cls'],
                'conf': t['conf_e'],
                'xyxy': t['xyxy'].astype(np.float32),
                'mask_xy': t.get('mask_xy'),
            })
        return out


def _dets_from_result(r, frame_hw: tuple[int, int], min_area_ratio: float) -> list[dict]:
    H, W = frame_hw
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    for i, box in enumerate(r.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
        ar = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / max(W * H, 1)
        if ar < min_area_ratio * 0.5:
            continue
        mask_xy = None
        if r.masks is not None and i < len(r.masks.xy):
            mask_xy = r.masks.xy[i]
        out.append({'cls': cls, 'conf': conf, 'xyxy': xyxy, 'mask_xy': mask_xy})
    out.sort(key=lambda x: -x['conf'])
    return out


def draw_stable(frame: np.ndarray, tracks: list[dict], fps: float) -> None:
    """在原图上绘制稳定后的检测（就地修改）。"""
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    total = len(tracks)

    for t in tracks:
        cls = t['cls']
        conf = t['conf']
        xyxy = t['xyxy']
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]

        if t.get('mask_xy') is not None:
            pts = t['mask_xy'].astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)

        x1, y1, x2, y2 = map(int, xyxy)
        label = '%s  %.2f' % (CLASS_NAMES[cls], conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(
        frame,
        'Jenga Detector  FPS %.1f  objects %d  (CLAHE+smooth+merge)' % (fps, total),
        (8, 28), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
    )


def _open_v4l_camera(source: int, device_path: str | None):
    if device_path:
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok and fr is not None:
                return cap, device_path
            cap.release()

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if cap.isOpened():
        ok, fr = cap.read()
        if ok and fr is not None:
            return cap, 'index %s' % source
        cap.release()

    best, sc = pick_best_color_path()
    if best:
        cap = cv2.VideoCapture(best, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok and fr is not None:
                return cap, '%s (auto color_score=%.3f)' % (best, sc)
            cap.release()

    for idx in (4, 2, 0):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        ok, fr = cap.read()
        if ok and fr is not None:
            return cap, 'fallback index %s' % idx
        cap.release()

    return None, None


def run_opencv(
    model_path: str,
    source: int,
    conf: float,
    device_path: str | None,
    imgsz: int,
    augment: bool,
    enhance: bool,
    stabilize: bool,
    smooth_alpha: float,
    min_area: float,
    max_det: int,
    merge_strength: float,
    max_per_class: int,
):
    global CLASS_NAMES
    from ultralytics import YOLO
    model = YOLO(model_path)
    if hasattr(model, 'names') and model.names:
        CLASS_NAMES = [model.names[i] for i in sorted(model.names.keys())]
        print(f'[INFO] 类别: {CLASS_NAMES}')

    cap, desc = _open_v4l_camera(source, device_path)
    if cap is None:
        print('[ERROR] 无法打开任何摄像头。可试: python3 v4l_color_probe.py')
        sys.exit(1)

    stab = TrackStabilizer(
        alpha=smooth_alpha,
        min_area_ratio=min_area,
        new_track_conf=max(0.18, conf),
    )

    print('[OpenCV] 源=%s  conf=%.2f  imgsz=%d  enhance=%s  stabilize=%s'
          % (desc, conf, imgsz, enhance, stabilize))
    print('[建议] 环境太暗请补光；长期仍不准请用 capture_real_frames.py 采图微调。')

    win = 'Jenga 检测'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            infer = enhance_low_light_bgr(frame) if enhance else frame
            results = _predict(model, infer, conf, imgsz, augment, max_det)

            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now

            h, w = frame.shape[:2]
            tracks: list[dict] = []
            for r in results:
                dets = _dets_from_result(r, (h, w), min_area_ratio=min_area)
                if stabilize:
                    tracks = stab.update(dets, (h, w))
                else:
                    tracks = [
                        {'cls': d['cls'], 'conf': d['conf'],
                         'xyxy': d['xyxy'], 'mask_xy': d.get('mask_xy')}
                        for d in dets
                    ]

            tracks = merge_overlapping_tracks(tracks, merge_strength)
            tracks = limit_per_class(tracks, max_per_class)

            draw_stable(frame, tracks, fps)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except KeyboardInterrupt:
        pass
    cap.release()
    cv2.destroyAllWindows()


def run_ros(
    model_path: str,
    conf: float,
    headless: bool,
    imgsz: int,
    augment: bool,
    enhance: bool,
    stabilize: bool,
    smooth_alpha: float,
    min_area: float,
    max_det: int,
    merge_strength: float,
    max_per_class: int,
):
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from ultralytics import YOLO

    class JengaDetectorNode(Node):
        def __init__(self):
            global CLASS_NAMES
            super().__init__('jenga_detector')
            self._model = YOLO(model_path)
            if hasattr(self._model, 'names') and self._model.names:
                CLASS_NAMES = [self._model.names[i] for i in sorted(self._model.names.keys())]
            self._conf = conf
            self._imgsz = imgsz
            self._augment = augment
            self._enhance = enhance
            self._stabilize = stabilize
            self._stab = TrackStabilizer(
                alpha=smooth_alpha,
                min_area_ratio=min_area,
                new_track_conf=max(0.18, conf),
            )
            self._min_area = min_area
            self._max_det = max_det
            self._merge_strength = merge_strength
            self._max_per_class = max_per_class
            self._bridge = CvBridge()
            self._show = (not headless) and bool(os.environ.get('DISPLAY'))
            self._win = 'Jenga 检测（ROS）'
            self._fps_t = time.monotonic()
            self._fps_n = 0
            self._fps = 0.0

            self.create_subscription(
                Image, '/camera/camera/color/image_raw', self._cb, 10)

            if self._show:
                try:
                    cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self._win, 960, 540)
                except cv2.error:
                    self._show = False

            self.get_logger().info(
                'Jenga  conf=%.2f imgsz=%d enhance=%s stabilize=%s'
                % (conf, imgsz, enhance, stabilize)
            )

        def _cb(self, msg):
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            infer = enhance_low_light_bgr(frame) if self._enhance else frame
            results = _predict(
                self._model, infer, self._conf, self._imgsz,
                self._augment, self._max_det,
            )
            self._fps_n += 1
            now = time.monotonic()
            if now - self._fps_t >= 0.5:
                self._fps = self._fps_n / (now - self._fps_t)
                self._fps_n, self._fps_t = 0, now

            h, w = frame.shape[:2]
            tracks = []
            for r in results:
                dets = _dets_from_result(r, (h, w), min_area_ratio=self._min_area)
                if self._stabilize:
                    tracks = self._stab.update(dets, (h, w))
                else:
                    tracks = [
                        {'cls': d['cls'], 'conf': d['conf'],
                         'xyxy': d['xyxy'], 'mask_xy': d.get('mask_xy')}
                        for d in dets
                    ]

            tracks = merge_overlapping_tracks(tracks, self._merge_strength)
            tracks = limit_per_class(tracks, self._max_per_class)

            draw_stable(frame, tracks, self._fps)

            for t in tracks:
                x1, y1, x2, y2 = map(int, t['xyxy'])
                self.get_logger().info(
                    '%s  conf=%.2f  bbox=[%d,%d,%d,%d]'
                    % (CLASS_NAMES[t['cls']], t['conf'], x1, y1, x2, y2)
                )

            if self._show:
                try:
                    cv2.imshow(self._win, frame)
                    cv2.waitKey(1)
                except cv2.error:
                    self._show = False

        def destroy_node(self):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            super().destroy_node()

    rclpy.init()
    node = JengaDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


def _default_model() -> str:
    """自动寻找同目录下 models/jenga_best.pt，方便直接运行。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, 'models', 'jenga_best.pt')
    if os.path.exists(candidate):
        return candidate
    return ''


def main():
    ap = argparse.ArgumentParser(description='Jenga 实时检测（YOLOv8）')
    ap.add_argument('--model', default=_default_model(),
                    help='模型权重路径（默认自动找 models/jenga_best.pt）')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='推理置信度阈值（默认 0.25；暗光可试 0.18）')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--max-det', type=int, default=8)
    ap.add_argument('--no-enhance', action='store_true',
                    help='关闭 CLAHE/双边滤波')
    ap.add_argument('--no-stabilize', action='store_true',
                    help='关闭轨迹平滑')
    ap.add_argument('--smooth-alpha', type=float, default=0.55,
                    help='框 EMA 系数，越大越跟手、越小越稳（默认 0.55）')
    ap.add_argument('--min-area', type=float, default=0.0015,
                    help='最小框面积占画面比例，过滤噪点（默认 0.0015）')
    ap.add_argument('--merge-strength', type=float, default=0.12,
                    help='同类框合并阈值：IoU 与 交/小框 的较大值超过则并框（默认 0.12）')
    ap.add_argument('--max-per-class', type=int, default=3,
                    help='每类最多显示几个目标；桌上只有一块积木可设 1（默认 3）')
    ap.add_argument('--source', type=int, default=4)
    ap.add_argument('--device-path', default=None)
    ap.add_argument('--ros', action='store_true')
    ap.add_argument('--headless', action='store_true')
    args = ap.parse_args()

    model_path = args.model
    if not model_path:
        print('[ERROR] 未找到默认模型，请用 --model 指定权重路径，例如：', file=sys.stderr)
        print('  python3 detect_jenga.py --model models/jenga_best.pt', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(model_path):
        root_rel = os.path.join(os.path.dirname(__file__), '..', model_path)
        if os.path.exists(root_rel):
            model_path = root_rel
        else:
            print('[ERROR] 找不到模型: %s' % args.model, file=sys.stderr)
            sys.exit(1)

    dp = args.device_path or os.environ.get('DEVICE_PATH', '').strip() or None
    enhance = not args.no_enhance
    stabilize = not args.no_stabilize

    if args.ros:
        run_ros(
            model_path,
            args.conf,
            args.headless,
            args.imgsz,
            args.tta,
            enhance,
            stabilize,
            args.smooth_alpha,
            args.min_area,
            args.max_det,
            args.merge_strength,
            args.max_per_class,
        )
    else:
        run_opencv(
            model_path,
            args.source,
            args.conf,
            dp,
            args.imgsz,
            args.tta,
            enhance,
            stabilize,
            args.smooth_alpha,
            args.min_area,
            args.max_det,
            args.merge_strength,
            args.max_per_class,
        )


if __name__ == '__main__':
    main()
