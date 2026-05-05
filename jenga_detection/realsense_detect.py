"""
realsense_detect.py — RealSense 摄像头实时 Jenga 积木检测

用法：
  python jenga_detection/realsense_detect.py
  python jenga_detection/realsense_detect.py --conf 0.1 --imgsz 1280
  按 q / Esc 退出，按 s 截图保存
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 优先用新训练的模型，fallback 旧分割模型
SEG_WEIGHTS  = os.path.join(BASE_DIR, "models", "best.pt")
DET_WEIGHTS  = os.path.join(BASE_DIR, "models", "best.pt")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "runs", "screenshots")

PALETTE = [
    (0, 220, 110), (255, 140, 0), (30, 144, 255),
    (220, 20, 60), (148, 0, 211), (255, 215, 0),
    (0, 200, 200), (255, 100, 200),
]


# ─── 图像增强（CLAHE） ───────────────────────────────────────────────────────

def enhance(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(out, 5, 40, 40)


# ─── 从 mask 算姿态（低分辨率 data 方式，fallback 用）────────────────────────

def pose_from_mask(mask_arr, W, H):
    m = (mask_arr * 255).astype(np.uint8)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle += 90
    return dict(cx=cx, cy=cy, angle=angle,
                length=max(w, h), width=min(w, h),
                rect=rect, contour=cnt)


# ─── 从 masks.xy 多边形算姿态（高精度，与 detect_pose.py 一致）───────────────

def pose_from_xy(xy):
    """xy: shape (N, 2) 原始图像坐标多边形"""
    if len(xy) < 3:
        return None
    pts = xy.astype(np.int32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle += 90
    # 用多边形轮廓用于半透明填充
    contour = pts.reshape(-1, 1, 2)
    return dict(cx=cx, cy=cy, angle=angle,
                length=max(w, h), width=min(w, h),
                rect=rect, contour=contour)


# ─── 从 bbox 算中心（纯检测模型用） ─────────────────────────────────────────

def pose_from_box(xyxy):
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    # 用长宽比估角度
    angle = 0.0 if w >= h else 90.0
    rect = ((cx, cy), (w, h), angle)
    return dict(cx=cx, cy=cy, angle=angle,
                length=max(w, h), width=min(w, h),
                rect=rect, contour=None)


# ─── 绘制 ────────────────────────────────────────────────────────────────────

def draw(frame, poses, fps, conf_thr, model_tag):
    overlay = frame.copy()
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, p in enumerate(poses):
        color = PALETTE[i % len(PALETTE)]

        # mask 半透明
        if p["contour"] is not None:
            cv2.fillPoly(overlay, [p["contour"]], color)

        # 旋转框
        box = cv2.boxPoints(p["rect"]).astype(np.int32)
        cv2.drawContours(frame, [box], 0, color, 2)

        cx_i, cy_i = int(p["cx"]), int(p["cy"])

        # 中心点
        cv2.circle(frame, (cx_i, cy_i), 9, color, -1)
        cv2.circle(frame, (cx_i, cy_i), 3, (255, 255, 255), -1)

        # 移动量计算（相机坐标系，单位 m）
        REF_X, REF_Y = 733.5, 521.0
        SCALE = 0.2 / (363.4 - 733.5)   # m/px
        move_x = (cx_i - REF_X) * SCALE
        move_y = (cy_i - REF_Y) * SCALE

        # 从参考点到积木中心画移动箭头（青色虚线效果用细箭头代替）
        ref_i = (int(REF_X), int(REF_Y))
        cv2.arrowedLine(frame, ref_i, (cx_i, cy_i), (0, 255, 255), 1,
                        tipLength=max(0.05, 12 / max(1, np.hypot(cx_i - ref_i[0], cy_i - ref_i[1]))))

        # 角度映射（仅用于显示文字）：-90° → 0°，范围 [-90, 90)
        display_angle = (p["angle"] + 90) % 180
        if display_angle > 90:
            display_angle -= 180

        # 方向箭头用原始角度，保证与积木实际朝向一致
        a_rad = np.deg2rad(p["angle"])
        alen = max(30, int(p["length"] * 0.4))
        ex, ey = int(cx_i + alen * np.cos(a_rad)), int(cy_i + alen * np.sin(a_rad))
        cv2.arrowedLine(frame, (cx_i, cy_i), (ex, ey), (255, 255, 255), 2, tipLength=0.28)

        # 坐标 + 角度 + 置信度 + 移动量文字（带深色背景）
        conf_val = p.get("conf", 0.0)
        # 目标坐标 = 初始位置 + 移动量（单位 mm）
        INIT_X, INIT_Y, INIT_Z = -150.0, -390.0, -7.0
        INIT_ROT = 2.0
        target_x   = INIT_X + move_x * 1000
        target_y   = INIT_Y - move_y * 1000
        target_rot = INIT_ROT - display_angle

        sign_x = "+" if move_x >= 0 else ""
        sign_y = "+" if move_y >= 0 else ""
        lines = [f"#{i+1}  ({cx_i}, {cy_i}) px",
                 f"angle:  {display_angle:.1f} deg",
                 f"conf:   {conf_val:.2f}",
                 f"move_x: {sign_x}{move_x*1000:.1f} mm",
                 f"move_y: {sign_y}{move_y*1000:.1f} mm",
                 f"pos: ({target_x:.1f}, {target_y:.1f}, {INIT_Z:.0f}) mm",
                 f"rot: {target_rot:.1f} deg"]
        fs, th = 0.52, 1
        sizes = [cv2.getTextSize(l, font, fs, th)[0] for l in lines]
        bw = max(s[0] for s in sizes) + 10
        lh = sizes[0][1]
        bh = lh * len(lines) + (len(lines) + 1) * 5
        tx = min(cx_i + 14, W - bw - 4)
        ty = max(cy_i - bh - 4, 4)
        cv2.rectangle(frame, (tx - 2, ty), (tx + bw, ty + bh), (20, 20, 20), -1)
        for j, (line, sz) in enumerate(zip(lines, sizes)):
            if j == 0:
                c = color
            elif j in (3, 4):
                c = (0, 255, 255)    # move_x/y 青色
            elif j == 5:
                c = (100, 220, 255)  # 目标坐标 浅蓝色
            elif j == 6:
                c = (180, 255, 180)  # 目标角度 浅绿色
            else:
                c = (200, 200, 200)
            cv2.putText(frame, line, (tx + 2, ty + (j + 1) * (lh + 5)),
                        font, fs, c, th, cv2.LINE_AA)

    # mask 叠加
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    # 参考点（相机原点）十字标
    rx, ry = int(733.5), int(521)
    cv2.drawMarker(frame, (rx, ry), (0, 255, 255),
                   cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(frame, "CAM origin", (rx + 8, ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # HUD 右上角
    hud = f"FPS {fps:.1f}   Blocks: {len(poses)}   conf>={conf_thr}   [{model_tag}]"
    (hw, hh), _ = cv2.getTextSize(hud, font, 0.55, 2)
    cv2.rectangle(frame, (W - hw - 12, 4), (W - 2, hh + 14), (0, 0, 0), -1)
    cv2.putText(frame, hud, (W - hw - 8, hh + 8), font, 0.55, (0, 255, 180), 2, cv2.LINE_AA)

    # 底部提示
    cv2.putText(frame, "q/Esc: quit   s: screenshot   +/-: conf",
                (8, H - 10), font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)


# ─── 主循环 ──────────────────────────────────────────────────────────────────

def run(weights, conf, imgsz, use_enhance):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] 未安装 pyrealsense2，请运行：")
        print("  C:\\python\\python.exe -m pip install pyrealsense2")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(weights)
    is_seg = hasattr(model, "task") and "segment" in str(model.task).lower()
    # 也根据文件名判断
    if "seg" in os.path.basename(weights).lower():
        is_seg = True
    model_tag = "seg" if is_seg else "det"
    print(f"[INFO] 模型: {os.path.basename(weights)}  类型: {model_tag}")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(cfg)
        dev_name = profile.get_device().get_info(rs.camera_info.name)
        print(f"[INFO] RealSense 已连接: {dev_name}")
    except Exception as e:
        print(f"[ERROR] 启动 RealSense 失败: {e}")
        sys.exit(1)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    win = "Jenga RealSense 实时检测"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    current_conf = conf
    print(f"[INFO] 置信度阈值: {current_conf:.2f}  增强: {use_enhance}")
    print("[INFO] 按 q/Esc 退出，s 截图，+/- 动态调整置信度阈值")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            H, W = frame.shape[:2]

            infer = enhance(frame) if use_enhance else frame
            results = model.predict(infer, conf=current_conf, imgsz=imgsz,
                                    verbose=False, iou=0.4)

            # 解析检测结果
            poses = []
            r = results[0]
            if is_seg and r.masks is not None:
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                for idx, xy in enumerate(r.masks.xy):
                    p = pose_from_xy(xy)      # 用高精度多边形坐标算角度
                    if p:
                        p["conf"] = float(confs[idx]) if idx < len(confs) else 0.0
                        poses.append(p)
            elif r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    p = pose_from_box(xyxy)
                    p["conf"] = float(box.conf[0].cpu())
                    poses.append(p)

            # FPS
            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now

            draw(frame, poses, fps, current_conf, model_tag)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                p = os.path.join(SCREENSHOT_DIR, f"jenga_{ts}.jpg")
                cv2.imwrite(p, frame)
                print(f"[截图] {p}")
            elif key == ord("+") or key == ord("="):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                print(f"[INFO] 置信度 → {current_conf:.2f}")
            elif key == ord("-"):
                current_conf = max(0.05, round(current_conf - 0.05, 2))
                print(f"[INFO] 置信度 → {current_conf:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 已退出")


# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RealSense 实时 Jenga 检测")
    # 默认先用检测模型（更鲁棒），如存在分割模型则优先
    default_w = SEG_WEIGHTS if os.path.exists(SEG_WEIGHTS) else DET_WEIGHTS
    ap.add_argument("--weights", default=default_w)
    ap.add_argument("--conf",   type=float, default=0.15, help="置信度阈值（默认 0.15）")
    ap.add_argument("--imgsz",  type=int,   default=640)
    ap.add_argument("--no-enhance", action="store_true", help="关闭 CLAHE 增强")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"[ERROR] 找不到模型: {args.weights}")
        sys.exit(1)

    run(args.weights, args.conf, args.imgsz, not args.no_enhance)


if __name__ == "__main__":
    main()
