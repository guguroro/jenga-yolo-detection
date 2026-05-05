"""capture_one_frame.py — 从 RealSense 截一帧，然后用两个模型都测一次，看置信度输出"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("[INFO] 等待摄像头稳定...")
    for _ in range(30):
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    frame = np.asanyarray(frames.get_color_frame().get_data())
    pipeline.stop()
except Exception as e:
    print(f"[ERROR] RealSense 失败: {e}")
    sys.exit(1)

save_path = os.path.join(BASE_DIR, "runs", "debug_frame.jpg")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, frame)
print(f"[INFO] 帧已保存: {save_path}")

from ultralytics import YOLO

for name, pt in [
    ("seg  (jenga_seg_best.pt)", os.path.join(BASE_DIR, "jenga_detection", "models", "jenga_seg_best.pt")),
    ("det  (jenga_best.pt)",     os.path.join(BASE_DIR, "jenga_detection", "models", "jenga_best.pt")),
]:
    print(f"\n── 模型: {name} ──")
    model = YOLO(pt)
    # 用极低阈值，把所有候选都打印出来
    res = model.predict(save_path, conf=0.01, verbose=False, iou=0.3)[0]
    if res.boxes is None or len(res.boxes) == 0:
        print("  ⚠  conf=0.01 下仍然 0 个检测 → 模型对这张图完全没响应")
    else:
        print(f"  找到 {len(res.boxes)} 个候选（conf≥0.01）:")
        for b in res.boxes:
            print(f"    conf={float(b.conf[0]):.3f}  xyxy={[round(x,1) for x in b.xyxy[0].tolist()]}")
