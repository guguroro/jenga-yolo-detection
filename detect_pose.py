import cv2
import numpy as np
from ultralytics import YOLO
import os

_REPO   = r"F:\AAA lund course\Project_jenga\Proj_control_yolo-20260504T073013Z-3-001\Proj_control_yolo\jenga-yolo-detection"
_ROOT   = r"F:\AAA lund course\Project_jenga\Proj_control_yolo-20260504T073013Z-3-001\Proj_control_yolo"
WEIGHTS    = os.path.join(_REPO, "models", "best.pt")
IMAGE_DIRS = {
    "real": os.path.join(_ROOT, "real_pic"),
    "test": os.path.join(_ROOT, "test", "images"),
}
OUTPUT_DIR = os.path.join(_ROOT, "runs", "predict", "new_model_test")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(WEIGHTS)

def imwrite_u(path, img):
    """支持中文路径的图片保存"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())

colors = [(0,200,255),(0,255,100),(255,80,0),(200,0,255),(255,200,0),(0,120,255),(100,255,0),(0,255,200)]

for src_tag, IMAGE_DIR in IMAGE_DIRS.items():
    if not os.path.isdir(IMAGE_DIR):
        print(f"[跳过] 目录不存在: {IMAGE_DIR}")
        continue
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    print(f"\n[{src_tag}] 共 {len(image_files)} 张图片")

    for img_file in sorted(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{img_file}: 读取失败")
            continue

        results = model.predict(img, conf=0.15, imgsz=1280, verbose=False)
        r = results[0]

        out_name = f"{src_tag}_{img_file}"
        if r.masks is None:
            print(f"{img_file}: 未检测到积木")
            imwrite_u(os.path.join(OUTPUT_DIR, out_name), img)
            continue

        print(f"\n{'='*50}")
        print(f"图片: {img_file}  →  检测到 {len(r.masks.xy)} 块积木")
        print(f"{'='*50}")

        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        for i, xy in enumerate(r.masks.xy):
            if len(xy) < 3:
                continue
            col = colors[i % len(colors)]
            pts = xy.astype(np.int32)

            rect = cv2.minAreaRect(pts)
            (cx, cy), (w2, h2), angle = rect
            if w2 < h2:
                angle += 90
            length, width = max(w2, h2), min(w2, h2)

            # 长宽比过滤：Jenga 积木约 3:1，比值小于 1.8 的视为误检跳过
            if width < 1 or length / width < 1.8:
                print(f"  [跳过] 检测 {i+1}: 长宽比 {length/width:.2f} 过低，非积木")
                continue

            conf_val = float(confs[i]) if i < len(confs) else 0.0
            print(f"  积木 {i+1}:")
            print(f"    中心坐标: ({cx:.1f}, {cy:.1f}) px")
            print(f"    旋转角度: {angle:.1f}°")
            print(f"    置信度:   {conf_val:.2f}")
            print(f"    尺寸:     {length:.1f} x {width:.1f} px")

            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(img, [box], 0, col, 2)
            cxi, cyi = int(cx), int(cy)
            cv2.circle(img, (cxi, cyi), 5, col, -1)
            cv2.circle(img, (cxi, cyi), 5, (255, 255, 255), 2)
            rad = np.deg2rad(angle)
            arr = int(length * 0.45)
            ex = int(cxi + arr * np.cos(rad))
            ey = int(cyi + arr * np.sin(rad))
            cv2.arrowedLine(img, (cxi, cyi), (ex, ey), col, 2, tipLength=0.2)
            label = f"#{i+1} ({cx:.0f},{cy:.0f}) {angle:.1f}deg  {conf_val:.2f}"
            cv2.putText(img, label, (cxi+8, cyi-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        imwrite_u(os.path.join(OUTPUT_DIR, out_name), img)

print(f"\n所有结果已保存到: {OUTPUT_DIR}")
