import cv2
import numpy as np
from ultralytics import YOLO
import os

WEIGHTS = r"F:\LUND course\Project in control\Proj_control_yolo\runs\segment\runs\jenga_seg\weights\best.pt"
IMAGE_DIR = r"F:\LUND course\Project in control\Proj_control_yolo\test\images"
OUTPUT_DIR = r"F:\LUND course\Project in control\Proj_control_yolo\runs\predict\jenga_pose"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(WEIGHTS)

def get_pose_from_mask(mask_array):
    """从分割mask提取中心坐标和角度"""
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None

    # 最小旋转外接矩形 → 中心、尺寸、角度
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # 统一角度定义：让长边方向对应角度，范围 [-90, 90]
    if w < h:
        angle += 90

    return {
        "cx": cx,
        "cy": cy,
        "angle": angle,
        "width": min(w, h),
        "length": max(w, h),
        "contour": contour,
        "rect": rect,
    }

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    results = model.predict(img_path, conf=0.25, verbose=False)

    result = results[0]
    if result.masks is None:
        print(f"{img_file}: 未检测到积木")
        continue

    masks = result.masks.data.cpu().numpy()
    h_orig, w_orig = img.shape[:2]

    print(f"\n{'='*50}")
    print(f"图片: {img_file}  →  检测到 {len(masks)} 块积木")
    print(f"{'='*50}")

    for i, mask in enumerate(masks):
        # 将mask缩放回原图尺寸
        mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        pose = get_pose_from_mask(mask_resized)
        if pose is None:
            continue

        print(f"  积木 {i+1}:")
        print(f"    中心坐标: ({pose['cx']:.1f}, {pose['cy']:.1f}) px")
        print(f"    旋转角度: {pose['angle']:.1f}°")
        print(f"    尺寸:     {pose['length']:.1f} x {pose['width']:.1f} px")

        # 在图上画出旋转框和坐标轴
        box = cv2.boxPoints(pose["rect"])
        box = box.astype(np.int32)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        cv2.drawContours(img, [box], 0, color, 2)

        # 画中心点
        cx_i, cy_i = int(pose["cx"]), int(pose["cy"])
        cv2.circle(img, (cx_i, cy_i), 5, color, -1)

        # 画角度方向箭头（长轴方向）
        angle_rad = np.deg2rad(pose["angle"])
        arrow_len = int(pose["length"] / 2)
        ex = int(cx_i + arrow_len * np.cos(angle_rad))
        ey = int(cy_i + arrow_len * np.sin(angle_rad))
        cv2.arrowedLine(img, (cx_i, cy_i), (ex, ey), color, 2, tipLength=0.2)

        # 标注文字
        label = f"#{i+1} ({pose['cx']:.0f},{pose['cy']:.0f}) {pose['angle']:.1f}deg"
        cv2.putText(img, label, (cx_i + 8, cy_i - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    out_path = os.path.join(OUTPUT_DIR, img_file)
    cv2.imwrite(out_path, img)

print(f"\n所有结果已保存到: {OUTPUT_DIR}")
