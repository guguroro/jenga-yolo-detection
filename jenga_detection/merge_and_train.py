"""
merge_and_train.py
把原始训练集（train/）和自采数据集（self_data/）合并为单类别 "jenga"，
按 85/15 随机切分 train/valid，生成 merged_data.yaml，然后启动训练。
"""
import os
import shutil
import random
import yaml

ROOT = r"F:\AAA lund course\Project_jenga\Proj_control_yolo-20260504T073013Z-3-001\Proj_control_yolo"
REPO = os.path.join(ROOT, "jenga-yolo-detection")

# ── 来源目录 ────────────────────────────────────────────────────────────────
SOURCES = [
    {
        "images": os.path.join(ROOT, "train", "images"),
        "labels": os.path.join(ROOT, "train", "labels"),
        "remap": None,          # 原来就是 class 0，不需要重映射
    },
    {
        "images": os.path.join(ROOT, "self_data", "Jenga_Lund_Lab.yolov8", "train", "images"),
        "labels": os.path.join(ROOT, "self_data", "Jenga_Lund_Lab.yolov8", "train", "labels"),
        "remap": {0: 0, 1: 0, 2: 0},   # 3 个类别全映射为 0
    },
]

# ── 输出目录 ────────────────────────────────────────────────────────────────
MERGED = os.path.join(ROOT, "merged_dataset")
for split in ("train", "valid"):
    os.makedirs(os.path.join(MERGED, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(MERGED, split, "labels"), exist_ok=True)

# ── 收集所有 (image_path, label_path, remap) ───────────────────────────────
all_samples = []
for src in SOURCES:
    img_dir, lbl_dir, remap = src["images"], src["labels"], src["remap"]
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(img_file)[0]
        lbl_file = stem + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_file)
        if not os.path.exists(lbl_path):
            print(f"[WARN] 找不到标签，跳过: {lbl_file}")
            continue
        all_samples.append((
            os.path.join(img_dir, img_file),
            lbl_path,
            remap,
            img_file,
        ))

print(f"[INFO] 总样本数: {len(all_samples)}")

# ── 随机打乱并切分 ──────────────────────────────────────────────────────────
random.seed(42)
random.shuffle(all_samples)
n_train = int(len(all_samples) * 0.85)
splits = {"train": all_samples[:n_train], "valid": all_samples[n_train:]}
print(f"[INFO] train: {n_train}  valid: {len(all_samples) - n_train}")

# ── 复制图片 + 重写标签 ─────────────────────────────────────────────────────
for split, samples in splits.items():
    for img_path, lbl_path, remap, img_file in samples:
        # 图片：直接复制
        shutil.copy2(img_path, os.path.join(MERGED, split, "images", img_file))

        # 标签：重映射 class id
        stem = os.path.splitext(img_file)[0]
        out_lbl = os.path.join(MERGED, split, "labels", stem + ".txt")
        with open(lbl_path, "r") as f:
            lines = f.readlines()
        with open(out_lbl, "w") as f:
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if remap is not None:
                    cls_id = remap.get(cls_id, 0)
                f.write(f"{cls_id} " + " ".join(parts[1:]) + "\n")

print("[INFO] 数据合并完成")

# ── 生成 data.yaml ──────────────────────────────────────────────────────────
data_yaml = {
    "train": os.path.join(MERGED, "train", "images").replace("\\", "/"),
    "val":   os.path.join(MERGED, "valid", "images").replace("\\", "/"),
    "nc": 1,
    "names": ["jenga"],
}
yaml_path = os.path.join(MERGED, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
print(f"[INFO] data.yaml 已生成: {yaml_path}")

# ── 启动训练 ────────────────────────────────────────────────────────────────
from ultralytics import YOLO

base_model = os.path.join(ROOT, "yolov8n-seg.pt")
if not os.path.exists(base_model):
    base_model = "yolov8n-seg.pt"   # 自动下载

print(f"\n[INFO] 开始训练，基础模型: {base_model}")
model = YOLO(base_model)
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=8,
    project=os.path.join(ROOT, "runs"),
    name="jenga_merged",
    exist_ok=True,
    patience=20,
    # 数据增强（弥补数据量不足）
    hsv_h=0.015, hsv_s=0.9, hsv_v=0.4,
    degrees=15,
    scale=0.5,
    mosaic=1.0,
    mixup=0.1,
    flipud=0.3,
    fliplr=0.5,
)

best_pt = os.path.join(ROOT, "runs", "jenga_merged", "weights", "best.pt")
dest_pt  = os.path.join(REPO, "jenga_detection", "models", "jenga_seg_best.pt")

if os.path.exists(best_pt):
    shutil.copy2(best_pt, dest_pt)
    print(f"\n✅ 训练完成！新模型已保存到: {dest_pt}")
    print("现在可以直接运行 detect_pose.py 使用新模型。")
else:
    print(f"[WARN] 未找到 {best_pt}，请手动复制到 {dest_pt}")
