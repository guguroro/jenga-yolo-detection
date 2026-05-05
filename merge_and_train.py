import os
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"F:\LUND course\Project in control\Proj_control_yolo")
ORIG_IMAGES = BASE / "train" / "images"
ORIG_LABELS = BASE / "train" / "labels"
SELF_IMAGES = BASE / "self_data" / "train" / "images"
SELF_LABELS = BASE / "self_data" / "train" / "labels"
MERGED_IMAGES = BASE / "train_merged" / "images"
MERGED_LABELS = BASE / "train_merged" / "labels"

MERGED_IMAGES.mkdir(parents=True, exist_ok=True)
MERGED_LABELS.mkdir(parents=True, exist_ok=True)

# 复制原始训练集（类别本来就是 0，直接复制）
print("复制原始训练集...")
for f in ORIG_IMAGES.iterdir():
    shutil.copy2(f, MERGED_IMAGES / f.name)
for f in ORIG_LABELS.iterdir():
    shutil.copy2(f, MERGED_LABELS / f.name)
print(f"  原始图片: {len(list(ORIG_IMAGES.iterdir()))} 张")

# 复制 self_data 图片，并把标注中的类别 1、2 重映射为 0
print("复制 self_data 并重映射类别...")
img_count = 0
label_fixed = 0
for img_file in SELF_IMAGES.iterdir():
    shutil.copy2(img_file, MERGED_IMAGES / img_file.name)
    img_count += 1

for label_file in SELF_LABELS.iterdir():
    lines = label_file.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        parts[0] = "0"          # 所有类别统一改为 0（jenga）
        new_lines.append(" ".join(parts))
    (MERGED_LABELS / label_file.name).write_text("\n".join(new_lines), encoding="utf-8")
    label_fixed += 1

print(f"  self_data 图片: {img_count} 张，标注已重映射: {label_fixed} 个")
print(f"\n合并后训练集总图片: {len(list(MERGED_IMAGES.iterdir()))} 张")

# 生成新的 data_merged.yaml
yaml_content = f"""train: {MERGED_IMAGES.as_posix()}
val:   {(BASE / 'valid' / 'images').as_posix()}
test:  {(BASE / 'test' / 'images').as_posix()}

nc: 1
names: ['jenga']
"""
yaml_path = BASE / "data_merged.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"已生成: {yaml_path}")

if __name__ == "__main__":
    print("\n开始训练（GPU）...")
    model = YOLO(str(BASE / "yolov8l-seg.pt"))
    model.train(
        data=str(yaml_path),
        epochs=60,
        imgsz=1280,
        batch=4,
        device=0,
        project=str(BASE / "runs"),
        name="jenga_seg_v3",
        exist_ok=True,
    )
    print("\n训练完成！")
    print(f"最佳权重: {BASE / 'runs' / 'jenga_seg_v3' / 'weights' / 'best.pt'}")
