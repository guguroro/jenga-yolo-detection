# Jenga YOLOv8 Detection — 积木检测与位姿估计

基于 **YOLOv8l-seg** 实例分割模型，检测 Jenga 积木并实时输出每块积木的：
- 2D 中心坐标 `(cx, cy)`（像素）
- 旋转角度 `θ`（度）
- 物理尺寸（长 × 宽，像素）

---

## 目录结构

```
jenga-yolo-detection/
├── detect_pose.py        # 主脚本：检测积木 + 绘制中心点与方向
├── predict_jenga.py      # 批量推理，保存带 mask 的结果图
├── train_jenga.py        # 基础训练脚本
├── merge_and_train.py    # 合并自有数据集后训练
├── train_v3.py           # 两阶段训练（推荐，效果最好）
├── data.yaml             # 原始数据集配置
├── data_merged.yaml      # 合并数据集配置
├── data_self.yaml        # 自有数据微调配置
└── models/
    └── best.pt           # 训练好的权重（YOLOv8l-seg，两阶段微调）
```

---

## 环境依赖

```bash
pip install ultralytics opencv-python numpy
```

> GPU 加速（推荐）：需安装对应 CUDA 版本的 PyTorch，例如 CUDA 11.8：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 快速使用

### 1. 检测积木并绘制中心点 + 方向（主要功能）

编辑 `detect_pose.py` 顶部的路径：

```python
WEIGHTS  = "models/best.pt"          # 权重路径
IMAGE_DIR = "path/to/your/images"    # 输入图片文件夹
OUTPUT_DIR = "runs/predict/output"   # 输出结果文件夹
```

运行：

```bash
python detect_pose.py
```

**输出示例：**
```
图片: test1.jpg  →  检测到 7 块积木
  积木 1:
    中心坐标: (756.2, 930.2) px
    旋转角度: 25.7°
    尺寸:     231.4 x 92.3 px
  ...
```

每张图片会在 `OUTPUT_DIR` 保存带标注的结果图，包含：
- 彩色旋转矩形框
- 中心点（带白色描边的彩色圆点）
- 方向箭头
- 坐标与角度标签

### 2. 批量推理（仅保存检测结果图）

```bash
python predict_jenga.py
```

---

## 训练

### 使用自有数据（推荐）

将自有图片放入 `self_data/images/`，标注文件（YOLO 格式 `.txt`）放入 `self_data/labels/`。

运行两阶段训练（先用合并数据集预训练，再用自有数据微调）：

```bash
python train_v3.py
```

- **Stage 1**：在合并数据集（~249张）上训练 50 epochs，`imgsz=640`
- **Stage 2**：在自有数据（~40张）上微调 30 epochs，`imgsz=1280`，低学习率

最终权重保存在：`runs/jenga_v3_stage2/weights/best.pt`

---

## 后处理过滤规则

`detect_pose.py` 内置两条过滤规则，自动排除误检：

| 规则 | 阈值 | 说明 |
|------|------|------|
| 长宽比过滤 | `长/宽 < 1.8` 跳过 | Jenga 积木约 3:1，近正方形的为误检 |
| 轮廓点数 | `点数 < 3` 跳过 | mask 轮廓过于简单，跳过 |

---

## 注意事项

- 图片文件名建议使用**英文**，避免中文路径导致 OpenCV 读写异常
- 若相机距离变化较大，需根据实际像素尺寸调整长宽比阈值
- 当前模型对**密集堆叠**或**严重遮挡**的积木识别精度有限，建议补充相应场景的标注数据后重新训练

---

## 数据集来源

- [Roboflow Jenga Detection Dataset](https://universe.roboflow.com/) — 209 张基础训练图片
- 自采实拍数据（Intel RealSense 相机）— 40 张场景微调图片

---

## 模型性能

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv8l-seg |
| 训练策略 | 两阶段（预训练 + 域适应微调） |
| 输入分辨率 | 1280×1280（Stage 2） |
| 推理速度 | ~120ms/张（RTX 4070） |
