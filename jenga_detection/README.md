# Jenga 积木检测 —— 使用说明

## 目录结构

```
jenga_detection/
├── README.md                ← 本文件
├── detect_jenga.py          ← 实时检测（主程序）
├── train_jenga.py           ← 重新训练模型
├── capture_real_frames.py   ← 采集新训练图片
├── v4l_color_probe.py       ← 摄像头自动选取工具（被 detect 调用）
└── models/
    └── jenga_best.pt        ← 已训练好的模型权重
```

数据集位置（相对项目根目录）：
```
dataset/data2/Jenga Detection.v1i.yolov8/
├── train/   (39 张)
├── valid/   (3 张)
└── test/    (3 张)
```

---

## 快速上手：实时检测

```bash
cd ~/jenga_yolo_project
python3 jenga_detection/detect_jenga.py \
    --model jenga_detection/models/jenga_best.pt \
    --conf 0.25 --imgsz 640
```

按 `q` 或 `Esc` 退出。

**常用参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | 模型权重路径（.pt 文件） |
| `--conf` | 0.22 | 置信度阈值；调高减少误检，调低减少漏检 |
| `--imgsz` | 1280 | 推理分辨率；越大越精准但越慢 |
| `--source` | 4 | 摄像头设备号（`/dev/video4`） |
| `--device-path` | 无 | 指定摄像头路径，如 `/dev/video2` |
| `--no-enhance` | 关闭 | 关闭 CLAHE 低光增强 |
| `--no-stabilize` | 关闭 | 关闭检测框平滑（会更跟手但抖动） |
| `--max-per-class` | 3 | 每类最多显示几个框；只有一块积木时设 1 |

---

## 重新训练（使用 data2 数据集）

```bash
cd ~/jenga_yolo_project
python3 jenga_detection/train_jenga.py
```

或者指定参数：

```bash
python3 jenga_detection/train_jenga.py \
    --dataset "dataset/data2/Jenga Detection.v1i.yolov8" \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --out jenga_detection/runs/my_run
```

训练完成后，最优权重在 `--out` 指定目录下的 `weights/best.pt`。

---

## 采集新训练数据（提升精度）

若检测效果仍不好，可以用自己的摄像头采集更多真实图片：

```bash
python3 jenga_detection/capture_real_frames.py \
    --out my_real_frames/ \
    --interval 0.5    # 每 0.5 秒保存一帧
```

采集完成后，上传到 [Roboflow](https://roboflow.com) 标注，再导出 YOLOv8 格式，然后用 `train_jenga.py` 训练。

---

## YOLO 工作原理简介

### 1. 什么是 YOLO？

**YOLO（You Only Look Once）** 是一种单阶段目标检测算法。

传统方法先找"哪里可能有物体"再分类（两阶段），YOLO 将两件事合并成**一次前向传播**完成：
- 把图像划分成若干网格单元（grid cell）
- 每个单元同时预测：有没有物体、物体在哪（bounding box）、是什么类（类别概率）

这使它速度极快，适合实时摄像头检测。

### 2. 我们用的是 YOLOv8

- **版本**：YOLOv8s（`yolov8s.pt`，s = small）
- **框架**：`ultralytics` 库
- **任务类型**：目标检测（bounding box），输出矩形框 + 类别标签

```
输入图像 (640×640)
      ↓
  骨干网络（CSP-DarkNet）提取特征
      ↓
  颈部网络（PANet/FPN）多尺度融合
      ↓
  检测头：每个位置预测 [x,y,w,h,conf,cls]
      ↓
  NMS（非极大值抑制）去掉重叠框
      ↓
  输出：框坐标 + 置信度 + 类别
```

### 3. 我们做了什么？

#### 第一步：准备数据集

使用来自 Roboflow 的 **Jenga Detection** 数据集（data2）：
- 39 张训练图，3 张验证图，3 张测试图
- 每张图的标注格式（YOLO 格式）：
  ```
  0  0.597  0.426  0.324  0.688
  ↑   ↑      ↑      ↑      ↑
 类别 x中心  y中心  宽度   高度  （均为相对图像尺寸的比例值）
  ```

#### 第二步：修正数据集路径

Roboflow 导出的 `data.yaml` 用的是相对路径，YOLO 找不到文件。
脚本自动生成 `data_fixed.yaml`，把路径改成绝对路径：
```yaml
path: /home/.../Jenga Detection.v1i.yolov8
train: train/images
val:   valid/images
```

#### 第三步：训练（迁移学习）

不从零训练，而是在 **ImageNet 预训练权重**（`yolov8s.pt`）基础上微调：
```python
model = YOLO("yolov8s.pt")   # 加载预训练权重
model.train(data="data_fixed.yaml", epochs=150, ...)
```

为弥补训练集小（39 张）且与 RealSense 摄像头有域差，开启了强数据增强：
- **HSV 颜色抖动**（hsv_s=0.9）：模拟不同光照颜色
- **几何变换**（旋转 ±15°、缩放 ±55%、错切）：模拟不同拍摄角度
- **Mosaic 拼图**（mosaic=1.0）：4 张图拼成 1 张，增加背景多样性
- **MixUp**（mixup=0.20）：两张图叠加，增加泛化能力

训练提前在第 42/150 轮停止（EarlyStopping），第 2 轮就达到最优。

#### 第四步：训练结果

| 指标 | 数值 |
|------|------|
| mAP@50 | **0.752** |
| mAP@50-95 | **0.697** |
| Precision（准确率） | 0.931 |
| Recall（召回率） | 0.714 |

mAP50 = 0.752 意思是：在 IoU 阈值 50% 下，平均精度 75.2%。

#### 第五步：实时检测后处理

为防止低光、噪点导致框乱跳，加了 3 层后处理：

1. **CLAHE 低光增强**：提亮暗部，让模型"看清楚"
2. **轨迹平滑（EMA）**：当前帧的框 = 55% × 新检测 + 45% × 上一帧，减少抖动
3. **同类框合并**：两个高度重叠的同类框合并为一个大框，避免碎片化检测

---

## 依赖安装

```bash
pip install ultralytics opencv-python numpy
```
