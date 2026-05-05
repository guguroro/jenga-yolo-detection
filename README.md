# Jenga YOLO Detection

基于 **YOLOv8l-seg** 的积木（Jenga）实例分割检测系统，支持：
- 实时 RealSense 摄像头检测
- 静态图片批量推理
- 输出积木中心坐标、旋转角度、机器人目标位置

---

## 文件结构

```
jenga-yolo-detection/
├── models/
│   └── best.pt                  # 训练好的 YOLOv8l-seg 模型（两阶段训练）
├── jenga_detection/
│   ├── realsense_detect.py      # RealSense 实时检测（主程序）
│   ├── capture_one_frame.py     # 从摄像头抓取单帧保存
│   └── merge_and_train.py       # 合并数据集并重新训练
├── detect_pose.py               # 静态图片批量推理
├── train_v3.py                  # 两阶段训练脚本
├── merge_and_train.py           # 数据合并 + 训练（一键版）
├── data.yaml                    # 原始数据集配置
├── data_merged.yaml             # 合并数据集配置
└── data_self.yaml               # 自采数据集配置
```

---

## 环境要求

- Python **3.8+**（推荐 3.11）
- CUDA（可选，无 GPU 自动用 CPU）

安装依赖：

```bash
pip install ultralytics opencv-python numpy pyrealsense2
```

> 国内网络加速：
> ```bash
> pip install ultralytics opencv-python numpy pyrealsense2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
> ```

---

## 快速开始

### 1. 实时检测（RealSense 摄像头）

连接 Intel RealSense 摄像头后运行：

```bash
cd jenga-yolo-detection
python jenga_detection/realsense_detect.py
```

可选参数：

```bash
python jenga_detection/realsense_detect.py --conf 0.2 --imgsz 640 --no-enhance
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--conf` | 0.15 | 置信度阈值 |
| `--imgsz` | 640 | 推理分辨率 |
| `--no-enhance` | 关闭 | 禁用 CLAHE 图像增强 |

**运行时快捷键：**

| 按键 | 功能 |
|------|------|
| `q` / `Esc` | 退出 |
| `s` | 截图保存到 `runs/screenshots/` |
| `+` | 提高置信度阈值 |
| `-` | 降低置信度阈值 |

**检测窗口显示内容（每个积木）：**

```
#1  (640, 480) px        ← 像素中心坐标
angle:  35.0 deg         ← 旋转角度（-90°~90°，0° 为水平）
conf:   0.87             ← 置信度
move_x: +32.4 mm         ← 相机到积木的 X 方向移动量
move_y: -18.1 mm         ← 相机到积木的 Y 方向移动量
pos: (-117.6, -408.1, -7) mm   ← 机器人目标绝对坐标
rot: -33.0 deg           ← 机器人目标旋转角
```

坐标系参数（可在 `realsense_detect.py` 的 `draw()` 函数中修改）：

```python
REF_X, REF_Y = 733.5, 521.0      # 图像中相机参考点（像素）
SCALE = 0.2 / (363.4 - 733.5)    # 像素→米比例（20cm=555px 标定）
INIT_X, INIT_Y, INIT_Z = -150.0, -390.0, -7.0   # 机器人初始位置（mm）
INIT_ROT = 2.0                    # 机器人初始旋转角（deg）
```

---

### 2. 静态图片推理

```bash
python detect_pose.py
```

默认对 `real_pic/` 和 `test/images/` 目录下的图片推理，结果保存到 `runs/predict/new_model_test/`。

修改推理目录（编辑 `detect_pose.py` 开头）：

```python
IMAGE_DIRS = {
    "real": r"路径/到/你的图片目录",
    "test": r"路径/到/测试集目录",
}
OUTPUT_DIR = r"路径/到/输出目录"
```

---

## 模型训练

### 使用已有数据重新训练

如需用自己的数据重新训练，使用两阶段训练脚本：

```bash
python train_v3.py
```

**两阶段训练策略：**
- **Stage 1**：在合并数据集（原始 ~200 张 + 自采 40 张）上训练 `yolov8l-seg`，50 轮，640px
- **Stage 2**：仅用自采数据微调，30 轮，1280px，低学习率（域适应）

### 数据集标注建议

1. 使用 [Roboflow](https://roboflow.com) 标注
2. 选择 **Instance Segmentation**（实例分割）
3. 导出格式选 **YOLOv8**
4. 所有类别统一命名为 `jenga`（单类）

### 合并数据集并训练（一键）

```bash
python merge_and_train.py
```

修改 `merge_and_train.py` 中的 `SOURCES` 列表添加自己的数据路径。

---

## 坐标标定

如需重新标定像素→毫米的比例：

```bash
python measure_pixels.py
```

在弹出窗口中点击图像上已知距离的两点，程序输出像素距离。
用实际距离（mm）除以像素数即得比例系数，更新到 `realsense_detect.py` 中的 `SCALE`。

---

## 常见问题

**Q: RealSense 启动失败 `Frame didn't arrive`**
- 确认摄像头已连接，检查设备管理器
- 关闭其他占用摄像头的程序后重试

**Q: 检测不到积木 / 置信度很低**
- 按 `-` 键降低置信度阈值至 0.1
- 检查光线是否充足
- 考虑补充标注同场景数据后重新训练

**Q: 角度显示不准确**
- 当前角度定义：0° = 水平，正值 = 顺时针，范围 -90°~+90°
- 如需调整映射关系，修改 `draw()` 中的 `display_angle` 计算

**Q: Python 版本问题**
- 需要 Python 3.8+，ultralytics 不支持 3.7
- Windows 下建议用 `C:\python\python.exe` 调用 Python 3.11
