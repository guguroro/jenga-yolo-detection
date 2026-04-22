#!/usr/bin/env python3
"""
训练 Jenga 积木检测模型（YOLOv8 detection）。

支持两个数据集:
  data1 (旧): yolo_aruco/dataset/traning_data/Jenga Vision Pic.v1i.yolov8  (3类 seg)
  data2 (新): yolo_aruco/dataset/data2/Jenga Detection.v1i.yolov8          (1类 det)

用法:
  cd ~/jenga_project
  python3 yolo_aruco/train_jenga.py                          # 用 data2，检测模型
  python3 yolo_aruco/train_jenga.py --dataset <路径> --model yolov8s-seg.pt  # 自定义
"""
import sys
import textwrap
from pathlib import Path

DATA2_DEFAULT = 'yolo_aruco/dataset/data2/Jenga Detection.v1i.yolov8'


def fix_data_yaml(dataset_root: Path, nc: int, names: list) -> Path:
    """生成含绝对路径的 data_fixed.yaml，避免 Roboflow 相对路径问题。"""
    fixed = dataset_root / 'data_fixed.yaml'
    names_str = str(names)
    content = textwrap.dedent(f"""\
        path: {dataset_root}
        train: train/images
        val:   valid/images
        test:  test/images

        nc: {nc}
        names: {names_str}
    """)
    fixed.write_text(content)
    print(f'[data.yaml] 已写入 {fixed}')
    return fixed


def main():
    import argparse
    ap = argparse.ArgumentParser(description='训练 Jenga YOLOv8 检测模型')
    ap.add_argument('--dataset', default=DATA2_DEFAULT,
                    help='数据集根目录（含 train/valid/test）')
    ap.add_argument('--model', default='yolov8s.pt',
                    help='基础模型（yolov8n/s/m.pt 或 *-seg.pt）')
    ap.add_argument('--epochs', type=int, default=150, help='训练轮数（默认 150）')
    ap.add_argument('--batch', type=int, default=16,
                    help='Batch size（默认 16；显存不够可改 8）')
    ap.add_argument('--imgsz', type=int, default=640,
                    help='训练输入边长（默认 640）')
    ap.add_argument('--device', default='cuda:0', help='cuda:0 / cpu')
    ap.add_argument('--out', default='yolo_aruco/runs/jenga_det_v1',
                    help='输出目录')
    ap.add_argument('--nc', type=int, default=1, help='类别数（data2=1）')
    ap.add_argument('--names', nargs='+', default=['Jenga'], help='类别名称列表')
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print('[ERROR] 未安装 ultralytics：pip install ultralytics', file=sys.stderr)
        sys.exit(1)

    root = Path(args.dataset).resolve()
    if not root.exists():
        print(f'[ERROR] 数据集目录不存在: {root}', file=sys.stderr)
        sys.exit(1)

    n_train = len(list((root / 'train' / 'images').glob('*')))
    n_val   = len(list((root / 'valid' / 'images').glob('*')))
    n_test  = len(list((root / 'test'  / 'images').glob('*')))
    print(f'数据集: {root.name}')
    print(f'  train={n_train}  val={n_val}  test={n_test}  nc={args.nc}  names={args.names}')

    data_yaml = fix_data_yaml(root, args.nc, args.names)

    model = YOLO(args.model)
    print(f'\n模型: {args.model}  epochs={args.epochs}  batch={args.batch}'
          f'  imgsz={args.imgsz}  device={args.device}')

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        exist_ok=True,
        patience=40,
        plots=True,
        verbose=True,
        # ── 强数据增强：小数据集 + RealSense 域差 ──
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.55,
        degrees=15.0,
        translate=0.12,
        scale=0.55,
        shear=3.0,
        flipud=0.15,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.20,
        close_mosaic=20,
    )

    best = Path(args.out) / 'weights' / 'best.pt'
    print(f'\n训练完成！最优权重: {best}')
    print(f'下一步运行检测: python3 yolo_aruco/detect_jenga.py --model {best}')


if __name__ == '__main__':
    main()
