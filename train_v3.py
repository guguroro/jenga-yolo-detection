from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"F:\LUND course\Project in control\Proj_control_yolo")

def main():
    # ── 阶段一：全量数据训练（打好通用基础）──────────────────────────
    print("=" * 55)
    print("阶段一：全量数据训练（249张，yolov8l-seg，640px）")
    print("=" * 55)

    model = YOLO(str(BASE / "yolov8l-seg.pt"))
    model.train(
        data=str(BASE / "data_merged.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        lr0=0.01,
        project=str(BASE / "runs"),
        name="jenga_v3_stage1",
        exist_ok=True,
    )
    stage1_best = BASE / "runs" / "jenga_v3_stage1" / "weights" / "best.pt"
    print(f"\n阶段一完成，权重: {stage1_best}")

    # ── 生成只包含自己数据的 yaml ──────────────────────────────────
    self_yaml = BASE / "data_self.yaml"
    self_yaml.write_text(
        f"train: {(BASE / 'self_data' / 'train' / 'images').as_posix()}\n"
        f"val:   {(BASE / 'valid' / 'images').as_posix()}\n"
        f"test:  {(BASE / 'self_data' / 'test').as_posix()}\n"
        f"nc: 1\nnames: ['jenga']\n",
        encoding="utf-8",
    )

    # ── 阶段二：仅用自己数据微调（域适应）────────────────────────────
    print("\n" + "=" * 55)
    print("阶段二：自拍数据域适应微调（40张，低学习率，1280px）")
    print("=" * 55)

    model2 = YOLO(str(stage1_best))
    model2.train(
        data=str(self_yaml),
        epochs=30,
        imgsz=1280,
        batch=4,
        device=0,
        lr0=0.001,
        warmup_epochs=3,
        project=str(BASE / "runs"),
        name="jenga_v3_stage2",
        exist_ok=True,
    )

    final_best = BASE / "runs" / "jenga_v3_stage2" / "weights" / "best.pt"
    print(f"\n全部完成！最终权重: {final_best}")

if __name__ == "__main__":
    main()
