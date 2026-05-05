from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs",
    name="jenga_seg",
    exist_ok=True,
)

print("\n训练完成！")
print(f"最佳权重保存在：runs/jenga_seg/weights/best.pt")
