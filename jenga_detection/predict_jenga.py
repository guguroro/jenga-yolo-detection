from ultralytics import YOLO
import os

model = YOLO(r"F:\LUND course\Project in control\Proj_control_yolo\runs\segment\runs\jenga_seg\weights\best.pt")

results = model.predict(
    source=r"F:\LUND course\Project in control\Proj_control_yolo\test\images",
    save=True,
    save_conf=True,
    conf=0.25,
    project=r"F:\LUND course\Project in control\Proj_control_yolo\runs\predict",
    name="jenga_test",
    exist_ok=True,
)

print("\n推理完成！")
print(r"结果图片保存在：F:\LUND course\Project in control\Proj_control_yolo\runs\predict\jenga_test")
