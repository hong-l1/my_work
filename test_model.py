import torch

from ultralytics import YOLO

# 加载自定义模型
model = YOLO("yolov8n_basketball.yaml")

# 打印模型结构，确认DCNv3和BGAM模块已插入
print(model.model)

# 测试前向传播
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    outputs = model(x)

print(f"\n模型输出形状: {[o.shape for o in outputs]}")
print("✅ 模型加载和前向传播成功！")
