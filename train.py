from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 使用nano版本，速度快、体积小

results = model.train(
    data="dataset.yaml",       # 数据集配置文件路径（包含train/val路径及类别）
    epochs=1000,                # 训练轮数
    imgsz=960,                 # 输入图像尺寸
    batch=16,                  # 批次大小（根据显存调整）
    device=0,                  # GPU编号，CPU则设为'cpu'
    workers=8,                 # 数据加载线程数
    lr0=0.01,                  # 初始学习率
    project="runs/train",      # 结果保存目录
    name="my_yolo_run",        # 本次实验名称
    exist_ok=True,             # 允许覆盖同名目录
    verbose=True               # 打印详细日志
)