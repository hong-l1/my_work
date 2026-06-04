from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

# Fixed training config
MODEL_PATH = ROOT / "yolov8n_basketball.yaml"
DATA_PATH = ROOT / "pappu-1" / "data.yaml"
PRETRAINED_PATH = ""  # e.g. str(ROOT / "pappu-1" / "yolo11n.pt")

EPOCHS = 100
IMGSZ = 640
BATCH = 16
DEVICE = "0"  # "0" / "0,1" / "cpu"
WORKERS = 8
PROJECT = "runs/train"
NAME = "basketball_gsgam"
OPTIMIZER = "auto"
LR0 = 0.01
LRF = 0.01
WEIGHT_DECAY = 0.0005
BOX_GAIN = 7.5
CLS_GAIN = 0.5
DFL_GAIN = 1.5
GDA_GAIN = 0.2
PATIENCE = 100
CACHE = False
COS_LR = False
CLOSE_MOSAIC = 10
SINGLE_CLS = False
RESUME = False
AMP = True
VERBOSE = True


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")

    model = YOLO(str(MODEL_PATH))

    train_kwargs = {
        "data": str(DATA_PATH),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "project": PROJECT,
        "name": NAME,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "weight_decay": WEIGHT_DECAY,
        "box": BOX_GAIN,
        "cls": CLS_GAIN,
        "dfl": DFL_GAIN,
        "gda": GDA_GAIN,
        "patience": PATIENCE,
        "cache": CACHE,
        "cos_lr": COS_LR,
        "close_mosaic": CLOSE_MOSAIC,
        "single_cls": SINGLE_CLS,
        "resume": RESUME,
        "amp": AMP,
        "verbose": VERBOSE,
    }

    if PRETRAINED_PATH:
        train_kwargs["pretrained"] = PRETRAINED_PATH

    print("Starting training with fixed config:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")

    results = model.train(**train_kwargs)

    print("\nTraining finished.")
    print(f"Results object: {results}")


if __name__ == "__main__":
    main()
