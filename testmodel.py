from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2f_DCNv3_GSGAM, Detect
from ultralytics.nn.tasks import DetectionModel


def main():
    model = DetectionModel("yolov8n_basketball.yaml", ch=3, nc=2, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, gda=0.2)
    criterion = model.init_criterion()

    custom_blocks = [m for m in model.model.modules() if isinstance(m, C2f_DCNv3_GSGAM)]
    assert custom_blocks, "未找到 C2f_DCNv3_GSGAM，说明自定义结构没有加载成功。"

    detect_head = model.model[-1]
    assert isinstance(detect_head, Detect), "模型最后一层不是 Detect。"
    assert detect_head.nl == 4, f"期望 4 个检测层(P2/P3/P4/P5)，实际为 {detect_head.nl}。"

    expected_stride = [4.0, 8.0, 16.0, 32.0]
    actual_stride = model.stride.tolist()
    assert actual_stride == expected_stride, f"stride 不匹配，期望 {expected_stride}，实际 {actual_stride}。"

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        preds = model(x)

    pred_tensor = None
    raw_outputs = None
    if isinstance(preds, tuple):
        pred_tensor, raw_outputs = preds
    elif isinstance(preds, list):
        raw_outputs = preds
    else:
        pred_tensor = preds

    print("✅ 自定义 YOLO 结构加载成功")
    print(f"C2f_DCNv3_GSGAM 数量: {len(custom_blocks)}")
    print(f"Detect 输入层索引: {detect_head.f}")
    print(f"Detect 检测层数 nl: {detect_head.nl}")
    print(f"模型 stride: {actual_stride}")
    print(f"是否挂载 GDA loss: {hasattr(criterion, 'gda_loss')}")
    print(f"GDA loss 权重: {getattr(criterion, 'gda_gain', None)}")
    if pred_tensor is not None:
        print(f"前向输出张量形状: {tuple(pred_tensor.shape)}")
    if raw_outputs is not None:
        print("各检测层输出形状:")
        for i, feat in enumerate(raw_outputs):
            print(f"  P{i + 2}: {tuple(feat.shape)}")


if __name__ == "__main__":
    main()
