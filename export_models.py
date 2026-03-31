"""Export YOLOv8n and MobileNetV3-Small to ONNX for both Triton and FastAPI."""

import shutil
from pathlib import Path

import onnx
import torch
import timm
from ultralytics import YOLO


ROOT = Path(__file__).parent

TRITON_YOLO = ROOT / "triton" / "model_repository" / "yolo_detection" / "1" / "model.onnx"
TRITON_CLS = ROOT / "triton" / "model_repository" / "classification" / "1" / "model.onnx"
FASTAPI_YOLO = ROOT / "fastapi_app" / "models" / "yolo_detection" / "model.onnx"
FASTAPI_CLS = ROOT / "fastapi_app" / "models" / "classification" / "model.onnx"


def export_yolo() -> Path:
    """Export YOLOv8n to ONNX with dynamic batch axis."""
    print("=== Exporting YOLOv8n to ONNX ===")
    model = YOLO("yolov8n.pt")
    export_path = model.export(format="onnx", imgsz=640, simplify=True, dynamic=True)
    export_path = Path(export_path)

    m = onnx.load(str(export_path))
    print(f"  Inputs:  {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
    print(f"  Outputs: {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")
    return export_path


def export_mobilenet() -> Path:
    """Export MobileNetV3-Small to ONNX with dynamic batch axis."""
    print("\n=== Exporting MobileNetV3-Small to ONNX ===")
    model = timm.create_model("mobilenetv3_small_100", pretrained=True)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    export_path = ROOT / "mobilenetv3_small.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(export_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    m = onnx.load(str(export_path))
    print(f"  Inputs:  {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
    print(f"  Outputs: {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")
    return export_path


def copy_models(yolo_path: Path, cls_path: Path) -> None:
    """Copy ONNX files to both Triton and FastAPI directories."""
    print("\n=== Copying models ===")
    for dest in [TRITON_YOLO, FASTAPI_YOLO]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yolo_path, dest)
        print(f"  {dest}")

    for dest in [TRITON_CLS, FASTAPI_CLS]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cls_path, dest)
        print(f"  {dest}")


def main() -> None:
    yolo_path = export_yolo()
    cls_path = export_mobilenet()
    copy_models(yolo_path, cls_path)
    print("\nDone! Models exported and copied to triton/ and fastapi_app/.")


if __name__ == "__main__":
    main()
