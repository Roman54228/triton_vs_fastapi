"""FastAPI inference client — demonstrates both HTTP and gRPC protocols."""

import argparse
import sys
from pathlib import Path

import requests


def predict_http(image_path: str, url: str = "http://localhost:8000") -> None:
    """Send image via HTTP multipart upload."""
    print(f"\n=== HTTP Predict ({url}/predict) ===")

    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{url}/predict",
            files={"file": (Path(image_path).name, f, "image/jpeg")},
        )

    if resp.status_code != 200:
        print(f"Error: {resp.status_code} — {resp.text}")
        return

    data = resp.json()
    detections = data.get("detections", [])
    print(f"Found {len(detections)} detection(s)\n")

    for i, det in enumerate(detections):
        box = det["box"]
        print(f"  [{i}] box=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})  "
              f"conf={det['confidence']:.3f}  yolo_class={det['yolo_class']}  "
              f"label={det['label']}")


def predict_grpc(image_path: str, url: str = "localhost:50051") -> None:
    """Send image via gRPC."""
    print(f"\n=== gRPC Predict ({url}) ===")

    try:
        import grpc
        # Import compiled protobuf stubs
        sys.path.insert(0, str(Path(__file__).parent / "app" / "proto"))
        import inference_pb2
        import inference_pb2_grpc
    except ImportError as e:
        print(f"gRPC dependencies not available: {e}")
        print("Run: pip install grpcio grpcio-tools && python -m grpc_tools.protoc ...")
        return

    image_data = Path(image_path).read_bytes()
    channel = grpc.insecure_channel(url)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    response = stub.Predict(inference_pb2.PredictRequest(image_data=image_data))

    print(f"Found {len(response.detections)} detection(s)\n")
    for i, det in enumerate(response.detections):
        print(f"  [{i}] box=({det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f})  "
              f"conf={det.confidence:.3f}  yolo_class={det.yolo_class}  "
              f"label={det.label}")

    channel.close()


def check_health(url: str = "http://localhost:8000") -> None:
    """Check server health."""
    print(f"\n=== Health Check ({url}/health) ===")
    resp = requests.get(f"{url}/health")
    print(f"Status: {resp.status_code}")
    print(f"Body:   {resp.json()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI inference client")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--protocol", choices=["http", "grpc", "both"], default="both")
    parser.add_argument("--http-url", default="http://localhost:8000")
    parser.add_argument("--grpc-url", default="localhost:50051")
    parser.add_argument("--health", action="store_true", help="Check health first")
    args = parser.parse_args()

    if args.health:
        check_health(args.http_url)

    if args.protocol in ("http", "both"):
        predict_http(args.image, args.http_url)

    if args.protocol in ("grpc", "both"):
        predict_grpc(args.image, args.grpc_url)


if __name__ == "__main__":
    main()
