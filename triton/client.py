"""Triton Inference Server client — sends a dummy request to the pipeline via HTTP."""

import argparse
import sys

import numpy as np
import requests


def predict(url: str = "http://localhost:8000") -> None:
    resp = requests.get(f"{url}/v2/health/ready")
    if resp.status_code != 200:
        print("Server not ready")
        sys.exit(1)

    payload = {
        "inputs": [
            {
                "name": "DUMMY",
                "shape": [1],
                "datatype": "FP32",
                "data": [0.0],
            }
        ],
        "outputs": [
            {"name": "DETECTIONS_YOLO1"},
            {"name": "DETECTIONS_YOLO2"},
        ],
    }

    resp = requests.post(f"{url}/v2/models/pipeline/infer", json=payload)
    if not resp.ok:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)
    result = resp.json()

    outputs = {o["name"]: o for o in result["outputs"]}

    print(f"\n=== Results ===")
    for name in ("DETECTIONS_YOLO1", "DETECTIONS_YOLO2"):
        det_out = outputs[name]
        shape = det_out["shape"]
        flat = np.array(det_out["data"], dtype=np.float32)
        detections = flat.reshape(shape)
        print(f"\n{name}: {len(detections)} detection(s)")
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            print(f"  [{i}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})  "
                  f"conf={conf:.3f}  class={int(cls_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton HTTP client")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    predict(args.url)


if __name__ == "__main__":
    main()
