"""Triton Inference Server client — sends a dummy request to the pipeline via HTTP."""

import argparse
import base64
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
            {"name": "DETECTIONS"},
            {"name": "LABELS"},
        ],
    }

    resp = requests.post(f"{url}/v2/models/pipeline/infer", json=payload)
    resp.raise_for_status()
    result = resp.json()

    outputs = {o["name"]: o for o in result["outputs"]}

    det_out = outputs["DETECTIONS"]
    n = det_out["shape"][0]
    detections = np.array(det_out["data"], dtype=np.float32).reshape(n, 6)

    lbl_out = outputs["LABELS"]
    labels = [base64.b64decode(s).decode() for s in lbl_out["data"]]

    print(f"\n=== Results ===")
    print(f"Found {len(detections)} detection(s)\n")
    for i, (det, label) in enumerate(zip(detections, labels)):
        x1, y1, x2, y2, conf, cls_id = det
        print(f"  [{i}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})  "
              f"conf={conf:.3f}  yolo_class={int(cls_id)}  label={label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton HTTP client")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    predict(args.url)


if __name__ == "__main__":
    main()
