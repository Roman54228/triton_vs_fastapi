"""Triton BLS pipeline: YOLOv8 detection using ultralytics on CUDA."""

import numpy as np
from ultralytics import YOLO
import triton_python_backend_utils as pb_utils


def _log(msg: str) -> None:
    with open("/tmp/pipeline.log", "a") as f:
        f.write(msg + "\n")


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        _log("Loading YOLOv8n with CUDA...")
        self._model = YOLO("yolov8n.pt")
        self._model.to("cuda")
        _log("YOLOv8n loaded on CUDA")

    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            try:
                # Read image from file (same as before)
                image_source = "/models/test.jpg"
                _log(f"Running inference on: {image_source}")

                results = self._model(image_source, verbose=False)

                # Extract detections: [x1, y1, x2, y2, confidence, class_id]
                detections = np.zeros((0, 6), dtype=np.float32)
                if results and len(results[0].boxes):
                    boxes = results[0].boxes
                    detections = np.concatenate([
                        boxes.xyxy.cpu().numpy(),
                        boxes.conf.cpu().numpy().reshape(-1, 1),
                        boxes.cls.cpu().numpy().reshape(-1, 1),
                    ], axis=1).astype(np.float32)

                _log(f"Detections: {len(detections)}")

                if len(detections) == 0:
                    detections = np.zeros((1, 6), dtype=np.float32)

                det1_tensor = pb_utils.Tensor("DETECTIONS_YOLO1", detections)
                det2_tensor = pb_utils.Tensor("DETECTIONS_YOLO2", detections)

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[det1_tensor, det2_tensor]
                ))
            except Exception as e:
                import traceback
                _log(f"Pipeline error: {e}\n{traceback.format_exc()}")
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self) -> None:
        _log("Pipeline finalized")
