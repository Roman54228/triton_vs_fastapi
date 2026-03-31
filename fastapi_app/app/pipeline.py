"""Detection pipeline using ultralytics YOLO (no ONNX)."""

import logging

import cv2
import numpy as np

from .models import ModelManager

logger = logging.getLogger(__name__)


class InferencePipeline:
    """YOLO detection pipeline using ultralytics."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def predict(self, image_bytes: bytes) -> dict:
        """Run YOLO detection on raw image bytes."""
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")

        model = self._mm.get_model()
        results = model(image, verbose=False)

        detections = []
        if results and len(results[0].boxes):
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "yolo_class": cls_id,
                })

        return {"detections": detections}
