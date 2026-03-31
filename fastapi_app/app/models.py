"""Model manager using ultralytics YOLO (no ONNX)."""

import logging
import threading

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages a YOLO model on GPU."""

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._lock = threading.RLock()

    def load_model(self) -> None:
        """Load YOLOv8n onto GPU."""
        logger.info("Loading YOLOv8n with CUDA...")
        self._model = YOLO("yolov8n.pt")
        self._model.to("cuda")
        logger.info("YOLOv8n loaded on CUDA")

    def get_model(self) -> YOLO:
        """Get the YOLO model. Thread-safe."""
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            return self._model

    def warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        logger.info("Model warmed up")

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._model is not None
