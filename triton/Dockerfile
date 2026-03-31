FROM nvcr.io/nvidia/tritonserver:24.01-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    'numpy<2' \
    Pillow

# Pre-download YOLOv8n weights into the image
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
