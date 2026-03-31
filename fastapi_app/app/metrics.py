"""Manual Prometheus metrics and middleware.

~60 lines that replicate what Triton exposes for free on :8002/metrics
(40+ metrics including per-model latency, queue time, batch sizes, etc.).
"""

import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Request metrics
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Model metrics
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Model inference latency",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Dynamic batch sizes",
    ["model_name"],
    buckets=[1, 2, 4, 8, 16, 32],
)

QUEUE_SIZE = Gauge(
    "inference_queue_size",
    "Current items waiting in batch queue",
    ["model_name"],
)

MODELS_LOADED = Gauge(
    "models_loaded_total",
    "Number of models currently loaded",
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request count and latency for all HTTP endpoints."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start

        method = request.method
        path = request.url.path

        REQUEST_COUNT.labels(method=method, endpoint=path, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)

        return response


def get_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metrics output."""
    return generate_latest(), CONTENT_TYPE_LATEST
