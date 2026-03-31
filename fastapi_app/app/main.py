"""FastAPI application: HTTP endpoints, lifecycle, model serving."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .metrics import MODELS_LOADED, MetricsMiddleware, get_metrics
from .models import ModelManager
from .pipeline import InferencePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + warmup. Shutdown: cleanup."""
    logger.info("Starting up — loading YOLO model")

    mm = ModelManager()
    mm.load_model()
    mm.warmup()
    MODELS_LOADED.set(1)

    app.state.model_manager = mm
    app.state.pipeline = InferencePipeline(mm)

    # Start gRPC server in background
    from .grpc_server import serve_grpc
    grpc_task = asyncio.create_task(serve_grpc(app.state.pipeline))
    logger.info("gRPC server starting on port 50051")

    yield

    # Shutdown
    grpc_task.cancel()
    try:
        await grpc_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete")


app = FastAPI(title="FastAPI Inference Server", lifespan=lifespan)
app.add_middleware(MetricsMiddleware)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Run YOLO detection pipeline."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, app.state.pipeline.predict, image_bytes
    )
    return JSONResponse(content=result)


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    mm: ModelManager = app.state.model_manager
    if mm.is_ready:
        return JSONResponse(content={"status": "healthy"})
    raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    body, content_type = get_metrics()
    return Response(content=body, media_type=content_type)
