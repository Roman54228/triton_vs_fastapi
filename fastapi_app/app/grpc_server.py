"""gRPC inference server — ~200 lines of manual implementation.

Triton provides gRPC support by default on port 8001.
This file shows the effort needed to add gRPC to a FastAPI app:
  1. Define .proto service (inference.proto)
  2. Compile protobuf (grpc_tools.protoc)
  3. Implement servicer class
  4. Server lifecycle management
  5. Serialization/deserialization
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import grpc
from grpc import aio as grpc_aio

from .pipeline import InferencePipeline

logger = logging.getLogger(__name__)

PROTO_DIR = Path(__file__).parent / "proto"
_compiled = False


def _compile_proto() -> None:
    """Compile inference.proto → Python gRPC stubs."""
    global _compiled
    if _compiled:
        return

    proto_file = PROTO_DIR / "inference.proto"
    if not proto_file.exists():
        raise FileNotFoundError(f"Proto file not found: {proto_file}")

    # Check if stubs already exist
    pb2_file = PROTO_DIR / "inference_pb2.py"
    grpc_file = PROTO_DIR / "inference_pb2_grpc.py"
    if pb2_file.exists() and grpc_file.exists():
        _compiled = True
        return

    logger.info("Compiling inference.proto...")
    from grpc_tools import protoc
    result = protoc.main([
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={PROTO_DIR}",
        f"--grpc_python_out={PROTO_DIR}",
        str(proto_file),
    ])
    if result != 0:
        raise RuntimeError(f"protoc failed with code {result}")

    # Fix import path in generated grpc file
    grpc_content = grpc_file.read_text()
    grpc_content = grpc_content.replace(
        "import inference_pb2",
        "from . import inference_pb2",
    )
    grpc_file.write_text(grpc_content)

    # Create __init__.py for proto package
    init_file = PROTO_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    _compiled = True
    logger.info("Proto compilation complete")


def _get_stubs():
    """Import compiled protobuf stubs."""
    _compile_proto()
    from .proto import inference_pb2, inference_pb2_grpc
    return inference_pb2, inference_pb2_grpc


class InferenceServiceServicer:
    """gRPC servicer wrapping the inference pipeline."""

    def __init__(self, pipeline: InferencePipeline) -> None:
        self._pipeline = pipeline

    async def Predict(self, request, context):
        """Handle a gRPC prediction request."""
        pb2, _ = _get_stubs()

        if not request.image_data:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty image data")
            return pb2.PredictResponse()

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._pipeline.predict, request.image_data
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.PredictResponse()

        response = pb2.PredictResponse()
        for det in result.get("detections", []):
            detection = pb2.Detection(
                x1=det["box"][0],
                y1=det["box"][1],
                x2=det["box"][2],
                y2=det["box"][3],
                confidence=det["confidence"],
                yolo_class=det["yolo_class"],
                label=det["label"],
            )
            response.detections.append(detection)

        return response

    async def Health(self, request, context):
        """Handle a gRPC health check request."""
        pb2, _ = _get_stubs()

        mm = self._pipeline._mm
        status = "healthy" if mm.is_ready else "unhealthy"

        return pb2.HealthResponse(status=status, models={})


async def serve_grpc(pipeline: InferencePipeline, port: int = 50051) -> None:
    """Start the async gRPC server."""
    _, pb2_grpc = _get_stubs()

    server = grpc_aio.server()
    servicer = InferenceServiceServicer(pipeline)
    pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    await server.start()
    logger.info("gRPC server listening on port %d", port)

    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        await server.stop(grace=5)
        logger.info("gRPC server stopped")
