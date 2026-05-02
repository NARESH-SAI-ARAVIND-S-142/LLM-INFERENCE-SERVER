"""
miniServe — gRPC Server
Provides gRPC interface to the same inference engine and batch scheduler.

This runs alongside the REST server, sharing the same InferenceEngine
and BatchScheduler instances. gRPC is faster than REST for internal
microservice communication due to:
1. Binary serialization (protobuf) instead of JSON
2. HTTP/2 multiplexing
3. Bidirectional streaming support
"""

import asyncio
import time
import uuid
import logging
from concurrent import futures
from typing import Optional

import grpc
from grpc import aio as grpc_aio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add proto directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import config
from server.inference_engine import InferenceEngine
from server.batch_scheduler import BatchScheduler, InferenceRequest
from server import metrics as m

import inference_pb2
import inference_pb2_grpc

logger = logging.getLogger("miniServe.gRPC")


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    gRPC service implementation.
    
    Each RPC method receives a protobuf request, submits it to
    the shared batch scheduler, and returns a protobuf response.
    """

    def __init__(self, engine: InferenceEngine, scheduler: BatchScheduler):
        self.engine = engine
        self.scheduler = scheduler

    async def Generate(self, request, context):
        """Handle a Generate RPC call."""
        request_start = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        max_tokens = request.max_tokens if request.max_tokens > 0 else config.MAX_NEW_TOKENS
        temperature = request.temperature if request.temperature > 0 else config.TEMPERATURE

        m.record_request("grpc")

        try:
            inference_request = InferenceRequest(
                prompt=request.prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=request_id,
            )
            response = await self.scheduler.submit(inference_request)

            # Record metrics
            total_latency = time.time() - request_start
            m.record_inference(
                latency_seconds=response.latency_ms / 1000,
                batch_size=response.batch_size,
                tokens_generated=response.tokens_generated,
                queue_wait_seconds=response.queue_wait_ms / 1000,
            )
            m.record_queue_wait(response.queue_wait_ms / 1000)
            m.record_total_latency(total_latency)
            m.update_cache_metrics(self.engine.cache_stats)

            return inference_pb2.GenerateResponse(
                generated_text=response.generated_text,
                prompt=request.prompt,
                tokens_generated=response.tokens_generated,
                latency_ms=response.latency_ms,
                batch_size=response.batch_size,
                queue_wait_ms=response.queue_wait_ms,
                from_cache=response.from_cache,
                request_id=request_id,
            )

        except Exception as e:
            m.record_error("grpc_inference_error")
            logger.error(f"gRPC Generate failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Generation failed: {str(e)}")
            return inference_pb2.GenerateResponse()

    async def HealthCheck(self, request, context):
        """Handle a HealthCheck RPC call."""
        return inference_pb2.HealthCheckResponse(
            status="healthy" if self.engine is not None else "not_ready",
            model=config.MODEL_NAME,
            device=config.DEVICE,
            queue_depth=self.scheduler.queue_depth,
        )


async def serve_grpc(engine: InferenceEngine, scheduler: BatchScheduler):
    """
    Start the async gRPC server.
    
    Args:
        engine: Shared InferenceEngine instance.
        scheduler: Shared BatchScheduler instance.
    """
    server = grpc_aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    servicer = InferenceServicer(engine, scheduler)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{config.GRPC_HOST}:{config.GRPC_PORT}"
    server.add_insecure_port(listen_addr)
    
    logger.info(f"gRPC server starting on {listen_addr}")
    await server.start()
    logger.info(f"gRPC server is READY on {listen_addr}")
    
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        logger.info("gRPC server shutting down...")
        await server.stop(grace=5)


async def main():
    """
    Standalone gRPC server entry point.
    
    For standalone use:
        python -m server.grpc_server
        
    In production, the gRPC server runs alongside the REST server,
    sharing the same engine and scheduler.
    """
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting standalone gRPC server...")
    config.print_config()

    engine = InferenceEngine()
    m.MODEL_LOADED.set(1)

    scheduler = BatchScheduler(engine)
    await scheduler.start()

    await serve_grpc(engine, scheduler)


if __name__ == "__main__":
    asyncio.run(main())
