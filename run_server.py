"""
miniServe — Combined Server Runner
Starts both REST (FastAPI) and gRPC servers simultaneously.

Usage:
    python run_server.py

This script:
1. Loads the model once (shared InferenceEngine)
2. Starts the BatchScheduler (shared between REST and gRPC)
3. Launches REST API on port 8000
4. Launches gRPC server on port 50051
5. Both servers share the same engine and scheduler
"""

import asyncio
import logging
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from server.inference_engine import InferenceEngine
from server.batch_scheduler import BatchScheduler
from server import metrics as m

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miniServe")


async def run_rest_server(engine: InferenceEngine, scheduler: BatchScheduler):
    """Run the FastAPI REST server using uvicorn."""
    import uvicorn
    from server.main import app

    # Inject shared engine and scheduler into the app state
    # We override the lifespan-created ones
    import server.main as main_module
    main_module.engine = engine
    main_module.scheduler = scheduler

    uvi_config = uvicorn.Config(
        app=app,
        host=config.REST_HOST,
        port=config.REST_PORT,
        log_level=config.LOG_LEVEL.lower(),
        lifespan="off",  # We manage lifecycle ourselves
    )
    server = uvicorn.Server(uvi_config)
    await server.serve()


async def run_grpc_server(engine: InferenceEngine, scheduler: BatchScheduler):
    """Run the gRPC server."""
    from server.grpc_server import serve_grpc
    await serve_grpc(engine, scheduler)


async def main():
    """Start all server components."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              miniServe — LLM Inference Server           ║")
    print("║         Dynamic Batching | KV-Cache | REST + gRPC       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    config.print_config()

    # ─── Step 1: Load model ───────────────────────────────────────────
    logger.info("Loading model...")
    engine = InferenceEngine()
    m.MODEL_LOADED.set(1)
    logger.info("Model loaded successfully!")

    # ─── Step 2: Start batch scheduler ────────────────────────────────
    scheduler = BatchScheduler(engine)
    await scheduler.start()
    logger.info("Batch scheduler started!")

    # ─── Step 3: Launch servers ───────────────────────────────────────
    logger.info(f"Starting REST API on http://{config.REST_HOST}:{config.REST_PORT}")
    logger.info(f"Starting gRPC server on {config.GRPC_HOST}:{config.GRPC_PORT}")
    logger.info(f"API Docs: http://{config.REST_HOST}:{config.REST_PORT}/docs")
    logger.info(f"Metrics:  http://{config.REST_HOST}:{config.REST_PORT}/metrics")
    print()
    print("  🚀 miniServe is READY!")
    print(f"  📡 REST:    http://localhost:{config.REST_PORT}/v1/generate")
    print(f"  📡 gRPC:    localhost:{config.GRPC_PORT}")
    print(f"  📖 Docs:    http://localhost:{config.REST_PORT}/docs")
    print(f"  📊 Metrics: http://localhost:{config.REST_PORT}/metrics")
    print()

    # Run both servers concurrently
    try:
        await asyncio.gather(
            run_rest_server(engine, scheduler),
            run_grpc_server(engine, scheduler),
        )
    except asyncio.CancelledError:
        logger.info("Servers cancelled")
    finally:
        await scheduler.stop()
        m.MODEL_LOADED.set(0)
        logger.info("miniServe shut down gracefully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  👋 Shutting down miniServe...")
