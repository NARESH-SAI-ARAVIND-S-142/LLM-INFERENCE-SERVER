"""
miniServe — FastAPI REST Server
Main entry point for the LLM inference server.

Provides:
- POST /v1/generate — Submit text generation requests (batched automatically)
- GET  /health      — Health check with model/cache/queue status
- GET  /metrics     — Prometheus metrics endpoint
- GET  /stats       — Detailed server statistics

The FastAPI app starts the batch scheduler on startup and accepts
requests that are automatically grouped into batches.
"""

import asyncio
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from server.inference_engine import InferenceEngine
from server.continuous_batch_scheduler import ContinuousBatchScheduler, QueueFullError
from server import metrics as m

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miniServe.REST")


# ─── Global Components ───────────────────────────────────────────────────────

engine: Optional[InferenceEngine] = None
scheduler: Optional[ContinuousBatchScheduler] = None


# ─── Lifespan (Startup + Shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine and scheduler on startup, cleanup on shutdown."""
    global engine, scheduler

    logger.info("=" * 60)
    logger.info("  miniServe — Starting Up")
    logger.info("=" * 60)
    config.print_config()

    # Load model
    engine = InferenceEngine()
    m.MODEL_LOADED.set(1)

    # Start batch scheduler
    scheduler = ContinuousBatchScheduler(engine)
    await scheduler.start()

    logger.info("miniServe is READY — accepting requests")
    logger.info(f"  REST API: http://{config.REST_HOST}:{config.REST_PORT}")
    logger.info(f"  Docs:     http://{config.REST_HOST}:{config.REST_PORT}/docs")
    logger.info("=" * 60)

    yield  # Server is running

    # Shutdown
    logger.info("miniServe shutting down...")
    await scheduler.stop()
    m.MODEL_LOADED.set(0)
    logger.info("Goodbye!")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="miniServe",
    description="Production-grade LLM inference server with dynamic batching and KV-cache",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request body for text generation."""
    prompt: str = Field(..., description="Input text to generate from", min_length=1)
    max_tokens: int = Field(default=50, description="Maximum tokens to generate", ge=1, le=500)
    temperature: float = Field(default=1.0, description="Sampling temperature", ge=0.0, le=2.0)
    request_id: Optional[str] = Field(default=None, description="Request ID for KV-cache reuse")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "prompt": "Once upon a time in a land far away",
                "max_tokens": 50,
                "temperature": 0.8,
            }
        ]
    }}


class GenerateResponse(BaseModel):
    """Response body for text generation."""
    generated_text: str
    prompt: str
    tokens_generated: int
    latency_ms: float
    batch_size: int
    queue_wait_ms: float
    from_cache: bool
    request_id: str


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    model: str
    device: str
    queue_depth: int
    cache_stats: dict
    scheduler_stats: dict


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.
    
    The request is automatically queued and batched with other
    concurrent requests for optimal throughput.
    """
    request_start = time.time()
    request_id = request.request_id or str(uuid.uuid4())

    # Record incoming request
    m.record_request("rest")
    m.update_queue_depth(scheduler.queue_depth)

    try:
        # Submit to continuous batch scheduler
        request.request_id = request_id
        response = await scheduler.submit(request)

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
        m.update_cache_metrics(engine.cache_stats)
        m.update_queue_depth(scheduler.queue_depth)

        return GenerateResponse(
            generated_text=response.generated_text,
            prompt=request.prompt,
            tokens_generated=response.tokens_generated,
            latency_ms=round(response.latency_ms, 2),
            batch_size=response.batch_size,
            queue_wait_ms=round(response.queue_wait_ms, 2),
            from_cache=response.from_cache,
            request_id=request_id,
        )

    except QueueFullError as e:
        m.record_error("queue_full")
        logger.warning(f"Generation rejected: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        m.record_error("inference_error")
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with detailed server status."""
    return HealthResponse(
        status="healthy" if engine is not None else "not_ready",
        model=config.MODEL_NAME,
        device=config.DEVICE,
        queue_depth=scheduler.queue_depth if scheduler else 0,
        cache_stats=engine.cache_stats if engine else {},
        scheduler_stats=scheduler.stats if scheduler else {},
    )


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(
        content=m.get_metrics(),
        media_type=m.get_content_type(),
    )


@app.get("/stats")
async def stats():
    """Detailed server statistics."""
    return JSONResponse({
        "model": config.MODEL_NAME,
        "device": config.DEVICE,
        "config": {
            "max_batch_size": config.MAX_BATCH_SIZE,
            "max_wait_time_ms": config.MAX_WAIT_TIME_MS,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "kv_cache_max_entries": config.KV_CACHE_MAX_ENTRIES,
        },
        "scheduler": scheduler.stats if scheduler else {},
        "cache": engine.cache_stats if engine else {},
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    """Run the REST server."""
    uvicorn.run(
        "server.main:app",
        host=config.REST_HOST,
        port=config.REST_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
