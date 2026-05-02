"""
miniServe — Dynamic Batch Scheduler
Groups incoming requests into batches for efficient inference.

HOW DYNAMIC BATCHING WORKS:
1. Requests arrive one at a time via submit()
2. Each request gets a Future — the client awaits this Future
3. A background loop collects requests into batches
4. A batch is fired when EITHER:
   a) max_batch_size is reached, OR
   b) max_wait_time has elapsed since the first request in the batch
5. The batch is sent to the InferenceEngine
6. Results are dispatched to individual Futures, waking up clients

WHY THIS IS IMPORTANT:
- Single-request inference wastes compute (GPU/CPU BLAS kernels are
  optimized for larger matrices)
- Batching amortizes the fixed overhead of model forward pass
- But waiting too long hurts latency — hence the max_wait_time cap
- This is the exact tradeoff Google faces with Gemini serving
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from server.inference_engine import InferenceEngine, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request waiting to be batched."""
    prompt: str
    max_tokens: int = config.MAX_NEW_TOKENS
    temperature: float = config.TEMPERATURE
    request_id: Optional[str] = None
    future: asyncio.Future = field(default=None, repr=False)
    enqueue_time: float = field(default_factory=time.time)

    @property
    def queue_wait_ms(self) -> float:
        """Time spent waiting in queue."""
        return (time.time() - self.enqueue_time) * 1000


@dataclass
class InferenceResponse:
    """Response returned to the client after batched inference."""
    generated_text: str
    tokens_generated: int
    latency_ms: float
    batch_size: int
    queue_wait_ms: float
    from_cache: bool
    request_id: Optional[str] = None


class BatchScheduler:
    """
    Asynchronous dynamic batch scheduler.
    
    Collects incoming inference requests and groups them into
    batches based on max_batch_size and max_wait_time thresholds.
    
    Usage:
        scheduler = BatchScheduler(engine)
        await scheduler.start()  # Start background batch loop
        
        # From API handler:
        response = await scheduler.submit(InferenceRequest(prompt="Hello"))
        print(response.generated_text)
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = None,
        max_wait_time_ms: float = None,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size or config.MAX_BATCH_SIZE
        self.max_wait_time = (max_wait_time_ms or config.MAX_WAIT_TIME_MS) / 1000.0

        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Metrics
        self._total_batches: int = 0
        self._total_requests: int = 0

        logger.info(
            f"BatchScheduler initialized: "
            f"max_batch_size={self.max_batch_size}, "
            f"max_wait_time={self.max_wait_time * 1000:.0f}ms"
        )

    async def start(self) -> None:
        """Start the background batch processing loop."""
        if self._running:
            logger.warning("BatchScheduler is already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())
        logger.info("BatchScheduler started — batch loop running")

    async def stop(self) -> None:
        """Stop the batch processing loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("BatchScheduler stopped")

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """
        Submit a request for batched inference.
        
        This method:
        1. Creates a Future for this request
        2. Puts the request into the queue
        3. Awaits the Future (client is suspended here)
        4. Returns the result when the batch completes
        
        Args:
            request: The inference request.
            
        Returns:
            InferenceResponse with generated text and metrics.
        """
        # Create a Future that will hold the result
        loop = asyncio.get_event_loop()
        request.future = loop.create_future()
        request.enqueue_time = time.time()

        # Put request into the queue
        await self._queue.put(request)

        logger.debug(
            f"Request submitted: '{request.prompt[:50]}...' "
            f"(queue_depth={self._queue.qsize()})"
        )

        # Await the future — this suspends the client coroutine
        # until the batch scheduler processes this request
        response = await request.future
        return response

    async def _batch_loop(self) -> None:
        """
        Background loop that collects and processes batches.
        
        Algorithm:
        1. Block until at least one request arrives
        2. Start a timer (max_wait_time)
        3. Collect more requests until:
           - max_batch_size is reached, OR
           - max_wait_time expires
        4. Run inference on the batch
        5. Dispatch results to individual clients
        6. Repeat
        """
        logger.info("Batch loop started — waiting for requests...")

        while self._running:
            try:
                batch: list[InferenceRequest] = []

                # ─── Step 1: Wait for first request (blocking) ───────────
                try:
                    first_request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,  # Check _running flag every 1s
                    )
                    batch.append(first_request)
                except asyncio.TimeoutError:
                    continue  # No requests — loop back and check _running

                # ─── Step 2: Collect more requests until deadline ─────────
                deadline = time.time() + self.max_wait_time

                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break  # Timer expired — fire the batch

                    try:
                        request = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=remaining,
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break  # Timer expired — fire the batch

                # ─── Step 3: Run batched inference ────────────────────────
                batch_size = len(batch)
                prompts = [r.prompt for r in batch]
                request_ids = [r.request_id for r in batch]
                max_tokens = batch[0].max_tokens  # Use first request's setting
                temperature = batch[0].temperature

                self._total_batches += 1
                self._total_requests += batch_size

                logger.info(
                    f"Firing batch #{self._total_batches}: "
                    f"{batch_size} requests"
                )

                try:
                    # Run inference in a thread pool to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: self.engine.generate_batch(
                            prompts=prompts,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            request_ids=request_ids if any(request_ids) else None,
                        ),
                    )

                    # ─── Step 4: Dispatch results to clients ──────────────
                    for request, result in zip(batch, results):
                        response = InferenceResponse(
                            generated_text=result.generated_text,
                            tokens_generated=result.tokens_generated,
                            latency_ms=result.inference_time_ms,
                            batch_size=batch_size,
                            queue_wait_ms=round(request.queue_wait_ms, 2),
                            from_cache=result.from_cache,
                            request_id=result.request_id,
                        )
                        # Set the result on the Future — wakes up the client
                        if not request.future.done():
                            request.future.set_result(response)

                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    # Propagate error to all waiting clients
                    for request in batch:
                        if not request.future.done():
                            request.future.set_exception(e)

            except asyncio.CancelledError:
                logger.info("Batch loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in batch loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Avoid tight error loop

    @property
    def queue_depth(self) -> int:
        """Current number of requests waiting in queue."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        """Batch scheduler statistics."""
        avg_batch = (
            self._total_requests / self._total_batches
            if self._total_batches > 0
            else 0
        )
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": round(avg_batch, 2),
            "queue_depth": self.queue_depth,
            "max_batch_size": self.max_batch_size,
            "max_wait_time_ms": self.max_wait_time * 1000,
        }
