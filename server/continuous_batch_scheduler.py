"""
miniServe — Continuous Batch Scheduler
Iteration-level scheduling as described in the Orca paper.
"""

import asyncio
import time
import logging
import json
from typing import Optional, Dict, AsyncGenerator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from server.inference_engine import InferenceEngine
from server.sequence import Sequence
from server import metrics as m
from server.batch_scheduler import InferenceResponse # Re-use response schema

logger = logging.getLogger(__name__)

class QueueFullError(Exception):
    """Raised when the waiting queue is full."""
    pass

class ContinuousBatchScheduler:
    """
    Iteration-level scheduler.
    Runs one forward pass per step across all currently running sequences.
    """
    def __init__(
        self,
        engine: InferenceEngine,
        max_running: Optional[int] = None,
        max_waiting: Optional[int] = None,
        max_timeout: Optional[float] = None,
    ):
        self.engine = engine
        self.max_running = max_running if max_running is not None else config.MAX_RUNNING_SEQUENCES
        self.max_waiting = max_waiting if max_waiting is not None else config.MAX_WAITING_QUEUE_SIZE
        self.max_timeout = max_timeout if max_timeout is not None else config.MAX_SEQUENCE_TIMEOUT_S
        
        self.waiting_queue: asyncio.Queue[Sequence] = asyncio.Queue(maxsize=self.max_waiting)
        self.running_sequences: Dict[str, Sequence] = {}
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._new_request_event = asyncio.Event()

        # Stats
        self._total_requests = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._step_loop())
        logger.info(f"ContinuousBatchScheduler started (max_running={self.max_running})")

    async def stop(self) -> None:
        self._running = False
        self._new_request_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ContinuousBatchScheduler stopped")

    async def submit(self, request, stream=False):
        """
        Submit a request to the continuous batcher.
        Raises QueueFullError if the waiting queue is full.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        stream_queue = asyncio.Queue() if stream else None
        
        seq = Sequence(
            request_id=request.request_id or str(time.time()),
            prompt=request.prompt,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            future=future,
            stream_queue=stream_queue,
        )

        try:
            self.waiting_queue.put_nowait(seq)
        except asyncio.QueueFull:
            raise QueueFullError("The server is currently overloaded. Waiting queue is full.")

        self._new_request_event.set()
        
        if stream:
            return self._stream_generator(seq)
        
        # Await completion
        return await future

    async def _stream_generator(self, seq: Sequence) -> AsyncGenerator[str, None]:
        """Generator that yields Server-Sent Events (SSE) for streaming responses."""
        try:
            while True:
                token = await seq.stream_queue.get()
                if token is None:
                    if seq.error:
                        yield f"data: {json.dumps({'error': str(seq.error)})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                yield f"data: {json.dumps({'text': token})}\n\n"
        except asyncio.CancelledError:
            # If the client disconnects, we should ideally evict the sequence
            logger.info(f"Client disconnected during stream for request {seq.request_id}")
            seq.error = Exception("Client disconnected")
            raise

    async def _step_loop(self) -> None:
        """Background loop that runs continuous batching steps."""
        while self._running:
            try:
                # 1. Admit new sequences from waiting queue
                while len(self.running_sequences) < self.max_running:
                    if self.waiting_queue.empty():
                        break
                    seq = self.waiting_queue.get_nowait()
                    seq.start_time = time.time()
                    m.QUEUE_WAIT_TIME.observe(seq.queue_wait_ms / 1000.0)
                    self.running_sequences[seq.request_id] = seq
                    self._total_requests += 1

                # 2. If nothing is running, wait for new requests
                if not self.running_sequences:
                    m.SCHEDULER_RUNNING_SEQUENCES.set(0)
                    m.SCHEDULER_HEARTBEAT.set(time.monotonic())
                    self._new_request_event.clear()
                    # Wait until _new_request_event is set or timeout to emit heartbeat
                    try:
                        await asyncio.wait_for(self._new_request_event.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                
                m.SCHEDULER_RUNNING_SEQUENCES.set(len(self.running_sequences))
                m.SCHEDULER_HEARTBEAT.set(time.monotonic())

                # 3. Check for timeouts
                now = time.time()
                to_evict = []
                for req_id, seq in self.running_sequences.items():
                    if seq.running_time_s > self.max_timeout:
                        logger.warning(f"Sequence {req_id} timed out after {seq.running_time_s:.1f}s")
                        seq.error = TimeoutError(f"Sequence generation timed out after {self.max_timeout}s")
                        to_evict.append(req_id)
                
                for req_id in to_evict:
                    self._evict_sequence(self.running_sequences[req_id])

                if not self.running_sequences:
                    continue

                # 4. Run a single step
                sequences_to_step = list(self.running_sequences.values())
                
                step_start = time.time()
                
                # Run in executor to avoid blocking event loop
                loop = asyncio.get_running_loop()
                with m.STEP_LATENCY.time():
                    await loop.run_in_executor(
                        None,
                        lambda: self.engine._generate_batch_step(sequences_to_step)
                    )
                    
                # Push newly generated tokens to stream queues
                for seq in sequences_to_step:
                    if seq.stream_queue is not None and seq.latest_token_text:
                        seq.stream_queue.put_nowait(seq.latest_token_text)
                        seq.latest_token_text = ""

                # 5. Evict finished sequences
                finished = [s for s in sequences_to_step if s.is_finished or s.error]
                for seq in finished:
                    self._evict_sequence(seq)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous batching step: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    def _evict_sequence(self, seq: Sequence):
        """Clean up a sequence and resolve its future."""
        if seq.request_id in self.running_sequences:
            del self.running_sequences[seq.request_id]
        
        # Move past_key_values to CPU to free GPU memory
        seq.move_cache_to_cpu()
        
        if seq.stream_queue is not None:
            seq.stream_queue.put_nowait(None)
        
        if not seq.future.done():
            if seq.error:
                seq.future.set_exception(seq.error)
            else:
                response = InferenceResponse(
                    generated_text=seq.generated_text,
                    tokens_generated=seq.tokens_generated,
                    latency_ms=seq.running_time_s * 1000.0,
                    batch_size=1, # conceptually 1 for the client
                    queue_wait_ms=seq.queue_wait_ms,
                    from_cache=False,
                    request_id=seq.request_id,
                )
                seq.future.set_result(response)

    @property
    def queue_depth(self) -> int:
        return self.waiting_queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "running_sequences": len(self.running_sequences),
            "queue_depth": self.queue_depth,
            "max_running": self.max_running,
            "max_waiting": self.max_waiting,
            "total_requests": self._total_requests,
        }
