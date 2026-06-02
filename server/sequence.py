"""
miniServe — Sequence Representation
Dataclass representing a single generation sequence in the continuous batching scheduler.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

@dataclass
class Sequence:
    """
    A single sequence being processed by the continuous batcher.
    
    Stores the prompt, intermediate generation state (input_ids, past_key_values),
    and metrics (arrival_time, start_time).
    """
    request_id: str
    prompt: str
    max_tokens: int = config.MAX_NEW_TOKENS
    temperature: float = config.TEMPERATURE
    
    # Generation state
    input_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    past_key_values: Optional[Any] = None
    tokens_generated: int = 0
    generated_text: str = ""
    is_finished: bool = False
    error: Optional[Exception] = None
    
    # Asynchronous future for the client to await
    future: Optional[asyncio.Future] = field(default=None, repr=False)
    
    # Timing metrics
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None

    @property
    def queue_wait_ms(self) -> float:
        """Time spent waiting in queue before generation started."""
        if self.start_time is None:
            return (time.time() - self.arrival_time) * 1000.0
        return (self.start_time - self.arrival_time) * 1000.0
        
    @property
    def running_time_s(self) -> float:
        """Time spent running (since generation started)."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def total_time_ms(self) -> float:
        """Total time since request arrival."""
        return (time.time() - self.arrival_time) * 1000.0

    def move_cache_to_cpu(self):
        """Moves past_key_values to CPU to free GPU memory upon eviction."""
        if self.past_key_values is not None and isinstance(self.past_key_values, tuple):
            # past_key_values is typically a tuple of tuples of tensors
            cpu_past_key_values = []
            for layer in self.past_key_values:
                if isinstance(layer, tuple):
                    cpu_layer = tuple(tensor.cpu() for tensor in layer if hasattr(tensor, 'cpu'))
                    cpu_past_key_values.append(cpu_layer)
            if cpu_past_key_values:
                self.past_key_values = tuple(cpu_past_key_values)
