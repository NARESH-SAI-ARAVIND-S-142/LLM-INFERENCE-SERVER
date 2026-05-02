"""
miniServe — KV-Cache Manager
LRU-based cache for storing attention key/value states.

WHY KV-CACHE MATTERS:
In autoregressive generation, each new token requires attending to ALL
previous tokens. Without caching, generating token N requires recomputing
attention for tokens 1..N-1. With KV-cache, we store the key/value pairs
from previous steps and only compute attention for the new token.

For a 50-token generation, this avoids recomputing attention 1+2+3+...+49
= 1,225 times. That's a ~50x speedup for long sequences.

This module manages an LRU (Least Recently Used) cache to:
1. Store past_key_values per request_id (for multi-turn conversations)
2. Evict oldest entries when memory is constrained
3. Track cache hit/miss rates as metrics
"""

import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class KVCacheEntry:
    """A single cached KV-state entry."""
    request_id: str
    past_key_values: Any  # Tuple of (key, value) tensors per layer
    prompt_tokens: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def memory_estimate_mb(self) -> float:
        """Rough estimate of memory used by this cache entry."""
        if self.past_key_values is None:
            return 0.0
        # Each layer has (key, value), each is [batch, heads, seq_len, head_dim]
        # For GPT-2 small: 12 layers, 12 heads, 64 head_dim
        # Memory ≈ 2 * num_layers * seq_len * num_heads * head_dim * 4 bytes
        try:
            num_layers = len(self.past_key_values)
            # past_key_values[layer] = (key_tensor, value_tensor)
            key_tensor = self.past_key_values[0][0]
            total_elements = sum(
                k.numel() + v.numel()
                for k, v in self.past_key_values
            )
            return (total_elements * 4) / (1024 * 1024)  # 4 bytes per float32, convert to MB
        except (IndexError, AttributeError):
            return 0.0


class KVCacheManager:
    """
    Thread-safe LRU KV-Cache Manager.
    
    Stores past_key_values from HuggingFace model.generate() so that
    multi-turn conversations can reuse previously computed attention states.
    
    Usage:
        cache = KVCacheManager(max_entries=100)
        
        # Check if we have cached state for this conversation
        entry = cache.get("conversation-123")
        if entry:
            past_kv = entry.past_key_values  # Reuse cached attention
        
        # After generation, store the new state
        cache.put("conversation-123", new_past_key_values, prompt_tokens=15)
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Metrics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        logger.info(f"KV-Cache initialized with max_entries={max_entries}")

    def get(self, request_id: str) -> Optional[KVCacheEntry]:
        """
        Retrieve cached KV-state for a request.
        Returns None on cache miss. Moves entry to end (most recent) on hit.
        """
        with self._lock:
            if request_id in self._cache:
                self._hits += 1
                entry = self._cache[request_id]
                entry.last_accessed = time.time()
                entry.access_count += 1
                # Move to end (most recently used)
                self._cache.move_to_end(request_id)
                logger.debug(
                    f"KV-Cache HIT: {request_id} "
                    f"(accesses={entry.access_count}, age={entry.age_seconds:.1f}s)"
                )
                return entry
            else:
                self._misses += 1
                logger.debug(f"KV-Cache MISS: {request_id}")
                return None

    def put(
        self,
        request_id: str,
        past_key_values: Any,
        prompt_tokens: int,
    ) -> None:
        """
        Store KV-cache entry. Evicts LRU entries if at capacity.
        """
        with self._lock:
            # If updating existing entry, remove old one first
            if request_id in self._cache:
                del self._cache[request_id]

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_entries:
                self._evict_lru()

            # Store new entry
            entry = KVCacheEntry(
                request_id=request_id,
                past_key_values=past_key_values,
                prompt_tokens=prompt_tokens,
            )
            self._cache[request_id] = entry
            logger.debug(
                f"KV-Cache PUT: {request_id} "
                f"(tokens={prompt_tokens}, size={len(self._cache)}/{self.max_entries})"
            )

    def _evict_lru(self) -> None:
        """Remove the least recently used entry (front of OrderedDict)."""
        if self._cache:
            evicted_id, evicted_entry = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(
                f"KV-Cache EVICT: {evicted_id} "
                f"(age={evicted_entry.age_seconds:.1f}s, "
                f"accesses={evicted_entry.access_count})"
            )

    def invalidate(self, request_id: str) -> bool:
        """Remove a specific entry from cache. Returns True if found."""
        with self._lock:
            if request_id in self._cache:
                del self._cache[request_id]
                logger.debug(f"KV-Cache INVALIDATE: {request_id}")
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"KV-Cache CLEARED: removed {count} entries")

    # ─── Metrics ──────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a ratio [0.0, 1.0]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def total_memory_mb(self) -> float:
        """Estimated total memory used by cache in MB."""
        with self._lock:
            return sum(entry.memory_estimate_mb for entry in self._cache.values())

    def stats(self) -> dict:
        """Return cache statistics as a dictionary."""
        return {
            "size": self.size,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self.hit_rate, 4),
            "memory_mb": round(self.total_memory_mb, 2),
        }

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(size={self.size}/{self.max_entries}, "
            f"hit_rate={self.hit_rate:.2%}, "
            f"memory={self.total_memory_mb:.1f}MB)"
        )
