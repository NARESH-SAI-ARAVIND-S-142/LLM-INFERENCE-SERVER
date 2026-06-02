"""
miniServe — Centralized Configuration
All settings with environment variable overrides.
"""

import os
import torch


# ─── Model Settings ──────────────────────────────────────────────────────────

MODEL_NAME: str = os.getenv("MINISERVE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE: str = os.getenv("MINISERVE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS: int = int(os.getenv("MINISERVE_MAX_TOKENS", "50"))
TEMPERATURE: float = float(os.getenv("MINISERVE_TEMPERATURE", "1.0"))
TOP_K: int = int(os.getenv("MINISERVE_TOP_K", "50"))


# ─── Batch Scheduler Settings ────────────────────────────────────────────────

MAX_BATCH_SIZE: int = int(os.getenv("MINISERVE_MAX_BATCH_SIZE", "8"))
MAX_WAIT_TIME_MS: float = float(os.getenv("MINISERVE_MAX_WAIT_MS", "50"))
MAX_RUNNING_SEQUENCES: int = int(os.getenv("MINISERVE_MAX_RUNNING", "8"))
MAX_WAITING_QUEUE_SIZE: int = int(os.getenv("MINISERVE_MAX_WAITING_QUEUE", "100"))
MAX_SEQUENCE_TIMEOUT_S: float = float(os.getenv("MINISERVE_MAX_SEQ_TIMEOUT", "30.0"))


# ─── KV-Cache Settings ───────────────────────────────────────────────────────

KV_CACHE_MAX_ENTRIES: int = int(os.getenv("MINISERVE_KV_CACHE_MAX", "100"))


# ─── Server Settings ─────────────────────────────────────────────────────────

REST_HOST: str = os.getenv("MINISERVE_REST_HOST", "0.0.0.0")
REST_PORT: int = int(os.getenv("MINISERVE_REST_PORT", "8000"))
GRPC_HOST: str = os.getenv("MINISERVE_GRPC_HOST", "0.0.0.0")
GRPC_PORT: int = int(os.getenv("MINISERVE_GRPC_PORT", "50051"))
PROMETHEUS_PORT: int = int(os.getenv("MINISERVE_PROM_PORT", "9090"))


# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("MINISERVE_LOG_LEVEL", "INFO")


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("  miniServe Configuration")
    print("=" * 60)
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Device:         {DEVICE}")
    print(f"  Max Tokens:     {MAX_NEW_TOKENS}")
    print(f"  Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"  Max Wait (ms):  {MAX_WAIT_TIME_MS}")
    print(f"  Max Running:    {MAX_RUNNING_SEQUENCES}")
    print(f"  Max Queue:      {MAX_WAITING_QUEUE_SIZE}")
    print(f"  Timeout (s):    {MAX_SEQUENCE_TIMEOUT_S}")
    print(f"  KV Cache Max:   {KV_CACHE_MAX_ENTRIES}")
    print(f"  REST Port:      {REST_PORT}")
    print(f"  gRPC Port:      {GRPC_PORT}")
    print(f"  Log Level:      {LOG_LEVEL}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
