"""
miniServe — gRPC Client
Sample client demonstrating gRPC API usage.

Usage:
    # Single request:
    python client/grpc_client.py
    
    # Custom prompt:
    python client/grpc_client.py --prompt "The meaning of life is"
    
    # Concurrent requests:
    python client/grpc_client.py --concurrent 10
"""

import argparse
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import grpc
from grpc import aio as grpc_aio

import inference_pb2
import inference_pb2_grpc


DEFAULT_TARGET = "localhost:50051"

SAMPLE_PROMPTS = [
    "Once upon a time in a magical forest",
    "The future of artificial intelligence is",
    "In the year 2050, humanity discovered",
    "The secret to happiness lies in",
    "Deep beneath the ocean, scientists found",
    "The last star in the universe flickered",
    "A robot walked into a coffee shop and",
    "The ancient library contained books that",
    "On Mars, the first colony celebrated",
    "The quantum computer finally solved",
]


async def send_request(
    stub: inference_pb2_grpc.InferenceServiceStub,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
) -> dict:
    """Send a single gRPC generate request."""
    request = inference_pb2.GenerateRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    start = time.time()
    response = await stub.Generate(request)
    elapsed = (time.time() - start) * 1000

    return {
        "generated_text": response.generated_text,
        "prompt": response.prompt,
        "tokens_generated": response.tokens_generated,
        "latency_ms": response.latency_ms,
        "batch_size": response.batch_size,
        "queue_wait_ms": response.queue_wait_ms,
        "from_cache": response.from_cache,
        "request_id": response.request_id,
        "client_latency_ms": round(elapsed, 2),
    }


async def single_request(target: str, prompt: str, max_tokens: int):
    """Send a single gRPC request."""
    print(f"\n{'='*60}")
    print(f"  miniServe gRPC Client — Single Request")
    print(f"{'='*60}\n")
    print(f"  Target: {target}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Max tokens: {max_tokens}\n")

    async with grpc_aio.insecure_channel(target) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        result = await send_request(stub, prompt, max_tokens)

    print(f"  ┌─ Generated Text ─────────────────────────────")
    print(f"  │ {result['generated_text']}")
    print(f"  └──────────────────────────────────────────────\n")
    print(f"  Tokens Generated: {result['tokens_generated']}")
    print(f"  Inference Time:   {result['latency_ms']:.2f} ms")
    print(f"  Queue Wait:       {result['queue_wait_ms']:.2f} ms")
    print(f"  Batch Size:       {result['batch_size']}")
    print(f"  From Cache:       {result['from_cache']}")
    print(f"  Client Latency:   {result['client_latency_ms']:.2f} ms (includes network)")
    print()


async def concurrent_requests(target: str, num_requests: int, max_tokens: int):
    """Send concurrent gRPC requests to test batching."""
    print(f"\n{'='*60}")
    print(f"  miniServe gRPC Client — {num_requests} Concurrent Requests")
    print(f"{'='*60}\n")

    start = time.time()
    async with grpc_aio.insecure_channel(target) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        tasks = [
            send_request(stub, SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)], max_tokens)
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = (time.time() - start) * 1000
    successful = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]

    print(f"  Results: {len(successful)} success, {len(errors)} errors\n")

    if successful:
        latencies = sorted([r["client_latency_ms"] for r in successful])
        tokens = [r["tokens_generated"] for r in successful]
        batch_sizes = [r["batch_size"] for r in successful]

        print(f"  ┌─ Performance Summary (gRPC) ────────────────")
        print(f"  │ Total time:        {total_time:.0f} ms")
        print(f"  │ Throughput:         {len(successful)/(total_time/1000):.1f} req/s")
        print(f"  │ Total tokens:       {sum(tokens)}")
        print(f"  │ Tokens/sec:         {sum(tokens)/(total_time/1000):.1f}")
        print(f"  │")
        print(f"  │ Client Latency:")
        print(f"  │   p50:   {latencies[len(latencies)//2]:.1f} ms")
        print(f"  │   p95:   {latencies[int(len(latencies)*0.95)]:.1f} ms")
        print(f"  │   p99:   {latencies[int(len(latencies)*0.99)]:.1f} ms")
        print(f"  │   max:   {latencies[-1]:.1f} ms")
        print(f"  │")
        print(f"  │ Avg batch size:     {sum(batch_sizes)/len(batch_sizes):.1f}")
        print(f"  └──────────────────────────────────────────────\n")

    for e in errors:
        print(f"  ERROR: {e}")


async def health_check(target: str):
    """gRPC health check."""
    async with grpc_aio.insecure_channel(target) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        request = inference_pb2.HealthCheckRequest()
        response = await stub.HealthCheck(request)
        print(f"  Status: {response.status}")
        print(f"  Model:  {response.model}")
        print(f"  Device: {response.device}")
        print(f"  Queue:  {response.queue_depth}")


def main():
    parser = argparse.ArgumentParser(description="miniServe gRPC Client")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="gRPC server target")
    parser.add_argument("--prompt", default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens")
    parser.add_argument("--concurrent", type=int, default=0, help="Concurrent requests")
    parser.add_argument("--health", action="store_true", help="Health check")
    args = parser.parse_args()

    if args.health:
        asyncio.run(health_check(args.target))
    elif args.concurrent > 0:
        asyncio.run(concurrent_requests(args.target, args.concurrent, args.max_tokens))
    else:
        asyncio.run(single_request(args.target, args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
