"""
miniServe — REST Client
Sample client demonstrating REST API usage.

Usage:
    # Single request:
    python client/rest_client.py
    
    # Custom prompt:
    python client/rest_client.py --prompt "The meaning of life is"
    
    # Concurrent requests (to test batching):
    python client/rest_client.py --concurrent 10
"""

import argparse
import asyncio
import time
import json
import sys
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp")
    sys.exit(1)


DEFAULT_URL = "http://localhost:8000"

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
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    request_id: Optional[str] = None,
) -> dict:
    """Send a single generation request."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if request_id:
        payload["request_id"] = request_id

    start = time.time()
    async with session.post(f"{url}/v1/generate", json=payload) as resp:
        elapsed = (time.time() - start) * 1000
        data = await resp.json()
        data["client_latency_ms"] = round(elapsed, 2)
        return data


async def single_request(url: str, prompt: str, max_tokens: int):
    """Send a single request and print the result."""
    print(f"\n{'='*60}")
    print(f"  miniServe REST Client — Single Request")
    print(f"{'='*60}\n")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Max tokens: {max_tokens}\n")

    async with aiohttp.ClientSession() as session:
        result = await send_request(session, url, prompt, max_tokens)

    print(f"  ┌─ Generated Text ─────────────────────────────")
    print(f"  │ {result.get('generated_text', 'ERROR')}")
    print(f"  └──────────────────────────────────────────────\n")
    print(f"  Tokens Generated: {result.get('tokens_generated', 'N/A')}")
    print(f"  Inference Time:   {result.get('latency_ms', 'N/A')} ms")
    print(f"  Queue Wait:       {result.get('queue_wait_ms', 'N/A')} ms")
    print(f"  Batch Size:       {result.get('batch_size', 'N/A')}")
    print(f"  From Cache:       {result.get('from_cache', 'N/A')}")
    print(f"  Client Latency:   {result.get('client_latency_ms', 'N/A')} ms")
    print(f"  Request ID:       {result.get('request_id', 'N/A')}")
    print()


async def concurrent_requests(url: str, num_requests: int, max_tokens: int):
    """Send multiple concurrent requests to test batching."""
    print(f"\n{'='*60}")
    print(f"  miniServe REST Client — Concurrent Requests")
    print(f"{'='*60}\n")
    print(f"  Sending {num_requests} concurrent requests...\n")

    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(
                session, url,
                prompt=SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)],
                max_tokens=max_tokens,
            )
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = (time.time() - start) * 1000

    # Print results
    successful = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]

    print(f"  Results: {len(successful)} success, {len(errors)} errors\n")

    if successful:
        latencies = [r.get("client_latency_ms", 0) for r in successful]
        inference_times = [r.get("latency_ms", 0) for r in successful]
        batch_sizes = [r.get("batch_size", 1) for r in successful]
        tokens = [r.get("tokens_generated", 0) for r in successful]

        latencies.sort()
        print(f"  ┌─ Performance Summary ────────────────────────")
        print(f"  │ Total time:        {total_time:.0f} ms")
        print(f"  │ Throughput:         {len(successful) / (total_time / 1000):.1f} req/s")
        print(f"  │ Total tokens:       {sum(tokens)}")
        print(f"  │ Tokens/sec:         {sum(tokens) / (total_time / 1000):.1f}")
        print(f"  │")
        print(f"  │ Client Latency:")
        print(f"  │   p50:   {latencies[len(latencies)//2]:.1f} ms")
        print(f"  │   p95:   {latencies[int(len(latencies)*0.95)]:.1f} ms")
        print(f"  │   p99:   {latencies[int(len(latencies)*0.99)]:.1f} ms")
        print(f"  │   max:   {latencies[-1]:.1f} ms")
        print(f"  │")
        print(f"  │ Avg batch size:     {sum(batch_sizes)/len(batch_sizes):.1f}")
        print(f"  │ Max batch size:     {max(batch_sizes)}")
        print(f"  └──────────────────────────────────────────────\n")

        # Print first 3 responses as samples
        print(f"  Sample responses:")
        for i, r in enumerate(successful[:3]):
            text = r.get("generated_text", "")[:80]
            print(f"  [{i+1}] \"{text}...\"")
        print()

    for e in errors:
        print(f"  ERROR: {e}")


async def health_check(url: str):
    """Check server health."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/health") as resp:
            data = await resp.json()
            print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="miniServe REST Client")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server URL")
    parser.add_argument("--prompt", default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens")
    parser.add_argument("--concurrent", type=int, default=0, help="Number of concurrent requests")
    parser.add_argument("--health", action="store_true", help="Check server health")
    args = parser.parse_args()

    if args.health:
        asyncio.run(health_check(args.url))
    elif args.concurrent > 0:
        asyncio.run(concurrent_requests(args.url, args.concurrent, args.max_tokens))
    else:
        asyncio.run(single_request(args.url, args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
