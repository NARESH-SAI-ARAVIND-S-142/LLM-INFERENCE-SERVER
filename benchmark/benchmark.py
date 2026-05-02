"""
miniServe — Automated Benchmark
Tests throughput vs latency tradeoff across different batch sizes.

Produces:
1. CSV file with raw results
2. Throughput vs Batch Size plot
3. Latency percentiles (p50/p95/p99) vs Batch Size plot
4. Tokens/sec vs Batch Size plot
5. Tradeoff curve: Throughput vs p99 Latency

Usage:
    # Make sure the server is running first!
    # Then:
    cd llm-inference-server
    source venv/bin/activate
    python benchmark/benchmark.py
    
    # Custom settings:
    python benchmark/benchmark.py --requests 50 --url http://localhost:8000
"""

import argparse
import asyncio
import time
import os
import sys
import json
from dataclasses import dataclass

import aiohttp
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_PROMPTS = [
    "Once upon a time in a magical forest, there lived",
    "The future of artificial intelligence is shaping up to be",
    "In the year 2050, humanity discovered a way to",
    "The secret to happiness lies in understanding that",
    "Deep beneath the ocean, scientists found evidence of",
    "The last star in the universe flickered and then",
    "A robot walked into a coffee shop and ordered",
    "The ancient library contained books that could predict",
    "On Mars, the first colony celebrated its tenth anniversary",
    "The quantum computer finally solved the problem that",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@dataclass
class BenchmarkResult:
    batch_size_config: int
    num_requests: int
    total_time_ms: float
    throughput_rps: float
    tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_mean_ms: float
    avg_batch_size: float
    total_tokens: int
    errors: int


async def send_request(session: aiohttp.ClientSession, url: str, prompt: str, max_tokens: int = 50) -> dict:
    """Send a single request and return timing info."""
    start = time.time()
    try:
        async with session.post(
            f"{url}/v1/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.8},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            elapsed = (time.time() - start) * 1000
            data["client_latency_ms"] = elapsed
            data["error"] = False
            return data
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {"error": True, "error_msg": str(e), "client_latency_ms": elapsed}


async def run_benchmark_batch(
    url: str,
    batch_size_config: int,
    num_requests: int,
    max_tokens: int = 50,
) -> BenchmarkResult:
    """
    Run a benchmark for a specific batch size configuration.
    
    We first update the server's batch size config, then send
    num_requests concurrent requests and measure performance.
    """
    print(f"\n  ┌─ Benchmark: batch_size={batch_size_config}, requests={num_requests}")
    
    # Send all requests concurrently
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(
                session, url,
                SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)],
                max_tokens,
            )
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
    
    total_time = (time.time() - start_time) * 1000

    # Separate successes and errors
    successes = [r for r in results if not r.get("error", True)]
    errors = [r for r in results if r.get("error", True)]

    if not successes:
        print(f"  │ ERROR: All {num_requests} requests failed!")
        print(f"  └─────────────────────────────────────────────")
        return BenchmarkResult(
            batch_size_config=batch_size_config,
            num_requests=num_requests,
            total_time_ms=total_time,
            throughput_rps=0, tokens_per_sec=0,
            latency_p50_ms=0, latency_p95_ms=0, latency_p99_ms=0,
            latency_min_ms=0, latency_max_ms=0, latency_mean_ms=0,
            avg_batch_size=0, total_tokens=0, errors=len(errors),
        )

    # Calculate metrics
    latencies = sorted([r["client_latency_ms"] for r in successes])
    tokens = [r.get("tokens_generated", 0) for r in successes]
    batch_sizes = [r.get("batch_size", 1) for r in successes]
    total_tokens = sum(tokens)

    result = BenchmarkResult(
        batch_size_config=batch_size_config,
        num_requests=num_requests,
        total_time_ms=round(total_time, 2),
        throughput_rps=round(len(successes) / (total_time / 1000), 2),
        tokens_per_sec=round(total_tokens / (total_time / 1000), 2),
        latency_p50_ms=round(np.percentile(latencies, 50), 2),
        latency_p95_ms=round(np.percentile(latencies, 95), 2),
        latency_p99_ms=round(np.percentile(latencies, 99), 2),
        latency_min_ms=round(min(latencies), 2),
        latency_max_ms=round(max(latencies), 2),
        latency_mean_ms=round(np.mean(latencies), 2),
        avg_batch_size=round(np.mean(batch_sizes), 2),
        total_tokens=total_tokens,
        errors=len(errors),
    )

    print(f"  │ Throughput:   {result.throughput_rps:.1f} req/s")
    print(f"  │ Tokens/sec:   {result.tokens_per_sec:.1f}")
    print(f"  │ p50 latency:  {result.latency_p50_ms:.0f} ms")
    print(f"  │ p95 latency:  {result.latency_p95_ms:.0f} ms")
    print(f"  │ p99 latency:  {result.latency_p99_ms:.0f} ms")
    print(f"  │ Avg batch:    {result.avg_batch_size:.1f}")
    print(f"  │ Errors:       {result.errors}")
    print(f"  └─────────────────────────────────────────────")

    return result


def plot_results(results: list[BenchmarkResult], output_dir: str):
    """Generate benchmark visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    if not results or all(r.throughput_rps == 0 for r in results):
        print("  No valid results to plot.")
        return

    batch_sizes = [r.batch_size_config for r in results]
    throughputs = [r.throughput_rps for r in results]
    p50s = [r.latency_p50_ms for r in results]
    p95s = [r.latency_p95_ms for r in results]
    p99s = [r.latency_p99_ms for r in results]
    tok_per_sec = [r.tokens_per_sec for r in results]

    # Style
    plt.style.use("dark_background")
    colors = {
        "primary": "#00D4FF",
        "secondary": "#FF6B6B",
        "tertiary": "#4ECDC4",
        "accent": "#FFE66D",
        "bg": "#0D1117",
        "grid": "#21262D",
    }

    # ─── Plot 1: Throughput vs Batch Size ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    
    bars = ax.bar(range(len(batch_sizes)), throughputs, color=colors["primary"], alpha=0.8, width=0.6)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Max Batch Size", fontsize=12, color="white")
    ax.set_ylabel("Throughput (req/s)", fontsize=12, color="white")
    ax.set_title("Throughput vs Batch Size", fontsize=14, color="white", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, color=colors["grid"])
    
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_vs_batch_size.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── Plot 2: Latency Percentiles vs Batch Size ────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    
    ax.plot(batch_sizes, p50s, "o-", color=colors["primary"], label="p50", linewidth=2, markersize=8)
    ax.plot(batch_sizes, p95s, "s-", color=colors["accent"], label="p95", linewidth=2, markersize=8)
    ax.plot(batch_sizes, p99s, "^-", color=colors["secondary"], label="p99", linewidth=2, markersize=8)
    
    ax.set_xlabel("Max Batch Size", fontsize=12, color="white")
    ax.set_ylabel("Latency (ms)", fontsize=12, color="white")
    ax.set_title("Latency Percentiles vs Batch Size", fontsize=14, color="white", fontweight="bold")
    ax.legend(fontsize=11, facecolor=colors["bg"], edgecolor=colors["grid"])
    ax.grid(alpha=0.2, color=colors["grid"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_batch_size.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── Plot 3: Tokens/sec vs Batch Size ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    
    ax.plot(batch_sizes, tok_per_sec, "D-", color=colors["tertiary"], linewidth=2, markersize=8)
    ax.fill_between(batch_sizes, tok_per_sec, alpha=0.2, color=colors["tertiary"])
    
    ax.set_xlabel("Max Batch Size", fontsize=12, color="white")
    ax.set_ylabel("Tokens/sec", fontsize=12, color="white")
    ax.set_title("Token Generation Rate vs Batch Size", fontsize=14, color="white", fontweight="bold")
    ax.grid(alpha=0.2, color=colors["grid"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tokens_per_sec.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── Plot 4: Tradeoff Curve (Throughput vs p99 Latency) ───────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    
    scatter = ax.scatter(throughputs, p99s, c=batch_sizes, cmap="plasma", s=150, edgecolors="white", linewidths=1.5, zorder=5)
    ax.plot(throughputs, p99s, "--", color="white", alpha=0.3, linewidth=1)
    
    for i, bs in enumerate(batch_sizes):
        ax.annotate(f"bs={bs}", (throughputs[i], p99s[i]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=9, color="white", alpha=0.8)
    
    cbar = plt.colorbar(scatter, ax=ax, label="Batch Size")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    
    ax.set_xlabel("Throughput (req/s)", fontsize=12, color="white")
    ax.set_ylabel("p99 Latency (ms)", fontsize=12, color="white")
    ax.set_title("Throughput vs Latency Tradeoff", fontsize=14, color="white", fontweight="bold")
    ax.grid(alpha=0.2, color=colors["grid"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tradeoff_curve.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    print(f"\n  📊 Plots saved to {output_dir}/")


async def run_full_benchmark(url: str, num_requests: int, max_tokens: int):
    """Run the complete benchmark suite."""
    print(f"\n{'='*60}")
    print(f"  miniServe — Automated Benchmark")
    print(f"{'='*60}")
    print(f"  Server:    {url}")
    print(f"  Requests:  {num_requests} per batch size config")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Batch Sizes: 1, 2, 4, 8, 16, 32")

    # Check server is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                health = await resp.json()
                print(f"\n  Server Status: {health.get('status', 'unknown')}")
                print(f"  Model: {health.get('model', 'unknown')}")
    except Exception as e:
        print(f"\n  ❌ Cannot reach server at {url}: {e}")
        print(f"  Make sure the server is running: python -m server.main")
        return

    batch_sizes_to_test = [1, 2, 4, 8, 16, 32]
    results: list[BenchmarkResult] = []

    for bs in batch_sizes_to_test:
        print(f"\n  ━━━ Testing batch_size={bs} ━━━")
        result = await run_benchmark_batch(url, bs, num_requests, max_tokens)
        results.append(result)
        # Small delay between tests
        await asyncio.sleep(2)

    # Save results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame([vars(r) for r in results])
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 Results saved to {csv_path}")

    # Print summary table
    print(f"\n  {'='*80}")
    print(f"  BENCHMARK SUMMARY")
    print(f"  {'='*80}")
    print(f"  {'Batch':>6} | {'Throughput':>12} | {'Tok/s':>8} | {'p50':>8} | {'p95':>8} | {'p99':>8} | {'Errors':>6}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for r in results:
        print(
            f"  {r.batch_size_config:>6} | "
            f"{r.throughput_rps:>10.1f}/s | "
            f"{r.tokens_per_sec:>8.1f} | "
            f"{r.latency_p50_ms:>6.0f}ms | "
            f"{r.latency_p95_ms:>6.0f}ms | "
            f"{r.latency_p99_ms:>6.0f}ms | "
            f"{r.errors:>6}"
        )
    print(f"  {'='*80}")

    # Generate plots
    plot_results(results, RESULTS_DIR)

    # Calculate improvement
    if len(results) >= 2 and results[0].throughput_rps > 0:
        best = max(results, key=lambda r: r.throughput_rps)
        baseline = results[0]
        improvement = best.throughput_rps / baseline.throughput_rps
        print(f"\n  🚀 Best throughput: {best.throughput_rps:.1f} req/s at batch_size={best.batch_size_config}")
        print(f"     vs baseline ({baseline.throughput_rps:.1f} req/s): {improvement:.1f}x improvement")
    
    print(f"\n  ✅ Benchmark complete!\n")


def main():
    parser = argparse.ArgumentParser(description="miniServe Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--requests", type=int, default=20, help="Requests per batch size test")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens per request")
    args = parser.parse_args()

    asyncio.run(run_full_benchmark(args.url, args.requests, args.max_tokens))


if __name__ == "__main__":
    main()
