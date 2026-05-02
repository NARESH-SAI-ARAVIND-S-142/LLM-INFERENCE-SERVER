<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/gRPC-1.59%2B-244c5a?style=for-the-badge&logo=google&logoColor=white" alt="gRPC"/>
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

<h1 align="center">⚡ miniServe</h1>

<p align="center">
  <strong>A production-grade LLM inference server with dynamic batching, KV-cache management, and dual REST/gRPC APIs — built from scratch.</strong>
</p>

<p align="center">
  <em>Mirrors the architecture of serving infrastructure at Google (Gemini), OpenAI, and Anthropic.</em>
</p>

---

## 📌 Overview

**miniServe** is a high-performance inference server designed to serve Large Language Models efficiently at scale. It implements the same core techniques used in production LLM serving systems:

- **Dynamic request batching** to maximize hardware utilization
- **KV-cache management** to eliminate redundant attention computation
- **Dual-protocol API layer** (REST + gRPC) for flexible integration
- **Full observability stack** with Prometheus metrics and Grafana dashboards

> Built to demonstrate deep understanding of ML systems infrastructure — not just ML models, but the engineering required to serve them reliably under load.

---

## 🏗️ System Architecture

```
                        ┌──────────────────────────────────────────┐
                        │             miniServe                     │
                        │                                          │
  ┌─────────┐          │  ┌──────────────────────────────────┐    │
  │  REST    │──HTTP──▶ │  │         API Gateway               │    │
  │  Client  │          │  │   (FastAPI / gRPC Servicer)       │    │
  └─────────┘          │  └──────────┬───────────────────────┘    │
                        │             │                            │
  ┌─────────┐          │             ▼                            │
  │  gRPC   │──H2───▶  │  ┌──────────────────────────────────┐    │
  │  Client  │          │  │       Request Queue               │    │
  └─────────┘          │  │    (asyncio.Queue)                 │    │
                        │  └──────────┬───────────────────────┘    │
                        │             │                            │
                        │             ▼                            │
                        │  ┌──────────────────────────────────┐    │
                        │  │     Dynamic Batch Scheduler       │    │
                        │  │  max_wait=50ms │ max_batch=8      │    │
                        │  └──────────┬───────────────────────┘    │
                        │             │                            │
                        │             ▼                            │
                        │  ┌──────────────────────────────────┐    │
                        │  │      Inference Engine             │    │
                        │  │  HuggingFace Transformers + KV$   │    │
                        │  └──────────┬───────────────────────┘    │
                        │             │                            │
                        │             ▼                            │
                        │  ┌──────────────────────────────────┐    │
                        │  │     KV-Cache Manager (LRU)        │    │
                        │  │  Thread-safe │ Memory-tracked     │    │
                        │  └──────────────────────────────────┘    │
                        │                                          │
                        │  ┌──────────────────────────────────┐    │
                        │  │   Prometheus Metrics Exporter     │──────▶ Grafana
                        │  │   14 metrics │ /metrics endpoint  │    │
                        │  └──────────────────────────────────┘    │
                        └──────────────────────────────────────────┘
```

### Request Lifecycle

1. Client sends a prompt via **REST** (`POST /v1/generate`) or **gRPC** (`Generate` RPC)
2. Request enters the **async queue** with a `Future` attached
3. **Batch Scheduler** collects requests until `max_batch_size` (8) is reached or `max_wait_time` (50ms) expires — whichever comes first
4. Padded batch is sent to the **Inference Engine** for generation
5. Engine checks **KV-Cache** for reusable attention states (multi-turn optimization)
6. Results are dispatched back to individual client `Futures`
7. **Prometheus** records latency, throughput, batch size, and cache metrics at every step

---

## ✨ Key Features

| Feature | Implementation | Impact |
|---------|---------------|--------|
| **Dynamic Batching** | asyncio queue with dual-trigger (time + count) | Up to **4.3×** throughput improvement over single-request baseline |
| **KV-Cache** | Thread-safe LRU `OrderedDict` with memory tracking | Eliminates redundant attention computation in multi-turn conversations |
| **Dual API** | FastAPI (REST) + grpcio (gRPC), shared engine | Flexible integration — REST for external clients, gRPC for microservices |
| **14 Prometheus Metrics** | Counters, Histograms, Gauges, Summaries | Full observability: latency percentiles, batch distribution, cache hit rate |
| **Grafana Dashboard** | Pre-configured JSON with 8 panels | Real-time monitoring out of the box |
| **Automated Benchmarking** | Sweep across batch sizes 1–32, generate plots | Quantified throughput-vs-latency tradeoff curves |
| **Locust Load Testing** | Configurable concurrent users with web UI | Realistic production traffic simulation |
| **Docker Compose** | Multi-stage build + Prometheus + Grafana | One-command production deployment |
| **Zero GPU Required** | Runs on CPU with GPT-2 (124M params, ~475MB) | Accessible on any machine with 4GB+ free RAM |

---

## 📂 Project Structure

```
llm-inference-server/
│
├── server/                          # ── Core Server ──────────────
│   ├── main.py                      #    FastAPI REST API server
│   ├── grpc_server.py               #    Async gRPC server
│   ├── batch_scheduler.py           #    Dynamic batching engine
│   ├── inference_engine.py          #    Model loading & generation
│   ├── kv_cache.py                  #    LRU KV-cache manager
│   └── metrics.py                   #    Prometheus instrumentation
│
├── proto/                           # ── Protocol Buffers ─────────
│   ├── inference.proto              #    Service & message definitions
│   ├── inference_pb2.py             #    Auto-generated message code
│   └── inference_pb2_grpc.py        #    Auto-generated service stubs
│
├── benchmark/                       # ── Performance Testing ──────
│   ├── benchmark.py                 #    Automated benchmark suite
│   ├── load_test.py                 #    Locust load test config
│   └── results/                     #    Output: CSVs & PNGs
│
├── client/                          # ── Sample Clients ───────────
│   ├── rest_client.py               #    REST client (single + batch)
│   └── grpc_client.py               #    gRPC client (single + batch)
│
├── docker/                          # ── Containerization ─────────
│   ├── Dockerfile                   #    Multi-stage production build
│   ├── docker-compose.yml           #    Server + Prometheus + Grafana
│   ├── prometheus.yml               #    Scrape configuration
│   └── grafana/                     #    Dashboard & datasource provisioning
│       ├── dashboards/
│       │   └── miniserve.json       #    Pre-built Grafana dashboard
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboards.yml
│           └── datasources/
│               └── datasources.yml
│
├── config.py                        #    Centralized configuration
├── run_server.py                    #    Combined REST + gRPC launcher
├── requirements.txt                 #    Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **pip** (Python package manager)
- **~4GB free RAM** (GPT-2 uses ~475MB + overhead for batching)
- **Internet** (one-time, to download the model from HuggingFace)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/llm-inference-server.git
cd llm-inference-server

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate gRPC stubs from protobuf schema
python -m grpc_tools.protoc \
    -I proto \
    --python_out=proto \
    --grpc_python_out=proto \
    proto/inference.proto
```

### Launch the Server

```bash
python run_server.py
```

On first run, the GPT-2 model (~500MB) is automatically downloaded from HuggingFace and cached locally. Subsequent starts load from cache in ~2 seconds.

```
╔══════════════════════════════════════════════════════════╗
║              miniServe — LLM Inference Server           ║
║         Dynamic Batching | KV-Cache | REST + gRPC       ║
╚══════════════════════════════════════════════════════════╝

  🚀 miniServe is READY!
  📡 REST:    http://localhost:8000/v1/generate
  📡 gRPC:    localhost:50051
  📖 Docs:    http://localhost:8000/docs
  📊 Metrics: http://localhost:8000/metrics
```

### Verify It Works

```bash
# Send a single request via curl
curl -s -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of artificial intelligence is", "max_tokens": 50}' \
  | python3 -m json.tool
```

```json
{
    "generated_text": "in the hands of the people, who will choose...",
    "tokens_generated": 50,
    "latency_ms": 5983.8,
    "batch_size": 1,
    "queue_wait_ms": 12.34,
    "from_cache": false,
    "request_id": "4b02c3ac-b772-4823-8e1f-81136ebd7a7f"
}
```

---

## 📡 API Reference

### REST API (FastAPI)

Interactive API documentation is automatically available at **http://localhost:8000/docs** (Swagger UI).

#### `POST /v1/generate` — Text Generation

Submit a prompt for text generation. Requests are automatically batched with other concurrent requests.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | *required* | Input text to generate from |
| `max_tokens` | `int` | `50` | Maximum new tokens to generate (1–500) |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0–2.0). Lower = more deterministic |
| `request_id` | `string` | `auto` | Optional ID for KV-cache reuse across turns |

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `generated_text` | `string` | The model's generated continuation |
| `prompt` | `string` | Echo of the original prompt |
| `tokens_generated` | `int` | Number of new tokens produced |
| `latency_ms` | `float` | Inference time (excludes queue wait) |
| `batch_size` | `int` | Number of requests in this batch |
| `queue_wait_ms` | `float` | Time spent waiting in the batch queue |
| `from_cache` | `bool` | Whether KV-cache was used |
| `request_id` | `string` | Request identifier |

#### `GET /health` — Health Check

Returns server readiness status, model info, queue depth, and cache statistics.

#### `GET /metrics` — Prometheus Metrics

Exports all metrics in Prometheus text exposition format. Scrape this endpoint with Prometheus.

#### `GET /stats` — Server Statistics

Returns detailed JSON statistics including scheduler and cache state.

---

### gRPC API

Defined in [`proto/inference.proto`](proto/inference.proto). The gRPC server listens on port **50051**.

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `Generate` | `GenerateRequest` | `GenerateResponse` | Text generation (same fields as REST) |
| `HealthCheck` | `HealthCheckRequest` | `HealthCheckResponse` | Server health status |

---

## 🧠 Core Concepts

### Dynamic Batching

Traditional inference servers process requests one at a time, wasting compute capacity. miniServe implements **dynamic batching** — incoming requests are held in an async queue and grouped into batches using a dual-trigger mechanism:

```
                    ┌─ Request 1 arrives ──▶ Start timer (50ms)
                    │
                    ├─ Request 2 arrives ──▶ Add to batch
                    │
  Batch fires ◀──── ├─ Request 3 arrives ──▶ Add to batch
  when EITHER:      │
                    ├─ ...
  • batch_size = 8  │
    is reached      ├─ Request 8 arrives ──▶ FIRE (batch full)
         OR         │
  • 50ms elapsed    └─ Timer expires ──────▶ FIRE (timeout)
    since first
    request
```

**Why it matters:** On the CPU benchmark, batching 5 requests together achieved **13.8 tok/s** vs **1.8 tok/s** for single requests — a **~7× improvement** in token throughput.

### KV-Cache Management

In autoregressive (token-by-token) generation, each new token must attend to every previous token. Without caching, generating token *N* recomputes attention for tokens *1* through *N-1*.

miniServe implements an **LRU (Least Recently Used) cache** that stores `past_key_values` from the transformer's attention layers:

- **Cache hit**: Reuse stored attention states → skip redundant computation
- **Cache miss**: Compute from scratch → store result for future reuse
- **Eviction**: When cache is full, evict the least-recently-accessed entry
- **Thread safety**: All operations are protected by a mutex lock
- **Monitoring**: Cache size, hit rate, and memory usage are exported as Prometheus metrics

### Throughput vs. Latency Tradeoff

This is the fundamental tension in serving systems:

| Strategy | Throughput | Latency | When to use |
|----------|-----------|---------|-------------|
| Small batches (1–2) | Low | Low | Latency-sensitive, interactive applications |
| Medium batches (4–8) | Medium | Medium | Balanced workloads |
| Large batches (16–32) | High | High | Throughput-oriented, batch processing |

The automated benchmark suite measures this tradeoff empirically and generates visualization plots.

---

## 📊 Benchmarking

### Automated Benchmark Suite

Run the full benchmark to measure throughput and latency across batch sizes 1–32:

```bash
# Ensure the server is running in another terminal
python benchmark/benchmark.py --requests 20 --max-tokens 50
```

**Outputs** (saved to `benchmark/results/`):

| File | Description |
|------|-------------|
| `benchmark_results.csv` | Raw metrics for every batch size configuration |
| `throughput_vs_batch_size.png` | Bar chart: requests/sec at each batch size |
| `latency_vs_batch_size.png` | Line chart: p50/p95/p99 latency curves |
| `tokens_per_sec.png` | Line chart: token generation throughput |
| `tradeoff_curve.png` | Scatter plot: throughput vs. p99 latency tradeoff |

### Locust Load Testing

For realistic traffic simulation with a web UI:

```bash
# Start Locust (web UI opens at http://localhost:8089)
locust -f benchmark/load_test.py --host http://localhost:8000

# Headless mode (for CI/CD pipelines)
locust -f benchmark/load_test.py \
    --host http://localhost:8000 \
    --headless \
    -u 20 -r 5 -t 60s \
    --csv=benchmark/results/locust
```

### Sample Clients

```bash
# ── REST ──────────────────────────────────────────────────────
# Single request
python client/rest_client.py --prompt "Once upon a time" --max-tokens 50

# Concurrent requests (triggers dynamic batching)
python client/rest_client.py --concurrent 10

# Health check
python client/rest_client.py --health

# ── gRPC ──────────────────────────────────────────────────────
# Single request
python client/grpc_client.py --prompt "The meaning of life is"

# Concurrent requests
python client/grpc_client.py --concurrent 10

# Health check
python client/grpc_client.py --health
```

---

## 📈 Observability

### Prometheus Metrics

All metrics are exported at `GET /metrics` in Prometheus text exposition format.

| Metric | Type | Description |
|--------|------|-------------|
| `miniserve_requests_total` | Counter | Total requests received (labeled by `method`: rest/grpc) |
| `miniserve_tokens_generated_total` | Counter | Cumulative tokens generated |
| `miniserve_batches_processed_total` | Counter | Total batches executed |
| `miniserve_errors_total` | Counter | Error count (labeled by `error_type`) |
| `miniserve_inference_latency_seconds` | Histogram | Model inference latency (buckets: 10ms → 10s) |
| `miniserve_queue_wait_seconds` | Histogram | Time spent in batch queue (buckets: 1ms → 1s) |
| `miniserve_total_request_latency_seconds` | Histogram | End-to-end request latency (queue + inference) |
| `miniserve_batch_size` | Histogram | Distribution of batch sizes processed |
| `miniserve_tokens_per_request` | Histogram | Tokens generated per individual request |
| `miniserve_queue_depth` | Gauge | Current number of pending requests |
| `miniserve_kv_cache_entries` | Gauge | Current KV-cache occupancy |
| `miniserve_kv_cache_hit_rate` | Gauge | Cache hit ratio (0.0 – 1.0) |
| `miniserve_kv_cache_memory_mb` | Gauge | Estimated cache memory footprint |
| `miniserve_model_loaded` | Gauge | Model readiness flag (0 or 1) |

### Grafana Dashboard

A pre-configured dashboard (`docker/grafana/dashboards/miniserve.json`) includes 8 panels:

- Request rate (req/s) by method
- Inference latency heatmap (p50 / p95 / p99)
- Batch size distribution histogram
- Queue depth over time
- KV-cache hit rate and entry count
- Total tokens generated
- Total requests processed
- Queue wait time gauge (p95)

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
cd docker

# Build and launch the full stack
docker compose up --build -d

# Verify services are running
docker compose ps
```

| Service | URL | Credentials |
|---------|-----|-------------|
| miniServe REST API | http://localhost:8000 | — |
| miniServe gRPC | localhost:50051 | — |
| Swagger Docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `miniserve` |

### Standalone Docker Build

```bash
# Build the image
docker build -t miniserve:latest -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 -p 50051:50051 miniserve:latest
```

The Dockerfile uses a **multi-stage build**:
1. **Builder stage**: Installs dependencies and compiles protobuf stubs
2. **Runtime stage**: Copies only what's needed for a minimal image
3. **Model pre-download**: GPT-2 weights are baked into the image for instant cold starts
4. **Health check**: Built-in Docker HEALTHCHECK against `/health` endpoint

---

## ⚙️ Configuration

All settings are centralized in `config.py` and can be overridden via environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MINISERVE_MODEL` | `gpt2` | HuggingFace model identifier |
| `MINISERVE_DEVICE` | `auto` | Compute device — auto-detects CUDA, falls back to CPU |
| `MINISERVE_MAX_BATCH_SIZE` | `8` | Maximum requests per inference batch |
| `MINISERVE_MAX_WAIT_MS` | `50` | Maximum milliseconds to wait before firing a batch |
| `MINISERVE_MAX_TOKENS` | `50` | Default max tokens per generation |
| `MINISERVE_TEMPERATURE` | `1.0` | Default sampling temperature |
| `MINISERVE_TOP_K` | `50` | Default top-k sampling parameter |
| `MINISERVE_KV_CACHE_MAX` | `100` | Maximum KV-cache entries before LRU eviction |
| `MINISERVE_REST_HOST` | `0.0.0.0` | REST server bind address |
| `MINISERVE_REST_PORT` | `8000` | REST server port |
| `MINISERVE_GRPC_HOST` | `0.0.0.0` | gRPC server bind address |
| `MINISERVE_GRPC_PORT` | `50051` | gRPC server port |
| `MINISERVE_LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

**Example: Run with custom settings**

```bash
MINISERVE_MAX_BATCH_SIZE=16 \
MINISERVE_MAX_WAIT_MS=100 \
MINISERVE_MAX_TOKENS=100 \
python run_server.py
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | Python 3.12 | Async-first ML ecosystem |
| **Model** | GPT-2 124M (HuggingFace) | Lightweight, free, CPU-compatible |
| **ML Framework** | PyTorch + Transformers | Industry-standard model inference |
| **REST API** | FastAPI + Uvicorn | High-performance async HTTP server |
| **RPC** | gRPC + Protocol Buffers | Binary serialization, HTTP/2 multiplexing |
| **Scheduling** | asyncio + `asyncio.Queue` | Non-blocking concurrent batch collection |
| **Caching** | `collections.OrderedDict` | O(1) LRU cache with thread-safe locking |
| **Metrics** | prometheus-client | Counters, histograms, gauges, summaries |
| **Monitoring** | Grafana | Real-time dashboard visualization |
| **Load Testing** | Locust | Distributed, scriptable traffic generation |
| **Analysis** | pandas + matplotlib | Benchmark data processing and plotting |
| **Container** | Docker + Docker Compose | Reproducible multi-service deployment |

---

## 🗂️ System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Linux / macOS / Windows | Linux (Ubuntu 22.04+) |
| **Python** | 3.10 | 3.12 |
| **RAM** | 4 GB free | 8 GB+ |
| **CPU** | Any x86-64 | Multi-core (benefits batching) |
| **GPU** | Not required | CUDA-capable (for faster inference) |
| **Storage** | 3 GB (model + deps) | 5 GB |
| **Network** | One-time download | — |

---

## 📄 License

This project is licensed under the **MIT License** — free for personal, academic, and commercial use.

---

<p align="center">
  <sub>Built with ❤️ to understand LLM infrastructure from the ground up.</sub>
</p>
