---
title: miniServe LLM Inference
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

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
  <strong>A production-grade LLM inference server with continuous batching, SSE streaming, KV-cache management, and dual REST/gRPC APIs — built from scratch.</strong>
</p>

<p align="center">
  <em>Mirrors the architecture of serving infrastructure at Google (Gemini), OpenAI, and Anthropic.</em>
</p>

---

## 📌 Overview

**miniServe** is a high-performance inference server designed to serve Large Language Models efficiently at scale. It implements the same core techniques used in production LLM serving systems like vLLM and TGI:

- **Continuous (iteration-level) batching** to maximize hardware utilization
- **Server-Sent Events (SSE) streaming** for real-time chat experiences
- **KV-cache management** to eliminate redundant attention computation
- **Instruction-tuned model support** (defaulting to Qwen 2.5) with dynamic chat templates
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
  │          │          │  │    (asyncio.Queue)                 │    │
  └─────────┘          │  └──────────┬───────────────────────┘    │
                        │             │                            │
                        │             ▼                            │
                        │  ┌──────────────────────────────────┐    │
                        │  │ Continuous Batch Scheduler       │    │
                        │  │  max_sequences=8 │ async loop     │    │
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

1. Client sends a prompt via **REST** (`POST /v1/generate`) or **gRPC** (`Generate` RPC).
2. Request enters the **async queue** as a `Sequence` object with its own token queue.
3. **Continuous Batch Scheduler** runs a background loop, pulling requests from the queue until `max_running_sequences` (8) is reached.
4. The scheduler submits the active sequences to the **Inference Engine** for a single generation step.
5. Engine checks **KV-Cache** for reusable attention states, performs prefill or decode, and adds the generated token to each sequence's stream queue.
6. The API Gateway yields tokens dynamically via **SSE (Server-Sent Events)** for real-time streaming, or waits until finished for a single JSON response.
7. Finished sequences are evicted from the batch immediately, and new waiting requests are swapped in without halting the engine.

---

## ✨ Key Features

| Feature | Implementation | Impact |
|---------|---------------|--------|
| **Continuous Batching** | Iteration-level scheduling loop (Orca/vLLM style) | Massively higher throughput by evicting early-finished requests immediately |
| **Token Streaming** | SSE (Server-Sent Events) via `StreamingResponse` | Chatbot-like real-time UX (first token latency < 100ms) |
| **Instruction Tuning** | Native `apply_chat_template` support | Accurately handles prompt formatting for models like Llama, Qwen, and Mistral |
| **KV-Cache** | Thread-safe LRU `DynamicCache` / Tuple manager | Eliminates redundant attention computation in multi-turn conversations |
| **Dual API** | FastAPI (REST) + grpcio (gRPC), shared scheduler | Flexible integration — REST for web clients, gRPC for microservices |
| **Docker Space** | Optimized Dockerfile for Hugging Face Spaces | Pre-downloads model weights at build time for instant cold starts |

---

## 📂 Project Structure

```
llm-inference-server/
│
├── server/                          # ── Core Server ──────────────
│   ├── main.py                      #    FastAPI REST API server
│   ├── grpc_server.py               #    Async gRPC server
│   ├── continuous_batch_scheduler.py#    Continuous batching engine
│   ├── inference_engine.py          #    Model loading & generation
│   ├── sequence.py                  #    Sequence state definitions
│   ├── kv_cache.py                  #    LRU KV-cache manager
│   └── metrics.py                   #    Prometheus instrumentation
│
├── proto/                           # ── Protocol Buffers ─────────
│   ├── inference.proto              #    Service & message definitions
│   ├── inference_pb2.py             #    Auto-generated message code
│   └── inference_pb2_grpc.py        #    Auto-generated service stubs
│
├── docker/                          # ── Containerization ─────────
│   ├── Dockerfile                   #    Multi-stage production build
│   ├── docker-compose.yml           #    Server + Prometheus + Grafana
│   └── prometheus.yml               #    Scrape configuration
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
- **~4GB free RAM** 
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

On first run, the default `Qwen/Qwen2.5-0.5B-Instruct` model is automatically downloaded from HuggingFace and cached locally. 

```
╔══════════════════════════════════════════════════════════╗
║              miniServe — LLM Inference Server           ║
║         Continuous Batching | KV-Cache | Streaming      ║
╚══════════════════════════════════════════════════════════╝

  🚀 miniServe is READY!
  📡 REST:    http://0.0.0.0:8000/v1/generate
  📡 gRPC:    0.0.0.0:50051
  📖 Docs:    http://0.0.0.0:8000/docs
  📊 Metrics: http://0.0.0.0:8000/metrics
```

### Verify It Works (Streaming)

```bash
curl -N -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a short poem about a server running in the cloud."}
    ],
    "stream": true,
    "max_tokens": 100
  }'
```

---

## 📡 API Reference

### REST API (FastAPI)

Interactive API documentation is automatically available at **http://localhost:8000/docs** (Swagger UI).

#### `POST /v1/generate` — Text Generation

Submit a prompt or chat messages for text generation. 

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `null` | OpenAI-style chat messages `[{"role": "user", "content": "hello"}]` |
| `prompt` | `string` | `null` | Legacy raw text prompt |
| `stream` | `bool` | `false` | Whether to stream the response via Server-Sent Events |
| `max_tokens` | `int` | `50` | Maximum new tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature |

**Response (Non-Streaming):**
Returns a JSON object with `generated_text`, `tokens_generated`, and `latency_ms`.

**Response (Streaming):**
Returns standard SSE data chunks containing string deltas:
```text
data: {"text": " The"}
data: {"text": " cloud"}
...
```

#### `GET /health` — Health Check
Returns server readiness status, model info, queue depth, and cache statistics.

#### `GET /metrics` — Prometheus Metrics
Exports all metrics in Prometheus text exposition format. Scrape this endpoint with Prometheus.

---

## 🧠 Core Concepts

### Continuous Batching

Traditional static batching processes requests in lockstep, wasting compute capacity if one sequence is shorter than others. miniServe implements **continuous batching**:

1. Sequences are batched dynamically at the **iteration level**.
2. If sequence A finishes in 10 tokens and sequence B takes 100 tokens, sequence A is immediately evicted from the batch.
3. The server immediately slots in sequence C from the wait queue in the very next step, keeping hardware utilization at 100%.

### Safe BPE Streaming

To stream partial tokens without corrupting UTF-8 characters or splitting BPE subwords unexpectedly, miniServe decodes the entire generated sequence state at every step and mathematically computes the string delta to yield to the client. This guarantees flawless emoji and multibyte character streaming.

---

## 🐳 Docker Deployment

### Hugging Face Spaces & Standalone

The project includes an optimized `Dockerfile` specifically designed to be deployed directly to Hugging Face Spaces or any cloud environment:

```bash
# Build the image
docker build -t miniserve:latest -f Dockerfile .

# Run the container
docker run -p 8000:8000 -p 50051:50051 miniserve:latest
```

**Note:** The Dockerfile automatically pre-downloads `Qwen/Qwen2.5-0.5B-Instruct` so the Docker image contains the weights at build time, preventing slow startup times on cloud platforms.

---

## ⚙️ Configuration

All settings are centralized in `config.py` and can be overridden via environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MINISERVE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model identifier |
| `MINISERVE_MAX_RUNNING_SEQUENCES` | `8` | Maximum concurrent generations in continuous batching |
| `MINISERVE_MAX_WAITING_QUEUE_SIZE`| `100` | Maximum backlog of requests before 503 is returned |
| `MINISERVE_MAX_TOKENS` | `50` | Default max tokens per generation |
| `MINISERVE_KV_CACHE_MAX` | `100` | Maximum KV-cache entries before LRU eviction |

---

## 📄 License

This project is licensed under the **MIT License** — free for personal, academic, and commercial use.
