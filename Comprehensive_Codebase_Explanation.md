# The Complete "Pin-to-Pin" Codebase Explanation for Interviews

This document is your ultimate script for an interview. It explains the entire `miniServe` codebase line-by-line, file-by-file, in a logical order so you can explain exactly how the architecture works to a Senior Engineer or Hiring Manager.

When explaining a complex system in an interview, **always start from the data structures (the bottom), move to the engine, then the scheduler, and finally the API (the top).**

---

## 1. `config.py` (The Central Configuration)
**What to say in an interview:**
"I hate hardcoding values. In `config.py`, I centralized all parameters using `os.getenv`. This allows the server to be completely configurable via Docker environment variables without changing code. It handles the `MODEL_NAME`, `DRAFT_MODEL` for speculative decoding, maximum batch sizes, and port bindings."

---

## 2. `server/sequence.py` (The Data Structure)
**What to say in an interview:**
"The core data structure of my continuous batcher is the `Sequence` dataclass. When a user request arrives, it is instantly converted into a `Sequence` object."
* **State Tracking:** It holds `input_ids`, `attention_mask`, and `past_key_values`. 
* **Async Communication:** Crucially, it holds an `asyncio.Queue` and an `asyncio.Future`. As the PyTorch engine generates tokens in the background thread, it pushes them into this specific Sequence's queue. The FastAPI layer simply awaits this queue to stream tokens to the user via Server-Sent Events (SSE).
* **Metrics:** It records `arrival_time` and `start_time` so we can accurately measure queue starvation and Time-To-First-Token (TTFT).

---

## 3. `server/kv_cache.py` (Memory Management)
**What to say in an interview:**
"To prevent Out-Of-Memory (OOM) crashes, I built a `KVCacheManager`. It enforces a strict upper limit on the number of cached sequences."
* **How it works:** It acts as an LRU (Least Recently Used) cache or strict boundary. If the cache is full and a new request arrives, the request is forced to wait in the queue. When a sequence finishes generating, `kv_cache.py` evicts its Keys and Values, freeing up exactly enough RAM to allow the next sequence in.

---

## 4. `server/inference_engine.py` (The Core Machine Learning Logic)
**What to say in an interview:**
"This is the heaviest file in the project. It entirely replaces the standard `model.generate()` function with a custom loop to support Continuous Batching and Speculative Decoding."

**Pin-to-Pin Breakdown:**
* **`__init__`**: 
  * I load two models: A 1.5B Main model and a 0.5B Draft model. 
  * **Quantization**: Because the Hugging Face Free Tier CPU only has 16GB of RAM, I apply `torch.quantization.quantize_dynamic` to squash the `nn.Linear` layers into `qint8` (8-bit integers). This cuts memory usage in half and dramatically speeds up CPU matrix multiplications.
* **`_generate_batch_step`**: 
  * **The Split**: Every iteration, it splits the incoming batch into `prefill_seqs` (brand new requests) and `decode_seqs` (requests already generating).
  * **Prefill**: Runs a standard forward pass for new prompts to generate their first token and establish their initial `past_key_values`.
  * **Decode (Speculative Decoding)**:
    1. **Draft Phase**: I loop the tiny Draft model $K$ times (e.g., 3 times) to predict the next 3 tokens rapidly.
    2. **Padding**: Because this is a continuous batch, sequences are of different lengths. I dynamically calculate the `max_seq_len` and left-pad the `past_key_values` and `attention_mask` tensors with zeros so PyTorch can process them as one clean rectangular batch.
    3. **Verification Phase**: I pass the original token plus the 3 draft guesses into the 1.5B Main model as a sequence of length 4. The Main model verifies all of them in parallel.
    4. **Acceptance & Rollback**: I compare the Main model's output to the Draft model's output. If the Draft model guessed wrong on token 3, I accept tokens 1 and 2, and dynamically **slice** the `past_key_values` tensors (`k[:, :, :target_len, :]`) to roll the memory state back. This fixes the cache corruption without dropping the sequence.

---

## 5. `server/continuous_batch_scheduler.py` (The Traffic Cop)
**What to say in an interview:**
"Because PyTorch matrix math blocks the CPU, I had to decouple the API from the ML engine."

**Pin-to-Pin Breakdown:**
* **The Background Loop**: I create an infinite `_step_loop` running inside a separate thread pool (`run_in_executor`).
* **Admission Control**: Every step, the scheduler looks at the `waiting_queue`. If there is space in the batch (e.g., less than 8 running sequences) and space in the KV-cache, it pulls waiting sequences and upgrades them to running sequences.
* **Execution**: It hands the batch to `inference_engine._generate_batch_step()`.
* **Eviction**: It inspects the sequences after the step. If any sequence hit the `eos_token` (End of String) or max tokens, it marks it finished, evicts it from the batch, and cleans up the memory. 

---

## 6. `server/main.py` (The REST API)
**What to say in an interview:**
"I used FastAPI for the public-facing REST endpoints. Its job is purely routing and input validation using Pydantic."

**Pin-to-Pin Breakdown:**
* **Lifespan**: Uses FastAPI's modern `@asynccontextmanager` to load the `InferenceEngine` and start the `ContinuousBatchScheduler` in the background immediately when the server boots.
* **`/v1/generate` endpoint**: 
  * Accepts a JSON payload, creates a `Sequence` object, and calls `await scheduler.submit()`.
  * **Streaming**: If the user requests `stream=True`, I use `StreamingResponse` to read from the sequence's internal `asyncio.Queue` and yield tokens back to the user via Server-Sent Events (SSE). This provides the real-time ChatGPT-like typing effect.

---

## 7. `server/grpc_server.py` (The Internal Microservice API)
**What to say in an interview:**
"REST is great for browsers, but JSON serialization overhead is terrible for backend microservices. I built a gRPC server using Protobufs for high-speed internal communication."

**Pin-to-Pin Breakdown:**
* It implements the exact same logic as `main.py`, but uses `grpc.aio.server()`.
* Instead of JSON, it decodes highly compressed binary Protobuf messages defined in `proto/inference.proto`. 
* It interacts with the exact same central `scheduler` instance, proving that the architecture is perfectly decoupled.

---

## 8. `server/metrics.py` (Observability)
**What to say in an interview:**
"You can't optimize a system you can't measure. I instrumented the entire codebase using the `prometheus_client`."

**Pin-to-Pin Breakdown:**
* I defined `Counter`, `Histogram`, and `Gauge` objects.
* I track `Tokens Per Second`, `Inference Latency`, `Queue Wait Time`, and critically, the `Speculative Acceptance Rate` (to measure how accurate the Draft model actually is).
* These metrics are exposed on the `/metrics` endpoint to be scraped by Grafana dashboards.

---

## 9. `run_server.py` (The Bootloader)
**What to say in an interview:**
"Finally, because I have both a FastAPI server and an async gRPC server, I wrote a master bootloader script. It uses `asyncio.gather()` to launch the Uvicorn web server and the gRPC binary server concurrently on different ports, tying the whole microservice together."

---

### Interview Strategy Tip
If they ask you what the hardest part of the project was, **point them to Phase 4 (Inference Engine)**. Explain that managing PyTorch tensor shapes (padding KV-caches dynamically) and handling the KV-cache rollback logic for Speculative Decoding took the most architectural forethought. This proves you understand low-level memory mechanics.
