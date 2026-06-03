# FAANG Interview Guide: Defending the miniServe Architecture

When interviewing at top-tier tech companies (Meta, Google, Amazon, OpenAI) for an AI Systems Engineer or ML Infra role, interviewers don't just care that your code works. They care about **why** you built it that way, what **trade-offs** you made, and how deeply you understand the **low-level mechanics** of the hardware and algorithms.

This guide maps FAANG-level interview questions to each phase of your `miniServe` project, providing the strategic approach and the ideal technical response.

---

## Phase 1: Architecture & Concurrency

### Q1: "Why did you use `asyncio.Queue` and a background thread for inference instead of just making the FastAPI endpoint `async def`?"
* **The Trap:** Junior engineers think `async` makes Python execute things in parallel. It does not.
* **How to Approach:** Demonstrate your understanding of Python's Global Interpreter Lock (GIL) and the difference between I/O-bound and CPU-bound tasks.
* **Ideal Answer:** "FastAPI and `asyncio` are great for I/O-bound tasks (like waiting for a database), but PyTorch matrix multiplications are heavily CPU-bound. If I ran the PyTorch forward pass directly inside an `async def` endpoint, it would block the entire event loop, meaning the server couldn't even accept new HTTP requests. By decoupling the API layer from the ML layer using an `asyncio.Queue` and running the inference engine in a separate thread pool (`run_in_executor`), I ensured the web server remained non-blocking and highly responsive to incoming traffic."

### Q2: "You implemented both REST and gRPC. When would you use one over the other in a microservices architecture?"
* **How to Approach:** Discuss serialization overhead and human-readability.
* **Ideal Answer:** "REST with JSON is excellent for public-facing APIs because it's human-readable, stateless, and easily consumable by browsers using SSE. However, JSON serialization/deserialization is slow and bulky. For internal service-to-service communication, I implemented gRPC. gRPC uses Protobufs, which are tightly packed binary formats, and runs over HTTP/2, allowing for persistent, multiplexed connections. This drastically reduces network overhead and latency in a high-throughput backend environment."

---

## Phase 2: Continuous Batching & Scheduling

### Q3: "Explain the fundamental difference between Static Batching and Continuous Batching. What is the mathematical advantage?"
* **How to Approach:** Explain the "wasted compute" problem in static batching using the concept of sequence lengths.
* **Ideal Answer:** "In static batching, requests are grouped together, and the batch isn't finished until the *longest* sequence in the batch emits an EOS token. If one sequence is 10 tokens and another is 100 tokens, the 10-token slot is 'locked' but essentially doing useless padded compute for 90 iterations. Continuous Batching operates at the iteration level. After every single forward pass, the scheduler inspects the batch. If a sequence finishes, it is evicted immediately, its memory is freed, and a new sequence from the queue is dynamically injected into the batch for the very next iteration. This turns an uneven, ragged batch into a dense computational block, significantly increasing total tokens-per-second throughput."

### Q4: "How do you handle queue starvation in your continuous batcher? What if a massive request blocks smaller ones?"
* **How to Approach:** Discuss fairness and scheduling algorithms (e.g., FCFS vs Round Robin).
* **Ideal Answer:** "Currently, my scheduler uses a First-Come-First-Serve (FCFS) queue. However, to prevent a massive request from monopolizing the batch indefinitely, I implemented a strict `MAX_SEQUENCE_TIMEOUT_S` and `MAX_NEW_TOKENS` limit. In a more advanced scenario, I would implement token-level preemption—if a high-priority request arrives, we can swap out a running sequence's KV-cache to CPU RAM, run the priority request, and then swap the original sequence back in. This ensures fairness and bounds the maximum wait time for any request."

---

## Phase 3: Memory Management (The KV-Cache)

### Q5: "What is the Big-O time complexity of the Attention mechanism, and exactly why does a KV-cache solve this?"
* **How to Approach:** Break down the math. FAANG interviewers love mathematical fundamentals.
* **Ideal Answer:** "Standard self-attention has a time complexity of $O(N^2 \cdot D)$, where $N$ is the sequence length and $D$ is the embedding dimension. Every time you generate a new token, the model normally has to recalculate the Key and Value vectors for all $N$ previous tokens to compute the attention weights. A KV-cache stores the previously computed Key and Value matrices in memory. During generation, the new token only computes its *own* Query, Key, and Value, and attends to the cached KV matrices. This reduces the time complexity of generating the next token from $O(N^2)$ down to $O(N)$, trading memory for compute."

### Q6: "How did you manage the memory for sequences of completely different lengths inside the same batch?"
* **How to Approach:** Explain tensor manipulation (padding) and cache memory management.
* **Ideal Answer:** "Because sequences are dynamically injected, their KV-caches are at different lengths. Before passing them into PyTorch, I have to intercept the caches. I find the `max_seq_len` of the current active batch, and I dynamically apply left-padding (`torch.nn.functional.pad`) with zeros to the Keys and Values of the shorter sequences so they form a perfect rectangular tensor. I also have to dynamically adjust the `attention_mask` so the model ignores those zero-padded KV elements."

---

## Phase 4: Extreme Optimization (Speculative Decoding)

### Q7: "Why did you implement Speculative Decoding? Why couldn't you just make the main model run faster?"
* **How to Approach:** Show that you understand the difference between Compute-Bound and Memory-Bandwidth Bound operations. This is the "golden ticket" concept for ML Systems roles.
* **Ideal Answer:** "Autoregressive generation (batch size 1) is fundamentally **Memory-Bandwidth Bound**, not compute-bound. The CPU spends more time fetching the massive 1.5B model weights from RAM into the CPU cache than it does performing the actual matrix multiplications. Because Transformers can process multiple tokens in parallel, passing 1 token vs passing 3 tokens takes almost the exact same amount of time. Speculative decoding exploits this. By using a tiny 0.5B draft model to quickly generate 3 guesses, I pass all 3 guesses to the main model at once. If the guesses are right, I get 3 tokens for the latency cost of 1 memory fetch. It bypasses the memory-bandwidth bottleneck by increasing the arithmetic intensity (compute per byte transferred)."

### Q8: "What happens if the draft model is wrong? Doesn't that corrupt the KV-cache?"
* **How to Approach:** Explain state rollback and tensor slicing.
* **Ideal Answer:** "Yes, it would, which is why manual KV-cache rollback is critical. If the draft model generates 3 tokens, and the main model accepts the first 2 but rejects the 3rd, the current KV-cache of both models is now polluted with the rejected 3rd token. To fix this, I calculate the target valid sequence length ($L_{valid}$) and slice the `past_key_values` tensors: `k = k[:, :, :target_len, :]`. This instantly truncates the memory back to the exact valid state, allowing the model to smoothly continue generating without having to restart the context."

---

## Phase 5: Hardware Optimization (Quantization)

### Q9: "Why did you choose `qint8` Dynamic Quantization over FP16 or FP32?"
* **How to Approach:** Discuss memory footprint vs CPU architecture support.
* **Ideal Answer:** "On a constrained Hugging Face CPU Free Tier, 16GB of RAM is the hard limit. The 1.5B model in FP32 would consume ~6GB, and the draft model would consume another ~2GB, leaving barely any room for the KV-cache or OS overhead. Furthermore, standard CPUs do not typically have hardware acceleration for FP16 math (unlike GPUs). By converting the `nn.Linear` weights to `qint8` integers, I halved the memory footprint immediately. Dynamic quantization keeps the activations in FP32 but quantizes weights to INT8, providing a massive speedup on CPU vector instruction sets (like AVX2/AVX-512) while maintaining generation quality."

### Q10: "If I gave you an infinite budget and a cluster of NVIDIA H100 GPUs, how would you change this architecture?"
* **How to Approach:** Show that you understand scalability and distributed systems.
* **Ideal Answer:** "If I had H100s, I would immediately drop manual CPU quantization and switch to a highly optimized backend like `vLLM` or `TensorRT-LLM`. I would replace my manual KV-cache padding with **PagedAttention**, which stores KV blocks in non-contiguous GPU memory pages to completely eliminate memory fragmentation. Finally, I would deploy the models using Tensor Parallelism across multiple GPUs, and shard the API layer using a load balancer (like HAProxy or Nginx) routing requests to a fleet of inference workers."
