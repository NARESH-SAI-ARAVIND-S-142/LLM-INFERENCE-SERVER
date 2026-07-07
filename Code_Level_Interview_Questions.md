# Hard Code-Level Interview Questions for miniServe

While the previous guides covered the high-level architecture, interviewers love to ask "gotcha" questions to see if you *actually* wrote the code and understand the low-level mechanics of Python and PyTorch. 

Here are the hardest code-specific questions an interviewer might ask about your codebase, and exactly how to answer them.

---

### Q1. "In `continuous_batch_scheduler.py`, I see you used `loop.run_in_executor`. Why didn't you just use `await engine._generate_batch_step()`?"

**The Trap:** Testing if you understand Python's event loop and PyTorch's blocking nature.
**Your Answer:** 
> "Because PyTorch model inference is strictly CPU/GPU bound, not I/O bound. The `_generate_batch_step()` method relies on heavy matrix multiplications using C++ underneath. If I just used `await`, it would still execute on the main thread and block the Python Global Interpreter Lock (GIL) and the `asyncio` event loop. By using `loop.run_in_executor(None, ...)`, I offload the heavy PyTorch compute to a background thread pool, allowing the main thread to immediately return to the FastAPI event loop so it can continue receiving incoming HTTP requests and pushing SSE streams."

---

### Q2. "In `inference_engine.py` during Speculative Decoding, you do `k = k[:, :, :target_len, :]`. What exactly is happening here and why are you slicing dimension 2?"

**The Trap:** Testing your knowledge of PyTorch tensor shapes and Transformer internals.
**Your Answer:** 
> "That line executes the KV-cache rollback if the draft model guesses a token incorrectly. The tensor shape for `past_key_values` in modern transformers is `(batch_size, num_heads, sequence_length, head_dimension)`. Dimension 2 (0-indexed) is the `sequence_length`. If the draft model predicted 3 tokens, but the main model only accepted 1, my KV-cache is polluted with 2 wrong tokens. I calculate `target_len = original_length + accepted_tokens`, and slice the tensor precisely at dimension 2. This instantly truncates the memory back to the exact valid token state without needing to drop the request and start over."

---

### Q3. "In your `sequence.py`, you use an `asyncio.Queue`. How exactly do you bridge the synchronous PyTorch thread and the asynchronous FastAPI stream?"

**The Trap:** Bridging sync and async code in Python is notoriously difficult and error-prone.
**Your Answer:** 
> "Because PyTorch runs in a synchronous background thread pool, it cannot safely call `await queue.put()` natively. To bridge the gap, when the engine generates a token, I use `loop.call_soon_threadsafe(queue.put_nowait, new_token)`. This allows the background PyTorch thread to safely push data into the `asyncio.Queue` belonging to the main event loop. FastAPI then reads from this queue asynchronously via an `async generator` using `yield`, which powers the Server-Sent Events (SSE) streaming."

---

### Q4. "In `inference_engine.py`, you use `torch.nn.functional.pad` on the KV-cache before passing it to the model. Why is this necessary?"

**The Trap:** Testing if you understand the actual mechanics of Continuous Batching.
**Your Answer:** 
> "In Continuous Batching, sequences are dynamically injected into the running batch at different times. Sequence A might have 50 tokens in its KV-cache, while Sequence B just entered and has 0. PyTorch requires all tensors in a batch to be perfectly rectangular. Before I run the forward pass, I find the maximum sequence length in the batch. I then apply *left-padding* with zeros to the KV-caches of the shorter sequences so they match the longest sequence. I also dynamically pad their `attention_mask` with zeros so the model knows to mathematically ignore those empty padded slots during the attention calculation."

---

### Q5. "What happens if a user maliciously asks the model to write an infinite loop of code, and it never emits an EOS (End of String) token? Does your batcher hang forever?"

**The Trap:** Testing your edge-case handling and system resilience.
**Your Answer:** 
> "No, the batcher will not hang. In `continuous_batch_scheduler.py`, after every generation step, the engine checks two things: `if next_token == eos_token_id OR seq.tokens_generated >= seq.max_tokens`. Every incoming request is strictly bounded by a `max_tokens` limit (defaulting to 50 in config, hard-capped at a maximum value). Even if the model gets stuck in a loop, it will violently evict the sequence from the batch as soon as it hits the maximum token limit, freeing the memory for other users."

---

### Q6. "You used `torch.quantization.quantize_dynamic` to get 8-bit precision. Why dynamic instead of static quantization?"

**The Trap:** Testing your understanding of Machine Learning deployment formats.
**Your Answer:** 
> "Static quantization requires running calibration data through the model beforehand to calculate exact clipping ranges for the activations, which is complex and can heavily degrade accuracy for LLMs. Dynamic quantization only converts the weights of the `nn.Linear` layers into `qint8` upfront, while leaving the activations in FP32. During the forward pass, it dynamically quantizes the incoming activations on the fly, multiplies them using fast 8-bit CPU vector math (like AVX2), and converts the output back to FP32. It gave me a 50% memory reduction instantly with almost zero degradation in generation quality, perfect for the Hugging Face CPU Free Tier."
