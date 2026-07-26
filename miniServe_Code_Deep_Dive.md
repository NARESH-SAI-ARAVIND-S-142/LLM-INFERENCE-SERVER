# miniServe — Full Code Deep-Dive (Learn This Cold)

This is based on your ACTUAL code, not the generic README. Read this end to end once,
then re-read just the "Say this in the interview" boxes until they're automatic.

---

## ⚠️ IMPORTANT — Fix this before the interview

Your **resume says GPT-2 (124M)**. Your **actual code** (`config.py`) shows:

```python
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"      # the main model
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"      # a SECOND, smaller model
```

You are running **two models at once** — a 1.5B main model and a 0.5B draft model — for
**speculative decoding** (explained in detail below). This is a genuinely advanced technique.
GPT-2 may have been an early version, but your current code is Qwen-based with speculative
decoding, which is significantly more impressive than the resume lets on.

**Decide now:** tell Deepak the truth — "I started with GPT-2, then upgraded to a Qwen2.5
1.5B main model with a 0.5B draft model for speculative decoding." This is a BETTER story
than the resume, not a worse one. Don't hide it — lead with it.

---

## 1. The big picture — what actually happens, file by file

```
run_server.py
  → server/main.py           (FastAPI app, starts everything)
      → server/inference_engine.py     (loads 2 models: main + draft)
      → server/continuous_batch_scheduler.py  (the loop that runs forever)
          → server/sequence.py          (one object per in-flight request)
          → server/kv_cache.py          (LRU cache, mostly for multi-turn chat)
          → server/metrics.py           (Prometheus counters/gauges)
      → server/grpc_server.py           (same scheduler, gRPC transport)
```

One engine. One scheduler. Two front doors (REST + gRPC). That's the architecture in one line.

---

## 2. `config.py` — the dials that control everything

```python
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SPECULATIVE_K = 3          # draft model proposes 3 tokens ahead
MAX_RUNNING_SEQUENCES = 8  # batch size cap
MAX_WAITING_QUEUE_SIZE = 100
KV_CACHE_MAX_ENTRIES = 100
```

**Say this in the interview:** "Everything is centralized in config.py with environment
variable overrides — so I can change the model, batch size, or cache size without touching
code, just by setting `MINISERVE_MODEL` etc."

---

## 3. `server/inference_engine.py` — the actual model-running code

### 3a. What loads at startup (`__init__`)

```python
self.tokenizer.padding_side = "left"          # LEFT padding, not right — see §5
self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.float32).to(self.device)
self.draft_model = AutoModelForCausalLM.from_pretrained(self.draft_model_name, ...)

if self.device == "cpu":
    self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
    self.draft_model = torch.quantization.quantize_dynamic(self.draft_model, {torch.nn.Linear}, dtype=torch.qint8)
```

**Two things happening, both worth knowing cold:**

1. **Two models loaded, not one.** The 1.5B "main" model is the accurate one. The 0.5B
   "draft" model is smaller/faster and used to *guess* ahead (§6).
2. **Dynamic INT8 quantization on CPU.** `torch.quantization.quantize_dynamic` converts the
   `nn.Linear` layers' weights from float32 to int8 **after training** (post-training
   quantization), at load time. This shrinks memory ~4x and speeds up CPU matrix multiplies,
   at a small accuracy cost. It's applied *only on CPU* — GPUs have better native fp16/bf16
   paths so this specific trick is a CPU-serving optimization.

**Say this in the interview:** "Since I'm serving on CPU, I apply dynamic INT8 quantization
to the Linear layers at load time — it roughly quarters the memory footprint of the weight
matrices and speeds up matmuls, which matters a lot given CPU is already the bottleneck
resource here."

### 3b. `generate_batch()` — the simple, non-streaming path

This is the **older, simpler** code path (used by `generate_single`/`generate_batch` directly,
NOT by the continuous batching scheduler — that's `_generate_batch_step`, see §4). It just
calls HuggingFace's built-in `model.generate(...)`:

```python
encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

outputs = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_tokens,
    do_sample=(temperature > 0),
    use_cache=True,   # KV-cache ON inside generate()
)

# Strip off the prompt, keep only new tokens
new_token_ids = outputs[i][input_ids.shape[1]:]
```

**Key detail:** `outputs[i][prompt_length:]` — since `generate()` returns prompt + generated
tokens concatenated, you slice off everything before `prompt_length` to get just the new text.

This path is what you'd use for a quick single-shot benchmark. It's **not** what runs during
live continuous batching — that's a hand-written loop (§4), which is the harder, more
interview-relevant code.

---

## 4. `_generate_batch_step()` — YOUR hand-written generation loop (the important one)

This is the function the scheduler calls **once per iteration**. It does NOT call
`model.generate()` — you wrote the prefill/decode loop yourself. This is the part to know
in the most depth.

### 4a. Splitting sequences into two groups

```python
prefill_seqs = [s for s in sequences if s.past_key_values is None]   # brand new requests
decode_seqs  = [s for s in sequences if s.past_key_values is not None]  # already generating
```

Every step, some sequences in the batch are just starting (need a full prefill pass over
their prompt) and some are mid-generation (only need one new token processed, reusing their
cache). Both groups get **separate batched forward passes** in the same step.

### 4b. Prefill — first pass for new sequences

```python
encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
next_token_logits = outputs.logits[:, -1, :]     # only care about the LAST position
next_tokens = self._sample(next_token_logits, prefill_seqs)
```

Then it does the **same prefill on the draft model too** (`draft_outputs = self.draft_model(...)`)
— because speculative decoding needs the draft model's cache primed and ready from token 1,
not just the main model's.

`seq.past_key_values = unbatched_pkv[i]` — after this, the sequence now "has a cache," so on
the *next* scheduler iteration it moves from `prefill_seqs` to `decode_seqs`.

**Say this in the interview:** "A sequence's very first step is always a prefill — full
prompt in, one token out, cache created. Every step after that is decode — reusing the cache.
My code checks `past_key_values is None` to route each sequence to the right path every
single iteration, since the batch composition changes constantly."

### 4c. `_unbatch_pkv()` and `_build_cache()` — the trickiest plumbing in the file

Here's the subtlety that's easy to get wrong and great to explain if asked:

- The model's forward pass processes a **batch** of sequences together, so its output
  `past_key_values` is one big batched tensor: shape `[batch_size, heads, seq_len, head_dim]`.
- But each `Sequence` object needs to hold **its own individual** cache, because next
  iteration the batch composition might be completely different (some evicted, new ones
  admitted).
- `_unbatch_pkv()` slices the batched cache tensor back into N separate per-sequence caches:
  `k = batched_pkv[layer][0][i:i+1]` — grabs just row `i` for sequence `i`, keeping the batch
  dimension as size-1.
- Later, `_build_cache()` does the **reverse**: takes N separate per-sequence caches and
  concatenates/pads them back into one batched tensor for the next forward pass.

**Say this in the interview:** "Because the batch's membership changes every iteration, I
can't just keep one shared cache tensor — I unbatch it into per-sequence pieces after every
forward pass, store those on each Sequence object, then re-batch whichever sequences are
selected for the *next* step. That re-batching step is also where padding happens, since
different sequences' caches can be different lengths."

---

## 5. Padding — why left-padding, and how the mask handles it

```python
self.tokenizer.padding_side = "left"
```

**Why left, not right:** with left-padding, every sequence's most recent real token always
lands in the **same final column** of the tensor, regardless of how long that sequence's
cache has grown. This means `outputs.logits[:, -1, :]` always correctly grabs "the next-token
prediction for every sequence" in one clean slice — no per-sequence index bookkeeping needed.
Right-padding would put the real "current" token at a different column for every sequence.

In the speculative decoding code, you can see this explicitly:

```python
pad_len = max_pkv_len - key.size(2)
key = torch.nn.functional.pad(key, (0, 0, pad_len, 0))   # pad on the LEFT of the seq_len dim
...
mask = [0]*pad_len + base_masks[i] + [1]                  # 0s for padding, 1s for real tokens
```

**Say this in the interview:** "I pad on the left specifically so the last column always
holds each sequence's true most-recent token — that keeps `logits[:, -1, :]` valid across
the whole batch without extra indexing logic."

---

## 6. Speculative decoding — your most advanced feature (know this deeply)

This is genuinely the most interview-impressive part of your code, and it's **not** in the
generic README at all — it's something you (or a previous version of you) implemented on
top of the basic idea. Let's build the intuition first, then the mechanics.

### The plain-English idea

Normal decoding: the big, accurate 1.5B model generates one token, then another, then
another — one at a time, each one costing a full (memory-bandwidth-bound) forward pass.

Speculative decoding's trick: use a **small, fast draft model** (0.5B) to *guess* several
tokens ahead cheaply, THEN run the big model **once** to check all those guesses **in
parallel**. Any guesses that match what the big model would have said anyway are accepted
for free. The first guess that's wrong gets thrown away, and the big model's own correct
token is used instead.

**Analogy:** Imagine you're translating a document. Instead of translating word-by-word
yourself (slow, careful), you let a fast-but-imperfect assistant draft a whole paragraph
ahead. You then skim it once and only stop to fix wherever the assistant guessed wrong.
You still get a fully correct translation, but you did much less "slow, careful" work per
correct word — as long as the assistant is right often enough.

### Why this helps with the memory-bandwidth-bound problem specifically

Recall: decoding one token normally means moving the ENTIRE big model's weights through
memory just to produce ONE token. That's incredibly wasteful — huge memory movement for
tiny output. Speculative decoding's real trick is: **run the big model once, but verify K
tokens at once instead of generating 1** — so you get up to K tokens for the cost of roughly
one big-model memory-bandwidth pass, instead of K passes. Your `SPECULATIVE_K = 3` means:
draft proposes 3 tokens ahead, main model checks all 3 in one pass.

**Say this in the interview (this is a great line):** "The core bottleneck in decoding is
memory bandwidth — you move the full model's weights for every single token. Speculative
decoding amortizes that cost: the big model does one pass but verifies K candidate tokens
simultaneously, so you can get multiple tokens per expensive memory-bandwidth pass instead
of one."

### The three phases in your code

**Phase 1 — Drafting** (`for k in range(K)`): the small draft model generates K tokens in a
normal one-at-a-time loop (fast, since it's a small model), building up `draft_tokens_per_seq`.

**Phase 2 — Verification**: the K draft tokens are all fed into the **main** model **at once**
as a batch of K positions:
```python
inputs = [seq.input_ids[-1]] + draft_tokens_per_seq[i][:-1]   # feed all K draft tokens together
main_outputs = self.model(input_ids=main_input_ids, ...)      # ONE forward pass checks all K
seq_logits = main_outputs.logits[i]   # shape (K, vocab) — one prediction per draft position
```
The main model computes what token IT would have picked at each of those K positions.

**Phase 3 — Acceptance & rollback**:
```python
n = 0
for j in range(K-1):
    if draft_toks[j] == main_toks[j]:
        n += 1          # draft guessed right — accept it
    else:
        break            # first mismatch — stop accepting
accepted_tokens = draft_toks[:n] + [main_toks[n]]   # accepted guesses + the main model's correction
```
Whichever prefix of draft tokens matches the main model's own predictions gets accepted "for
free." The moment there's a mismatch, everything after is discarded and replaced by the main
model's actual token — this **guarantees the output is identical to what plain main-model
decoding would have produced**, just potentially faster.

Then the KV-caches (both draft and main) get **sliced back down** (`slice_pkv`) to the actual
accepted length, since they were built assuming all K tokens would be accepted.

**Say this in the interview:** "Acceptance is a simple prefix match — I walk the draft
tokens against what the main model independently predicted at each position, accept the
matching prefix, and take the main model's own token as the correction at the first
mismatch. That's what guarantees the output distribution is mathematically identical to
standard decoding — speculative decoding is a *speedup* technique, not an approximation."

### Metrics tracking this exact thing

```python
metrics.SPECULATIVE_TOKENS_GENERATED.inc(K-1)
metrics.SPECULATIVE_TOKENS_ACCEPTED.inc(n)
```
This lets you compute an **acceptance rate** — the fraction of draft guesses that turned out
correct. High acceptance rate = draft model is well-matched to the main model = big speedup.
Low acceptance rate = you're wasting time drafting tokens that just get thrown away.

**If asked "what determines acceptance rate":** the draft and main model need similar
"opinions" — typically the draft model is a smaller version of the same family (Qwen 0.5B
drafting for Qwen 1.5B, as in your code), so their token distributions are correlated.

---

## 7. `server/kv_cache.py` — the LRU cache manager

This is a **separate, simpler cache** from the per-sequence caches used during active
generation (§4c) — it's for **reusing state across separate requests**, e.g. multi-turn
conversations where a `request_id` returns later.

```python
self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()

def get(self, request_id):
    if request_id in self._cache:
        self._cache.move_to_end(request_id)   # mark as "recently used"
        return entry
    return None   # cache miss

def put(self, request_id, past_key_values, prompt_tokens):
    while len(self._cache) >= self.max_entries:
        self._evict_lru()    # pop the LEAST recently used (front of dict)
    self._cache[request_id] = entry
```

**Why `OrderedDict` specifically:** it preserves insertion order AND supports O(1)
`move_to_end()`. Every cache hit moves that entry to the end (most-recently-used position).
Eviction always pops from the **front** (`popitem(last=False)`) — which is guaranteed to be
the least-recently-used entry, since anything used recently was just moved to the end. This
is the standard, clean way to implement LRU without a separate linked list.

**Say this in the interview:** "I used Python's OrderedDict because it gives O(1) move-to-end
and pop-from-front, which is exactly what LRU needs — no need for a custom doubly-linked-list
implementation."

**Memory estimate function:**
```python
total_elements = sum(k.numel() + v.numel() for k, v in past_key_values)
return (total_elements * 4) / (1024 * 1024)   # 4 bytes/float32, convert to MB
```
This is literally the formula we discussed earlier — element count × 4 bytes, converted to MB.
Good to connect this: "this is the exact same calculation I derived — num_layers × 2 (K,V) ×
heads × head_dim × seq_len × 4 bytes."

---

## 8. `server/continuous_batch_scheduler.py` — the loop that runs forever

This is the background `asyncio` task started at server startup that never stops until shutdown.

```python
async def _step_loop(self):
    while self._running:
        # 1. Admit new sequences from waiting queue until max_running (8) is hit
        while len(self.running_sequences) < self.max_running:
            if self.waiting_queue.empty(): break
            seq = self.waiting_queue.get_nowait()
            self.running_sequences[seq.request_id] = seq

        # 2. If nothing running, sleep/wait for new work
        if not self.running_sequences:
            await asyncio.wait_for(self._new_request_event.wait(), timeout=1.0)
            continue

        # 3. Timeout check — evict sequences running too long
        # (skipped here, see code — 30s default timeout)

        # 4. Run ONE step for the whole batch (in an executor, off the event loop)
        await loop.run_in_executor(None, lambda: self.engine._generate_batch_step(sequences_to_step))

        # 5. Push newly generated tokens into each sequence's stream_queue (for SSE)
        for seq in sequences_to_step:
            if seq.stream_queue is not None and seq.latest_token_text:
                seq.stream_queue.put_nowait(seq.latest_token_text)

        # 6. Evict anything finished
        finished = [s for s in sequences_to_step if s.is_finished or s.error]
        for seq in finished:
            self._evict_sequence(seq)
```

**This IS the continuous batching definition, in code.** Step 6 evicts finished sequences
immediately after this iteration; step 1 (next loop iteration) immediately backfills from the
waiting queue. No fixed "wait for whole batch" behavior anywhere — the composition of
`running_sequences` can be completely different from one iteration to the next.

**Why `run_in_executor`, specifically:** the actual model forward pass
(`engine._generate_batch_step`) is **synchronous, CPU-heavy PyTorch code**. If you called it
directly inside the async loop, it would **block the entire event loop** — no other requests
could even be admitted or have their SSE tokens delivered while the model runs. Running it in
an executor (a thread pool) lets the async event loop stay responsive to new connections while
the actual computation happens on a separate thread.

**Say this in the interview:** "The scheduler's step loop is async, but model inference itself
is blocking, synchronous PyTorch code. I run it via `run_in_executor` so the event loop isn't
frozen during the forward pass — new requests can still be accepted and SSE tokens can still
be flushed to already-connected clients while a batch step is computing."

### `submit()` — how a request enters the system

```python
async def submit(self, request, stream=False):
    future = loop.create_future()
    stream_queue = asyncio.Queue() if stream else None
    seq = Sequence(..., future=future, stream_queue=stream_queue)
    self.waiting_queue.put_nowait(seq)
    self._new_request_event.set()      # wake up the scheduler if it was sleeping

    if stream:
        return self._stream_generator(seq)    # returns immediately, an async generator
    return await future                        # BLOCKS here until scheduler resolves it
```

Two different waiting mechanisms for two different client types:
- **Non-streaming:** the caller `await`s a `Future` directly — the scheduler resolves it
  (`seq.future.set_result(...)`) once the sequence finishes completely.
- **Streaming:** the caller gets an async generator immediately, which pulls tokens one at a
  time off `seq.stream_queue` as the scheduler pushes them in, until it sees `None` (the
  sentinel meaning "done").

---

## 9. `server/main.py` — REST layer, thin on purpose

```python
@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    response_or_gen = await scheduler.submit(request, stream=request.stream)
    if request.stream:
        return StreamingResponse(response_or_gen, media_type="text/event-stream")
    return GenerateResponse(...)
```

Notice how little logic is here. The handler's entire job is: validate input → hand off to
`scheduler.submit()` → format whatever comes back. **All the real work happens in the
scheduler and engine.** This is intentional and exactly what you should say if asked "why is
your API layer so thin": "the gateway is deliberately dumb — it's just a translation layer
between HTTP and the scheduler's queue, so REST and gRPC can share identical business logic."

Startup (`lifespan`) does two things in order: load the engine (`InferenceEngine()` — this is
where both models load and quantize), then start the scheduler's background loop
(`await scheduler.start()`). Both happen once, before the server starts accepting traffic.

---

## 10. Metrics — what's actually being tracked and why each one matters

From `server/metrics.py`, the ones worth being able to explain:

- **`INFERENCE_LATENCY` / `STEP_LATENCY`** — histograms with specific bucket boundaries
  (0.01s to 10s) so Prometheus can compute p50/p95/p99 without storing every raw sample.
- **`QUEUE_WAIT_TIME`** — separately tracks time spent waiting vs time spent actually
  generating — lets you diagnose "is my server slow because of queuing (overloaded) or
  slow because generation itself is slow (model/hardware limited)."
- **`SPECULATIVE_TOKENS_GENERATED` / `SPECULATIVE_TOKENS_ACCEPTED`** — lets you compute
  acceptance rate live, in production, not just in offline benchmarks.
- **`SCHEDULER_HEARTBEAT`** (a Gauge set to `time.monotonic()` every loop iteration) — a
  classic liveness signal: if this stops increasing, the scheduler loop has died/hung, even
  if the process is still technically running.

**Say this in the interview if asked "why track queue wait separately from inference time":**
"So I can tell overload from slowness — if queue wait is high but inference latency is
normal, I need more capacity or bigger max_running; if inference latency itself is high,
that's a model/hardware problem, not a scheduling one."

---

## 11. The full request lifecycle, now with real function names

1. Client → `POST /v1/generate` → `server/main.py:generate()`
2. → `scheduler.submit(request, stream=...)` → builds a `Sequence`, pushes to `waiting_queue`,
   sets `_new_request_event`
3. `_step_loop()` (already running in the background) wakes up, pulls from `waiting_queue`
   into `running_sequences` up to `max_running=8`
4. Every iteration: `engine._generate_batch_step(sequences_to_step)` runs — splits into
   prefill/decode groups, does the (possibly speculative) forward pass(es), samples next
   tokens, updates each `Sequence`'s `input_ids`, `past_key_values`, `generated_text`
5. New token text pushed to `seq.stream_queue` → `_stream_generator()` yields it as an SSE
   `data: {...}` line → client sees it
6. When `seq.is_finished` (EOS or `max_tokens` hit) → `_evict_sequence()` — moves cache to
   CPU, resolves the `future` or pushes `None` to end the stream, removes from
   `running_sequences`
7. Next loop iteration immediately backfills the freed slot from `waiting_queue`

---

## 12. Quick-reference: hardest likely questions and your best answers

**"Why two models loaded?"**
→ Speculative decoding — 0.5B draft model proposes tokens fast, 1.5B main model verifies
them in one batched pass, guaranteeing identical output to plain decoding while using less
memory bandwidth per accepted token.

**"Why left-padding?"**
→ Keeps every sequence's current token in the same last column so `logits[:, -1, :]` works
uniformly across the batch without per-sequence indexing.

**"What's the difference between `generate_batch()` and `_generate_batch_step()`?"**
→ `generate_batch()` calls HuggingFace's built-in `.generate()` — simple, used for one-shot/
benchmark calls. `_generate_batch_step()` is my own hand-written prefill/decode loop, used by
the continuous batching scheduler, because I need fine-grained control over per-sequence
state (mixing prefill and decode sequences in one batch, speculative decoding, mid-generation
eviction) that `.generate()` doesn't expose.

**"Why run inference in an executor instead of directly in the async loop?"**
→ Model inference is blocking, synchronous PyTorch code — running it directly would freeze
the event loop and block new connections/SSE delivery. `run_in_executor` moves it to a thread
pool so the loop stays responsive.

**"How does the LRU cache work mechanically?"**
→ `OrderedDict` — `get()` calls `move_to_end()` on hit, `put()` evicts via `popitem(last=False)`
which is always the least-recently-used entry, since recently accessed entries get moved to
the end. O(1) for both operations.

**"What guarantees speculative decoding doesn't change the output?"**
→ The main model independently computes what IT would have generated at every draft position.
Acceptance is a strict prefix match against that — any draft token that doesn't match the
main model's own prediction is discarded and replaced, so the final output is provably
identical to what plain main-model-only decoding would have produced.

---

## What to do next

Read section 6 (speculative decoding) and section 4 (the hand-written loop) twice more —
those are your strongest, most differentiated material and very likely where Deepak spends
the most time once he sees them in your code. Everything else in this doc is solid supporting
detail, but those two sections are your headline.
