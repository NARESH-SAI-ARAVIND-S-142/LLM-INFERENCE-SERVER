# How to Truly Learn and Own This Codebase

The biggest mistake engineers make is reading code top-to-bottom like a book. Code doesn't flow like a story—it flows like a **phone call chain**. To truly understand this project, you need to trace the path a single user request takes from the moment it hits the server to the moment a token lands in their browser.

This guide gives you a **5-day learning plan** with the exact reading order, exercises, and mental models.

---

## Day 1: Understand the Data Flow (Don't Read Code Yet)

Before opening any file, draw this diagram on paper:

```
User sends HTTP POST
       │
       ▼
  main.py (FastAPI)
    Converts JSON → Sequence object
       │
       ▼
  continuous_batch_scheduler.py
    Puts Sequence into a waiting queue
    Background loop picks it up
       │
       ▼
  inference_engine.py
    Runs PyTorch forward pass
    Pushes generated tokens into Sequence.queue
       │
       ▼
  main.py (FastAPI)
    Reads from Sequence.queue
    Streams tokens back to user via SSE
```

**Exercise:** Without looking at any code, write down in plain English what you think happens when two users send a request at the exact same time. Where do they wait? Who decides when they get processed?

---

## Day 2: Read the Foundation Files (Bottom-Up)

Read these files in this exact order. Start from the simplest building blocks.

### File 1: `config.py`
- **Time:** 10 minutes
- **What to focus on:** Every setting that controls the server's behavior. Notice how everything uses `os.getenv()` with sensible defaults.
- **Exercise:** Change `MAX_BATCH_SIZE` from 8 to 2 in your head. What would happen to throughput? What about latency per user?

### File 2: `server/sequence.py`
- **Time:** 15 minutes
- **What to focus on:** This is the "passport" of every user request. Every field in this dataclass exists for a reason.
- **Exercise:** For each field (`input_ids`, `attention_mask`, `past_key_values`, `output_queue`, `future`), write one sentence explaining why it exists. If you can't, that's the gap you need to fill.

### File 3: `server/kv_cache.py`
- **Time:** 20 minutes
- **What to focus on:** How it stores and retrieves `past_key_values`. Think of it as a dictionary with a strict size limit.
- **Exercise:** What happens when the cache is full and a new user arrives? Trace the code path.

---

## Day 3: Read the Core Engine (The Hardest Part)

### File 4: `server/inference_engine.py`
- **Time:** 2 hours (yes, this file deserves it)
- **Read it in 4 passes:**

**Pass 1 — The `__init__` method (lines ~60-120):**
- How are the two models loaded?
- Where does quantization happen?
- **Exercise:** Comment out the quantization lines in your head. How much more RAM would the server use?

**Pass 2 — The prefill logic:**
- When a brand-new prompt arrives, how does the engine process it for the first time?
- **Key concept:** "Prefill" means running the entire prompt through the model once to build the initial KV-cache.

**Pass 3 — The speculative decode loop:**
- This is the most complex part. Read it like this:
  1. The Draft model generates K=3 tokens quickly (the "guesses")
  2. The Main model receives the original token + 3 guesses as a batch
  3. The Main model checks each guess one by one
  4. If guess #2 is wrong, everything after it gets thrown away
- **Exercise:** On paper, walk through this scenario:
  - Draft model guesses: ["the", "cat", "sat"]
  - Main model agrees with: ["the", "cat"] but rejects "sat" and says "ran"
  - What tokens get added to the output? Answer: ["the", "cat", "ran"]
  - What happens to the KV-cache? It gets sliced back to remove "sat"

**Pass 4 — The tensor padding logic:**
- Why does the engine need to pad KV-caches before a forward pass?
- **Key concept:** PyTorch needs all tensors in a batch to be the same shape. If Sequence A has 50 cached tokens and Sequence B has 10, we must pad B's cache with 40 zeros on the left.

---

## Day 4: Read the Scheduler and API

### File 5: `server/continuous_batch_scheduler.py`
- **Time:** 1 hour
- **What to focus on:**
  - The `submit()` method: How does a request enter the queue?
  - The `_step_loop()` method: This is the infinite background loop. Every iteration it:
    1. Checks for new waiting requests
    2. Moves them into the "running" batch (if there's room)
    3. Calls `engine._generate_batch_step()`
    4. Checks if any sequence is finished (hit EOS or max_tokens)
    5. Evicts finished sequences and frees their memory
- **Exercise:** If `MAX_RUNNING_SEQUENCES = 8` and there are already 8 sequences running, what happens to a 9th request? It stays in the waiting queue until one of the 8 finishes.

### File 6: `server/main.py`
- **Time:** 30 minutes
- **What to focus on:**
  - The `lifespan()` function: How the engine and scheduler boot up
  - The `/v1/generate` endpoint: How it creates a Sequence, submits it, and either returns a full JSON response or a `StreamingResponse`
- **Exercise:** Find the line where streaming happens. Trace exactly how tokens flow from `inference_engine.py` → `Sequence.output_queue` → `StreamingResponse` → User's browser.

### File 7: `server/grpc_server.py`
- **Time:** 20 minutes
- **What to focus on:** It does the same thing as `main.py` but over a different protocol. Notice how it talks to the same `scheduler` object.

---

## Day 5: Read the Support Files and Run Experiments

### File 8: `server/metrics.py`
- **Time:** 15 minutes
- Understand `Counter` vs `Histogram` vs `Gauge` in Prometheus.

### File 9: `run_server.py`
- **Time:** 10 minutes
- See how `asyncio.gather()` launches both servers simultaneously.

### File 10: `Dockerfile`
- **Time:** 15 minutes
- **Exercise:** Identify which line pre-downloads the model weights. Why is this important? (Answer: Without it, the container would download 3GB of weights every single time it boots, taking 5+ minutes.)

---

## The Mental Models That Make Everything Click

### Mental Model 1: The Restaurant
Think of the server as a restaurant:
- **`main.py`** = The waiter (takes orders from customers)
- **`continuous_batch_scheduler.py`** = The kitchen manager (groups orders together)
- **`inference_engine.py`** = The chef (actually cooks the food)
- **`kv_cache.py`** = The pantry (stores partially prepared ingredients so the chef doesn't start from scratch)
- **`sequence.py`** = The order ticket (tracks what the customer wanted and what's been prepared so far)

### Mental Model 2: The Assembly Line
Think of token generation as an assembly line:
- Without Continuous Batching: The factory builds one car at a time. If car A is on step 100/100 and car B is on step 1/100, car B must wait for car A to finish completely.
- With Continuous Batching: The factory has 8 assembly slots. The moment car A rolls off the line, car C is immediately placed on the empty slot. No slot ever sits idle.

### Mental Model 3: The Exam Cheat Sheet
Think of Speculative Decoding as a student and a teacher:
- The student (Draft model) quickly guesses the next 3 answers
- The teacher (Main model) checks all 3 at once (because checking is faster than solving)
- If the student got answers 1 and 2 right but answer 3 wrong, the teacher keeps 1 and 2, corrects answer 3, and moves on

---

## The Single Most Important Thing

After reading through all the files, do this one final exercise:

**Open two terminal windows.** In window 1, start the server locally. In window 2, send a request using curl. Then go back to the code and add `print()` statements at these 5 critical points:

1. Inside `main.py` right when the request arrives
2. Inside `scheduler.submit()` right when the sequence enters the queue
3. Inside `_step_loop()` right when the scheduler picks up the sequence
4. Inside `_generate_batch_step()` right when a token is generated
5. Inside the SSE streaming generator right when a token is yielded

Watch the prints appear in order in terminal 1. **This single exercise will make the entire architecture click in your brain forever.**
