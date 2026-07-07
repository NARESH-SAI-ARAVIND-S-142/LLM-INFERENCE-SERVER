# The "Must-Convey" Interview Checklist

If an interview is short and you only have 5 minutes to talk about `miniServe`, you must prioritize hitting these **5 critical engineering concepts**. 

These are the specific signals that Senior Engineers and Hiring Managers look for to separate a "Junior Developer who followed a tutorial" from a "Senior Systems Engineer who designed an architecture."

---

## 1. I Engineered for Strict Constraints
**What to convey:** Never present the architecture in a vacuum. Always start by explaining *why* you built it this way.
**The Key Phrase:** *"I was deploying on the Hugging Face Free Tier, which has strict constraints: No GPU, only CPU, and a hard 16GB RAM limit. Every architectural decision I made—from 8-bit quantization to Speculative Decoding—was specifically chosen to squeeze maximum throughput out of that exact hardware profile."*
**Why it matters:** It proves you are pragmatic. You don't just throw cloud money at a problem; you engineer solutions around hardware limits.

---

## 2. I Decoupled the I/O from the Compute
**What to convey:** You must prove you understand Python's threading and blocking limitations.
**The Key Phrase:** *"PyTorch inference is completely synchronous and heavily CPU-bound. If I ran it directly in the FastAPI endpoint, it would have blocked the entire web server. I strictly decoupled the API layer from the Machine Learning layer by putting the PyTorch engine in a background `run_in_executor` thread pool, communicating with the API exclusively via asynchronous queues."*
**Why it matters:** Failing to decouple ML models from web servers is the #1 mistake juniors make. Proving you solved this immediately elevates your seniority.

---

## 3. I Understand the "Memory Bandwidth" Bottleneck
**What to convey:** You must explain *why* you built Speculative Decoding, not just *how*.
**The Key Phrase:** *"Autoregressive LLM generation on CPUs is fundamentally memory-bandwidth bound, not compute-bound. The processor spends all its time waiting for weights to load from RAM. I implemented Speculative Decoding because it allows me to bypass this bottleneck by trading cheap excess compute (a tiny draft model) for a massive reduction in memory fetches, verifying multiple tokens in a single parallel forward pass."*
**Why it matters:** This is the holy grail of ML Systems interviews. If you say "Memory-Bandwidth Bound," the interviewer will instantly know you understand low-level hardware mechanics.

---

## 4. I Operated at the "Iteration Level", not the "Sequence Level"
**What to convey:** You must highlight the mathematical superiority of Continuous Batching.
**The Key Phrase:** *"Standard static batching is wildly inefficient because the batch is locked until the longest sequence finishes, wasting precious compute slots. I built a custom scheduler to operate at the iteration level. After every single token is generated, my scheduler dynamically evicts finished sequences and instantly injects new ones, ensuring the batch is mathematically dense at all times."*
**Why it matters:** It shows you understand how to maximize GPU/CPU utilization in a high-traffic production environment.

---

## 5. I Built for Production Observability
**What to convey:** The project didn't end when the code compiled; it ended when it was measurable in production.
**The Key Phrase:** *"Because this is a continuous system, I couldn't just print logs. I instrumented the entire engine with Prometheus metrics. I specifically tracked the 'Speculative Acceptance Rate' to ensure my draft model was actually saving time, and tracked 'Tokens Per Second' to monitor my CPU throughput."*
**Why it matters:** Companies don't hire people to write code; they hire people to run code in production. Mentioning Prometheus proves you care about Day-2 operations, reliability, and monitoring.
