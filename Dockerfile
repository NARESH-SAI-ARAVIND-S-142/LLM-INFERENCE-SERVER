# ══════════════════════════════════════════════════════════════════
# miniServe — Hugging Face Spaces Dockerfile
# ══════════════════════════════════════════════════════════════════
# HF Spaces requirements:
#   - Port 7860 (mandatory)
#   - Non-root user recommended
#   - Model downloaded at build time for fast startup
# ══════════════════════════════════════════════════════════════════

FROM python:3.12-slim

# ─── System setup ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ─── Install Python dependencies ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ───────────────────────────────────────
COPY . .

# ─── Generate gRPC stubs ─────────────────────────────────────────
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=proto \
    --grpc_python_out=proto \
    proto/inference.proto

# ─── Pre-download model (cached in Docker layer) ─────────────
# This avoids downloading on every container start
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# ─── Fix permissions for HF Spaces ───────────────────────────────
RUN chown -R appuser:appuser /app
# HuggingFace cache directory
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser/.cache

USER appuser

# ─── HF Spaces requires port 7860 ────────────────────────────────
ENV MINISERVE_REST_PORT=7860
ENV MINISERVE_REST_HOST=0.0.0.0
ENV MINISERVE_GRPC_PORT=50051
ENV MINISERVE_LOG_LEVEL=INFO
ENV MINISERVE_MAX_BATCH_SIZE=8
ENV MINISERVE_MAX_WAIT_MS=50
ENV MINISERVE_MAX_TOKENS=50

EXPOSE 7860

# ─── Launch the server ────────────────────────────────────────────
CMD ["python", "run_server.py"]
