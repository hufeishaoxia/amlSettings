FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (vllm needs devel image for compilation)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference_server.py .

# Model checkpoint will be mounted at /model or baked in at build time
# To bake in: COPY checkpoint-1128/ /model/

EXPOSE 8080

# Health check for DLIS
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["python3", "inference_server.py"]
CMD ["--model_path", "/model", "--port", "8080", "--max_len", "4096", "--tp", "1"]
