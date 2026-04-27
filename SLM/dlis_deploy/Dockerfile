FROM vllm/vllm-openai:v0.19.1

RUN pip install --no-cache-dir tornado protobuf && \
    echo "=== Build-time verification ===" && \
    python3 -c "import vllm; print('vllm:', vllm.__version__)" && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)" && \
    python3 -c "import triton; print('triton:', triton.__version__)" && \
    echo "=== Build OK ==="

WORKDIR /app

COPY dlis_server/ /app/dlis_server/
COPY qwen3_model/ /qwen3_model/

RUN chmod +x /app/dlis_server/run.sh

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV QWEN3_MODEL_PATH=/qwen3_model
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.9
ENV TENSOR_PARALLEL_SIZE=1
ENV _ModelPath_=/app/dlis_server/run.sh
ENV _ListeningPort_=8888

EXPOSE 8888

ENTRYPOINT []
CMD ["bash", "/app/dlis_server/run.sh"]
