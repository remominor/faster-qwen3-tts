FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/remominor/faster-qwen3-tts"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_SIZE=5
ENV ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-0.6B-Base
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    build-essential \
    git ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchaudio \
    && python3 -m pip install ".[server]"

EXPOSE 8000
CMD ["python3", "openai_server.py", "--host", "0.0.0.0", "--port", "8000"]
