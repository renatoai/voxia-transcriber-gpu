FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch + audio + vision (matching versions, CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install whisperx, pyannote, runpod
RUN pip install --no-cache-dir \
    whisperx \
    "pyannote.audio>=4.0" \
    runpod \
    requests

WORKDIR /app
COPY src/handler.py .

CMD ["python", "-u", "handler.py"]
