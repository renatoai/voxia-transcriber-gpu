FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Remove pre-installed torchvision/torchaudio to avoid conflicts
RUN pip uninstall -y torchvision torchaudio torchmetrics 2>/dev/null; true

# Install matching versions
RUN pip install --no-cache-dir \
    torchaudio==2.6.0 \
    torchvision==0.21.0

RUN pip install --no-cache-dir \
    whisperx \
    "pyannote.audio>=4.0" \
    runpod \
    requests

WORKDIR /app
COPY src/handler.py .

CMD ["python", "-u", "handler.py"]
