FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV PATH="/usr/local/bin:$PATH"

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    ffmpeg git curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch + torchaudio ONLY (no torchvision — not needed, causes circular import)
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Pin torchmetrics before 1.6 to avoid arniqa importing torchvision
RUN pip install --no-cache-dir "torchmetrics<1.6.0"

# whisperx + pyannote + runpod
RUN pip install --no-cache-dir \
    whisperx \
    "pyannote.audio>=4.0" \
    runpod \
    requests

# Verify imports
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
    && python -c "import whisperx; print('whisperx OK')" \
    && python -c "from pyannote.audio import Pipeline; print('pyannote OK')" \
    && python -c "import runpod; print('runpod OK')"

WORKDIR /app
COPY src/handler.py .

CMD ["python", "-u", "handler.py"]
