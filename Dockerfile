# Official PyTorch image already pins torch==2.6.0 + CUDA 12.4, matching
# exactly what chatterbox-tts's pyproject.toml requires -- avoids the
# manual "install python3.11 via deadsnakes + bootstrap pip + pick a CUDA
# wheel index" toolchain the F5-TTS worker needed (fewer moving parts,
# fewer build iterations).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_XET=0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# chatterbox-tts pins torch==2.6.0 in its own pyproject.toml -- already
# satisfied by the base image, so this should not trigger a reinstall.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download and assemble the model checkpoint at BUILD time (not on
# first request): bakes ~GBs of weights into the image so cold starts don't
# pay a multi-minute download on top of the container start. Reuses the
# same assembly logic as the handler.
COPY app /app
RUN python3 -c "import sys; sys.path.insert(0, '/app'); from handler import assemble_checkpoint_dir; assemble_checkpoint_dir()"

CMD ["python3", "-u", "/app/handler.py"]
