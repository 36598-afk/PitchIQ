# Dockerfile — RunPod Serverless container for ZoneArc's inference pipeline.
#
# Build and push this to a container registry (Docker Hub, or RunPod's own),
# then point a RunPod Serverless endpoint at the image. RunPod handles
# scaling/starting containers on GPU workers — you just provide the image.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ffmpeg is required by infer_pitch71.py for audio extraction (extract_audio)
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching — this layer only
# rebuilds when requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY infer_pitch71.py .
COPY path_scoring.py .
COPY runpod_handler.py .

# Model weights — downloaded from GitHub Releases at build time, NOT copied
# from the repo checkout. Git LFS pointer files (not the real binaries) were
# being pulled in during RunPod's build, causing "invalid load key" errors.
# GitHub Release assets are real binary downloads, sidestepping that entirely.
RUN mkdir -p /app/models/Models/Audio_Impact /app/models/Models/Visual_Impact && \
    curl -f -L -o /app/models/Models/Audio_Impact/audio_classifier.pt "https://github.com/36598-afk/PitchIQ/releases/download/v1.0-models/audio_classifier.pt" && \
    curl -f -L -o /app/models/Models/Visual_Impact/classifier_mitt_vs_bat.pt "https://github.com/36598-afk/PitchIQ/releases/download/v1.0-models/classifier_mitt_vs_bat.pt" && \
    curl -f -L -o /app/models/Models/Visual_Impact/best.pt "https://github.com/36598-afk/PitchIQ/releases/download/v1.0-models/best.pt" && \
    echo "=== Downloaded model sizes (verify these look right in the build log) ===" && \
    ls -la /app/models/Models/Audio_Impact/audio_classifier.pt \
           /app/models/Models/Visual_Impact/classifier_mitt_vs_bat.pt \
           /app/models/Models/Visual_Impact/best.pt

ENV ZONEARC_MODELS_DIR=/app/models

CMD ["python3", "-u", "runpod_handler.py"]

# rebuild trigger
