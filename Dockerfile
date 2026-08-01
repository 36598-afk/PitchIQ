# Dockerfile — RunPod Serverless container for ZoneArc's inference pipeline.
#
# Build and push this to a container registry (Docker Hub, or RunPod's own),
# then point a RunPod Serverless endpoint at the image. RunPod handles
# scaling/starting containers on GPU workers — you just provide the image.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ffmpeg is required by infer_pitch71.py for audio extraction (extract_audio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching — this layer only
# rebuilds when requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY infer_pitch71.py .
COPY path_scoring.py .
COPY runpod_handler.py .

# Model weights — baked into the image so there's no cold-start download.
# These paths match what infer_pitch71.py's MODELS dict expects: BASE +
# "/Models/Audio_Impact/..." etc, where BASE = ZONEARC_MODELS_DIR below.
COPY models/audio_classifier.pt          /app/models/Models/Audio_Impact/audio_classifier.pt
COPY models/classifier_mitt_vs_bat.pt    /app/models/Models/Visual_Impact/classifier_mitt_vs_bat.pt
COPY models/best.pt                      /app/models/Models/Visual_Impact/best.pt

ENV ZONEARC_MODELS_DIR=/app/models

CMD ["python3", "-u", "runpod_handler.py"]
