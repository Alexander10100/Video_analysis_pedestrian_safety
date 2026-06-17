# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем CPU-версию PyTorch (~700 МБ вместо ~2.5 ГБ с CUDA)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 1000 torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 1000 -r requirements.txt

# Pre-download default YOLO weights (n/s/m) so runtime never fetches a partial file.
# l/x are large (~170/+MB) — they will auto-download on first use if selected.
RUN python - <<'EOF'
from ultralytics import YOLO
for sz in ("n", "s", "m"):
    YOLO(f"yolov8{sz}.pt")
    print(f"yolov8{sz}.pt cached")
EOF

COPY . .

FROM base AS stream_detect
CMD ["python", "stream_detect.py", "--folder", "/app/video"]

FROM base AS heatmap
CMD ["python", "heatmap_web.py"]

FROM base AS camera_map
CMD ["python", "camera_map_web.py"]
