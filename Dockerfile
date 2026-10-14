# syntax=docker/dockerfile:1.7
# ── Базовый образ: зависимости + пакет pedestrian_safety ───────────────────────
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PSAFETY_DATA_DIR=/app \
    FFMPEG_PATH=/usr/bin/ffmpeg

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt генерируется из poetry.lock (pre-commit хук poetry-export),
# поэтому версии в образе совпадают с локальным окружением разработчика.
COPY requirements.txt .

# 1) PyTorch той же версии, что в lock-файле, но CPU-сборка (~700 МБ вместо ~2.5 ГБ с CUDA)
# 2) остальные зависимости — без torch/torchvision/CUDA-пакетов
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH=$(grep -E '^torch==' requirements.txt | cut -d';' -f1 | tr -d ' ') && \
    TORCHVISION=$(grep -E '^torchvision==' requirements.txt | cut -d';' -f1 | tr -d ' ') && \
    pip install "$TORCH" "$TORCHVISION" --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^(torch|torchvision|triton|nvidia-)' requirements.txt > /tmp/requirements-nogpu.txt && \
    pip install -r /tmp/requirements-nogpu.txt

# Заранее скачиваем веса YOLO (n/s/m), чтобы в рантайме не было частичных загрузок.
# l/x большие — скачаются при первом выборе.
RUN python - <<'PY'
from ultralytics import YOLO
for sz in ("n", "s", "m"):
    YOLO(f"yolov8{sz}.pt")
    print(f"yolov8{sz}.pt cached")
PY

# Сам пакет (зависимости уже стоят — ставим без них)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-cache-dir .

# Данные по умолчанию (в docker-compose перекрываются volume-ами)
COPY zones.json heatmap_camera_positions.json ./
COPY calibrations ./calibrations
COPY heatmap_assets ./heatmap_assets

# ── Сервисы ────────────────────────────────────────────────────────────────────
FROM base AS stream_detect
EXPOSE 5000
CMD ["psafety-stream", "--folder", "/app/video"]

FROM base AS heatmap
EXPOSE 5055
CMD ["psafety-heatmap"]

FROM base AS camera_map
EXPOSE 5060
CMD ["psafety-camera-map"]
