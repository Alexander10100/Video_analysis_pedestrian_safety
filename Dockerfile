FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем CPU-версию PyTorch (~700 МБ вместо ~2.5 ГБ с CUDA)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download default YOLO weights (n/s/m) so runtime never fetches a partial file.
# l/x are large (~170/+MB) — they will auto-download on first use if selected.
RUN python - <<'EOF'
from ultralytics import YOLO
for sz in ("n", "s", "m"):
    YOLO(f"yolov8{sz}.pt")
    print(f"yolov8{sz}.pt cached")
EOF

COPY . .
