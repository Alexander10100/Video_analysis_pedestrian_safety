"""
offline_detect.py — Офлайн-обработка видео: аннотация кадров и сохранение результата.

MERGE v1 + v2: объединены все функции обеих версий без потери функционала.

  Из версии 1 (оригинал):
    - AsyncVideoWriter / AsyncVideoReader — фоновая запись и чтение
    - Batched inference (--inference-batch-size) при --preload-video
    - --preload-video: загрузка всего видео в RAM
    - --no-annotated-video: режим без записи mp4 (только статистика)
    - --writer-queue-size / --reader-queue-size
    - GPU-оптимизации (torch.inference_mode, half, cudnn.benchmark)

  Из версии 2 (правки):
    - Расширенный load_yolo: v8/v8p6/v9/v10/v11/rtdetr — любая модель одним ключом
    - enhance_frame: предобработка кадра (--enhance clahe/gamma/both/none)
    - Тайловая детекция верхней части (--tile / --tile-ratio)
    - draw_person_boxes: рисование bbox людей (зелёный=взрослый, оранжевый=ребёнок)
    - person_inside_vehicle: фильтрация людей внутри машин
    - Детекция транспорта (классы 2,3,5,7) для фильтрации
    - --ema-alpha: управление сглаживанием bbox (1.0 = выкл)
    - --debug: режим отладки (raw bbox, отброшенные, транспорт)
    - Предупреждение при --detect-every > 5

Использование:
    python offline_detect.py --video input.mp4 --camera cam_01

    # Тайловая детекция (мелкие/далёкие объекты)
    python offline_detect.py --video input.mp4 --camera cam_01 --tile

    # Папка с файлами
    python offline_detect.py --folder /path/to/videos --camera cam_01

    # Полный набор опций
    python offline_detect.py --video input.mp4 --camera cam_01 \\
        --output result.mp4 --model v8m6 --conf 0.3 --detect-every 3 \\
        --tile --enhance clahe --ema-alpha 1.0 \\
        --preload-video --writer-queue-size 64 --inference-batch-size 16

    # Только статистика без записи mp4 (самый быстрый режим)
    python offline_detect.py --video input.mp4 --camera cam_01 --no-annotated-video

    # Отладка: посмотреть что детектируется на первых 300 кадрах
    python offline_detect.py --video input.mp4 --camera cam_01 --debug --max-frames 300

Зависимости:
    pip install ultralytics opencv-python pillow
    (zone_manager, traffic_light, violation_detector, age_classifier — из текущего проекта)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Callable

import cv2
import numpy as np

from age_classifier import AgeClassifier, AgeTracker, BboxEMA
from traffic_light import TrafficLightAnalyzer
from violation_detector import (
    ViolationDetector,
    draw_violations,
    draw_zones,
    draw_traffic_light_states,
    _put_text_pil,
    person_inside_vehicle,
)
from zone_manager import ZoneManager

# ── Поддерживаемые форматы ────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}

# ── Кодек для выходного файла ─────────────────────────────────────────────────
OUTPUT_FOURCC = "mp4v"
OUTPUT_EXT    = ".mp4"

# ── Периодичность чистки мёртвых треков ──────────────────────────────────────
_EVICT_EVERY = 150

# ── Цвета bbox людей ─────────────────────────────────────────────────────────
_COLOR_ADULT   = (50, 205, 50)     # зелёный   — взрослый
_COLOR_CHILD   = (255, 165, 0)     # оранжевый — ребёнок
_COLOR_TILE    = (200, 200, 0)     # жёлтый    — из тайловой детекции
_COLOR_UNKNOWN = (180, 180, 180)   # серый     — неизвестно

# ── Классы транспортных средств COCO ─────────────────────────────────────────
_VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorbike, bus, truck


# ══════════════════════════════════════════════════════════════════════════════
# GPU / runtime
# ══════════════════════════════════════════════════════════════════════════════

def _configure_runtime():
    cv2.setUseOptimized(True)
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return torch


_TORCH = _configure_runtime()


def _resolve_device(device: str = "auto") -> str:
    requested = (device or "auto").strip().lower()
    if requested != "auto":
        if requested.isdigit():
            return f"cuda:{requested}"
        return requested
    if _TORCH is not None and _TORCH.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _use_half_for_device(device: str) -> bool:
    if _TORCH is None or not _TORCH.cuda.is_available():
        return False
    return device not in {"cpu", "mps"}


# ══════════════════════════════════════════════════════════════════════════════
# Async IO
# ══════════════════════════════════════════════════════════════════════════════

def _preload_video_frames(cap: cv2.VideoCapture, limit: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


class AsyncVideoWriter:
    def __init__(self, writer: cv2.VideoWriter, queue_size: int):
        self._writer = writer
        self._queue: Queue[np.ndarray | None] = Queue(maxsize=max(1, queue_size))
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            try:
                if frame is None:
                    return
                self._writer.write(frame)
            except BaseException as exc:
                self._errors.append(exc)
                return
            finally:
                self._queue.task_done()

    def write(self, frame: np.ndarray) -> None:
        if self._errors:
            raise RuntimeError("Async VideoWriter failed") from self._errors[0]
        self._queue.put(frame)

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._writer.release()
        if self._errors:
            raise RuntimeError("Async VideoWriter failed") from self._errors[0]


class AsyncVideoReader:
    def __init__(self, cap: cv2.VideoCapture, limit: int, queue_size: int):
        self._cap = cap
        self._limit = limit
        self._queue: Queue[np.ndarray | None] = Queue(maxsize=max(1, queue_size))
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            count = 0
            while count < self._limit:
                ok, frame = self._cap.read()
                if not ok:
                    break
                self._queue.put(frame)
                count += 1
        except BaseException as exc:
            self._errors.append(exc)
        finally:
            self._queue.put(None)

    def read(self) -> np.ndarray | None:
        frame = self._queue.get()
        self._queue.task_done()
        if self._errors:
            raise RuntimeError("Async VideoReader failed") from self._errors[0]
        return frame

    def close(self) -> None:
        self._thread.join()
        self._cap.release()
        if self._errors:
            raise RuntimeError("Async VideoReader failed") from self._errors[0]


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _progress_bar(current: int, total: int, width: int = 38) -> str:
    if total <= 0:
        return f"[{'?' * width}] ??%"
    pct  = current / total
    done = int(pct * width)
    bar  = "█" * done + "░" * (width - done)
    return f"[{bar}] {pct:5.1%}"


def _format_eta(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _make_output_path(input_path: Path, output_arg: str | None) -> Path:
    """Сформировать путь выходного файла."""
    if output_arg:
        p = Path(output_arg)
        if p.is_dir():
            return p / (input_path.stem + "_annotated" + OUTPUT_EXT)
        if not p.suffix:
            return p.with_suffix(OUTPUT_EXT)
        return p
    return input_path.parent / (input_path.stem + "_annotated" + OUTPUT_EXT)


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка модели — расширенный реестр (v2)
# ══════════════════════════════════════════════════════════════════════════════

def load_yolo(model_key: str, device: str = "auto"):
    """
    Загружает модель по короткому ключу.

    Поддерживаемые ключи:
      YOLOv8 стандартные:
        v8n, v8s, v8m, v8l, v8x
        (обратная совместимость: n, s, m, l, x -> v8*)

      YOLOv8-P6 (1280px, 6-уровневый FPN — лучше для мелких объектов):
        v8n6, v8s6, v8m6, v8l6, v8x6

      YOLOv9:
        v9c, v9e

      YOLOv10:
        v10n, v10s, v10m, v10l, v10x

      YOLO11 (ultralytics, актуальное поколение):
        v11n, v11s, v11m, v11l, v11x

      RT-DETR (трансформер, при перекрытиях):
        rtdetr-l, rtdetr-x

    Для видеонаблюдения с мелкими людьми рекомендуется:
      v8m6  — лучший баланс качества и скорости для мелких объектов
      v8x6  — максимальное качество
      v11m  — актуальнее v8
      rtdetr-l — при сильных перекрытиях
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)

    MODEL_MAP = {
        # YOLOv8
        "v8n": "yolov8n.pt", "v8s": "yolov8s.pt", "v8m": "yolov8m.pt",
        "v8l": "yolov8l.pt", "v8x": "yolov8x.pt",
        # YOLOv8-P6 — 6-уровневый FPN, обучены на 1280px
        "v8n6": "yolov8n6.pt", "v8s6": "yolov8s6.pt", "v8m6": "yolov8m6.pt",
        "v8l6": "yolov8l6.pt", "v8x6": "yolov8x6.pt",
        # YOLOv9
        "v9c": "yolov9c.pt", "v9e": "yolov9e.pt",
        # YOLOv10
        "v10n": "yolov10n.pt", "v10s": "yolov10s.pt", "v10m": "yolov10m.pt",
        "v10l": "yolov10l.pt", "v10x": "yolov10x.pt",
        # YOLO11
        "v11n": "yolo11n.pt", "v11s": "yolo11s.pt", "v11m": "yolo11m.pt",
        "v11l": "yolo11l.pt", "v11x": "yolo11x.pt",
        # RT-DETR
        "rtdetr-l": "rtdetr-l.pt", "rtdetr-x": "rtdetr-x.pt",
    }

    # Обратная совместимость со старыми ключами (n/s/m/l/x)
    _COMPAT = {"n": "v8n", "s": "v8s", "m": "v8m", "l": "v8l", "x": "v8x"}
    model_key = _COMPAT.get(model_key, model_key)

    if model_key not in MODEL_MAP:
        valid = ", ".join(MODEL_MAP.keys())
        print(f"[ERROR] Неизвестная модель '{model_key}'. Доступные: {valid}")
        sys.exit(1)

    name = MODEL_MAP[model_key]
    resolved_device = _resolve_device(device)
    use_half = _use_half_for_device(resolved_device)
    print(f"[INFO] Загружаем модель {name}  (ключ: {model_key})...")
    print(f"[INFO] YOLO device: {resolved_device}, half: {use_half}")

    model = YOLO(name)
    model.to(resolved_device)
    model.model_name  = name
    model.device_name = resolved_device
    model.use_half    = use_half
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Предобработка кадра (v2)
# ══════════════════════════════════════════════════════════════════════════════

def enhance_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    """
    Улучшает контраст кадра перед подачей в модель.
    Оригинальный кадр НЕ меняется — используется только для детекции.
    На выходное видео пишется оригинал (annotated = frame.copy()).

    Режимы (--enhance):
      clahe  — локальная нормализация гистограммы (рекомендуется при тенях)
      gamma  — глобальное осветление тёмных зон (гамма=0.6)
      both   — gamma + clahe (максимальный эффект)
      none   — без предобработки (по умолчанию)
    """
    if mode == "none":
        return frame

    out = frame.copy()

    if mode in ("gamma", "both"):
        gamma = 0.6
        inv = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)

    if mode in ("clahe", "both"):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Рисование bbox людей (v2)
# ══════════════════════════════════════════════════════════════════════════════

def draw_person_boxes(
    frame: np.ndarray,
    norm_boxes: list,
    tile_flags: list[bool],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Рисует bbox каждого человека.
      norm_boxes[i]: (tid, nx1, ny1, nx2, ny2, conf, age_label, age_conf)
      tile_flags[i]: True если бокс из тайловой детекции

    Цвета:
      Зелёный   — взрослый (основная детекция)
      Оранжевый — ребёнок
      Жёлтый    — из тайла (tid == -1)
    """
    for i, item in enumerate(norm_boxes):
        tid, nx1, ny1, nx2, ny2, conf_val, age_label, age_conf = item
        from_tile = tile_flags[i] if i < len(tile_flags) else False

        x1 = int(nx1 * frame_w)
        y1 = int(ny1 * frame_h)
        x2 = int(nx2 * frame_w)
        y2 = int(ny2 * frame_h)

        if from_tile:
            color     = _COLOR_TILE
            label_str = f"TILE {conf_val:.0%}"
        elif age_label == "child":
            color     = _COLOR_CHILD
            label_str = f"РЕБ {age_conf:.0%}"
        else:
            color     = _COLOR_ADULT
            label_str = f"ВЗР {age_conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, lbl_y - th - 4), (x1 + tw + 4, lbl_y + baseline), color, -1)
        cv2.putText(frame, label_str, (x1 + 2, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Тайловая детекция (v2)
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _run_tile_detection(
    model,
    frame: np.ndarray,
    frame_w: int,
    frame_h: int,
    conf: float,
    imgsz: int,
    tile_top_ratio: float = 0.65,
) -> list[dict]:
    """
    Детектируем людей в верхней части кадра (дальние/мелкие объекты).
    Возвращаем боксы в координатах полного кадра.
    """
    cut_y = int(frame_h * tile_top_ratio)
    tile = frame[:cut_y, :]
    tile_conf = max(conf * 0.7, 0.05)

    results = model(
        tile, classes=[0], conf=tile_conf, iou=0.45,
        imgsz=imgsz, verbose=False,
    )[0]

    boxes: list[dict] = []
    if results.boxes is None:
        return boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0]) if box.conf is not None else 0.0
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf_val})
    return boxes


def _merge_tile_boxes(
    norm_boxes: list,
    tile_raw: list[dict],
    frame_w: int,
    frame_h: int,
    iou_thresh: float = 0.35,
) -> list[dict]:
    """Возвращает только те боксы из тайла, которые не дублируют уже найденные."""
    existing = [
        (int(nb[1] * frame_w), int(nb[2] * frame_h),
         int(nb[3] * frame_w), int(nb[4] * frame_h))
        for nb in norm_boxes
    ]
    new_boxes = []
    for tb in tile_raw:
        coord = (tb["x1"], tb["y1"], tb["x2"], tb["y2"])
        if not any(_iou(coord, ex) > iou_thresh for ex in existing):
            new_boxes.append(tb)
    return new_boxes


# ══════════════════════════════════════════════════════════════════════════════
# Легенда
# ══════════════════════════════════════════════════════════════════════════════

def _draw_legend_offline(
    frame: np.ndarray,
    persons: int,
    ms: float,
    violations: int,
    adults: int,
    children: int,
    model_sz: str,
    camera_id: str,
    age_calibrated: bool,
    frame_idx: int,
    total_frames: int,
    detect_every: int,
    tile_enabled: bool = False,
    tile_extra: int = 0,
) -> np.ndarray:
    cam_str    = (camera_id or "—")[:18]
    age_status = "ДА" if age_calibrated else "НЕТ"
    progress   = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)
    tile_str   = f"ВКЛ (+{tile_extra})" if tile_enabled else "ВЫКЛ"

    lines = [
        (f"Inference: {ms:.0f} ms",           (180, 180, 180)),
        (f"Кадр:      {progress}",             (130, 130, 130)),
        (f"Людей:     {persons}",              ( 50, 205,  50)),
        (f"  взрослых: {adults}",              ( 50, 200,  50)),
        (f"  детей:    {children}",            (  0, 165, 255)),
        (f"Наруш.:    {violations}",           ( 60,  60, 230)),
        (f"Тайл:      {tile_str}",
         (50, 205, 50) if tile_enabled else (130, 130, 130)),
        (f"Модель:    {model_sz}",             ( 90, 130, 255)),
        (f"Обраб. 1/{detect_every} кадров",   (180, 130,   0)),
        (f"Камера:    {cam_str}",              (100, 200, 200)),
        (f"Калибр.:   {age_status}",
         (50, 205, 50) if age_calibrated else (180, 130, 0)),
    ]

    pad, lh   = 8, 22
    font_size = 13
    box_w     = 245
    box_h     = len(lines) * lh + pad * 2

    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10 + box_w, 10 + box_h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = 10 + pad + i * lh
        frame = _put_text_pil(frame, text, (10 + pad, y),
                               color_bgr=color, font_size=font_size)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Обработка одного видеофайла
# ══════════════════════════════════════════════════════════════════════════════

def process_video(
    input_path: Path,
    output_path: Path,
    model,
    zone_mgr: ZoneManager,
    tl_analyzer: TrafficLightAnalyzer,
    viol_det: ViolationDetector,
    age_clf: AgeClassifier | None,
    age_tracker: AgeTracker,
    bbox_ema: BboxEMA,
    camera_id: str,
    conf: float,
    imgsz: int,
    detect_every: int,
    max_frames: int,
    # Опции из v2
    use_tile: bool = False,
    tile_top_ratio: float = 0.65,
    enhance: str = "none",
    debug_mode: bool = False,
    # Опции из v1
    violation_callback: Callable[[list, float], None] | None = None,
    preload_video: bool = False,
    writer_queue_size: int = 64,
    reader_queue_size: int = 0,
    inference_batch_size: int = 16,
    write_annotated: bool = True,
) -> dict:
    """
    Обработать один видеофайл, записать аннотированный результат.
    Возвращает словарь со статистикой прогона.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть: {input_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit        = max_frames if max_frames > 0 else total_frames

    print(f"\n[INFO] Входной файл : {input_path.name}")
    print(f"       Разрешение    : {frame_w}x{frame_h}  FPS: {src_fps:.2f}")
    print(f"       Всего кадров  : {total_frames}  Лимит: {limit}")
    print(f"       Выходной файл : {output_path}")
    print(f"       Детекция 1/{detect_every} кадров")
    if detect_every > 5:
        print(f"       [WARN] detect-every={detect_every} — трекер будет терять людей! Рекомендуется <= 5")
    print(f"       Enhance       : {enhance.upper()}")
    if use_tile:
        cut_px = int(frame_h * tile_top_ratio)
        print(f"       Тайл          : ВКЛ  (верхние {tile_top_ratio:.0%} = {cut_px}px)")
    else:
        print("       Тайл          : ВЫКЛ (добавь --tile для мелких/далёких объектов)")
    if debug_mode:
        print("       [DEBUG]       : ВКЛ — raw bbox (голубой), отброшен (красный), транспорт (серый)")
    print(f"       Preload RAM    : {'on' if preload_video else 'off'}")
    print(f"       Reader queue   : {reader_queue_size if not preload_video else 0}")
    print(f"       Writer queue   : {writer_queue_size}")
    print(f"       Infer batch    : {inference_batch_size}")
    print(f"       Annotated mp4  : {'on' if write_annotated else 'off'}\n")

    # Предзагрузка в RAM (v1)
    preloaded_frames: list[np.ndarray] | None = None
    if preload_video:
        t_preload = time.perf_counter()
        preloaded_frames = _preload_video_frames(cap, limit)
        cap.release()
        cap = None
        limit = len(preloaded_frames)
        if limit == 0:
            print(f"[ERROR] Не удалось прочитать кадры: {input_path}")
            return {}
        bytes_total = sum(frame.nbytes for frame in preloaded_frames)
        print(
            f"[INFO] Preloaded frames: {limit}, "
            f"RAM: {bytes_total / (1024 ** 3):.2f} GB, "
            f"time: {time.perf_counter() - t_preload:.1f}s"
        )

    # AgeClassifier
    if age_clf is None or age_clf.frame_height != frame_h:
        if camera_id:
            age_clf = AgeClassifier.load_for_camera(camera_id, frame_h)
        else:
            age_clf = AgeClassifier(frame_h)
    age_calibrated = age_clf.is_calibrated()

    # Сбрасываем историю: треки предыдущего файла не применимы
    age_tracker.reset()
    bbox_ema.reset()

    # VideoWriter
    writer       = None
    async_writer = None
    if write_annotated:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(f"[ERROR] Не удалось создать VideoWriter: {output_path}")
            if cap is not None:
                cap.release()
            return {}
        async_writer = AsyncVideoWriter(writer, writer_queue_size) if writer_queue_size > 0 else None

    # Статистика
    stat = {
        "total_frames":       0,
        "detected_frames":    0,
        "total_persons":      0,
        "total_violations":   0,
        "total_adults":       0,
        "total_children":     0,
        "inference_ms_sum":   0.0,
        "elapsed_sec":        0.0,
        "annotated_written":  write_annotated,
        "tile_extra_persons": 0,
    }

    cw_zones  = zone_mgr.crosswalk_zones()
    all_zones = zone_mgr.get_all()

    # Async reader (v1)
    async_reader = (
        AsyncVideoReader(cap, limit, reader_queue_size)
        if cap is not None and not preload_video and reader_queue_size > 0
        else None
    )

    # Имя модели для легенды
    raw_name    = getattr(model, "model_name", None) or str(getattr(model, "ckpt_path", "?"))
    model_label = Path(raw_name).stem

    # track() kwargs — используется в поштучном режиме
    track_kwargs = {
        "classes": [0, 2, 3, 5, 7],   # людей + транспорт для фильтрации
        "conf":    conf,
        "iou":     0.45,
        "imgsz":   imgsz,
        "persist": True,
        "verbose": False,
        "device":  getattr(model, "device_name", "auto"),
        "half":    bool(getattr(model, "use_half", False)),
    }

    last_annotated: np.ndarray | None = None
    frame_idx  = 0
    detect_cnt = 0
    t_start    = time.perf_counter()
    t_progress = t_start

    # Режим batched inference (v1): preload + detect_every==1 + batch > 1
    # В batched режиме тайловая детекция и enhance НЕ применяются
    # (тайл несовместим с batch API; enhance — тоже).
    use_batched_inference = (
        preloaded_frames is not None
        and detect_every == 1
        and inference_batch_size > 1
        and not use_tile
        and enhance == "none"
    )

    if use_batched_inference:
        # Переопределяем track_kwargs: batch API не поддерживает vehicle-классы через persist
        batch_track_kwargs = {
            "classes": [0],
            "conf":    conf,
            "iou":     0.45,
            "imgsz":   imgsz,
            "persist": True,
            "verbose": False,
            "device":  getattr(model, "device_name", "auto"),
            "half":    bool(getattr(model, "use_half", False)),
        }
        for batch_start in range(0, limit, inference_batch_size):
            batch_frames = preloaded_frames[batch_start:batch_start + inference_batch_size]
            if not batch_frames:
                break

            t0 = time.perf_counter()
            if _TORCH is not None:
                with _TORCH.inference_mode():
                    batch_results = model.track(batch_frames, **batch_track_kwargs)
            else:
                batch_results = model.track(batch_frames, **batch_track_kwargs)
            batch_elapsed_ms = (time.perf_counter() - t0) * 1000
            per_frame_ms = batch_elapsed_ms / max(1, len(batch_results))

            for local_idx, (frame, results) in enumerate(zip(batch_frames, batch_results)):
                frame_idx = batch_start + local_idx + 1
                detect_cnt += 1

                if cw_zones:
                    tl_analyzer.process_frame(frame, cw_zones)
                tl_states = tl_analyzer.get_all_states()

                annotated = frame.copy() if write_annotated else None
                if write_annotated and all_zones:
                    annotated = draw_zones(annotated, all_zones, tl_states)

                norm_boxes: list[tuple] = []
                tile_flags: list[bool] = []
                adults_cnt   = 0
                children_cnt = 0
                active_tids: set[int] = set()

                boxes = results.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    ids = (
                        boxes.id.detach().cpu().numpy().astype(np.int32, copy=False)
                        if boxes.id is not None
                        else np.full(len(xyxy), -1, dtype=np.int32)
                    )
                    confs = (
                        boxes.conf.detach().cpu().numpy()
                        if boxes.conf is not None
                        else np.zeros(len(xyxy), dtype=np.float32)
                    )
                    for i in range(len(xyxy)):
                        raw_x1, raw_y1, raw_x2, raw_y2 = np.rint(xyxy[i]).astype(np.int32)
                        tid      = int(ids[i])
                        conf_val = float(confs[i])
                        if tid >= 0:
                            active_tids.add(tid)
                        x1, y1, x2, y2 = bbox_ema.smooth(tid, raw_x1, raw_y1, raw_x2, raw_y2)
                        raw_label, raw_conf = age_clf.classify(x1, y1, x2, y2)
                        age_label, age_conf = age_tracker.update(tid, raw_label, raw_conf)
                        if age_label == "child":
                            children_cnt += 1
                        else:
                            adults_cnt += 1
                        norm_boxes.append((
                            tid,
                            x1 / frame_w, y1 / frame_h,
                            x2 / frame_w, y2 / frame_h,
                            conf_val, age_label, age_conf,
                        ))
                        tile_flags.append(False)

                if detect_cnt % _EVICT_EVERY == 0 and active_tids:
                    bbox_ema.evict(active_tids)
                    age_tracker.evict(active_tids)

                violations = viol_det.analyze(norm_boxes) if norm_boxes else []
                if violation_callback is not None and violations:
                    violation_callback(violations, frame_idx / src_fps)

                if write_annotated:
                    annotated = draw_person_boxes(annotated, norm_boxes, tile_flags, frame_w, frame_h)
                    annotated, vcount = draw_violations(annotated, violations, frame_w, frame_h)
                    if cw_zones:
                        annotated = draw_traffic_light_states(
                            annotated, cw_zones, tl_states, frame_w, frame_h)
                    annotated = _draw_legend_offline(
                        annotated,
                        persons        = len(norm_boxes),
                        ms             = per_frame_ms,
                        violations     = vcount,
                        adults         = adults_cnt,
                        children       = children_cnt,
                        model_sz       = model_label,
                        camera_id      = camera_id,
                        age_calibrated = age_calibrated,
                        frame_idx      = frame_idx,
                        total_frames   = limit,
                        detect_every   = detect_every,
                        tile_enabled   = False,
                        tile_extra     = 0,
                    )
                    last_annotated = annotated
                    if async_writer is not None:
                        async_writer.write(annotated)
                    elif writer is not None:
                        writer.write(annotated)
                else:
                    vcount = sum(1 for item in violations if item.violation != "none")

                stat["total_frames"]     += 1
                stat["detected_frames"]  += 1
                stat["total_persons"]    += len(norm_boxes)
                stat["total_violations"] += vcount
                stat["total_adults"]     += adults_cnt
                stat["total_children"]   += children_cnt
                stat["inference_ms_sum"] += per_frame_ms

            now = time.perf_counter()
            if now - t_progress >= 2.0:
                t_progress = now
                elapsed    = now - t_start
                rate       = frame_idx / elapsed if elapsed > 0 else 0
                eta        = (limit - frame_idx) / rate if rate > 0 else 0
                avg_ms     = (stat["inference_ms_sum"] / stat["detected_frames"]
                              if stat["detected_frames"] > 0 else 0)
                print(
                    f"\r  {_progress_bar(frame_idx, limit)}  "
                    f"ETA:{_format_eta(eta)}  "
                    f"inf:{avg_ms:.0f}ms  "
                    f"viol:{stat['total_violations']}  ",
                    end="", flush=True,
                )

    # Поштучный режим — основной (с тайлом, enhance, debug, фильтром транспорта)
    else:
        while frame_idx < limit:
            if preloaded_frames is not None:
                frame = preloaded_frames[frame_idx]
            elif async_reader is not None:
                frame = async_reader.read()
                if frame is None:
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            frame_idx += 1

            if frame_idx % detect_every == 0:
                detect_cnt += 1

                t0 = time.perf_counter()

                # Предобработка для детекции (v2)
                detect_frame = enhance_frame(frame, enhance)

                # Основная детекция: люди + транспорт
                if _TORCH is not None:
                    with _TORCH.inference_mode():
                        results = model.track(detect_frame, **track_kwargs)[0]
                else:
                    results = model.track(detect_frame, **track_kwargs)[0]

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Тайловая детекция (v2)
                tile_raw: list[dict] = []
                if use_tile:
                    t1 = time.perf_counter()
                    tile_raw = _run_tile_detection(
                        model, detect_frame, frame_w, frame_h,
                        conf=conf, imgsz=imgsz, tile_top_ratio=tile_top_ratio,
                    )
                    elapsed_ms += (time.perf_counter() - t1) * 1000

                if cw_zones:
                    tl_analyzer.process_frame(frame, cw_zones)
                tl_states = tl_analyzer.get_all_states()

                annotated = frame.copy() if write_annotated else None
                if write_annotated and all_zones:
                    annotated = draw_zones(annotated, all_zones, tl_states)

                # Первый проход: собираем боксы транспорта
                vehicle_boxes: list[tuple] = []
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        if cls_id in _VEHICLE_CLASSES:
                            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                            vehicle_boxes.append((vx1, vy1, vx2, vy2))

                norm_boxes: list[tuple] = []
                tile_flags: list[bool]  = []
                adults_cnt   = 0
                children_cnt = 0
                active_tids: set[int] = set()

                # [DEBUG] транспорт серыми рамками
                if debug_mode and write_annotated:
                    for vx1, vy1, vx2, vy2 in vehicle_boxes:
                        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (120, 120, 120), 1)
                        cv2.putText(annotated, "VEH", (vx1 + 2, vy1 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

                # Второй проход: люди из основной детекции
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        if cls_id != 0:
                            continue

                        raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                        tid      = int(box.id[0]) if box.id is not None else -1
                        conf_val = float(box.conf[0]) if box.conf is not None else 0.0

                        if tid >= 0:
                            active_tids.add(tid)

                        # [DEBUG] raw bbox голубым
                        if debug_mode and write_annotated:
                            cv2.rectangle(annotated, (raw_x1, raw_y1), (raw_x2, raw_y2),
                                          (0, 255, 255), 1)
                            cv2.putText(annotated, f"RAW {conf_val:.0%}",
                                        (raw_x1 + 2, raw_y1 + 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

                        # Фильтрация людей внутри транспорта (v2)
                        if person_inside_vehicle(raw_x1, raw_y1, raw_x2, raw_y2, vehicle_boxes):
                            if debug_mode and write_annotated:
                                cv2.rectangle(annotated, (raw_x1, raw_y1), (raw_x2, raw_y2),
                                              (0, 0, 255), 2)
                                cv2.putText(annotated, "IN_VEH",
                                            (raw_x1 + 2, raw_y2 - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            continue

                        x1, y1, x2, y2 = bbox_ema.smooth(tid, raw_x1, raw_y1, raw_x2, raw_y2)
                        raw_label, raw_conf = age_clf.classify(x1, y1, x2, y2)
                        age_label, age_conf = age_tracker.update(tid, raw_label, raw_conf)

                        if age_label == "child":
                            children_cnt += 1
                        else:
                            adults_cnt += 1

                        norm_boxes.append((
                            tid,
                            x1 / frame_w, y1 / frame_h,
                            x2 / frame_w, y2 / frame_h,
                            conf_val, age_label, age_conf,
                        ))
                        tile_flags.append(False)

                # Тайловые боксы (только уникальные)
                tile_new = _merge_tile_boxes(norm_boxes, tile_raw, frame_w, frame_h)
                stat["tile_extra_persons"] += len(tile_new)

                for tb in tile_new:
                    x1, y1, x2, y2 = tb["x1"], tb["y1"], tb["x2"], tb["y2"]
                    if person_inside_vehicle(x1, y1, x2, y2, vehicle_boxes):
                        continue
                    raw_label, raw_conf = age_clf.classify(x1, y1, x2, y2)
                    if raw_label == "child":
                        children_cnt += 1
                    else:
                        adults_cnt += 1
                    norm_boxes.append((
                        -1,
                        x1 / frame_w, y1 / frame_h,
                        x2 / frame_w, y2 / frame_h,
                        tb["conf"], raw_label, raw_conf,
                    ))
                    tile_flags.append(True)

                # Периодическая чистка мёртвых треков
                if detect_cnt % _EVICT_EVERY == 0 and active_tids:
                    bbox_ema.evict(active_tids)
                    age_tracker.evict(active_tids)

                violations = viol_det.analyze(norm_boxes) if norm_boxes else []
                if violation_callback is not None and violations:
                    violation_callback(violations, frame_idx / src_fps)

                if write_annotated:
                    # Сначала bbox людей, потом нарушения поверх (v2)
                    annotated = draw_person_boxes(
                        annotated, norm_boxes, tile_flags, frame_w, frame_h)
                    annotated, vcount = draw_violations(
                        annotated, violations, frame_w, frame_h)
                    if cw_zones:
                        annotated = draw_traffic_light_states(
                            annotated, cw_zones, tl_states, frame_w, frame_h)
                    annotated = _draw_legend_offline(
                        annotated,
                        persons        = len(norm_boxes),
                        ms             = elapsed_ms,
                        violations     = vcount,
                        adults         = adults_cnt,
                        children       = children_cnt,
                        model_sz       = model_label,
                        camera_id      = camera_id,
                        age_calibrated = age_calibrated,
                        frame_idx      = frame_idx,
                        total_frames   = limit,
                        detect_every   = detect_every,
                        tile_enabled   = use_tile,
                        tile_extra     = len(tile_new),
                    )
                    last_annotated = annotated
                else:
                    vcount = sum(1 for item in violations if item.violation != "none")

                stat["detected_frames"]  += 1
                stat["total_persons"]    += len(norm_boxes)
                stat["total_violations"] += vcount
                stat["total_adults"]     += adults_cnt
                stat["total_children"]   += children_cnt
                stat["inference_ms_sum"] += elapsed_ms

            else:
                annotated = last_annotated if last_annotated is not None else frame

            if write_annotated:
                if async_writer is not None:
                    async_writer.write(annotated)
                elif writer is not None:
                    writer.write(annotated)

            stat["total_frames"] += 1

            now = time.perf_counter()
            if now - t_progress >= 2.0:
                t_progress = now
                elapsed    = now - t_start
                rate       = frame_idx / elapsed if elapsed > 0 else 0
                eta        = (limit - frame_idx) / rate if rate > 0 else 0
                avg_ms     = (stat["inference_ms_sum"] / stat["detected_frames"]
                              if stat["detected_frames"] > 0 else 0)
                tile_extra_total = stat["tile_extra_persons"]
                print(
                    f"\r  {_progress_bar(frame_idx, limit)}  "
                    f"ETA:{_format_eta(eta)}  "
                    f"inf:{avg_ms:.0f}ms  "
                    f"viol:{stat['total_violations']}  "
                    f"tile+:{tile_extra_total}  ",
                    end="", flush=True,
                )

    # Освобождение ресурсов
    if async_reader is not None:
        async_reader.close()
        cap = None
    if cap is not None:
        cap.release()
    if async_writer is not None:
        async_writer.close()
    elif writer is not None:
        writer.release()

    stat["elapsed_sec"] = time.perf_counter() - t_start
    print()
    return stat


# ══════════════════════════════════════════════════════════════════════════════
# Вывод итоговой статистики
# ══════════════════════════════════════════════════════════════════════════════

def _print_stats(stat: dict, input_path: Path, output_path: Path):
    elapsed  = stat.get("elapsed_sec", 0)
    det_f    = stat.get("detected_frames", 1) or 1
    avg_ms   = stat["inference_ms_sum"] / det_f
    real_fps = stat["total_frames"] / elapsed if elapsed > 0 else 0

    print()
    print("=" * 58)
    print(f"  Обработан файл   : {input_path.name}")
    if stat.get("annotated_written", True):
        print(f"  Записан файл     : {output_path}")
    else:
        print("  Аннот. видео     : отключено")
    print(f"  Всего кадров     : {stat['total_frames']}")
    print(f"  Кадров с детекц. : {stat['detected_frames']}")
    print(f"  Время обработки  : {_format_eta(elapsed)}")
    print(f"  Ср. скорость     : {real_fps:.1f} кадр/с")
    print(f"  Ср. inference    : {avg_ms:.1f} мс")
    print(f"  Всего людей      : {stat['total_persons']}")
    print(f"    взрослых       : {stat['total_adults']}")
    print(f"    детей          : {stat['total_children']}")
    print(f"  Всего нарушений  : {stat['total_violations']}")
    print(f"  Доп. от тайла    : {stat.get('tile_extra_persons', 0)}")
    print("=" * 58)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Инициализация pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _build_pipeline(camera_id: str):
    zone_mgr    = ZoneManager()
    tl_analyzer = TrafficLightAnalyzer()
    if camera_id:
        n = zone_mgr.reload_for_camera(camera_id)
        print(f"[INFO] Камера '{camera_id}': загружено {n} зон")
    else:
        print("[INFO] Камера не указана — зоны не загружены")
    viol_det = ViolationDetector(zone_mgr, tl_analyzer)
    return zone_mgr, tl_analyzer, viol_det


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Офлайн-аннотация видео: детекция людей + нарушения ПДД",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video",  type=str, help="Путь к одному видеофайлу")
    src_group.add_argument("--folder", type=str, help="Папка с видеофайлами (обработает все .mp4/.avi/...)")

    parser.add_argument("--camera", type=str, default="",
                        help="ID камеры (для загрузки зон и калибровки возраста)")
    parser.add_argument("--output", type=str, default="",
                        help="Путь выходного файла или папки")
    parser.add_argument(
        "--model", type=str, default="v8m",
        help=(
            "Модель детекции. Варианты:\n"
            "  YOLOv8:    v8n v8s v8m v8l v8x\n"
            "  YOLOv8-P6: v8n6 v8s6 v8m6 v8l6 v8x6  <- лучше для мелких объектов\n"
            "  YOLOv9:    v9c v9e\n"
            "  YOLOv10:   v10n v10s v10m v10l v10x\n"
            "  YOLO11:    v11n v11s v11m v11l v11x\n"
            "  RT-DETR:   rtdetr-l rtdetr-x          <- при перекрытиях\n"
            "Рекомендуется: v8m6 или v11m"
        ),
    )
    parser.add_argument("--device",       type=str,   default="auto",
                        help="YOLO device: auto, cpu, 0, 1, ...")
    parser.add_argument("--conf",         type=float, default=0.3,
                        help="Порог уверенности. Рекомендуется 0.25-0.35 для видеонаблюдения")
    parser.add_argument("--imgsz",        type=int,   default=1280,
                        help="Размер входа модели")
    parser.add_argument("--detect-every", type=int,   default=3,
                        help="Детектировать каждый N-й кадр. Рекомендуется 1-5")
    parser.add_argument("--max-frames",   type=int,   default=0,
                        help="Ограничить число обрабатываемых кадров (0 = всё видео)")
    parser.add_argument("--ema-alpha",    type=float, default=1.0,
                        help="EMA-сглаживание bbox (1.0 = выключено, 0.3 = плавно)")
    parser.add_argument(
        "--enhance", type=str, default="none",
        choices=["none", "clahe", "gamma", "both"],
        help=(
            "Предобработка кадра перед детекцией:\n"
            "  none  — без предобработки\n"
            "  clahe — локальная нормализация контраста (лучший выбор при тенях)\n"
            "  gamma — глобальное осветление (гамма=0.6)\n"
            "  both  — gamma + clahe"
        ),
    )
    parser.add_argument("--tile",         action="store_true", default=False,
                        help="Тайловая детекция верхней части кадра (далёкие/мелкие объекты)")
    parser.add_argument("--tile-ratio",   type=float, default=0.65,
                        help="Доля высоты кадра для тайла (0.4-0.8)")
    parser.add_argument("--debug",        action="store_true", default=False,
                        help=(
                            "Режим отладки: raw bbox (голубой), "
                            "отброшен фильтром (красный), транспорт (серый). "
                            "Используй с --max-frames 300"
                        ))
    parser.add_argument("--preload-video",        action="store_true",
                        help="Загрузить видео целиком в RAM перед обработкой")
    parser.add_argument("--writer-queue-size",    type=int, default=64,
                        help="Очередь кадров для асинхронной записи; 0 отключает async writer")
    parser.add_argument("--reader-queue-size",    type=int, default=0,
                        help="Очередь кадров для фонового чтения; 0 отключает async reader")
    parser.add_argument("--inference-batch-size", type=int, default=16,
                        help="Кадров на YOLO batch при preload + detect-every=1 (без --tile/--enhance)")
    parser.add_argument("--no-annotated-video",   action="store_true",
                        help="Не записывать аннотированный mp4 (только статистика, самый быстрый режим)")

    args = parser.parse_args()

    camera_id    = args.camera.strip()
    detect_every = max(1, args.detect_every)
    tile_ratio   = max(0.3, min(0.9, args.tile_ratio))
    ema_alpha    = max(0.1, min(1.0, args.ema_alpha))

    if detect_every > 5:
        print(f"[WARN] --detect-every {detect_every} велико — трекер будет терять людей!")
    if args.tile:
        print(f"[INFO] Тайловая детекция: ВКЛ (верхние {tile_ratio:.0%} кадра)")
    if args.preload_video and (args.tile or args.enhance != "none"):
        print("[WARN] --preload-video с --tile или --enhance: batched inference отключён, "
              "используется поштучная обработка")

    model = load_yolo(args.model, device=args.device)

    if args.video:
        sources = [Path(args.video)]
    else:
        folder  = Path(args.folder)
        sources = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not sources:
            print(f"[ERROR] Нет видеофайлов в папке: {folder}")
            sys.exit(1)
        print(f"[INFO] Найдено {len(sources)} файлов в {folder}")

    zone_mgr, _, _ = _build_pipeline(camera_id)
    age_clf: AgeClassifier | None = None

    age_tracker = AgeTracker(
        window         = 15,
        min_votes      = 5,
        flip_threshold = 0.70,
        warmup_scale   = 0.50,
    )
    bbox_ema = BboxEMA(alpha=ema_alpha)

    total_stat: dict = {
        "total_frames": 0, "detected_frames": 0,
        "total_persons": 0, "total_violations": 0,
        "total_adults": 0, "total_children": 0,
        "inference_ms_sum": 0.0, "elapsed_sec": 0.0,
        "tile_extra_persons": 0,
    }

    for src in sources:
        if not src.exists():
            print(f"[WARN] Файл не найден, пропуск: {src}")
            continue

        out_path = _make_output_path(src, args.output)

        # tl_analyzer и viol_det пересоздаём на каждый файл — чистое состояние светофоров
        tl_analyzer = TrafficLightAnalyzer()
        viol_det    = ViolationDetector(zone_mgr, tl_analyzer)

        stat = process_video(
            input_path           = src,
            output_path          = out_path,
            model                = model,
            zone_mgr             = zone_mgr,
            tl_analyzer          = tl_analyzer,
            viol_det             = viol_det,
            age_clf              = age_clf,
            age_tracker          = age_tracker,
            bbox_ema             = bbox_ema,
            camera_id            = camera_id,
            conf                 = args.conf,
            imgsz                = args.imgsz,
            detect_every         = detect_every,
            max_frames           = args.max_frames,
            use_tile             = args.tile,
            tile_top_ratio       = tile_ratio,
            enhance              = args.enhance,
            debug_mode           = args.debug,
            preload_video        = args.preload_video,
            writer_queue_size    = args.writer_queue_size,
            reader_queue_size    = args.reader_queue_size,
            inference_batch_size = args.inference_batch_size,
            write_annotated      = not args.no_annotated_video,
        )

        if stat:
            _print_stats(stat, src, out_path)
            for k in total_stat:
                total_stat[k] += stat.get(k, 0)

    if len(sources) > 1 and total_stat["total_frames"] > 0:
        print("=" * 58)
        print(f"  ИТОГО по {len(sources)} файлам")
        elapsed = total_stat["elapsed_sec"]
        det_f   = total_stat["detected_frames"] or 1
        print(f"  Кадров обработано  : {total_stat['total_frames']}")
        print(f"  Время              : {_format_eta(elapsed)}")
        print(f"  Ср. inference      : {total_stat['inference_ms_sum'] / det_f:.1f} мс")
        print(f"  Людей (суммарно)   : {total_stat['total_persons']}")
        print(f"  Нарушений (сумм.)  : {total_stat['total_violations']}")
        print(f"  Доп. от тайла      : {total_stat['tile_extra_persons']}")
        print("=" * 58)

    print("[✓] Готово.")


if __name__ == "__main__":
    main()