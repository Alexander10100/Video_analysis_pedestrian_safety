"""
offline_detect.py — Офлайн-обработка видео: аннотация кадров и сохранение результата.

ИСПРАВЛЕНИЯ:
  - [FIX] Добавлено рисование bbox всех обнаруженных людей (зелёный = взрослый, синий = ребёнок)
  - [FIX] BboxEMA отключён по умолчанию (alpha=1.0) — он сдвигал рамки от реальных людей
  - [FIX] Тайловая детекция верхней части кадра для мелких/далёких объектов (--tile)
  - [FIX] detect-every по умолчанию = 1, предупреждение при значении > 5

Использование:
    python offline_detect.py --video input.mp4 --camera cam_01
    python offline_detect.py --video input.mp4 --camera cam_01 --tile
    python offline_detect.py --folder /path/to/videos --camera cam_01 --tile
    python offline_detect.py --video input.mp4 --camera cam_01 \\
        --output result.mp4 --model m --conf 0.3 --detect-every 3 --tile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
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

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}
OUTPUT_FOURCC = "mp4v"
OUTPUT_EXT = ".mp4"
_EVICT_EVERY = 150

# Цвета bbox
_COLOR_ADULT = (50, 205, 50)   # зелёный  — взрослый
_COLOR_CHILD = (255, 165, 0)   # оранжевый — ребёнок
_COLOR_TILE = (200, 200, 0)   # жёлто-голубой — доп. бокс из тайла
_COLOR_UNKNOWN = (180, 180, 180)   # серый — без трекера / неизвестно


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _progress_bar(current: int, total: int, width: int = 38) -> str:
    if total <= 0:
        return f"[{'?' * width}] ??%"
    pct = current / total
    done = int(pct * width)
    bar = "█" * done + "░" * (width - done)
    return f"[{bar}] {pct:5.1%}"


def _format_eta(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _make_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        p = Path(output_arg)
        if p.is_dir():
            return p / (input_path.stem + "_annotated" + OUTPUT_EXT)
        if not p.suffix:
            return p.with_suffix(OUTPUT_EXT)
        return p
    return input_path.parent / (input_path.stem + "_annotated" + OUTPUT_EXT)


def load_yolo(model_key: str):
    """
    Загружает модель по короткому ключу.

    Поддерживаемые ключи:
      YOLOv8 стандартные:
        v8n, v8s, v8m, v8l, v8x

      YOLOv8-p6 (640→1280 обучены, лучше для мелких объектов):
        v8n6, v8s6, v8m6, v8l6, v8x6

      YOLOv9:
        v9c, v9e

      YOLOv10:
        v10n, v10s, v10m, v10l, v10x

      YOLO11 (ultralytics):
        v11n, v11s, v11m, v11l, v11x

      RT-DETR (трансформер, хорош для перекрытых объектов):
        rtdetr-l, rtdetr-x

    Для видеонаблюдения с мелкими людьми рекомендуется:
      v8m6  — v8m обученный на p6 пирамиде (лучший баланс)
      v8x6  — максимальное качество для мелких объектов
      v11m  — YOLO11 medium, актуальнее v8
      rtdetr-l — если люди частично перекрыты/сливаются с фоном
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)

    MODEL_MAP = {
        # YOLOv8 стандартные
        "v8n": "yolov8n.pt",
        "v8s": "yolov8s.pt",
        "v8m": "yolov8m.pt",
        "v8l": "yolov8l.pt",
        "v8x": "yolov8x.pt",
        # YOLOv8-P6 — обучены на 1280px, имеют 6 уровней FPN вместо 5
        # Значительно лучше детектируют мелкие объекты вдали
        "v8n6": "yolov8n6.pt",
        "v8s6": "yolov8s6.pt",
        "v8m6": "yolov8m6.pt",
        "v8l6": "yolov8l6.pt",
        "v8x6": "yolov8x6.pt",
        # YOLOv9
        "v9c": "yolov9c.pt",
        "v9e": "yolov9e.pt",
        # YOLOv10
        "v10n": "yolov10n.pt",
        "v10s": "yolov10s.pt",
        "v10m": "yolov10m.pt",
        "v10l": "yolov10l.pt",
        "v10x": "yolov10x.pt",
        # YOLO11 (актуальное поколение от Ultralytics)
        "v11n": "yolo11n.pt",
        "v11s": "yolo11s.pt",
        "v11m": "yolo11m.pt",
        "v11l": "yolo11l.pt",
        "v11x": "yolo11x.pt",
        # RT-DETR — трансформер, лучше при перекрытиях и слиянии объектов
        "rtdetr-l": "rtdetr-l.pt",
        "rtdetr-x": "rtdetr-x.pt",
    }

    # Обратная совместимость: старые ключи n/s/m/l/x → v8
    _COMPAT = {"n": "v8n", "s": "v8s", "m": "v8m", "l": "v8l", "x": "v8x"}
    model_key = _COMPAT.get(model_key, model_key)

    if model_key not in MODEL_MAP:
        valid = ", ".join(MODEL_MAP.keys())
        print(f"[ERROR] Неизвестная модель '{model_key}'. Доступные: {valid}")
        sys.exit(1)

    name = MODEL_MAP[model_key]
    print(f"[INFO] Загружаем модель {name}  (ключ: {model_key})…")
    return YOLO(name)


# ══════════════════════════════════════════════════════════════════════════════
# Предобработка кадра для улучшения детекции в зонах тени и бликов
# ══════════════════════════════════════════════════════════════════════════════

def enhance_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    """
    Улучшает контраст кадра перед подачей в модель.
    Оригинальный кадр НЕ меняется — используется только для детекции.
    На выходное видео пишется оригинал (annotated = frame.copy()).

    Режимы (--enhance):
      clahe   — локальная нормализация гистограммы по тайлам 8×8 (рекомендуется)
                Убирает эффект тени: тёмные и светлые зоны выравниваются независимо.
                Лучший выбор для камер с сильными тенями и бликами мокрого асфальта.

      gamma   — гамма-коррекция (осветление тёмных зон, γ=0.6)
                Работает глобально — поднимает тёмные пиксели сохраняя структуру.
                Проще чем CLAHE, полезно когда весь кадр темноват.

      both    — сначала gamma потом clahe (максимальный эффект, чуть медленнее)

      none    — без предобработки (по умолчанию)
    """
    if mode == "none":
        return frame

    out = frame.copy()

    if mode in ("gamma", "both"):
        # Гамма-коррекция: осветляем тёмные пиксели
        gamma = 0.6
        inv = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)

    if mode in ("clahe", "both"):
        # CLAHE: локальная адаптивная нормализация контраста
        # clipLimit=2.0 — ограничение усиления (убирает шум)
        # tileGridSize=(8,8) — размер тайлов для локального выравнивания
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


def draw_person_boxes(
    frame: np.ndarray,
    norm_boxes: list,
    tile_flags: list,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Рисует bbox каждого человека из norm_boxes.

    norm_boxes элемент: (tid, nx1, ny1, nx2, ny2, conf, age_label, age_conf)
    tile_flags[i]:      True если i-й бокс из тайловой детекции

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
            color = _COLOR_TILE
            label_str = f"TILE {conf_val:.0%}"
        elif age_label == "child":
            color = _COLOR_CHILD
            label_str = f"РЕБ {age_conf:.0%}"
        else:
            color = _COLOR_ADULT
            label_str = f"ВЗР {age_conf:.0%}"

        # Рамка
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Подпись над рамкой
        (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, lbl_y - th - 4), (x1 + tw + 4, lbl_y + baseline), color, -1)
        cv2.putText(frame, label_str, (x1 + 2, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Тайловая детекция
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
    Детектируем людей в верхней части кадра (там дальние/мелкие объекты).
    Возвращаем боксы в координатах полного кадра.
    """
    cut_y = int(frame_h * tile_top_ratio)
    tile = frame[:cut_y, :]
    tile_conf = max(conf * 0.7, 0.05)

    results = model(
        tile,
        classes=[0],
        conf=tile_conf,
        iou=0.45,
        imgsz=imgsz,
        verbose=False,
    )[0]

    boxes: list[dict] = []
    if results.boxes is None:
        return boxes

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0]) if box.conf is not None else 0.0
        boxes.append({
            "x1": x1, "y1": y1,   # y тайла == y полного кадра (тайл = верх)
            "x2": x2, "y2": y2,
            "conf": conf_val,
        })
    return boxes


def _merge_tile_boxes(
    norm_boxes: list,
    tile_raw: list[dict],
    frame_w: int,
    frame_h: int,
    iou_thresh: float = 0.35,
) -> list[dict]:
    """
    Возвращает только те боксы из тайла, которые не дублируют уже найденные.
    """
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
    tile_extra: int,
    model_sz: str,
    camera_id: str,
    age_calibrated: bool,
    frame_idx: int,
    total_frames: int,
    detect_every: int,
    tile_enabled: bool = False,
) -> np.ndarray:
    cam_str = (camera_id or "—")[:18]
    age_status = "ДА" if age_calibrated else "НЕТ"
    progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)
    tile_str = f"ВКЛ (+{tile_extra})" if tile_enabled else "ВЫКЛ"

    lines = [
        (f"Inference: {ms:.0f} ms", (180, 180, 180)),
        (f"Кадр:      {progress}", (130, 130, 130)),
        (f"Людей:     {persons}", (50, 205, 50)),
        (f"  взрослых: {adults}", (50, 200, 50)),
        (f"  детей:    {children}", (0, 165, 255)),
        (f"Наруш.:    {violations}", (60, 60, 230)),
        (f"Тайл:      {tile_str}",
         (50, 205, 50) if tile_enabled else (130, 130, 130)),
        (f"Модель:    yolov8{model_sz}", (90, 130, 255)),
        (f"Обраб. 1/{detect_every} кадров", (180, 130, 0)),
        (f"Камера:    {cam_str}", (100, 200, 200)),
        (f"Калибр.:   {age_status}",
         (50, 205, 50) if age_calibrated else (180, 130, 0)),
    ]

    pad, lh = 8, 22
    font_size = 13
    box_w = 235
    box_h = len(lines) * lh + pad * 2

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
    use_tile: bool = False,
    tile_top_ratio: float = 0.65,
    enhance: str = "none",
    debug_mode: bool = False,
    violation_callback: Callable[[list, float], None] | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть: {input_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = max_frames if max_frames > 0 else total_frames

    print(f"\n[INFO] Входной файл : {input_path.name}")
    print(f"       Разрешение    : {frame_w}×{frame_h}  FPS: {src_fps:.2f}")
    print(f"       Всего кадров  : {total_frames}  Лимит: {limit}")
    print(f"       Выходной файл : {output_path}")
    print(f"       Детекция 1/{detect_every} кадров")
    if detect_every > 5:
        print(f"       [WARN] detect-every={detect_every} — трекер будет терять людей! Рекомендуется ≤ 5")
    print(f"       Enhance       : {enhance.upper()}")
    if use_tile:
        cut_px = int(frame_h * tile_top_ratio)
        print(f"       Тайл          : ВКЛ  (верхние {tile_top_ratio:.0%} = {cut_px}px)")
    else:
        print(f"       Тайл          : ВЫКЛ (добавь --tile для мелких/далёких объектов)")
    if debug_mode:
        print(f"       [DEBUG]       : ВКЛ — raw bbox (голубой), отброшен (красный), авто (серый)")
    print()

    if age_clf is None or age_clf.frame_height != frame_h:
        if camera_id:
            age_clf = AgeClassifier.load_for_camera(camera_id, frame_h)
        else:
            age_clf = AgeClassifier(frame_h)
    age_calibrated = age_clf.is_calibrated()

    age_tracker.reset()
    bbox_ema.reset()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"[ERROR] Не удалось создать VideoWriter: {output_path}")
        cap.release()
        return {}

    stat = {
        "total_frames": 0,
        "detected_frames": 0,
        "total_persons": 0,
        "total_violations": 0,
        "total_adults": 0,
        "total_children": 0,
        "inference_ms_sum": 0.0,
        "elapsed_sec": 0.0,
        "tile_extra_persons": 0,
    }

    cw_zones = zone_mgr.crosswalk_zones()
    all_zones = zone_mgr.get_all()

    last_annotated: np.ndarray | None = None
    frame_idx = 0
    detect_cnt = 0
    t_start = time.perf_counter()
    t_progress = t_start

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        if frame_idx % detect_every == 0:
            detect_cnt += 1

            t0 = time.perf_counter()

            # ── Предобработка для улучшения детекции в тени/бликах ────────────
            detect_frame = enhance_frame(frame, enhance)

            # ── Основная детекция (полный кадр + трекер) ──────────────────────
            results = model.track(
                detect_frame, classes=[0, 2, 3, 5, 7], conf=conf, iou=0.45,
                imgsz=imgsz, persist=True, verbose=False,
            )[0]

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── Тайловая детекция верхней части кадра ─────────────────────────
            tile_raw: list[dict] = []
            if use_tile:
                t1 = time.perf_counter()
                tile_raw = _run_tile_detection(
                    model, detect_frame, frame_w, frame_h,
                    conf=conf, imgsz=imgsz,
                    tile_top_ratio=tile_top_ratio,
                )
                elapsed_ms += (time.perf_counter() - t1) * 1000

            if cw_zones:
                tl_analyzer.process_frame(frame, cw_zones)
            tl_states = tl_analyzer.get_all_states()

            annotated = frame.copy()
            if all_zones:
                annotated = draw_zones(annotated, all_zones, tl_states)

            VEHICLE_CLASSES = {2, 3, 5, 7}
            vehicle_boxes: list[tuple] = []
            adults_cnt = 0
            children_cnt = 0
            active_tids: set[int] = set()

            # Первый проход — транспорт
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    if cls_id in VEHICLE_CLASSES:
                        vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                        vehicle_boxes.append((vx1, vy1, vx2, vy2))

            # norm_boxes: (tid, nx1, ny1, nx2, ny2, conf, age_label, age_conf)
            # — ровно 8 элементов, как ожидает viol_det.analyze / _classify
            # tile_flags: bool для каждого бокса — True если из тайла (для draw_person_boxes)
            norm_boxes: list[tuple] = []
            tile_flags: list[bool] = []

            # ── [DEBUG] Рисуем bbox транспортных средств (серые рамки) ──────────
            if debug_mode:
                for vx1, vy1, vx2, vy2 in vehicle_boxes:
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (120, 120, 120), 1)
                    cv2.putText(annotated, "VEH", (vx1 + 2, vy1 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            # Второй проход — люди из основной детекции
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    if cls_id != 0:
                        continue

                    raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                    tid = int(box.id[0]) if box.id is not None else -1
                    conf_val = float(box.conf[0]) if box.conf is not None else 0.0

                    if tid >= 0:
                        active_tids.add(tid)

                    # ── [DEBUG] Рисуем все сырые bbox людей до фильтрации ─────
                    if debug_mode:
                        cv2.rectangle(annotated, (raw_x1, raw_y1), (raw_x2, raw_y2),
                                      (0, 255, 255), 1)  # жёлто-голубой = raw
                        cv2.putText(annotated, f"RAW {conf_val:.0%}",
                                    (raw_x1 + 2, raw_y1 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

                    if person_inside_vehicle(raw_x1, raw_y1, raw_x2, raw_y2, vehicle_boxes):
                        # ── [DEBUG] Показываем отфильтрованных красной рамкой ─
                        if debug_mode:
                            cv2.rectangle(annotated, (raw_x1, raw_y1), (raw_x2, raw_y2),
                                          (0, 0, 255), 2)  # красный = отброшен
                            cv2.putText(annotated, "IN_VEH",
                                        (raw_x1 + 2, raw_y2 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        continue

                    # [FIX] EMA с alpha=1.0 = без сглаживания (не сдвигаем рамку)
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

            # Тайловые боксы — только уникальные (не дублируют основные)
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

            # [FIX] Сначала рисуем bbox всех людей, потом нарушения поверх
            annotated = draw_person_boxes(annotated, norm_boxes, tile_flags, frame_w, frame_h)

            violations = viol_det.analyze(norm_boxes) if norm_boxes else []
            if violation_callback is not None and violations:
                violation_callback(violations, frame_idx / src_fps)
            annotated, vcount = draw_violations(annotated, violations, frame_w, frame_h)

            if cw_zones:
                annotated = draw_traffic_light_states(
                    annotated, cw_zones, tl_states, frame_w, frame_h)

            persons = len(norm_boxes)

            # Берём имя файла модели без расширения (работает для любого семейства)
            raw_name = getattr(model, "model_name", None) or str(getattr(model, "ckpt_path", "?"))
            model_sz = Path(raw_name).stem  # "yolov8m" → "yolov8m", "yolo11m" → "yolo11m"
            annotated = _draw_legend_offline(
                annotated,
                persons=persons,
                ms=elapsed_ms,
                violations=vcount,
                adults=adults_cnt,
                children=children_cnt,
                tile_extra=len(tile_new),
                model_sz=model_sz,
                camera_id=camera_id,
                age_calibrated=age_calibrated,
                frame_idx=frame_idx,
                total_frames=limit,
                detect_every=detect_every,
                tile_enabled=use_tile,
            )

            last_annotated = annotated

            stat["detected_frames"] += 1
            stat["total_persons"] += persons
            stat["total_violations"] += vcount
            stat["total_adults"] += adults_cnt
            stat["total_children"] += children_cnt
            stat["inference_ms_sum"] += elapsed_ms

        else:
            annotated = last_annotated if last_annotated is not None else frame

        writer.write(annotated)
        stat["total_frames"] += 1

        now = time.perf_counter()
        if now - t_progress >= 2.0:
            t_progress = now
            elapsed = now - t_start
            rate = frame_idx / elapsed if elapsed > 0 else 0
            eta = (limit - frame_idx) / rate if rate > 0 else 0
            avg_ms = (stat["inference_ms_sum"] / stat["detected_frames"]
                      if stat["detected_frames"] > 0 else 0)
            print(
                f"\r  {_progress_bar(frame_idx, limit)}  "
                f"ETA:{_format_eta(eta)}  "
                f"inf:{avg_ms:.0f}ms  "
                f"viol:{stat['total_violations']}  "
                f"tile+:{stat['tile_extra_persons']}  ",
                end="", flush=True,
            )

    cap.release()
    writer.release()
    stat["elapsed_sec"] = time.perf_counter() - t_start
    print()
    return stat


# ══════════════════════════════════════════════════════════════════════════════
# Статистика
# ══════════════════════════════════════════════════════════════════════════════

def _print_stats(stat: dict, input_path: Path, output_path: Path):
    elapsed = stat.get("elapsed_sec", 0)
    det_f = stat.get("detected_frames", 1) or 1
    avg_ms = stat["inference_ms_sum"] / det_f
    real_fps = stat["total_frames"] / elapsed if elapsed > 0 else 0

    print()
    print("═" * 58)
    print(f"  Обработан файл   : {input_path.name}")
    print(f"  Записан файл     : {output_path}")
    print(f"  Всего кадров     : {stat['total_frames']}")
    print(f"  Кадров с детекц. : {stat['detected_frames']}")
    print(f"  Время обработки  : {_format_eta(elapsed)}")
    print(f"  Ср. скорость     : {real_fps:.1f} кадр/с")
    print(f"  Ср. inference    : {avg_ms:.1f} мс")
    print(f"  Всего людей      : {stat['total_persons']}")
    print(f"    взрослых       : {stat['total_adults']}")
    print(f"    детей          : {stat['total_children']}")
    print(f"  Наруш. (сумм.)   : {stat['total_violations']}")
    print(f"  Доп. от тайла    : {stat.get('tile_extra_persons', 0)}")
    print("═" * 58)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _build_pipeline(camera_id: str):
    zone_mgr = ZoneManager()
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video", type=str, help="Путь к одному видеофайлу")
    src_group.add_argument("--folder", type=str, help="Папка с видеофайлами")

    parser.add_argument("--camera", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        default="v8m",
        help=(
            "Модель детекции. Варианты:\n"
            "  YOLOv8:    v8n v8s v8m v8l v8x\n"
            "  YOLOv8-P6: v8n6 v8s6 v8m6 v8l6 v8x6  ← лучше для мелких объектов\n"
            "  YOLOv9:    v9c v9e\n"
            "  YOLOv10:   v10n v10s v10m v10l v10x\n"
            "  YOLO11:    v11n v11s v11m v11l v11x\n"
            "  RT-DETR:   rtdetr-l rtdetr-x  ← при перекрытиях\n"
            "Рекомендуется для видеонаблюдения: v8m6 или v11m"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Порог уверенности. Рекомендуется 0.25–0.35 для видеонаблюдения")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--detect-every", type=int, default=3,
                        help="Детектировать каждый N-й кадр. Рекомендуется 1–5")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ema-alpha", type=float, default=1.0,
                        help="EMA-сглаживание bbox (1.0 = выключено, 0.3 = плавно)")
    parser.add_argument(
        "--enhance",
        type=str,
        default="none",
        choices=["none", "clahe", "gamma", "both"],
        help=(
            "Предобработка кадра перед детекцией (оригинал в видео не меняется):\n"
            "  none  — без предобработки\n"
            "  clahe — локальная нормализация контраста по тайлам (лучший выбор при тенях)\n"
            "  gamma — глобальное осветление тёмных зон (γ=0.6)\n"
            "  both  — gamma + clahe (максимальный эффект)"
        ),
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        default=False,
        help="Тайловая детекция верхней части кадра (для далёких/мелких объектов)",
    )
    parser.add_argument(
        "--tile-ratio",
        type=float,
        default=0.65,
        help="Доля высоты кадра для тайла (0.4–0.8)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Режим отладки: рисует все raw bbox людей (голубой), "
            "отброшенных фильтром (красный), bbox машин (серый). "
            "Используй с --max-frames 300 чтобы быстро проверить проблемный участок."
        ),
    )

    args = parser.parse_args()

    camera_id = args.camera.strip()
    detect_every = max(1, args.detect_every)
    tile_ratio = max(0.3, min(0.9, args.tile_ratio))
    ema_alpha = max(0.1, min(1.0, args.ema_alpha))

    if detect_every > 5:
        print(f"[WARN] --detect-every {detect_every} велико — трекер будет терять людей!")

    if args.tile:
        print(f"[INFO] Тайловая детекция: ВКЛ (верхние {tile_ratio:.0%} кадра)")

    model = load_yolo(args.model)

    if args.video:
        sources = [Path(args.video)]
    else:
        folder = Path(args.folder)
        sources = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not sources:
            print(f"[ERROR] Нет видеофайлов в папке: {folder}")
            sys.exit(1)
        print(f"[INFO] Найдено {len(sources)} файлов в {folder}")

    zone_mgr, _, _ = _build_pipeline(camera_id)
    age_clf: AgeClassifier | None = None

    age_tracker = AgeTracker(
        window=15,
        min_votes=5,
        flip_threshold=0.70,
        warmup_scale=0.50,
    )
    # [FIX] alpha=1.0 по умолчанию — без сглаживания, рамки точно на людях
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

        out_path = _make_output_path(src, args.output if len(sources) == 1 else args.output)

        tl_analyzer = TrafficLightAnalyzer()
        viol_det = ViolationDetector(zone_mgr, tl_analyzer)

        stat = process_video(
            input_path=src,
            output_path=out_path,
            model=model,
            zone_mgr=zone_mgr,
            tl_analyzer=tl_analyzer,
            viol_det=viol_det,
            age_clf=age_clf,
            age_tracker=age_tracker,
            bbox_ema=bbox_ema,
            camera_id=camera_id,
            conf=args.conf,
            imgsz=args.imgsz,
            detect_every=detect_every,
            max_frames=args.max_frames,
            use_tile=args.tile,
            tile_top_ratio=tile_ratio,
            enhance=args.enhance,
            debug_mode=args.debug,
        )

        if stat:
            _print_stats(stat, src, out_path)
            for k in total_stat:
                total_stat[k] += stat.get(k, 0)

    if len(sources) > 1 and total_stat["total_frames"] > 0:
        print("══════════════════════════════════════════════════════════")
        print(f"  ИТОГО по {len(sources)} файлам")
        elapsed = total_stat["elapsed_sec"]
        det_f = total_stat["detected_frames"] or 1
        print(f"  Кадров обработано  : {total_stat['total_frames']}")
        print(f"  Время              : {_format_eta(elapsed)}")
        print(f"  Ср. inference      : {total_stat['inference_ms_sum'] / det_f:.1f} мс")
        print(f"  Людей (суммарно)   : {total_stat['total_persons']}")
        print(f"  Нарушений (сумм.)  : {total_stat['total_violations']}")
        print(f"  Доп. от тайла      : {total_stat['tile_extra_persons']}")
        print("══════════════════════════════════════════════════════════")

    print("[✓] Готово.")


if __name__ == "__main__":
    main()