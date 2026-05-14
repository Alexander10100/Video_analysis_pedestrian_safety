"""
offline_detect.py — Офлайн-обработка видео: аннотация кадров и сохранение результата.

Запускает тот же pipeline визуализации, что и stream_detect.py, но без Flask и трансляции:
  - Рисует зоны, светофоры, bbox-детекции, нарушения, легенду — всё то же, что на трансляции.
  - Записывает аннотированное видео в выходной файл.
  - Выводит прогресс в консоль и итоговую статистику.

Использование:
    # Один файл
    python offline_detect.py --video input.mp4 --camera cam_01

    # Папка (обрабатывает все видео последовательно)
    python offline_detect.py --folder /path/to/videos --camera cam_01

    # Задать выходной файл/папку, модель, порог
    python offline_detect.py --video input.mp4 --camera cam_01 \\
        --output result.mp4 --model m --conf 0.45

    # Детектировать каждый N-й кадр (ускорение), но писать все кадры в файл
    python offline_detect.py --video input.mp4 --camera cam_01 --detect-every 3

    # Ограничить число обрабатываемых кадров
    python offline_detect.py --video input.mp4 --camera cam_01 --max-frames 1800

Выход:
    По умолчанию: <имя_входного_файла>_annotated.mp4 рядом с входным файлом.
    При --output: указанный путь (для папки — это директория под выходные файлы).

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

from age_classifier import AgeClassifier, AgeTracker, BboxEMA  # ← добавлены AgeTracker, BboxEMA
from traffic_light import TrafficLightAnalyzer
from violation_detector import (
    ViolationDetector,
    draw_violations,
    draw_zones,
    draw_traffic_light_states,
    _put_text_pil,
)
from zone_manager import ZoneManager

# ── Поддерживаемые форматы ────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}

# ── Кодек для выходного файла ─────────────────────────────────────────────────
OUTPUT_FOURCC = "mp4v"
OUTPUT_EXT    = ".mp4"

# ── Периодичность чистки мёртвых треков ──────────────────────────────────────
_EVICT_EVERY = 150   # детектируемых кадров


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


def load_yolo(size: str, device: str = "auto"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)
    name = f"yolov8{size}.pt"
    resolved_device = _resolve_device(device)
    use_half = _use_half_for_device(resolved_device)
    print(f"[INFO] Загружаем модель {name}…")
    print(f"[INFO] YOLO device: {resolved_device}, half: {use_half}")
    model = YOLO(name)
    model.to(resolved_device)
    model.model_name = name
    model.device_name = resolved_device
    model.use_half = use_half
    return model


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
) -> np.ndarray:
    cam_str    = (camera_id or "—")[:18]
    age_status = "ДА" if age_calibrated else "НЕТ"
    progress   = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)

    lines = [
        (f"Inference: {ms:.0f} ms",                (180, 180, 180)),
        (f"Кадр:      {progress}",                 (130, 130, 130)),
        (f"Людей:     {persons}",                  ( 50, 205,  50)),
        (f"  взрослых: {adults}",                  ( 50, 200,  50)),
        (f"  детей:    {children}",                (  0, 165, 255)),
        (f"Наруш.:    {violations}",               ( 60,  60, 230)),
        (f"Модель:    yolov8{model_sz}",            ( 90, 130, 255)),
        (f"Обраб. 1/{detect_every} кадров",        (180, 130,   0)),
        (f"Камера:    {cam_str}",                   (100, 200, 200)),
        (f"Калибр.:   {age_status}",
         (50, 205, 50) if age_calibrated else (180, 130, 0)),
    ]

    pad, lh   = 8, 22
    font_size = 13
    box_w     = 235
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
    age_tracker: AgeTracker,       # ← новый параметр
    bbox_ema: BboxEMA,             # ← новый параметр
    camera_id: str,
    conf: float,
    imgsz: int,
    detect_every: int,
    max_frames: int,
    violation_callback: Callable[[list, float], None] | None = None,
    preload_video: bool = False,
    writer_queue_size: int = 64,
    inference_batch_size: int = 16,
    write_annotated: bool = True,
    reader_queue_size: int = 0,
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
    print(f"       Разрешение    : {frame_w}×{frame_h}  FPS: {src_fps:.2f}")
    print(f"       Всего кадров  : {total_frames}  Лимит: {limit}")
    print(f"       Выходной файл : {output_path}")
    print(f"       Детекция 1/{detect_every} кадров")
    print(f"       Preload RAM    : {'on' if preload_video else 'off'}")
    print(f"       Reader queue   : {reader_queue_size if not preload_video else 0}")
    print(f"       Writer queue   : {writer_queue_size}\n")
    print(f"       Infer batch    : {inference_batch_size}\n")
    print(f"       Annotated mp4  : {'on' if write_annotated else 'off'}\n")

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

    # ── AgeClassifier: создать/переинициализировать под высоту кадра ─────────
    if age_clf is None or age_clf.frame_height != frame_h:
        if camera_id:
            age_clf = AgeClassifier.load_for_camera(camera_id, frame_h)
        else:
            age_clf = AgeClassifier(frame_h)
    age_calibrated = age_clf.is_calibrated()

    # ── Сбрасываем EMA и трекер: история предыдущего файла не применима ──────
    # track_id-ы между разными видеофайлами могут совпадать случайно,
    # а геометрия сцены меняется — старые EMA-состояния только навредят.
    age_tracker.reset()
    bbox_ema.reset()

    # ── VideoWriter ───────────────────────────────────────────────────────────
    writer = None
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

    # ── Статистика ────────────────────────────────────────────────────────────
    stat = {
        "total_frames":      0,
        "detected_frames":   0,
        "total_persons":     0,
        "total_violations":  0,
        "total_adults":      0,
        "total_children":    0,
        "inference_ms_sum":  0.0,
        "elapsed_sec":       0.0,
        "annotated_written": write_annotated,
    }

    cw_zones  = zone_mgr.crosswalk_zones()
    all_zones = zone_mgr.get_all()
    async_reader = (
        AsyncVideoReader(cap, limit, reader_queue_size)
        if cap is not None and not preload_video and reader_queue_size > 0
        else None
    )

    last_annotated: np.ndarray | None = None
    frame_idx   = 0
    detect_cnt  = 0          # счётчик именно детектируемых кадров (для evict)
    t_start     = time.perf_counter()
    t_progress  = t_start
    track_kwargs = {
        "classes": [0],
        "conf": conf,
        "iou": 0.45,
        "imgsz": imgsz,
        "persist": True,
        "verbose": False,
        "device": getattr(model, "device_name", "auto"),
        "half": bool(getattr(model, "use_half", False)),
    }

    use_batched_inference = (
        preloaded_frames is not None
        and detect_every == 1
        and inference_batch_size > 1
    )

    if use_batched_inference:
        for batch_start in range(0, limit, inference_batch_size):
            batch_frames = preloaded_frames[batch_start:batch_start + inference_batch_size]
            if not batch_frames:
                break

            t0 = time.perf_counter()
            if _TORCH is not None:
                with _TORCH.inference_mode():
                    batch_results = model.track(batch_frames, **track_kwargs)
            else:
                batch_results = model.track(batch_frames, **track_kwargs)
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

                norm_boxes = []
                adults_cnt = 0
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
                        tid = int(ids[i])
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
                            conf_val,
                            age_label,
                            age_conf,
                        ))

                if detect_cnt % _EVICT_EVERY == 0 and active_tids:
                    bbox_ema.evict(active_tids)
                    age_tracker.evict(active_tids)

                violations = viol_det.analyze(norm_boxes) if norm_boxes else []
                if violation_callback is not None and violations:
                    violation_callback(violations, frame_idx / src_fps)
                if write_annotated:
                    annotated, vcount = draw_violations(annotated, violations, frame_w, frame_h)
                else:
                    vcount = sum(1 for item in violations if item.violation != "none")

                if write_annotated and cw_zones:
                    annotated = draw_traffic_light_states(
                        annotated, cw_zones, tl_states, frame_w, frame_h)

                persons = len(norm_boxes)
                if write_annotated:
                    annotated = _draw_legend_offline(
                        annotated,
                        persons=persons,
                        ms=per_frame_ms,
                        violations=vcount,
                        adults=adults_cnt,
                        children=children_cnt,
                        model_sz=getattr(model, "model_name", "?").replace("yolov8", "").replace(".pt", ""),
                        camera_id=camera_id,
                        age_calibrated=age_calibrated,
                        frame_idx=frame_idx,
                        total_frames=limit,
                        detect_every=detect_every,
                    )

                    last_annotated = annotated
                    if async_writer is not None:
                        async_writer.write(annotated)
                    elif writer is not None:
                        writer.write(annotated)

                stat["total_frames"] += 1
                stat["detected_frames"] += 1
                stat["total_persons"] += persons
                stat["total_violations"] += vcount
                stat["total_adults"] += adults_cnt
                stat["total_children"] += children_cnt
                stat["inference_ms_sum"] += per_frame_ms

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
                    f"viol:{stat['total_violations']}  ",
                    end="", flush=True,
                )

    while frame_idx < limit and not use_batched_inference:
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

        # ── Детекция только на каждом detect_every-м кадре ───────────────────
        if frame_idx % detect_every == 0:
            detect_cnt += 1

            t0 = time.perf_counter()
            if _TORCH is not None:
                with _TORCH.inference_mode():
                    results = model.track(frame, **track_kwargs)[0]
            else:
                results = model.track(frame, **track_kwargs)[0]

            if cw_zones:
                tl_analyzer.process_frame(frame, cw_zones)
            tl_states = tl_analyzer.get_all_states()

            annotated = frame.copy() if write_annotated else None
            if write_annotated and all_zones:
                annotated = draw_zones(annotated, all_zones, tl_states)

            # ── Bbox + EMA + классификация + temporal smoothing ───────────────
            norm_boxes    = []
            adults_cnt    = 0
            children_cnt  = 0
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
                    tid = int(ids[i])
                    conf_val = float(confs[i])

                    if tid >= 0:
                        active_tids.add(tid)

                    # 1. EMA: сглаживаем координаты по истории трека
                    x1, y1, x2, y2 = bbox_ema.smooth(tid, raw_x1, raw_y1, raw_x2, raw_y2)

                    # 2. Однокадровая классификация на сглаженном bbox
                    raw_label, raw_conf = age_clf.classify(x1, y1, x2, y2)

                    # 3. Temporal smoothing: стабилизируем метку через скользящее окно
                    age_label, age_conf = age_tracker.update(tid, raw_label, raw_conf)

                    if age_label == "child":
                        children_cnt += 1
                    else:
                        adults_cnt += 1

                    norm_boxes.append((
                        tid,
                        x1 / frame_w, y1 / frame_h,
                        x2 / frame_w, y2 / frame_h,
                        conf_val,
                        age_label,
                        age_conf,
                    ))

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── Периодическая чистка мёртвых треков ──────────────────────────
            if detect_cnt % _EVICT_EVERY == 0 and active_tids:
                bbox_ema.evict(active_tids)
                age_tracker.evict(active_tids)

            violations        = viol_det.analyze(norm_boxes) if norm_boxes else []
            if violation_callback is not None and violations:
                violation_callback(violations, frame_idx / src_fps)
            if write_annotated:
                annotated, vcount = draw_violations(annotated, violations, frame_w, frame_h)
            else:
                vcount = sum(1 for item in violations if item.violation != "none")

            if write_annotated and cw_zones:
                annotated = draw_traffic_light_states(
                    annotated, cw_zones, tl_states, frame_w, frame_h)

            persons = len(norm_boxes)

            if write_annotated:
                annotated = _draw_legend_offline(
                    annotated,
                    persons        = persons,
                    ms             = elapsed_ms,
                    violations     = vcount,
                    adults         = adults_cnt,
                    children       = children_cnt,
                    model_sz       = getattr(model, "model_name", "?").replace("yolov8", "").replace(".pt", ""),
                    camera_id      = camera_id,
                    age_calibrated = age_calibrated,
                    frame_idx      = frame_idx,
                    total_frames   = limit,
                    detect_every   = detect_every,
                )

                last_annotated = annotated

            stat["detected_frames"]  += 1
            stat["total_persons"]    += persons
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
            print(
                f"\r  {_progress_bar(frame_idx, limit)}  "
                f"ETA:{_format_eta(eta)}  "
                f"inf:{avg_ms:.0f}ms  "
                f"viol:{stat['total_violations']}  ",
                end="", flush=True,
            )

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
    print("═" * 58)
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
    print("═" * 58)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Инициализация pipeline под камеру
# ══════════════════════════════════════════════════════════════════════════════

def _build_pipeline(
    camera_id: str,
) -> tuple[ZoneManager, TrafficLightAnalyzer, ViolationDetector]:
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video",  type=str,
                            help="Путь к одному видеофайлу")
    src_group.add_argument("--folder", type=str,
                            help="Папка с видеофайлами (обработает все .mp4/.avi/…)")

    parser.add_argument("--camera",       type=str, default="",
                        help="ID камеры (для загрузки зон и калибровки возраста)")
    parser.add_argument("--output",       type=str, default="",
                        help="Путь выходного файла или папки")
    parser.add_argument("--model",        type=str, default="m",
                        choices=["n", "s", "m", "l", "x"],
                        help="Размер модели YOLOv8")
    parser.add_argument("--device",       type=str, default="auto",
                        help="YOLO device: auto, cpu, or CUDA device index such as 0")
    parser.add_argument("--conf",         type=float, default=0.45,
                        help="Порог уверенности детекции")
    parser.add_argument("--imgsz",        type=int,   default=640,
                        help="Размер входа модели")
    parser.add_argument("--detect-every", type=int,   default=1,
                        help="Детектировать каждый N-й кадр (1 = каждый кадр)")
    parser.add_argument("--max-frames",   type=int,   default=0,
                        help="Ограничить число обрабатываемых кадров (0 = всё видео)")
    parser.add_argument("--preload-video", action="store_true",
                        help="Загрузить видео целиком в RAM перед обработкой")
    parser.add_argument("--writer-queue-size", type=int, default=64,
                        help="Очередь кадров для асинхронной записи видео; 0 отключает async writer")
    parser.add_argument("--reader-queue-size", type=int, default=0,
                        help="Очередь кадров для фонового чтения видео; 0 отключает async reader")
    parser.add_argument("--inference-batch-size", type=int, default=16,
                        help="Кадров на один YOLO batch при preload и detect-every=1")
    parser.add_argument("--no-annotated-video", action="store_true",
                        help="Не рисовать и не сохранять аннотированное mp4; самый быстрый режим для отчетов")

    args = parser.parse_args()

    camera_id    = args.camera.strip()
    detect_every = max(1, args.detect_every)

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

    # ── AgeTracker и BboxEMA — одни на весь запуск, сбрасываются между файлами
    # внутри process_video через .reset(), поэтому создаём здесь один раз.
    age_tracker = AgeTracker(
        window         = 15,
        min_votes      = 5,
        flip_threshold = 0.70,
        warmup_scale   = 0.50,
    )
    bbox_ema = BboxEMA(alpha=0.35)

    total_stat: dict = {
        "total_frames": 0, "detected_frames": 0,
        "total_persons": 0, "total_violations": 0,
        "total_adults": 0, "total_children": 0,
        "inference_ms_sum": 0.0, "elapsed_sec": 0.0,
    }

    for src in sources:
        if not src.exists():
            print(f"[WARN] Файл не найден, пропуск: {src}")
            continue

        out_path = _make_output_path(src, args.output if len(sources) == 1 else args.output)

        # tl_analyzer и viol_det пересоздаём на каждый файл — чистое состояние светофоров
        tl_analyzer = TrafficLightAnalyzer()
        viol_det    = ViolationDetector(zone_mgr, tl_analyzer)

        stat = process_video(
            input_path   = src,
            output_path  = out_path,
            model        = model,
            zone_mgr     = zone_mgr,
            tl_analyzer  = tl_analyzer,
            viol_det     = viol_det,
            age_clf      = age_clf,
            age_tracker  = age_tracker,   # ← передаём
            bbox_ema     = bbox_ema,       # ← передаём
            camera_id    = camera_id,
            conf         = args.conf,
            imgsz        = args.imgsz,
            detect_every = detect_every,
            max_frames   = args.max_frames,
            preload_video = args.preload_video,
            writer_queue_size = args.writer_queue_size,
            inference_batch_size = args.inference_batch_size,
            write_annotated = not args.no_annotated_video,
            reader_queue_size = args.reader_queue_size,
        )

        if stat:
            _print_stats(stat, src, out_path)
            for k in total_stat:
                total_stat[k] += stat.get(k, 0)

    if len(sources) > 1 and total_stat["total_frames"] > 0:
        print("══════════════════════════════════════════════════════════")
        print(f"  ИТОГО по {len(sources)} файлам")
        elapsed = total_stat["elapsed_sec"]
        det_f   = total_stat["detected_frames"] or 1
        print(f"  Кадров обработано  : {total_stat['total_frames']}")
        print(f"  Время              : {_format_eta(elapsed)}")
        print(f"  Ср. inference      : {total_stat['inference_ms_sum'] / det_f:.1f} мс")
        print(f"  Людей (суммарно)   : {total_stat['total_persons']}")
        print(f"  Нарушений (сумм.)  : {total_stat['total_violations']}")
        print("══════════════════════════════════════════════════════════")

    print("[✓] Готово.")


if __name__ == "__main__":
    main()
