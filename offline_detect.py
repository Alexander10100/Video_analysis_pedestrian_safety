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
import time
from pathlib import Path

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


def load_yolo(size: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)
    name = f"yolov8{size}.pt"
    print(f"[INFO] Загружаем модель {name}…")
    return YOLO(name)


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
    print(f"       Детекция 1/{detect_every} кадров\n")

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"[ERROR] Не удалось создать VideoWriter: {output_path}")
        cap.release()
        return {}

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
    }

    cw_zones  = zone_mgr.crosswalk_zones()
    all_zones = zone_mgr.get_all()

    last_annotated: np.ndarray | None = None
    frame_idx   = 0
    detect_cnt  = 0          # счётчик именно детектируемых кадров (для evict)
    t_start     = time.perf_counter()
    t_progress  = t_start

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # ── Детекция только на каждом detect_every-м кадре ───────────────────
        if frame_idx % detect_every == 0:
            detect_cnt += 1

            t0 = time.perf_counter()
            results = model.track(
                frame, classes=[0], conf=conf, iou=0.45,
                imgsz=imgsz, persist=True, verbose=False,
            )[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if cw_zones:
                tl_analyzer.process_frame(frame, cw_zones)
            tl_states = tl_analyzer.get_all_states()

            annotated = frame.copy()
            if all_zones:
                annotated = draw_zones(annotated, all_zones, tl_states)

            # ── Bbox + EMA + классификация + temporal smoothing ───────────────
            norm_boxes    = []
            adults_cnt    = 0
            children_cnt  = 0
            active_tids: set[int] = set()

            if results.boxes is not None:
                for box in results.boxes:
                    raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                    tid      = int(box.id[0])     if box.id   is not None else -1
                    conf_val = float(box.conf[0]) if box.conf is not None else 0.0

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

            # ── Периодическая чистка мёртвых треков ──────────────────────────
            if detect_cnt % _EVICT_EVERY == 0 and active_tids:
                bbox_ema.evict(active_tids)
                age_tracker.evict(active_tids)

            violations        = viol_det.analyze(norm_boxes) if norm_boxes else []
            annotated, vcount = draw_violations(annotated, violations, frame_w, frame_h)

            if cw_zones:
                annotated = draw_traffic_light_states(
                    annotated, cw_zones, tl_states, frame_w, frame_h)

            persons = len(norm_boxes)

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

    cap.release()
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
    print(f"  Записан файл     : {output_path}")
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
    parser.add_argument("--conf",         type=float, default=0.45,
                        help="Порог уверенности детекции")
    parser.add_argument("--imgsz",        type=int,   default=640,
                        help="Размер входа модели")
    parser.add_argument("--detect-every", type=int,   default=1,
                        help="Детектировать каждый N-й кадр (1 = каждый кадр)")
    parser.add_argument("--max-frames",   type=int,   default=0,
                        help="Ограничить число обрабатываемых кадров (0 = всё видео)")

    args = parser.parse_args()

    camera_id    = args.camera.strip()
    detect_every = max(1, args.detect_every)

    model = load_yolo(args.model)

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