"""
stream_detect.py — Веб-визуализация детекции людей с HLS-потока или папки с видео
                   + детекция нарушений ПДД (зоны дороги / пешеходный переход / светофор)

Установка зависимостей:
    pip install ultralytics opencv-python flask pillow

Запуск:
    python stream_detect.py --url "https://..."
    python stream_detect.py --folder "/path/to/videos"

Открыть браузер: http://localhost:5000
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from stream_detect_web import create_app

from zone_manager        import ZoneManager
from traffic_light       import TrafficLightAnalyzer
from violation_detector  import (
    ViolationDetector, draw_violations, draw_zones,
    draw_traffic_light_states, _put_text_pil,
)

# Инициализируем глобальные объекты разметки
zone_mgr    = ZoneManager()
tl_analyzer = TrafficLightAnalyzer()
viol_det    = ViolationDetector(zone_mgr, tl_analyzer)

# ──────────────────────────────────────────────────────────────────────────────
# Глобальное состояние
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "conf":          0.45,
    "imgsz":         640,
    "fpm":           60,
    "persons":       0,
    "fps":           0.0,
    "dfps":          0.0,
    "ms":            0.0,
    "frames":        0,
    "frame_skip":    1,
    "source_type":   "INIT",
    "ts":            time.time(),
    "model_size":    "m",
    "model_loading": None,
    "violations":    0,
    "camera_id":     None,    # активная камера разметки
    "detect_enabled": True,  # флаг включения детекции
    "lock":          threading.Lock(),
}

_source_url:    str | None = None
_source_folder: str | None = None
_current_video: str | None = None
_restart_event      = threading.Event()
_model_reload_event = threading.Event()
_clear_cache_event  = threading.Event()

_frame_queue: queue.Queue = queue.Queue(maxsize=2)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}

# ──────────────────────────────────────────────────────────────────────────────
# YOLO — горячая замена модели
# ──────────────────────────────────────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()


def load_yolo(size: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)
    name = f"yolov8{size}.pt"
    print(f"[INFO] Загружаем {name}…")
    return YOLO(name)


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            with state["lock"]:
                sz = state["model_size"]
            _model = load_yolo(sz)
    return _model


def _do_reload_model(size: str):
    """Фоновая загрузка новой модели с атомарной заменой."""
    global _model
    new_m = load_yolo(size)
    with _model_lock:
        _model = new_m
    with state["lock"]:
        state["model_size"]    = size
        state["model_loading"] = None
    _model_reload_event.set()
    print(f"[INFO] Модель переключена → yolov8{size}")


# ──────────────────────────────────────────────────────────────────────────────
# Переключение камеры разметки
# ──────────────────────────────────────────────────────────────────────────────

def set_camera(camera_id: str):
    """
    Перезагрузить зоны для выбранной камеры из zones.json.

    camera_id пустой / None → сброс: очищаем зоны, тех. состояния светофоров,
    ставим camera_id = None. Поведение идентично выбору камеры с пустым набором зон.
    """
    global tl_analyzer, viol_det

    # ── Сброс камеры (пустой camera_id) ──────────────────────────────────────
    if not camera_id:
        with zone_mgr._lock:
            zone_mgr._zones.clear()
        # Сбрасываем анализатор светофоров — старые состояния не должны висеть
        tl_analyzer = TrafficLightAnalyzer()
        viol_det    = ViolationDetector(zone_mgr, tl_analyzer)
        with state["lock"]:
            state["camera_id"] = None
        _clear_cache_event.set()
        print("[camera] Камера сброшена, зоны и светофоры очищены")
        return

    # ── Загрузка зон для камеры ───────────────────────────────────────────────
    n = zone_mgr.reload_for_camera(camera_id)
    # Сброс истории светофоров (новая камера — новые ROI)
    tl_analyzer = TrafficLightAnalyzer()
    viol_det    = ViolationDetector(zone_mgr, tl_analyzer)
    with state["lock"]:
        state["camera_id"] = camera_id
    print(f"[camera] Активна: '{camera_id}'  ({n} зон)")


# ──────────────────────────────────────────────────────────────────────────────
# Детекция + анализ нарушений
# ──────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {0: (50, 205, 50)}


def detect_and_analyze(model, frame: np.ndarray, conf: float, imgsz: int):
    """
    Основная функция кадра:
      1. YOLO inference с трекингом (ByteTrack через persist=True)
      2. Обновление состояний светофоров по ROI-зонам
      3. Классификация нарушений для каждого bbox
      4. Отрисовка: зоны → bbox с цветами → состояния светофоров → легенда
    """
    t0 = time.perf_counter()

    results = model.track(
        frame, classes=[0], conf=conf, iou=0.45,
        imgsz=imgsz, persist=True, verbose=False
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Обновляем светофоры
    cw_zones  = zone_mgr.crosswalk_zones()
    tl_analyzer.process_frame(frame, cw_zones)
    tl_states = tl_analyzer.get_all_states()

    fh, fw = frame.shape[:2]
    annotated = frame.copy()

    # Получаем зоны только если камера выбрана
    all_zones = []
    with state["lock"]:
        if state.get("camera_id"):
            all_zones = zone_mgr.get_all()

    # Зоны под bbox-ами
    if all_zones:
        annotated = draw_zones(annotated, all_zones, tl_states)

    # Нормализованные bbox (0..1) + уверенность
    norm_boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid      = int(box.id[0])     if box.id   is not None else -1
            conf_val = float(box.conf[0]) if box.conf is not None else 0.0
            norm_boxes.append((
                tid,
                x1 / fw, y1 / fh,
                x2 / fw, y2 / fh,
                conf_val,
            ))

    # Классифицируем нарушения
    violations            = viol_det.analyze(norm_boxes) if norm_boxes else []
    annotated, vcount     = draw_violations(annotated, violations, fw, fh)

    # Состояния светофоров поверх всего
    if cw_zones:
        annotated = draw_traffic_light_states(annotated, cw_zones, tl_states, fw, fh)

    persons = len(norm_boxes)
    with state["lock"]:
        state["violations"] = vcount

    _draw_legend(annotated, persons, elapsed_ms, vcount)
    return annotated, persons, elapsed_ms


def _draw_legend(frame: np.ndarray, persons: int, ms: float, violations: int = 0):
    with state["lock"]:
        model_sz = state["model_size"]
        fpm      = state["fpm"]
        skip     = state["frame_skip"]
        cam_id   = state["camera_id"] or "—"

    lines = [
        (f"Inference: {ms:.0f} ms",        (180, 180, 180)),
        (f"Людей:     {persons}",           ( 50, 205,  50)),
        (f"Наруш.:    {violations}",        ( 60,  60, 230)),
        (f"Модель:    yolov8{model_sz}",    ( 90, 130, 255)),
        (f"FPM лим.:  {fpm}  (1/{skip})",   (180, 130,   0)),
        (f"Камера:    {cam_id[:18]}",        (100, 200, 200)),
    ]

    pad, lh   = 8, 22
    font_size = 13
    box_w     = 210
    box_h     = len(lines) * lh + pad * 2

    # Полупрозрачный фон
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10 + box_w, 10 + box_h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = 10 + pad + i * lh
        frame = _put_text_pil(
            frame, text,
            (10 + pad, y),
            color_bgr=color,
            font_size=font_size,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Frame-skip из FPM
# ──────────────────────────────────────────────────────────────────────────────
def compute_skip(stream_fps: float, fpm: int) -> int:
    if stream_fps <= 0:
        return 1
    fps_target = fpm / 60.0
    return max(1, int(round(stream_fps / fps_target)))


# ──────────────────────────────────────────────────────────────────────────────
# Поток захвата + детекции
# ──────────────────────────────────────────────────────────────────────────────
def _iter_video_sources():
    global _current_video
    if _source_folder:
        files = sorted(
            p for p in Path(_source_folder).iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        )
        if not files:
            print(f"[WARN] Нет видео в {_source_folder}")
            return
        idx = 0
        while True:
            p = files[idx % len(files)]
            _current_video = p.name
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                print(f"[INFO] Воспроизводим: {p.name}")
                yield cap, p.name
            else:
                print(f"[WARN] Не удалось открыть: {p}")
            idx += 1
    else:
        url = _source_url
        while True:
            if _restart_event.is_set():
                url = _source_url
                _restart_event.clear()
            print(f"[INFO] Подключаемся: {url}")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if cap.isOpened():
                yield cap, url
            else:
                print("[WARN] Повтор через 3 с…")
                time.sleep(3)


def capture_thread():
    get_model()   # прогрев до начала

    sfps_timer, sfps_cnt = time.perf_counter(), 0
    dfps_timer, dfps_cnt = time.perf_counter(), 0
    sfps_cur = 0.0

    for cap, label in _iter_video_sources():
        with state["lock"]:
            state["source_type"] = label[:50]

        frame_idx      = 0
        last_annotated = None

        while True:
            if _restart_event.is_set():
                cap.release()
                break

            if _model_reload_event.is_set():
                _model_reload_event.clear()

            if _clear_cache_event.is_set():
                _clear_cache_event.clear()
                last_annotated = None
                continue

            ok, frame = cap.read()
            if not ok:
                print(f"[INFO] Конец потока: {label}")
                cap.release()
                break

            frame_idx += 1

            sfps_cnt += 1
            el = time.perf_counter() - sfps_timer
            if el >= 1.0:
                sfps_cur = sfps_cnt / el
                sfps_cnt = 0
                sfps_timer = time.perf_counter()
                with state["lock"]:
                    state["fps"] = sfps_cur

            with state["lock"]:
                fpm   = state["fpm"]
                conf  = state["conf"]
                imgsz = state["imgsz"]

            skip = compute_skip(sfps_cur if sfps_cur > 0 else 25.0, fpm)
            with state["lock"]:
                state["frame_skip"] = skip

            if frame_idx % skip == 0:
                with state["lock"]:
                    detect_enabled = state["detect_enabled"]

                if detect_enabled:
                    with _model_lock:
                        mdl = _model
                    annotated, persons, ms = detect_and_analyze(mdl, frame, conf, imgsz)
                    last_annotated = annotated

                    dfps_cnt += 1
                    del_t = time.perf_counter() - dfps_timer
                    if del_t >= 1.0:
                        with state["lock"]:
                            state["dfps"] = dfps_cnt / del_t
                        dfps_cnt = 0
                        dfps_timer = time.perf_counter()

                    with state["lock"]:
                        state["persons"] = persons
                        state["ms"] = ms
                        state["frames"] += 1
                        state["ts"] = time.time()
                else:
                    # Детекция отключена — показываем только кадр с зонами
                    fh, fw    = frame.shape[:2]
                    annotated = frame.copy()

                    with state["lock"]:
                        camera_selected = state.get("camera_id")

                    if camera_selected:
                        all_zones = zone_mgr.get_all()
                        if all_zones:
                            tl_states = tl_analyzer.get_all_states()
                            annotated = draw_zones(annotated, all_zones, tl_states)
                            cw_zones  = zone_mgr.crosswalk_zones()
                            if cw_zones:
                                annotated = draw_traffic_light_states(
                                    annotated, cw_zones, tl_states, fw, fh
                                )

                    last_annotated = annotated

                    with state["lock"]:
                        state["persons"]    = 0
                        state["ms"]         = 0
                        state["violations"] = 0
            else:
                annotated = last_annotated if last_annotated is not None else frame

            try:
                _frame_queue.put_nowait(annotated)
            except queue.Full:
                try:    _frame_queue.get_nowait()
                except queue.Empty: pass
                try:    _frame_queue.put_nowait(annotated)
                except queue.Full:  pass


# ──────────────────────────────────────────────────────────────────────────────
# Flask-приложение
# ──────────────────────────────────────────────────────────────────────────────
app = create_app(sys.modules[__name__])


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global _source_url, _source_folder

    parser = argparse.ArgumentParser(description="Веб-визуализация детекции людей (YOLOv8)")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",    type=str, help="URL HLS/RTSP-потока")
    group.add_argument("--folder", type=str, help="Папка с видеофайлами")

    parser.add_argument(
        "--model", type=str, default="m", choices=["n", "s", "m", "l", "x"],
        help="Начальная модель (n/s/m/l/x). По умолчанию: m",
    )
    parser.add_argument("--conf",  type=float, default=0.45, help="Порог уверенности")
    parser.add_argument("--imgsz", type=int,   default=640,  help="Размер входа модели")
    parser.add_argument(
        "--fpm", type=int, default=60,
        help="Лимит детекций в минуту. По умолчанию: 60",
    )
    parser.add_argument("--port", type=int, default=5000, help="Порт веб-сервера")

    args = parser.parse_args()

    with state["lock"]:
        state["model_size"] = args.model
        state["conf"]       = args.conf
        state["imgsz"]      = args.imgsz
        state["fpm"]        = max(1, args.fpm)

    _source_url    = args.url
    _source_folder = args.folder

    threading.Thread(target=capture_thread, daemon=True, name="capture").start()

    print(f"\n[✓] Открой браузер → http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()