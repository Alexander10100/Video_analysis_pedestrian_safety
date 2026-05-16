"""
stream_detect.py — Веб-визуализация детекции людей с HLS-потока или папки с видео
                   + детекция нарушений ПДД + классификация взрослый/ребёнок
                   + FFmpeg-захват HLS/RTSP потоков

Установка зависимостей:
    pip install ultralytics opencv-python flask pillow

Запуск:
    python stream_detect.py --url "https://..."
    python stream_detect.py --folder "/path/to/videos"

Калибровка (запускается отдельно перед боевым запуском):
    python calibrate_camera.py --video cam1.mp4 --camera cam_01

Открыть браузер: http://localhost:5000
"""

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from stream_detect_web import create_app
import json
from datetime import datetime

from age_classifier import AgeClassifier, AgeTracker, BboxEMA
from traffic_light import TrafficLightAnalyzer
from violation_detector import (
    ViolationDetector, draw_violations, draw_zones,
    draw_traffic_light_states, _put_text_pil,
    person_inside_vehicle,
)
from zone_manager import ZoneManager

# ── Глобальные объекты разметки ───────────────────────────────────────────────
zone_mgr = ZoneManager()
tl_analyzer = TrafficLightAnalyzer()
viol_det = ViolationDetector(zone_mgr, tl_analyzer)

# ── AgeClassifier — создаётся/пересоздаётся при смене камеры ─────────────────
_age_clf: AgeClassifier | None = None
_age_clf_lock = threading.Lock()

# ── AgeTracker и BboxEMA — глобальные, сбрасываются при смене камеры ─────────
# Не требуют lock: обращение только из одного capture_thread.
_age_tracker = AgeTracker(
    window=15,
    min_votes=5,
    flip_threshold=0.70,
    warmup_scale=0.50,
)
_bbox_ema = BboxEMA(alpha=0.35)

# Счётчик кадров для периодической чистки мёртвых треков
_evict_every = 150   # кадров

# ── Логирование нарушений для отчётов ───────────────────────────────────────
VIOLATION_LOG = Path("violations_log.jsonl")
_violation_log_lock = threading.Lock()

# Защита от повторных нарушений одного и того же человека
_violation_cooldown: dict[int, float] = {}
COOLDOWN_SECONDS = 10        # секунд между логированиями одного track_id


def log_violations(camera_id: str, violation_list: list, ts: float):
    if not violation_list:
        return

    timestamp_str = datetime.fromtimestamp(ts).isoformat()
    current_time = ts
    new_entries = []

    for v in violation_list:
        if v.violation == "none":
            continue

        tid = getattr(v, 'track_id', -1)

        if tid < 0:
            continue

        last_logged = _violation_cooldown.get(tid, 0)
        if current_time - last_logged < COOLDOWN_SECONDS:
            continue

        entry = {
            "timestamp": timestamp_str,
            "track_id": tid,
            "camera_id": camera_id or "unknown",
            "violation_type": v.violation,
            "zone_label": v.zone_label,
            "note": v.note or "",
            "age_label": v.age_label,
            "person_conf": float(v.conf),
            "age_conf": float(v.age_conf),
        }
        new_entries.append(entry)

        # Обновляем время последнего логирования
        _violation_cooldown[tid] = current_time

    # Записываем в файл
    if new_entries:
        with _violation_log_lock:
            with VIOLATION_LOG.open("a", encoding="utf-8") as f:
                for e in new_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"[LOG] Записано {len(new_entries)} новых нарушений")


# ──────────────────────────────────────────────────────────────────────────────
# Глобальное состояние
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "conf": 0.45,
    "imgsz": 640,
    "fpm": 60,
    "persons": 0,
    "fps": 0.0,
    "dfps": 0.0,
    "ms": 0.0,
    "frames": 0,
    "frame_skip": 1,
    "source_type": "INIT",
    "ts": time.time(),
    "model_size": "m",
    "model_loading": None,
    "violations": 0,
    "camera_id": None,
    "detect_enabled": True,
    "adults": 0,
    "children": 0,
    "age_calibrated": False,
    "lock": threading.Lock(),
}

_source_url: str | None = None
_source_folder: str | None = None
_current_video: str | None = None
_restart_event = threading.Event()
_model_reload_event = threading.Event()
_clear_cache_event = threading.Event()

_frame_queue: queue.Queue = queue.Queue(maxsize=2)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FRAME_SIZE = STREAM_WIDTH * STREAM_HEIGHT * 3
FFMPEG_PATH = Path(__file__).resolve().parent / "ffmpeg" / "bin" / "ffmpeg.exe"

# ── FFmpeg-параметры для HLS/RTSP потоков ────────────────────────────────────
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FRAME_SIZE = STREAM_WIDTH * STREAM_HEIGHT * 3
FFMPEG_PATH = Path(__file__).resolve().parent / "ffmpeg" / "bin" / "ffmpeg.exe"

# ── YOLO ──────────────────────────────────────────────────────────────────────
_model = None
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
    global _model
    new_m = load_yolo(size)
    with _model_lock:
        _model = new_m
    with state["lock"]:
        state["model_size"] = size
        state["model_loading"] = None
    _model_reload_event.set()
    print(f"[INFO] Модель переключена → yolov8{size}")


# ── FFmpeg-захват ─────────────────────────────────────────────────────────────

def _open_ffmpeg_stream(url: str):
    """Запустить FFmpeg и вернуть Popen-процесс, пишущий BGR-кадры в stdout."""
    if not FFMPEG_PATH.exists():
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_PATH}")

    command = [
        str(FFMPEG_PATH),
        "-loglevel", "quiet",
        "-re",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", url,
        "-vf", f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}",
        "-vsync", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10 ** 8,
    )


def _kill_process(process):
    """Мягко завершить FFmpeg-процесс."""
    if process is None:
        return
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1)
    except Exception:
        pass


# ── Переключение камеры ───────────────────────────────────────────────────────

def set_camera(camera_id: str):
    """Перезагрузить зоны и сбросить все временные состояния для новой камеры."""
    global tl_analyzer, viol_det, _age_clf

    # Сбрасываем EMA и трекер — история старой камеры не применима к новой
    _bbox_ema.reset()
    _age_tracker.reset()

    if not camera_id:
        with zone_mgr._lock:
            zone_mgr._zones.clear()
        tl_analyzer = TrafficLightAnalyzer()
        viol_det = ViolationDetector(zone_mgr, tl_analyzer)
        with _age_clf_lock:
            _age_clf = None
        with state["lock"]:
            state["camera_id"] = None
            state["age_calibrated"] = False
        _clear_cache_event.set()
        print("[camera] Камера сброшена, зоны и светофоры очищены")
        return

    # ── Загрузка зон для камеры ───────────────────────────────────────────────
    n = zone_mgr.reload_for_camera(camera_id)
    # Сброс истории светофоров (новая камера — новые ROI)
    tl_analyzer = TrafficLightAnalyzer()
    viol_det = ViolationDetector(zone_mgr, tl_analyzer)

    with _age_clf_lock:
        _age_clf = None   # пересоздастся в capture_thread с актуальной высотой кадра

    with state["lock"]:
        state["camera_id"] = camera_id
        state["age_calibrated"] = False

    print(f"[camera] Активна: '{camera_id}'  ({n} зон)")


# ── Детекция и анализ ─────────────────────────────────────────────────────────

def detect_and_analyze(
    model,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    frame_idx: int,
):
    """
    Запустить YOLO-трекинг, классифицировать возраст (с EMA + temporal smoothing)
    и определить нарушения ПДД. Вернуть аннотированный кадр, число людей и время.
    """
    global _age_clf

    t0 = time.perf_counter()

    # classes=[0,2,3,5,7]: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
    results = model.track(
        frame, classes=[0, 2, 3, 5, 7], conf=conf, iou=0.45,
        imgsz=imgsz, persist=True, verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Обновляем светофоры
    cw_zones = zone_mgr.crosswalk_zones()
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

    # ── AgeClassifier: пересоздаём если ещё нет или сменилась камера ─────────
    with _age_clf_lock:
        if _age_clf is None:
            cam_id = state.get("camera_id") or ""
            if cam_id:
                _age_clf = AgeClassifier.load_for_camera(cam_id, fh)
            else:
                _age_clf = AgeClassifier(fh)
            with state["lock"]:
                state["age_calibrated"] = _age_clf.is_calibrated()
        age_clf = _age_clf

    # ── Bbox + EMA-сглаживание + классификация возраста + temporal smoothing ──
    norm_boxes = []
    vehicle_boxes = []   # [(x1,y1,x2,y2)] в пикселях — для фильтрации водителей
    adults_cnt = 0
    children_cnt = 0
    active_tids: set[int] = set()

    # ── Первый проход: собираем bbox транспортных средств ─────────────────────
    # YOLO классы: 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASSES = {2, 3, 5, 7}
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id in VEHICLE_CLASSES:
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                vehicle_boxes.append((vx1, vy1, vx2, vy2))

    # ── Второй проход: обрабатываем людей с проверкой на водителя ────────────
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id != 0:   # пропускаем не-людей
                continue

            raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if box.id is not None else -1
            conf_val = float(box.conf[0]) if box.conf is not None else 0.0

            if tid >= 0:
                active_tids.add(tid)

            # 0. Фильтр водителей: если нижняя половина bbox человека перекрывается
            #    с bbox автомобиля — он внутри машины, не пешеход.
            if person_inside_vehicle(raw_x1, raw_y1, raw_x2, raw_y2, vehicle_boxes):
                continue

            # 1. EMA: сглаживаем координаты по истории трека
            x1, y1, x2, y2 = _bbox_ema.smooth(tid, raw_x1, raw_y1, raw_x2, raw_y2)

            # 2. Однокадровая классификация на сглаженном bbox
            raw_label, raw_conf = age_clf.classify(x1, y1, x2, y2)

            # 3. Temporal smoothing: стабилизируем метку через скользящее окно
            age_label, age_conf = _age_tracker.update(tid, raw_label, raw_conf)

            if age_label == "child":
                children_cnt += 1
            else:
                adults_cnt += 1

            norm_boxes.append((
                tid,
                x1 / fw, y1 / fh,
                x2 / fw, y2 / fh,
                conf_val,
                age_label,
                age_conf,
            ))

    # ── Периодическая чистка мёртвых треков ──────────────────────────────────
    if frame_idx % _evict_every == 0 and active_tids:
        evicted_ema = _bbox_ema.evict(active_tids)
        evicted_tracker = _age_tracker.evict(active_tids)
        if evicted_ema or evicted_tracker:
            print(
                f"[evict] кадр {frame_idx}: "
                f"BboxEMA={evicted_ema}, AgeTracker={evicted_tracker} треков удалено"
            )

    violations = viol_det.analyze(norm_boxes) if norm_boxes else []
    annotated, vcount = draw_violations(annotated, violations, fw, fh)

    # === Логируем нарушения для отчётов ===
    cam_id = state.get("camera_id", "") if "state" in globals() else ""
    log_violations(cam_id, violations, time.time())

    # Состояния светофоров поверх всего
    if cw_zones:
        annotated = draw_traffic_light_states(annotated, cw_zones, tl_states, fw, fh)

    persons = len(norm_boxes)
    with state["lock"]:
        state["violations"] = vcount
        state["adults"] = adults_cnt
        state["children"] = children_cnt

    _draw_legend(annotated, persons, elapsed_ms, vcount, adults_cnt, children_cnt)
    return annotated, persons, elapsed_ms


def _draw_legend(
    frame: np.ndarray,
    persons: int,
    ms: float,
    violations: int = 0,
    adults: int = 0,
    children: int = 0,
):
    with state["lock"]:
        model_sz = state["model_size"]
        fpm = state["fpm"]
        skip = state["frame_skip"]
        cam_id = state["camera_id"] or "—"
        age_calibrated = state["age_calibrated"]

    age_status = "ДА" if age_calibrated else "НЕТ"
    lines = [
        (f"Inference: {ms:.0f} ms", (180, 180, 180)),
        (f"Людей:     {persons}", (50, 205, 50)),
        (f"  взрослых: {adults}", (50, 200, 50)),
        (f"  детей:    {children}", (0, 165, 255)),
        (f"Наруш.:    {violations}", (60, 60, 230)),
        (f"Модель:    yolov8{model_sz}", (90, 130, 255)),
        (f"FPM лим.:  {fpm}  (1/{skip})", (180, 130, 0)),
        (f"Камера:    {cam_id[:18]}", (100, 200, 200)),
        (f"Калибр.:   {age_status}",
         (50, 205, 50) if age_calibrated else (180, 130, 0)),
    ]

    pad, lh = 8, 22
    font_size = 13
    box_w = 230
    box_h = len(lines) * lh + pad * 2

    # Полупрозрачный фон
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10 + box_w, 10 + box_h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = 10 + pad + i * lh
        frame = _put_text_pil(frame, text, (10 + pad, y),
                              color_bgr=color, font_size=font_size)


# ──────────────────────────────────────────────────────────────────────────────
# Frame-skip из FPM
# ──────────────────────────────────────────────────────────────────────────────
def compute_skip(stream_fps: float, fpm: int) -> int:
    if stream_fps <= 0:
        return 1
    fps_target = fpm / 60.0
    return max(1, int(round(stream_fps / fps_target)))


def _open_ffmpeg_stream(url: str):
    if not FFMPEG_PATH.exists():
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_PATH}")

    command = [
        str(FFMPEG_PATH),
        "-loglevel", "quiet",
        "-re",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", url,
        "-vf", f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}",
        "-vsync", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )


def _kill_process(process):
    if process is None:
        return
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Поток захвата + детекции
# ──────────────────────────────────────────────────────────────────────────────
def _iter_video_sources():
    """
    Для папки — yield (cv2.VideoCapture, filename).
    Для URL   — yield (subprocess.Popen,  url)  через FFmpeg.
    """
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
            print(f"[INFO] Подключаемся через ffmpeg: {url}")
            try:
                process = _open_ffmpeg_stream(url)
                yield process, url
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}")
                time.sleep(3)
            except Exception as exc:
                print(f"[WARN] FFmpeg error: {exc}")
                print("[WARN] Повтор через 3 с…")
                time.sleep(3)


# ── Поток захвата + детекции ──────────────────────────────────────────────────

def capture_thread():
    global _age_clf

    get_model()   # прогрев до начала стрима

    sfps_timer, sfps_cnt = time.perf_counter(), 0
    dfps_timer, dfps_cnt = time.perf_counter(), 0
    sfps_cur = 0.0

    for source, label in _iter_video_sources():
        with state["lock"]:
            state["source_type"] = label[:50]

        is_folder_source = _source_folder is not None
        cap = source if is_folder_source else None
        process = None if is_folder_source else source

        fh_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else STREAM_HEIGHT
        if fh_cap > 0:
            cam_id = state.get("camera_id") or ""
            with _age_clf_lock:
                if _age_clf is None:
                    if cam_id:
                        _age_clf = AgeClassifier.load_for_camera(cam_id, fh_cap)
                    else:
                        _age_clf = AgeClassifier(fh_cap)
                    with state["lock"]:
                        state["age_calibrated"] = _age_clf.is_calibrated()

        frame_idx = 0
        last_annotated = None
        is_folder_source = _source_folder is not None
        cap = source if is_folder_source else None
        process = None if is_folder_source else source

        while True:
            # ── Рестарт потока ────────────────────────────────────────────────
            if _restart_event.is_set():
                if cap is not None:
                    cap.release()
                else:
                    _kill_process(process)
                break

            # ── Горячая замена модели ─────────────────────────────────────────
            if _model_reload_event.is_set():
                _model_reload_event.clear()

            # ── Сброс кэша при смене камеры ───────────────────────────────────
            if cap is not None:
                ok, frame = cap.read()
            else:
                raw_frame = process.stdout.read(STREAM_FRAME_SIZE) if process and process.stdout else b""
                ok = len(raw_frame) == STREAM_FRAME_SIZE
                frame = None if not ok else np.frombuffer(
                    raw_frame, np.uint8
                ).reshape((STREAM_HEIGHT, STREAM_WIDTH, 3))

            if _clear_cache_event.is_set():
                _clear_cache_event.clear()
                last_annotated = None
                with _age_clf_lock:
                    _age_clf = None
                # BboxEMA и AgeTracker уже сброшены в set_camera()
                continue

            # ── Читаем кадр ───────────────────────────────────────────────────
            if cap is not None:
                ok, frame = cap.read()
            else:
                raw = (process.stdout.read(STREAM_FRAME_SIZE)
                       if process and process.stdout else b"")
                ok = len(raw) == STREAM_FRAME_SIZE
                frame = (None if not ok
                         else np.frombuffer(raw, np.uint8)
                                .reshape((STREAM_HEIGHT, STREAM_WIDTH, 3)))

            if not ok:
                print(f"[INFO] Конец/рестарт источника: {label}")
                if cap is not None:
                    cap.release()
                else:
                    _kill_process(process)
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
                fpm = state["fpm"]
                conf = state["conf"]
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
                    # Передаём frame_idx для периодической чистки мёртвых треков
                    annotated, persons, ms = detect_and_analyze(
                        mdl, frame, conf, imgsz, frame_idx
                    )
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
                    fh_f, fw_f = frame.shape[:2]
                    annotated = frame.copy()
                    with state["lock"]:
                        camera_selected = state.get("camera_id")

                    if camera_selected:
                        all_zones = zone_mgr.get_all()
                        if all_zones:
                            tl_states = tl_analyzer.get_all_states()
                            annotated = draw_zones(annotated, all_zones, tl_states)
                            cw_zones = zone_mgr.crosswalk_zones()
                            if cw_zones:
                                annotated = draw_traffic_light_states(
                                    annotated, cw_zones, tl_states, fw_f, fh_f)
                    last_annotated = annotated

                    with state["lock"]:
                        state["persons"] = 0
                        state["ms"] = 0
                        state["violations"] = 0
                        state["adults"] = 0
                        state["children"] = 0
            else:
                annotated = last_annotated if last_annotated is not None else frame

            try:
                _frame_queue.put_nowait(annotated)
            except queue.Full:
                try:
                    _frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _frame_queue.put_nowait(annotated)
                except queue.Full:
                    pass

        if process is not None:
            _kill_process(process)

        if process is not None:
            _kill_process(process)


# ── Flask-приложение ──────────────────────────────────────────────────────────
app = create_app(sys.modules[__name__])


# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    global _source_url, _source_folder

    parser = argparse.ArgumentParser(
        description="Веб-визуализация детекции людей (YOLOv8) + нарушения ПДД"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="HLS/RTSP URL потока")
    group.add_argument("--folder", type=str, help="Папка с видеофайлами")

    parser.add_argument("--model", type=str, default="m",
                        choices=["n", "s", "m", "l", "x"],
                        help="Начальная модель (n/s/m/l/x). По умолчанию: m")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Порог уверенности детекции")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Размер входного изображения для YOLO")
    parser.add_argument("--fpm", type=int, default=60,
                        help="Лимит детекций в минуту. По умолчанию: 60")
    parser.add_argument("--port", type=int, default=5000,
                        help="Порт веб-сервера. По умолчанию: 5000")
    parser.add_argument("--camera", type=str, default="",
                        help="ID камеры для автозагрузки зон при старте")

    args = parser.parse_args()

    with state["lock"]:
        state["model_size"] = args.model
        state["conf"] = args.conf
        state["imgsz"] = args.imgsz
        state["fpm"] = max(1, args.fpm)

    _source_url = args.url
    _source_folder = args.folder

    if args.camera.strip():
        set_camera(args.camera.strip())

    threading.Thread(target=capture_thread, daemon=True, name="capture").start()

    print(f"\n[✓] Открой браузер → http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()