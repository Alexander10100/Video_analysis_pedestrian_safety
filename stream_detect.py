"""
stream_detect.py — Веб-визуализация детекции людей с HLS-потока или папки с видео

Установка зависимостей:
    pip install ultralytics opencv-python flask

Запуск:
    python stream_detect.py --url "https://video2.interra.ru/glaz.naroda.125.../index.m3u8"
    
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

# ──────────────────────────────────────────────────────────────────────────────
# Глобальное состояние
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "conf":          0.45,
    "imgsz":         640,
    "fpm":           60,        # лимит кадров детекции в минуту
    "persons":       0,
    "fps":           0.0,       # FPS всего стрима
    "dfps":          0.0,       # FPS только детекции
    "ms":            0.0,
    "frames":        0,
    "frame_skip":    1,
    "source_type":   "INIT",
    "ts":            time.time(),
    "model_size":    "m",
    "model_loading": None,
    "lock":          threading.Lock(),
}

_source_url:    str | None = None
_source_folder: str | None = None
_current_video: str | None = None
_restart_event      = threading.Event()
_model_reload_event = threading.Event()

_frame_queue: queue.Queue = queue.Queue(maxsize=2)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FRAME_SIZE = STREAM_WIDTH * STREAM_HEIGHT * 3
FFMPEG_PATH = Path(__file__).resolve().parent / "ffmpeg" / "bin" / "ffmpeg.exe"

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
# Детекция
# ──────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {0: (50, 205, 50)}


def detect_persons(model, frame: np.ndarray, conf: float, imgsz: int):
    t0 = time.perf_counter()
    results    = model(frame, classes=[0], conf=conf, iou=0.45, imgsz=imgsz, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    annotated = frame.copy()
    count = 0
    for box in results.boxes:
        cls_id     = int(box.cls)
        confidence = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        count += 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        lbl = f"person  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        pad = 4
        cv2.rectangle(annotated, (x1, y1-th-pad*2), (x1+tw+pad*2, y1), color, -1)
        cv2.putText(annotated, lbl, (x1+pad, y1-pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    _draw_legend(annotated, count, elapsed_ms)
    return annotated, count, elapsed_ms


def _draw_legend(frame: np.ndarray, persons: int, ms: float):
    with state["lock"]:
        model_sz = state["model_size"]
        fpm      = state["fpm"]
        skip     = state["frame_skip"]
    lines = [
        f"Inference: {ms:.0f} ms",
        f"Person:    {persons}",
        f"Model:     yolov8{model_sz}",
        f"FPM limit: {fpm}  (1/{skip})",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.52, 1
    pad, lh = 8, 20
    w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + pad*2
    h = len(lines) * lh + pad*2
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10+w, 10+h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    colors = [(180,180,180), (50,205,50), (90,130,255), (180,130,0)]
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(frame, line, (10+pad, 10+pad + lh//2 + i*lh),
                    font, scale, color, thick, cv2.LINE_AA)


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
    global _current_video
    if _source_folder:
        files = sorted(p for p in Path(_source_folder).iterdir()
                       if p.suffix.lower() in VIDEO_EXTS)
        if not files:
            print(f"[WARN] Нет видео в {_source_folder}"); return
        idx = 0
        while True:
            p = files[idx % len(files)]
            _current_video = p.name
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                print(f"[INFO] Воспроизводим: {p.name}")
                yield cap, p.name
            else:
                print(f"[WARN] Не удалось: {p}")
            idx += 1
    else:
        url = _source_url
        while True:
            if _restart_event.is_set():
                url = _source_url; _restart_event.clear()
            print(f"[INFO] Подключаемся через ffmpeg: {url}")
            try:
                process = _open_ffmpeg_stream(url)
                yield process, url
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}")
                time.sleep(3)
            except Exception as exc:
                print(f"[WARN] FFmpeg error: {exc}")
                print("[WARN] Повтор через 3 с…"); time.sleep(3)


def capture_thread():
    get_model()   # прогрев до начала

    sfps_timer, sfps_cnt = time.perf_counter(), 0
    dfps_timer, dfps_cnt = time.perf_counter(), 0
    sfps_cur = 0.0

    for source, label in _iter_video_sources():
        with state["lock"]:
            state["source_type"] = label[:50]

        frame_idx      = 0
        last_annotated = None
        is_folder_source = _source_folder is not None
        cap = source if is_folder_source else None
        process = None if is_folder_source else source

        while True:
            if _restart_event.is_set():
                if cap is not None:
                    cap.release()
                else:
                    _kill_process(process)
                break

            # Горячая замена модели
            if _model_reload_event.is_set():
                _model_reload_event.clear()

            if cap is not None:
                ok, frame = cap.read()
            else:
                raw_frame = process.stdout.read(STREAM_FRAME_SIZE) if process and process.stdout else b""
                ok = len(raw_frame) == STREAM_FRAME_SIZE
                frame = None if not ok else np.frombuffer(raw_frame, np.uint8).reshape((STREAM_HEIGHT, STREAM_WIDTH, 3))

            if not ok:
                print(f"[INFO] Конец/рестарт источника: {label}")
                if cap is not None:
                    cap.release()
                else:
                    _kill_process(process)
                break

            frame_idx += 1

            # stream FPS
            sfps_cnt += 1
            el = time.perf_counter() - sfps_timer
            if el >= 1.0:
                sfps_cur = sfps_cnt / el
                sfps_cnt = 0; sfps_timer = time.perf_counter()
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
                with _model_lock:
                    mdl = _model
                annotated, persons, ms = detect_persons(mdl, frame, conf, imgsz)
                last_annotated = annotated

                dfps_cnt += 1
                del_t = time.perf_counter() - dfps_timer
                if del_t >= 1.0:
                    with state["lock"]:
                        state["dfps"] = dfps_cnt / del_t
                    dfps_cnt = 0; dfps_timer = time.perf_counter()

                with state["lock"]:
                    state["persons"] = persons
                    state["ms"]      = ms
                    state["frames"] += 1
                    state["ts"]      = time.time()
            else:
                annotated = last_annotated if last_annotated is not None else frame

            try:
                _frame_queue.put_nowait(annotated)
            except queue.Full:
                try:    _frame_queue.get_nowait()
                except queue.Empty: pass
                try:    _frame_queue.put_nowait(annotated)
                except queue.Full:  pass

        if process is not None:
            _kill_process(process)


# ──────────────────────────────────────────────────────────────────────────────

app = create_app(sys.modules[__name__])

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global _source_url, _source_folder

    parser = argparse.ArgumentParser(description="Веб-визуализация детекции людей (YOLOv8)")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",    type=str)
    group.add_argument("--folder", type=str)

    parser.add_argument("--model",  type=str,   default="m", choices=["n","s","m","l","x"],
                        help="Начальная модель (n/s/m/l/x). По умолчанию: m")
    parser.add_argument("--conf",   type=float, default=0.45)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--fpm",    type=int,   default=60,
                        help="Лимит детекций в минуту. По умолчанию: 60")
    parser.add_argument("--port",   type=int,   default=5000)

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
