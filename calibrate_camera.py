"""
calibrate_camera.py — Офлайн-калибровка AutoCalibrator по видеофайлу.

Запускается один раз для каждой камеры перед боевым запуском stream_detect.py.
Прогоняет видео через YOLO, накапливает bbox-статистику, сохраняет калибровку в JSON.
Никакого Flask и трансляции — только прогресс в консоли.

Использование:
    python calibrate_camera.py --video cam1.mp4 --camera cam_01

    # Выбрать модель, порог, ограничить число кадров
    python calibrate_camera.py --video cam1.mp4 --camera cam_01 \\
        --model s --conf 0.4 --max-frames 2000

    # Принудительно перезаписать существующую калибровку
    python calibrate_camera.py --video cam1.mp4 --camera cam_01 --force

Результат:
    calibrations/<camera_id>.json  — файл калибровки, подхватывается stream_detect.py

Логика защиты от ложной классификации "все — дети":
    Если за всё видео в зоне собрано < MIN_SAMPLES наблюдений — зона остаётся
    без эталона (classify вернёт "unknown"), что в stream_detect трактуется как взрослый.
    Таким образом, пустое видео (нет людей) не портит результат.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CALIBRATIONS_DIR = Path("calibrations")
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}


# ══════════════════════════════════════════════
# AutoCalibrator (идентичен detect_test.py, но
# теперь принимает frame_height при создании)
# ══════════════════════════════════════════════
class AutoCalibrator:
    """
    Делит кадр на N_BANDS горизонтальных полос по Y-координате ног (y2).
    В каждой полосе накапливает наблюдения bbox_h (высота рамки человека).
    После MIN_SAMPLES наблюдений считает 60-й перцентиль как эталон взрослого.

    Защита от ложной классификации:
        Если в зоне < MIN_SAMPLES наблюдений — classify возвращает ("unknown", 0.0),
        что в downstream-коде трактуется как взрослый (безопасная сторона).
    """

    CHILD_RATIO = 0.78
    UNKNOWN_RATIO = 0.88
    MIN_SAMPLES = 12
    N_BANDS = 10

    def __init__(self, frame_height: int):
        self.frame_height = frame_height
        self.band_h = frame_height / self.N_BANDS
        self._samples: dict[int, list[float]] = defaultdict(list)
        self._refs: dict[int, float] = {}

    def update(self, x1: int, y1: int, x2: int, y2: int):
        bbox_h = float(y2 - y1)
        if bbox_h < 10:
            return
        band = self._get_band(y2)
        buf = self._samples[band]
        buf.append(bbox_h)
        if len(buf) > 300:
            self._samples[band] = buf[-200:]

    def calibrate(self) -> int:
        updated = 0
        for band, heights in self._samples.items():
            if len(heights) >= self.MIN_SAMPLES:
                # 75-й вместо 60-го: устойчивее когда в кадре есть дети
                self._refs[band] = float(np.percentile(heights, 75))
                updated += 1
        return updated

    def classify(self, x1: int, y1: int, x2: int, y2: int) -> tuple[str, float]:
        ref = self._get_ref(self._get_band(y2))
        if ref is None:
            return "unknown", 0.0
        ratio = float(y2 - y1) / ref
        if ratio < self.CHILD_RATIO:
            conf = min(0.95, 0.65 + (self.CHILD_RATIO - ratio) * 3.0)
            return "child", round(conf, 2)
        if ratio < self.UNKNOWN_RATIO:
            return "unknown", 0.5
        conf = min(0.95, 0.70 + (ratio - self.UNKNOWN_RATIO) * 2.0)
        return "adult", round(conf, 2)

    def to_dict(self) -> dict:
        return {
            "frame_height": self.frame_height,
            "refs": {str(k): v for k, v in self._refs.items()},
            "samples_count": {str(k): len(v) for k, v in self._samples.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoCalibrator":
        obj = cls(frame_height=d["frame_height"])
        obj._refs = {int(k): v for k, v in d.get("refs", {}).items()}
        return obj

    def status(self) -> dict:
        ready = sum(1 for b in range(self.N_BANDS) if b in self._refs)
        return {
            "ready_bands": ready,
            "total_bands": self.N_BANDS,
            "percent": int(ready / self.N_BANDS * 100),
            "samples": {b: len(v) for b, v in self._samples.items()},
            "refs_px": dict(self._refs),
        }

    def _get_band(self, y: int) -> int:
        return max(0, min(self.N_BANDS - 1, int(y / self.band_h)))

    def _get_ref(self, band: int) -> float | None:
        if band in self._refs:
            return self._refs[band]
        for delta in range(1, self.N_BANDS):
            for b in (band - delta, band + delta):
                if 0 <= b < self.N_BANDS and b in self._refs:
                    return self._refs[b]
        return None


# ══════════════════════════════════════════════
# Загрузка YOLO
# ══════════════════════════════════════════════
def load_model(size: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)
    name = f"yolov8{size}.pt"
    print(f"[INFO] Загружаем модель {name}...")
    return YOLO(name)


# ══════════════════════════════════════════════
# Прогресс в консоли
# ══════════════════════════════════════════════
def _progress_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"[{'?' * width}] ?%"
    pct = current / total
    done = int(pct * width)
    bar = "█" * done + "░" * (width - done)
    return f"[{bar}] {pct:.1%}  ({current}/{total})"


# ══════════════════════════════════════════════
# Основная функция калибровки
# ══════════════════════════════════════════════
def calibrate_video(
    video_path: Path,
    camera_id: str,
    model_size: str = "m",
    conf: float = 0.45,
    imgsz: int = 640,
    max_frames: int = 0,
    calib_every: int = 30,
    force: bool = False,
) -> AutoCalibrator:

    CALIBRATIONS_DIR.mkdir(exist_ok=True)
    out_path = CALIBRATIONS_DIR / f"{camera_id}.json"

    if out_path.exists() and not force:
        print(f"[INFO] Калибровка уже существует: {out_path}")
        print(f"       Используйте --force для перезаписи.")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return AutoCalibrator.from_dict(data)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    limit = max_frames if max_frames > 0 else total_frames
    print(f"\n[INFO] Видео: {video_path.name}")
    print(f"       Разрешение: {frame_w}×{frame_h}  FPS: {video_fps:.1f}")
    print(f"       Всего кадров: {total_frames}  Лимит: {limit}")
    print(f"       Камера: {camera_id}  Выход: {out_path}\n")

    model = load_model(model_size)
    calibrator = AutoCalibrator(frame_height=frame_h)

    frame_idx = 0
    detect_cnt = 0
    total_boxes = 0
    t_start = time.perf_counter()
    t_last_print = t_start

    # Детектируем каждые N кадров из FPS (≈ 2 кадра/сек из видео достаточно)
    detect_every = max(1, int(video_fps / 2))

    print(f"[INFO] Детектируем каждый {detect_every}-й кадр из видео...")
    print()

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % detect_every != 0:
            continue

        results = model(
            frame, classes=[0], conf=conf,
            iou=0.45, imgsz=imgsz, verbose=False,
        )[0]

        detect_cnt += 1
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            calibrator.update(x1, y1, x2, y2)
            total_boxes += 1

        # Пересчёт эталонов
        if detect_cnt % calib_every == 0:
            calibrator.calibrate()

        # Прогресс каждые 2 секунды
        now = time.perf_counter()
        if now - t_last_print >= 2.0:
            t_last_print = now
            elapsed = now - t_start
            rate = frame_idx / elapsed if elapsed > 0 else 0
            eta_sec = (limit - frame_idx) / rate if rate > 0 else 0
            eta_str = f"{int(eta_sec // 60)}:{int(eta_sec % 60):02d}"
            st = calibrator.status()

            print(f"\r  {_progress_bar(frame_idx, limit)}  "
                  f"boxes:{total_boxes}  "
                  f"calib:{st['percent']}%  "
                  f"ETA:{eta_str}   ",
                  end="", flush=True)

    cap.release()
    print()  # новая строка после \r

    # Финальный пересчёт
    calibrator.calibrate()
    st = calibrator.status()

    elapsed = time.perf_counter() - t_start
    print(f"\n[✓] Готово! Обработано {frame_idx} кадров за {elapsed:.1f} с")
    print(f"    Всего людей замечено: {total_boxes}")
    print(f"    Калибровка: {st['percent']}%  ({st['ready_bands']}/{st['total_bands']} зон)")

    if st["refs_px"]:
        print("    Эталоны: " + "  ".join(
            f"зона{k}={v:.0f}px"
            for k, v in sorted(st["refs_px"].items())
        ))
    else:
        print("    [WARN] Эталоны не построены — людей в видео не обнаружено.")
        print("           В stream_detect все детекции будут трактоваться как взрослые.")

    # Сохранение
    out_path.write_text(
        json.dumps(calibrator.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"    Сохранено: {out_path}\n")

    return calibrator


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Офлайн-калибровка AutoCalibrator по видеофайлу"
    )
    parser.add_argument("--video", required=True, type=str,
                        help="Путь к видеофайлу")
    parser.add_argument("--camera", required=True, type=str,
                        help="ID камеры (например cam_01). "
                             "Определяет имя файла калибровки.")
    parser.add_argument("--model", type=str, default="s",
                        choices=["n", "s", "m", "l", "x"],
                        help="Размер модели YOLOv8. По умолчанию: s")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Порог уверенности детекции. По умолчанию: 0.40")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Размер входа модели. По умолчанию: 640")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Максимум кадров (0 = всё видео)")
    parser.add_argument("--calib-every", type=int, default=30,
                        help="Пересчитывать эталоны каждые N детекций. По умолчанию: 30")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписать существующую калибровку")

    args = parser.parse_args()
    path = Path(args.video)

    if not path.exists():
        print(f"[ERROR] Файл не найден: {path}")
        sys.exit(1)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[ERROR] Неподдерживаемый формат: {path.suffix}")
        sys.exit(1)

    calibrate_video(
        video_path=path,
        camera_id=args.camera,
        model_size=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        calib_every=args.calib_every,
        force=args.force,
    )


if __name__ == "__main__":
    main()