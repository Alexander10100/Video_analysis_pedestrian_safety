"""
age_classifier.py — AutoCalibrator для использования в stream_detect.py.

Загружает готовую калибровку из calibrations/<camera_id>.json
(созданную calibrate_camera.py) и классифицирует bbox на adult/child/unknown.

Если калибровка не найдена или в ней нет ни одного эталона — classify
возвращает ("adult", 0.0), что является безопасным умолчанием:
лучше ложно не обнаружить ребёнка, чем пометить всех взрослых детьми.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CALIBRATIONS_DIR = Path("calibrations")

# Цвета BGR для отрисовки
COLOR_ADULT   = (50, 205, 50)
COLOR_CHILD   = (0, 165, 255)
COLOR_UNKNOWN = (160, 160, 160)


class AgeClassifier:
    """
    Потокобезопасный (read-only после load) классификатор возраста.
    Используется в capture_thread, создаётся один раз при смене камеры.
    """

    CHILD_RATIO   = 0.78
    UNKNOWN_RATIO = 0.88
    N_BANDS       = 10

    def __init__(self, frame_height: int):
        self.frame_height = frame_height
        self.band_h       = frame_height / self.N_BANDS
        self._refs: dict[int, float] = {}
        self._calibrated = False   # True если загружена хоть одна зона

    @classmethod
    def load_for_camera(cls, camera_id: str, frame_height: int) -> "AgeClassifier":
        """
        Загрузить калибровку для камеры из calibrations/<camera_id>.json.
        Если файла нет — вернуть объект без эталонов (всё → adult).
        """
        obj  = cls(frame_height=frame_height)
        path = CALIBRATIONS_DIR / f"{camera_id}.json"

        if not path.exists():
            print(f"[AgeClassifier] Калибровка не найдена: {path}. "
                  f"Все детекции → adult.")
            return obj

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw_refs = data.get("refs", {})
            if not raw_refs:
                print(f"[AgeClassifier] Калибровка {path} пуста (нет эталонов). "
                      f"Все детекции → adult.")
                return obj

            # Масштабируем эталоны если высота кадра отличается
            saved_h = data.get("frame_height", frame_height)
            scale   = frame_height / saved_h if saved_h > 0 else 1.0

            obj._refs = {int(k): v * scale for k, v in raw_refs.items()}
            obj._calibrated = True
            zones_ready = len(obj._refs)
            print(f"[AgeClassifier] Камера '{camera_id}': загружено {zones_ready} "
                  f"зон из {cls.N_BANDS}  (масштаб ×{scale:.3f})")
        except Exception as e:
            print(f"[AgeClassifier] Ошибка загрузки {path}: {e}. Все → adult.")

        return obj

    def classify(self, x1: int, y1: int, x2: int, y2: int) -> tuple[str, float]:
        """
        Вернуть (label, confidence).
        label: "adult" | "child" | "unknown"

        Если калибровки нет — всегда "adult" с confidence=0.
        """
        if not self._calibrated:
            return "adult", 0.0

        ref = self._get_ref(self._get_band(y2))
        if ref is None:
            # Зона есть, но для этой конкретной полосы нет эталона
            return "adult", 0.0

        ratio = float(y2 - y1) / ref

        if ratio < self.CHILD_RATIO:
            conf = min(0.95, 0.65 + (self.CHILD_RATIO - ratio) * 3.0)
            return "child", round(conf, 2)

        if ratio < self.UNKNOWN_RATIO:
            # Серая зона — не уверены, но не ребёнок → adult (безопасно)
            return "adult", round(0.5, 2)

        conf = min(0.95, 0.70 + (ratio - self.UNKNOWN_RATIO) * 2.0)
        return "adult", round(conf, 2)

    def is_calibrated(self) -> bool:
        return self._calibrated

    def status(self) -> dict:
        return {
            "calibrated":  self._calibrated,
            "ready_bands": len(self._refs),
            "total_bands": self.N_BANDS,
            "percent":     int(len(self._refs) / self.N_BANDS * 100),
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


def get_age_color(label: str) -> tuple[int, int, int]:
    """BGR цвет для отрисовки по возрастному ярлыку."""
    return {
        "child":   COLOR_CHILD,
        "adult":   COLOR_ADULT,
        "unknown": COLOR_UNKNOWN,
    }.get(label, COLOR_UNKNOWN)