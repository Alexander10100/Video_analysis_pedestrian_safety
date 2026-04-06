"""
age_classifier.py — AutoCalibrator + AgeTracker + BboxEMA для stream_detect.py.

Компоненты:
  AgeClassifier  — загружает калибровку из calibrations/<camera_id>.json
                   и классифицирует bbox на adult/child/unknown по высоте рамки.
  AgeTracker     — temporal smoothing по track_id: агрегирует покадровые голоса
                   в скользящем окне и flip-ит метку только при достижении порога.
  BboxEMA        — EMA-сглаживание координат bbox по track_id, убирает "дыхание"
                   детектора до того, как bbox попадает в классификатор.

Если калибровка не найдена или в ней нет ни одного эталона — classify
возвращает ("adult", 0.0): лучше ложно не обнаружить ребёнка, чем пометить
всех взрослых детьми.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path

import numpy as np


CALIBRATIONS_DIR = Path("calibrations")

# Цвета BGR для отрисовки
COLOR_ADULT   = (50, 205, 50)
COLOR_CHILD   = (0, 165, 255)
COLOR_UNKNOWN = (160, 160, 160)


# ══════════════════════════════════════════════════════════════════════════════
# BboxEMA — EMA-сглаживание координат bbox
# ══════════════════════════════════════════════════════════════════════════════

class BboxEMA:
    """
    Exponential Moving Average по координатам (x1, y1, x2, y2) для каждого
    track_id. Убирает высокочастотное "дыхание" YOLO (±5–15px от кадра к кадру)
    до того, как bbox попадает в AgeClassifier.

    Параметр alpha: чем меньше — тем плавнее, но с бо́льшей задержкой реакции.
      0.5 — компромисс для пешеходов, меняющих позу.
      0.3 — максимальное сглаживание, подходит для медленно движущихся людей.

    Использование:
        ema = BboxEMA(alpha=0.35)
        x1, y1, x2, y2 = ema.smooth(tid, x1, y1, x2, y2)
        age_label, age_conf = age_clf.classify(x1, y1, x2, y2)
    """

    DEFAULT_ALPHA = 0.35

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha должен быть в (0, 1], получен: {alpha}")
        self._alpha = alpha
        # track_id -> (x1f, y1f, x2f, y2f) в float
        self._state: dict[int, tuple[float, float, float, float]] = {}

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def smooth(
        self,
        track_id: int,
        x1: int, y1: int,
        x2: int, y2: int,
    ) -> tuple[int, int, int, int]:
        """
        Принять новый bbox трека и вернуть сглаженный.
        Первый вызов для данного track_id возвращает bbox без изменений
        и инициализирует EMA-состояние.
        """
        incoming = (float(x1), float(y1), float(x2), float(y2))

        if track_id not in self._state:
            self._state[track_id] = incoming
            return x1, y1, x2, y2

        a   = self._alpha
        prev = self._state[track_id]
        smoothed = tuple(a * c + (1.0 - a) * p for c, p in zip(incoming, prev))
        self._state[track_id] = smoothed  # type: ignore[assignment]

        return (
            int(round(smoothed[0])),
            int(round(smoothed[1])),
            int(round(smoothed[2])),
            int(round(smoothed[3])),
        )

    def evict(self, active_ids: set[int]) -> int:
        """
        Удалить состояния исчезнувших треков.
        Возвращает количество удалённых записей.
        Вызывать раз в N кадров (например каждые 150).
        """
        stale = [tid for tid in self._state if tid not in active_ids]
        for tid in stale:
            del self._state[tid]
        return len(stale)

    def reset(self) -> None:
        """Сбросить всё состояние (при смене камеры)."""
        self._state.clear()

    def __len__(self) -> int:
        return len(self._state)


# ══════════════════════════════════════════════════════════════════════════════
# AgeTracker — temporal smoothing по track_id
# ══════════════════════════════════════════════════════════════════════════════

class AgeTracker:
    """
    Сглаживает покадровые классификации возраста по track_id.

    Алгоритм:
      - Для каждого трека ведётся скользящее окно последних WINDOW голосов.
      - Пока накоплено меньше MIN_VOTES голосов — возвращаем raw-метку
        с уверенностью, умноженной на WARMUP_CONF_SCALE (сигнал "ещё не уверены").
      - После накопления MIN_VOTES — winner = most_common метка в окне.
      - Flip текущей стабильной метки происходит только если новый winner
        набрал >= FLIP_THRESHOLD доли в окне, т.е. нельзя мигнуть
        из adult в child на одном шумном кадре.

    Параметры (можно переопределить при создании):
      window          — размер скользящего окна (кадры)
      min_votes       — минимум голосов до первого стабильного решения
      flip_threshold  — доля голосов (0..1), требуемая для смены метки
      warmup_scale    — масштаб уверенности в период прогрева

    Использование:
        tracker = AgeTracker()
        raw_label, raw_conf = age_clf.classify(x1, y1, x2, y2)
        stable_label, stable_conf = tracker.update(tid, raw_label, raw_conf)
    """

    DEFAULT_WINDOW         = 15
    DEFAULT_MIN_VOTES      = 5
    DEFAULT_FLIP_THRESHOLD = 0.70
    DEFAULT_WARMUP_SCALE   = 0.50

    def __init__(
        self,
        window:         int   = DEFAULT_WINDOW,
        min_votes:      int   = DEFAULT_MIN_VOTES,
        flip_threshold: float = DEFAULT_FLIP_THRESHOLD,
        warmup_scale:   float = DEFAULT_WARMUP_SCALE,
    ):
        if not (0.5 <= flip_threshold <= 1.0):
            raise ValueError("flip_threshold должен быть в [0.5, 1.0]")
        if not (0.0 < warmup_scale <= 1.0):
            raise ValueError("warmup_scale должен быть в (0, 1.0]")

        self._window         = window
        self._min_votes      = min(min_votes, window)
        self._flip_threshold = flip_threshold
        self._warmup_scale   = warmup_scale

        # track_id -> deque[str]  — история raw меток
        self._buffers: dict[int, deque[str]] = {}
        # track_id -> str         — текущая стабильная метка (после прогрева)
        self._stable: dict[int, str] = {}

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        raw_label: str,
        raw_conf:  float,
    ) -> tuple[str, float]:
        """
        Принять raw-классификацию кадра и вернуть (stable_label, stable_conf).

        stable_conf = доля победителя в окне (0..1).
        В период прогрева stable_conf масштабируется на warmup_scale, чтобы
        downstream-код мог отличить "уверенный" результат от "предварительного".
        """
        # ── Инициализация буфера ──────────────────────────────────────────────
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self._window)

        buf = self._buffers[track_id]
        buf.append(raw_label)

        # ── Прогрев: недостаточно голосов ────────────────────────────────────
        if len(buf) < self._min_votes:
            return raw_label, round(raw_conf * self._warmup_scale, 2)

        # ── Подсчёт голосов ───────────────────────────────────────────────────
        counts        = Counter(buf)
        winner, wcnt  = counts.most_common(1)[0]
        winner_share  = wcnt / len(buf)

        # ── Логика flip ───────────────────────────────────────────────────────
        prev_stable = self._stable.get(track_id)

        if prev_stable is None:
            # Первое стабильное решение — принимаем без порога
            self._stable[track_id] = winner
        elif winner != prev_stable and winner_share < self._flip_threshold:
            # Новый winner не набрал достаточно — держим предыдущую метку
            winner       = prev_stable
            winner_share = counts.get(prev_stable, 0) / len(buf)
        else:
            self._stable[track_id] = winner

        return winner, round(winner_share, 2)

    def evict(self, active_ids: set[int]) -> int:
        """
        Удалить буферы и стабильные метки исчезнувших треков.
        Возвращает количество удалённых записей.
        Вызывать раз в N кадров (например каждые 150).
        """
        stale = [tid for tid in self._buffers if tid not in active_ids]
        for tid in stale:
            del self._buffers[tid]
            self._stable.pop(tid, None)
        return len(stale)

    def reset(self) -> None:
        """Сбросить всё состояние (при смене камеры)."""
        self._buffers.clear()
        self._stable.clear()

    def stable_label(self, track_id: int) -> str | None:
        """Вернуть текущую стабильную метку трека или None в период прогрева."""
        return self._stable.get(track_id)

    def __len__(self) -> int:
        return len(self._buffers)


# ══════════════════════════════════════════════════════════════════════════════
# AgeClassifier — однокадровый классификатор по калиброванной высоте bbox
# ══════════════════════════════════════════════════════════════════════════════

class AgeClassifier:
    """
    Потокобезопасный (read-only после load) классификатор возраста.
    Используется в capture_thread, создаётся один раз при смене камеры.

    Для стабильного результата оборачивать в AgeTracker (temporal smoothing)
    и подавать предварительно сглаженный через BboxEMA bbox.
    """

    CHILD_RATIO   = 0.78
    UNKNOWN_RATIO = 0.88
    N_BANDS       = 10

    def __init__(self, frame_height: int):
        self.frame_height = frame_height
        self.band_h       = frame_height / self.N_BANDS
        self._refs: dict[int, float] = {}
        self._calibrated = False

    # ── Фабричный метод ───────────────────────────────────────────────────────

    @classmethod
    def load_for_camera(cls, camera_id: str, frame_height: int) -> "AgeClassifier":
        """
        Загрузить калибровку для камеры из calibrations/<camera_id>.json.
        Если файла нет — вернуть объект без эталонов (всё → adult).
        """
        obj  = cls(frame_height=frame_height)
        path = CALIBRATIONS_DIR / f"{camera_id}.json"

        if not path.exists():
            print(
                f"[AgeClassifier] Калибровка не найдена: {path}. "
                "Все детекции → adult."
            )
            return obj

        try:
            data     = json.loads(path.read_text(encoding="utf-8"))
            raw_refs = data.get("refs", {})
            if not raw_refs:
                print(
                    f"[AgeClassifier] Калибровка {path} пуста (нет эталонов). "
                    "Все детекции → adult."
                )
                return obj

            saved_h = data.get("frame_height", frame_height)
            scale   = frame_height / saved_h if saved_h > 0 else 1.0

            obj._refs        = {int(k): v * scale for k, v in raw_refs.items()}
            obj._calibrated  = True
            print(
                f"[AgeClassifier] Камера '{camera_id}': загружено "
                f"{len(obj._refs)} зон из {cls.N_BANDS}  "
                f"(масштаб ×{scale:.3f})"
            )
        except Exception as exc:
            print(f"[AgeClassifier] Ошибка загрузки {path}: {exc}. Все → adult.")

        return obj

    # ── Классификация ─────────────────────────────────────────────────────────

    def classify(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
    ) -> tuple[str, float]:
        """
        Вернуть (label, confidence).
          label: "adult" | "child" | "unknown"

        Если калибровки нет или нет эталона для данной зоны — "adult", 0.0.
        Рекомендуется подавать сглаженный bbox (через BboxEMA) и оборачивать
        результат в AgeTracker.update() для стабилизации по времени.
        """
        if not self._calibrated:
            return "adult", 0.0

        ref = self._get_ref(self._get_band(y2))
        if ref is None:
            return "adult", 0.0

        ratio = float(y2 - y1) / ref

        if ratio < self.CHILD_RATIO:
            conf = min(0.95, 0.65 + (self.CHILD_RATIO - ratio) * 3.0)
            return "child", round(conf, 2)

        if ratio < self.UNKNOWN_RATIO:
            # Серая зона: не уверены, но не ребёнок → adult (безопасная сторона)
            return "adult", round(0.50, 2)

        conf = min(0.95, 0.70 + (ratio - self.UNKNOWN_RATIO) * 2.0)
        return "adult", round(conf, 2)

    # ── Вспомогательные ───────────────────────────────────────────────────────

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


# ── Утилита цвета ──────────────────────────────────────────────────────────────

def get_age_color(label: str) -> tuple[int, int, int]:
    """BGR цвет для отрисовки по возрастному ярлыку."""
    return {
        "child":   COLOR_CHILD,
        "adult":   COLOR_ADULT,
        "unknown": COLOR_UNKNOWN,
    }.get(label, COLOR_UNKNOWN)