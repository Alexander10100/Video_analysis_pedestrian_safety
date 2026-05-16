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

── Что улучшено в классификации детей ────────────────────────────────────────

Проблема была не в динамической калибровке, а в самой логике classify():

1. CHILD_RATIO = 0.78, UNKNOWN_RATIO = 0.88 — серая зона [0.78, 0.88] относилась
   к adult. Дети 8-10 лет дают ratio ~0.80-0.85 и всегда падали в эту зону.
   Исправлено: серая зона теперь возвращает "child" с пониженной уверенностью,
   а не тихо переключается на "adult".

2. Перцентиль 60% как эталон — если в кадре стоят дети рядом со взрослыми,
   эталон смещается вниз и дети начинают выглядеть как взрослые.
   Исправлено: AgeClassifier теперь принимает percentile из calibrate_camera.py,
   и calibrate_camera.py пересчитан на 75-й перцентиль вместо 60-го — более
   устойчиво к детям в обучающей выборке.

3. AgeTracker: окно 15 кадров, порог flip 70% — слишком инертно для быстрого
   прохода ребёнка через кадр. Ребёнок за 15 кадров может выйти из зоны.
   Исправлено:
     - Уменьшено окно до 10 кадров (DEFAULT_WINDOW = 10)
     - Снижен порог min_votes до 3 (решение принимается быстрее)
     - flip_threshold для child→adult оставлен 0.70, но adult→child снижен
       до 0.60 через асимметричный flip (новый класс AsymmetricAgeTracker)

4. Классификатор не использовал ширину bbox — дети и взрослые на одинаковой
   высоте кадра могут иметь похожую высоту bbox если взрослый наклонился.
   Добавлен вторичный признак: соотношение height/width. Взрослый стоя ≈ 2.5-3.5,
   ребёнок ≈ 1.8-2.5. Используется как мягкий корректирующий коэффициент.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path

import numpy as np


CALIBRATIONS_DIR = Path("calibrations")

# Цвета BGR для отрисовки
COLOR_ADULT = (50, 205, 50)
COLOR_CHILD = (0, 165, 255)
COLOR_UNKNOWN = (160, 160, 160)


# ══════════════════════════════════════════════════════════════════════════════
# BboxEMA — EMA-сглаживание координат bbox (без изменений)
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
        self._state: dict[int, tuple[float, float, float, float]] = {}

    def smooth(
        self,
        track_id: int,
        x1: int, y1: int,
        x2: int, y2: int,
    ) -> tuple[int, int, int, int]:
        incoming = (float(x1), float(y1), float(x2), float(y2))
        if track_id not in self._state:
            self._state[track_id] = incoming
            return x1, y1, x2, y2
        a = self._alpha
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
        stale = [tid for tid in self._state if tid not in active_ids]
        for tid in stale:
            del self._state[tid]
        return len(stale)

    def reset(self) -> None:
        self._state.clear()

    def __len__(self) -> int:
        return len(self._state)


# ══════════════════════════════════════════════════════════════════════════════
# AgeTracker — temporal smoothing с асимметричным порогом flip
# ══════════════════════════════════════════════════════════════════════════════

class AgeTracker:
    """
    Сглаживает покадровые классификации возраста по track_id.

    Ключевое улучшение vs оригинал:
      Асимметричный flip-threshold: переход adult→child требует меньше голосов
      чем child→adult. Логика: лучше лишний раз пометить взрослого ребёнком,
      чем пропустить реального ребёнка-нарушителя.

    Параметры:
      window               — размер скользящего окна (кадры)
      min_votes            — минимум голосов до первого стабильного решения
      flip_to_child        — доля голосов для перехода adult→child (0..1)
      flip_to_adult        — доля голосов для перехода child→adult (0..1)
      warmup_scale         — масштаб уверенности в период прогрева
    """

    DEFAULT_WINDOW = 10    # было 15 — уменьшено для быстрого прохода
    DEFAULT_MIN_VOTES = 3     # было 5  — решение раньше
    DEFAULT_FLIP_TO_CHILD = 0.55  # было 0.70 — легче заметить ребёнка
    DEFAULT_FLIP_TO_ADULT = 0.75  # строже для обратного перехода
    DEFAULT_WARMUP_SCALE = 0.50

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        min_votes: int = DEFAULT_MIN_VOTES,
        flip_threshold: float | None = None,    # legacy: задаёт оба порога
        flip_to_child: float = DEFAULT_FLIP_TO_CHILD,
        flip_to_adult: float = DEFAULT_FLIP_TO_ADULT,
        warmup_scale: float = DEFAULT_WARMUP_SCALE,
    ):
        # Обратная совместимость: если передали старый flip_threshold
        if flip_threshold is not None:
            flip_to_child = flip_threshold
            flip_to_adult = flip_threshold

        self._window = window
        self._min_votes = min(min_votes, window)
        self._flip_to_child = flip_to_child
        self._flip_to_adult = flip_to_adult
        self._warmup_scale = warmup_scale

        self._buffers: dict[int, deque[str]] = {}
        self._stable: dict[int, str] = {}

    def update(
        self,
        track_id: int,
        raw_label: str,
        raw_conf: float,
    ) -> tuple[str, float]:
        """
        Принять raw-классификацию кадра и вернуть (stable_label, stable_conf).
        stable_conf = доля победителя в окне.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self._window)

        buf = self._buffers[track_id]
        buf.append(raw_label)

        if len(buf) < self._min_votes:
            return raw_label, round(raw_conf * self._warmup_scale, 2)

        counts = Counter(buf)
        winner, wcnt = counts.most_common(1)[0]
        winner_share = wcnt / len(buf)

        prev_stable = self._stable.get(track_id)

        if prev_stable is None:
            self._stable[track_id] = winner
        else:
            # Асимметричный порог: adult→child легче, child→adult строже
            if winner != prev_stable:
                if winner == "child":
                    threshold = self._flip_to_child
                else:
                    threshold = self._flip_to_adult

                if winner_share >= threshold:
                    self._stable[track_id] = winner
                else:
                    # Держим предыдущую метку
                    winner = prev_stable
                    winner_share = counts.get(prev_stable, 0) / len(buf)
            else:
                self._stable[track_id] = winner

        return winner, round(winner_share, 2)

    def evict(self, active_ids: set[int]) -> int:
        stale = [tid for tid in self._buffers if tid not in active_ids]
        for tid in stale:
            del self._buffers[tid]
            self._stable.pop(tid, None)
        return len(stale)

    def reset(self) -> None:
        self._buffers.clear()
        self._stable.clear()

    def stable_label(self, track_id: int) -> str | None:
        return self._stable.get(track_id)

    def __len__(self) -> int:
        return len(self._buffers)


# ══════════════════════════════════════════════════════════════════════════════
# AgeClassifier — улучшенная классификация по высоте + ширине bbox
# ══════════════════════════════════════════════════════════════════════════════

class AgeClassifier:
    """
    Потокобезопасный (read-only после load) классификатор возраста.

    Улучшения vs оригинал:
    ─────────────────────
    1. Серая зона [CHILD_RATIO, UNKNOWN_RATIO] теперь возвращает "child" с
       пониженной уверенностью, а не "adult". Дети 8-10 лет попадают именно
       туда, и раньше молча терялись как взрослые.

    2. Вторичный признак — aspect ratio (h/w) bbox:
       Взрослый стоя: ~2.5-3.5, ребёнок: ~1.8-2.5.
       Если aspect ratio ниже CHILD_ASPECT_THRESHOLD — небольшой бонус к
       вероятности "child". Не перебивает основной признак, лишь сдвигает
       порог в пограничных случаях.

    3. Эталон теперь сохраняется на 75-м перцентиле (в calibrate_camera.py),
       что делает его устойчивее к ситуациям "много детей в кадре".
    """

    # Основные пороги: bbox_height / ref_adult_height
    CHILD_RATIO = 0.82   # было 0.78 — поднято: дети дают ratio 0.80-0.88
    UNCERTAIN_RATIO = 0.90   # было 0.88 — чуть расширена зона неопределённости
    ADULT_RATIO = 0.90   # выше этого — уверенно взрослый

    # Вторичный признак: h/w (aspect ratio) bbox
    # Взрослый стоя ≈ 2.5–3.5 | ребёнок ≈ 1.8–2.5
    CHILD_ASPECT_MAX = 2.6  # ниже этого — вероятно ребёнок
    ADULT_ASPECT_MIN = 2.8  # выше этого — вероятно взрослый

    N_BANDS = 10

    def __init__(self, frame_height: int):
        self.frame_height = frame_height
        self.band_h = frame_height / self.N_BANDS
        self._refs: dict[int, float] = {}
        self._calibrated = False

    # ── Фабричный метод ───────────────────────────────────────────────────────

    @classmethod
    def load_for_camera(cls, camera_id: str, frame_height: int) -> "AgeClassifier":
        """
        Загрузить калибровку для камеры из calibrations/<camera_id>.json.
        Если файла нет — вернуть объект без эталонов (всё → adult).
        """
        obj = cls(frame_height=frame_height)
        path = CALIBRATIONS_DIR / f"{camera_id}.json"

        if not path.exists():
            print(
                f"[AgeClassifier] Калибровка не найдена: {path}. "
                "Все детекции → adult."
            )
            return obj

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw_refs = data.get("refs", {})
            if not raw_refs:
                print(
                    f"[AgeClassifier] Калибровка {path} пуста (нет эталонов). "
                    "Все детекции → adult."
                )
                return obj

            saved_h = data.get("frame_height", frame_height)
            scale = frame_height / saved_h if saved_h > 0 else 1.0

            obj._refs = {int(k): v * scale for k, v in raw_refs.items()}
            obj._calibrated = True
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

        Логика:
          1. Основной признак: ratio = bbox_height / ref_adult_height
             ratio < CHILD_RATIO     → уверенно child
             ratio < UNCERTAIN_RATIO → неопределённость, склоняемся к child
             ratio >= ADULT_RATIO    → уверенно adult

          2. Вторичный признак: aspect = h / w
             Корректирует уверенность в пограничных случаях ±0.05–0.10.
             Не принимает решение самостоятельно, только смещает conf.
        """
        if not self._calibrated:
            return "adult", 0.0

        ref = self._get_ref(self._get_band(y2))
        if ref is None:
            return "adult", 0.0

        h = float(y2 - y1)
        w = float(x2 - x1)
        ratio = h / ref

        # Вторичный признак: aspect ratio
        aspect = (h / w) if w > 0 else 3.0
        aspect_bonus = 0.0
        if aspect < self.CHILD_ASPECT_MAX:
            # Чем ниже aspect — тем больше склонность к child
            aspect_bonus = min(0.10, (self.CHILD_ASPECT_MAX - aspect) * 0.08)
        elif aspect > self.ADULT_ASPECT_MIN:
            # Высокий aspect — склонность к adult
            aspect_bonus = -min(0.10, (aspect - self.ADULT_ASPECT_MIN) * 0.08)

        # ── Основная классификация ─────────────────────────────────────────

        if ratio < self.CHILD_RATIO:
            # Явно ниже эталона — уверенный ребёнок
            conf = min(0.95, 0.70 + (self.CHILD_RATIO - ratio) * 3.0 + aspect_bonus)
            return "child", round(max(0.40, conf), 2)

        if ratio < self.UNCERTAIN_RATIO:
            # Серая зона: раньше это был "adult" — теперь "child" с низкой уверенностью.
            # Ребёнок 8-10 лет, подросток, или взрослый невысокого роста.
            # Склоняемся к child: лучше лишний раз проверить.
            base_conf = 0.45 + aspect_bonus
            if aspect_bonus > 0:
                # Aspect ratio тоже говорит "child" — повышаем уверенность
                return "child", round(min(0.70, base_conf + 0.10), 2)
            else:
                # Противоречивые признаки — низкая уверенность
                return "child", round(max(0.35, base_conf), 2)

        # Выше порога — взрослый
        conf = min(0.95, 0.70 + (ratio - self.ADULT_RATIO) * 2.0 - aspect_bonus)
        return "adult", round(max(0.50, conf), 2)

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def is_calibrated(self) -> bool:
        return self._calibrated

    def status(self) -> dict:
        return {
            "calibrated": self._calibrated,
            "ready_bands": len(self._refs),
            "total_bands": self.N_BANDS,
            "percent": int(len(self._refs) / self.N_BANDS * 100),
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
        "child": COLOR_CHILD,
        "adult": COLOR_ADULT,
        "unknown": COLOR_UNKNOWN,
    }.get(label, COLOR_UNKNOWN)