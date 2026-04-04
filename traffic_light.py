"""
traffic_light.py — Определение цвета светофора по ROI кадра.

"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ── Цветовые диапазоны HSV ────────────────────────────────────────────────────
# Красный: два диапазона (wrap-around в hue) + расширенные границы насыщенности/яркости
_RED_RANGES = [
    ((0,   120, 100), (12,  255, 255)),   # нижний красный
    ((158, 120, 100), (180, 255, 255)),   # верхний красный
]
# Жёлтый: расширен диапазон hue и снижен порог насыщенности
_YELLOW_RANGE = ((15, 80, 80), (38, 255, 255))
# Зелёный: расширен вверх по hue, снижены пороги
_GREEN_RANGE  = ((35, 60, 60), (95, 255, 255))

# ── Публичные константы состояния ─────────────────────────────────────────────
STATE_RED     = "red"
STATE_GREEN   = "green"
STATE_YELLOW  = "yellow"
STATE_UNKNOWN = "unknown"

# Задержка после зелёного (секунды): человек ещё успевает перейти
GREEN_GRACE_SECONDS = 5.0

# Сколько кадров кэшировать один и тот же ROI (пропускать повторный анализ)
CACHE_FRAMES = 3  # уменьшено для более быстрой реакции

# Размер очереди голосования (сглаживание мигания)
VOTE_WINDOW = 5  # уменьшено — быстрее реагируем на смену сигнала

# Минимальный процент площади яркого пятна от общей (порог обнаружения)
MIN_AREA_RATIO = 0.20

# Минимальное количество пикселей для считывания сигнала
MIN_TOTAL_PIXELS = 15

# Приоритет при агрегации нескольких светофоров одной зоны
# red > yellow > unknown > green (безопасная сторона)
_STATE_PRIORITY = {
    STATE_RED:     4,
    STATE_YELLOW:  3,
    STATE_UNKNOWN: 2,
    STATE_GREEN:   1,
}


@dataclass
class TrafficLightState:
    light_id:    str                  # zone_id светофора
    state:       str = STATE_UNKNOWN  # red | green | yellow | unknown
    confidence:  float = 0.0         # 0..1 — доля голосов за победивший цвет
    green_until: float = 0.0         # timestamp до которого действует green-grace
    _history:    deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    _frame_cnt:  int = 0             # счётчик кадров для кэша

    def update(self, raw_state: str):
        """Добавить наблюдение, пересчитать итоговое состояние."""
        self._history.append(raw_state)
        counts = {s: self._history.count(s) for s in
                  (STATE_RED, STATE_GREEN, STATE_YELLOW, STATE_UNKNOWN)}
        best = max(counts, key=counts.__getitem__)
        self.confidence = counts[best] / len(self._history)

        prev = self.state
        self.state = best

        # Если переключился в зелёный — запускаем grace-period
        if best == STATE_GREEN and prev != STATE_GREEN:
            self.green_until = time.time() + GREEN_GRACE_SECONDS

        # Продлеваем grace пока горит зелёный
        if best == STATE_GREEN:
            self.green_until = time.time() + GREEN_GRACE_SECONDS

    def is_green_or_grace(self) -> bool:
        """True → переход разрешён (горит зелёный ИЛИ прошло < grace секунд)."""
        if self.state == STATE_GREEN:
            return True
        if self.state in (STATE_RED, STATE_YELLOW, STATE_UNKNOWN):
            return time.time() < self.green_until
        return False


def _mask_area(hsv: np.ndarray, lo: tuple, hi: tuple) -> int:
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    # убираем шум
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return int(cv2.countNonZero(mask))


def _enhance_roi(roi: np.ndarray) -> np.ndarray:
    """
    Предобработка ROI для лучшего распознавания:
    - CLAHE для выравнивания гистограммы (важно при ночных/пасмурных условиях)
    - небольшое размытие для удаления артефактов компрессии
    """
    # Гауссово размытие — убираем артефакты
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    # CLAHE на канале V (яркость в HSV) — улучшаем контраст
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return roi


def classify_roi(roi: np.ndarray) -> str:
    """
    Определить цвет светофора по вырезанному ROI.
    Возвращает STATE_RED | STATE_GREEN | STATE_YELLOW | STATE_UNKNOWN.
    """
    if roi is None or roi.size == 0:
        return STATE_UNKNOWN

    # Масштаб до фиксированного размера для стабильности
    h, w = roi.shape[:2]
    if max(h, w) < 8:   # слишком маленький ROI
        return STATE_UNKNOWN

    scale = min(1.0, 120 / max(h, w, 1))
    if scale < 1.0:
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    # Предобработка
    roi = _enhance_roi(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_area    = sum(_mask_area(hsv, lo, hi) for lo, hi in _RED_RANGES)
    yellow_area = _mask_area(hsv, *_YELLOW_RANGE)
    green_area  = _mask_area(hsv, *_GREEN_RANGE)

    total = red_area + yellow_area + green_area
    if total < MIN_TOTAL_PIXELS:
        return STATE_UNKNOWN

    best_area = max(red_area, yellow_area, green_area)
    ratio     = best_area / total

    if ratio < MIN_AREA_RATIO:
        return STATE_UNKNOWN

    if best_area == green_area:
        return STATE_GREEN
    if best_area == red_area:
        return STATE_RED
    return STATE_YELLOW


def _aggregate_states(states: list[str]) -> str:
    """
    Агрегация нескольких состояний светофоров одной зоны.
    Приоритет: red > yellow > unknown > green (безопасная сторона).
    """
    if not states:
        return STATE_UNKNOWN
    return max(states, key=lambda s: _STATE_PRIORITY.get(s, 0))


class TrafficLightAnalyzer:
    """
    Менеджер состояний светофоров для всех crosswalk-зон.
    Вызывается из capture_thread на каждом кадре.

    Поддерживает несколько ROI на одну зону:
      zone.traffic_light_roi может быть:
        - одним ROI:  [x, y, w, h]
        - списком ROI: [[x,y,w,h], [x,y,w,h], ...]
    Итоговое состояние зоны — агрегация по всем ROI (приоритет красного).
    """

    def __init__(self):
        # zone_id → список TrafficLightState (по одному на каждый ROI)
        self._states: dict[str, list[TrafficLightState]] = {}

    def _get_or_create(self, zone_id: str, n_rois: int) -> list[TrafficLightState]:
        existing = self._states.get(zone_id, [])
        # Если количество ROI изменилось — пересоздаём
        if len(existing) != n_rois:
            self._states[zone_id] = [
                TrafficLightState(light_id=f"{zone_id}__roi{i}")
                for i in range(n_rois)
            ]
        return self._states[zone_id]

    @staticmethod
    def _parse_rois(raw) -> list[list]:
        """
        Нормализует traffic_light_roi к списку ROI.
        [x,y,w,h]           → [[x,y,w,h]]
        [[x,y,w,h], ...]     → [[x,y,w,h], ...]
        """
        if raw is None:
            return []
        if not raw:
            return []
        # Если первый элемент — число, это один ROI
        if isinstance(raw[0], (int, float)):
            return [raw]
        return list(raw)

    def process_frame(self, frame: np.ndarray,
                      crosswalk_zones: list) -> dict[str, TrafficLightState]:
        """
        Обновить состояния всех светофоров.
        crosswalk_zones — список RoadZone с type='crosswalk' и has_light=True.
        Возвращает словарь zone_id → агрегированный TrafficLightState.
        """
        fh, fw = frame.shape[:2]

        for zone in crosswalk_zones:
            if not zone.has_light or not zone.traffic_light_roi:
                continue

            rois = self._parse_rois(zone.traffic_light_roi)
            if not rois:
                continue

            states_list = self._get_or_create(zone.id, len(rois))

            for idx, (roi_def, st) in enumerate(zip(rois, states_list)):
                st._frame_cnt += 1

                # Пропускаем промежуточные кадры (кэш)
                if st._frame_cnt % CACHE_FRAMES != 0:
                    continue

                x, y, w, h = roi_def
                x1 = int(max(0, x * fw))
                y1 = int(max(0, y * fh))
                x2 = int(min(fw, (x + w) * fw))
                y2 = int(min(fh, (y + h) * fh))

                if x2 <= x1 or y2 <= y1:
                    continue

                roi_img   = frame[y1:y2, x1:x2]
                raw_state = classify_roi(roi_img)
                st.update(raw_state)

        # Строим агрегированный словарь zone_id → единое состояние
        return self._build_aggregated()

    def _build_aggregated(self) -> dict[str, "AggregatedTLState"]:
        result = {}
        for zone_id, states in self._states.items():
            raw_states = [s.state for s in states]
            agg_state  = _aggregate_states(raw_states)
            avg_conf   = (sum(s.confidence for s in states) / len(states)) if states else 0.0
            # green_grace: True если хотя бы один ROI даёт grace (но только если нет красного)
            any_grace  = any(s.is_green_or_grace() for s in states)
            result[zone_id] = AggregatedTLState(
                zone_id    = zone_id,
                state      = agg_state,
                confidence = avg_conf,
                green_grace = any_grace and agg_state not in (STATE_RED, STATE_YELLOW),
                green_until = max((s.green_until for s in states), default=0.0),
                per_roi     = [
                    {"state": s.state, "confidence": round(s.confidence, 2)}
                    for s in states
                ],
            )
        return result

    def get_state(self, zone_id: str) -> "AggregatedTLState | None":
        return self._build_aggregated().get(zone_id)

    def get_all_states(self) -> dict[str, dict]:
        return {
            zid: {
                "state":       s.state,
                "confidence":  round(s.confidence, 2),
                "green_grace": s.green_grace,
                "green_until": s.green_until,
                "per_roi":     s.per_roi,
            }
            for zid, s in self._build_aggregated().items()
        }


class AggregatedTLState:
    """Агрегированное состояние зоны (по нескольким ROI)."""
    __slots__ = ("zone_id", "state", "confidence", "green_grace", "green_until", "per_roi")

    def __init__(self, zone_id, state, confidence, green_grace, green_until, per_roi):
        self.zone_id     = zone_id
        self.state       = state
        self.confidence  = confidence
        self.green_grace = green_grace
        self.green_until = green_until
        self.per_roi     = per_roi

    def is_green_or_grace(self) -> bool:
        return self.green_grace or self.state == STATE_GREEN