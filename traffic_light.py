"""
traffic_light.py — Определение цвета светофора по ROI кадра.

Типы светофоров (light_type в RoadZone):
  "pedestrian" — пешеходный.  Зелёный = можно идти, красный = нарушение.
  "vehicle"    — автомобильный (едет НАВСТРЕЧУ пешеходам).
                 Зелёный для машин = пешеходам ЗАПРЕЩЕНО.
                 Красный для машин = пешеходам РАЗРЕШЕНО (инверсия).

Это позволяет разметить ситуацию когда пешеходного светофора не видно,
но виден автомобильный светофор потока, который едет на пешехода.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ── Цветовые диапазоны HSV ────────────────────────────────────────────────────
_RED_RANGES = [
    ((0,   120, 100), (12,  255, 255)),
    ((158, 120, 100), (180, 255, 255)),
]
_YELLOW_RANGE = ((15, 80, 80),  (38, 255, 255))
_GREEN_RANGE  = ((35, 60, 60),  (95, 255, 255))

# ── Публичные константы состояния (сигнал на самом светофоре) ────────────────
STATE_RED     = "red"
STATE_GREEN   = "green"
STATE_YELLOW  = "yellow"
STATE_UNKNOWN = "unknown"

# ── Типы светофоров ───────────────────────────────────────────────────────────
LIGHT_TYPE_PEDESTRIAN = "pedestrian"   # пешеходный
LIGHT_TYPE_VEHICLE    = "vehicle"      # автомобильный (инверсия)

# Задержка после «разрешающего» сигнала (секунды)
GREEN_GRACE_SECONDS = 5.0

CACHE_FRAMES = 3
VOTE_WINDOW  = 5
MIN_AREA_RATIO   = 0.20
MIN_TOTAL_PIXELS = 15

_STATE_PRIORITY = {
    STATE_RED:     4,
    STATE_YELLOW:  3,
    STATE_UNKNOWN: 2,
    STATE_GREEN:   1,
}


def pedestrian_allowed(raw_state: str, light_type: str) -> bool:
    """
    Возвращает True, если пешеходу РАЗРЕШЕНО идти, с учётом типа светофора.

    Пешеходный светофор:
        green  → разрешено
        red/yellow/unknown → запрещено

    Автомобильный светофор (инверсия):
        red/yellow → разрешено  (машины стоят — путь свободен)
        green      → запрещено  (машины едут — опасно)
        unknown    → запрещено  (неизвестно — безопаснее запретить)
    """
    if light_type == LIGHT_TYPE_VEHICLE:
        return raw_state in (STATE_RED, STATE_YELLOW)
    # pedestrian (default)
    return raw_state == STATE_GREEN


@dataclass
class TrafficLightState:
    light_id:    str
    state:       str   = STATE_UNKNOWN
    confidence:  float = 0.0
    green_until: float = 0.0   # timestamp до которого действует grace (по пешеходной логике)
    _history:    deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    _frame_cnt:  int   = 0

    def update(self, raw_state: str, light_type: str = LIGHT_TYPE_PEDESTRIAN):
        self._history.append(raw_state)
        counts = {s: self._history.count(s)
                  for s in (STATE_RED, STATE_GREEN, STATE_YELLOW, STATE_UNKNOWN)}
        best = max(counts, key=counts.__getitem__)
        self.confidence = counts[best] / len(self._history)

        prev       = self.state
        self.state = best

        # Grace-период: запускаем когда переходим в «разрешающий» сигнал
        was_allowed  = pedestrian_allowed(prev,  light_type)
        now_allowed  = pedestrian_allowed(best,  light_type)

        if now_allowed:
            self.green_until = time.time() + GREEN_GRACE_SECONDS
        elif was_allowed and not now_allowed:
            # Только что переключился в запрещающий — grace уже мог быть запущен
            pass

    def is_pedestrian_allowed_or_grace(self, light_type: str = LIGHT_TYPE_PEDESTRIAN) -> bool:
        """True → переход разрешён (сейчас ИЛИ в grace-период)."""
        if pedestrian_allowed(self.state, light_type):
            return True
        return time.time() < self.green_until


def _mask_area(hsv: np.ndarray, lo: tuple, hi: tuple) -> int:
    mask   = cv2.inRange(hsv,
                         np.array(lo, dtype=np.uint8),
                         np.array(hi, dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return int(cv2.countNonZero(mask))


def _enhance_roi(roi: np.ndarray) -> np.ndarray:
    roi  = cv2.GaussianBlur(roi, (3, 3), 0)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cl   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    hsv[:, :, 2] = cl.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def classify_roi(roi: np.ndarray) -> str:
    """Определить цвет светофора по вырезанному ROI (независимо от типа)."""
    if roi is None or roi.size == 0:
        return STATE_UNKNOWN
    h, w = roi.shape[:2]
    if max(h, w) < 8:
        return STATE_UNKNOWN
    scale = min(1.0, 120 / max(h, w, 1))
    if scale < 1.0:
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    roi = _enhance_roi(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_area    = sum(_mask_area(hsv, lo, hi) for lo, hi in _RED_RANGES)
    yellow_area = _mask_area(hsv, *_YELLOW_RANGE)
    green_area  = _mask_area(hsv, *_GREEN_RANGE)
    total = red_area + yellow_area + green_area
    if total < MIN_TOTAL_PIXELS:
        return STATE_UNKNOWN
    best_area = max(red_area, yellow_area, green_area)
    if best_area / total < MIN_AREA_RATIO:
        return STATE_UNKNOWN
    if best_area == green_area:
        return STATE_GREEN
    if best_area == red_area:
        return STATE_RED
    return STATE_YELLOW


def _aggregate_states(states: list[str]) -> str:
    if not states:
        return STATE_UNKNOWN
    return max(states, key=lambda s: _STATE_PRIORITY.get(s, 0))


class TrafficLightAnalyzer:
    """
    Менеджер состояний светофоров для всех crosswalk-зон.
    Учитывает light_type каждой зоны при вычислении «разрешён ли переход».
    """

    def __init__(self):
        self._states: dict[str, list[TrafficLightState]] = {}

    def _get_or_create(self, zone_id: str, n_rois: int) -> list[TrafficLightState]:
        existing = self._states.get(zone_id, [])
        if len(existing) != n_rois:
            self._states[zone_id] = [
                TrafficLightState(light_id=f"{zone_id}__roi{i}")
                for i in range(n_rois)
            ]
        return self._states[zone_id]

    @staticmethod
    def _parse_rois(raw) -> list[list]:
        if not raw:
            return []
        if isinstance(raw[0], (int, float)):
            return [raw]
        return list(raw)

    def process_frame(self, frame: np.ndarray,
                      crosswalk_zones: list) -> dict[str, "AggregatedTLState"]:
        fh, fw = frame.shape[:2]

        for zone in crosswalk_zones:
            if not zone.has_light or not zone.traffic_light_roi:
                continue
            rois = self._parse_rois(zone.traffic_light_roi)
            if not rois:
                continue
            light_type  = getattr(zone, "light_type", LIGHT_TYPE_PEDESTRIAN)
            states_list = self._get_or_create(zone.id, len(rois))

            for idx, (roi_def, st) in enumerate(zip(rois, states_list)):
                st._frame_cnt += 1
                if st._frame_cnt % CACHE_FRAMES != 0:
                    continue
                x, y, w, h = roi_def
                x1 = int(max(0, x * fw));  y1 = int(max(0, y * fh))
                x2 = int(min(fw, (x + w) * fw)); y2 = int(min(fh, (y + h) * fh))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_state = classify_roi(frame[y1:y2, x1:x2])
                st.update(raw_state, light_type)

        return self._build_aggregated()

    def _build_aggregated(self) -> dict[str, "AggregatedTLState"]:
        # Нужно знать light_type — берём из _zone_types (заполняется в process_frame)
        result = {}
        for zone_id, states in self._states.items():
            raw_states  = [s.state for s in states]
            agg_state   = _aggregate_states(raw_states)
            avg_conf    = (sum(s.confidence for s in states) / len(states)) if states else 0.0
            light_type  = self._zone_types.get(zone_id, LIGHT_TYPE_PEDESTRIAN)

            # Grace: разрешено сейчас или в период grace
            any_allowed = any(s.is_pedestrian_allowed_or_grace(light_type) for s in states)
            # Но если агрегированный сигнал красный (автомобильный = зелёный для машин),
            # то «разрешено» определяется по пешеходной логике
            pedestrian_ok = pedestrian_allowed(agg_state, light_type) or (
                any_allowed and not pedestrian_allowed(agg_state, light_type)
                and agg_state not in (STATE_RED, STATE_GREEN)  # yellow/unknown — grace может быть
            )
            # Упрощённо: используем any_allowed напрямую
            result[zone_id] = AggregatedTLState(
                zone_id     = zone_id,
                state       = agg_state,
                confidence  = avg_conf,
                light_type  = light_type,
                green_grace = any_allowed,
                green_until = max((s.green_until for s in states), default=0.0),
                per_roi     = [
                    {"state": s.state, "confidence": round(s.confidence, 2)}
                    for s in states
                ],
            )
        return result

    def process_frame(self, frame: np.ndarray,
                      crosswalk_zones: list) -> dict[str, "AggregatedTLState"]:
        """Перегружаем чтобы запомнить light_type каждой зоны."""
        if not hasattr(self, "_zone_types"):
            self._zone_types: dict[str, str] = {}

        fh, fw = frame.shape[:2]

        for zone in crosswalk_zones:
            if not zone.has_light or not zone.traffic_light_roi:
                continue
            rois = self._parse_rois(zone.traffic_light_roi)
            if not rois:
                continue
            light_type = getattr(zone, "light_type", LIGHT_TYPE_PEDESTRIAN)
            self._zone_types[zone.id] = light_type

            states_list = self._get_or_create(zone.id, len(rois))
            for idx, (roi_def, st) in enumerate(zip(rois, states_list)):
                st._frame_cnt += 1
                if st._frame_cnt % CACHE_FRAMES != 0:
                    continue
                x, y, w, h = roi_def
                x1 = int(max(0, x * fw));  y1 = int(max(0, y * fh))
                x2 = int(min(fw, (x + w) * fw)); y2 = int(min(fh, (y + h) * fh))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_state = classify_roi(frame[y1:y2, x1:x2])
                st.update(raw_state, light_type)

        return self._build_aggregated()

    def get_state(self, zone_id: str) -> "AggregatedTLState | None":
        return self._build_aggregated().get(zone_id)

    def get_all_states(self) -> dict[str, dict]:
        return {
            zid: {
                "state":       s.state,
                "confidence":  round(s.confidence, 2),
                "light_type":  s.light_type,
                "green_grace": s.green_grace,
                "green_until": s.green_until,
                "per_roi":     s.per_roi,
                # Удобный флаг для фронта: разрешён ли переход прямо сейчас
                "pedestrian_allowed": s.green_grace,
            }
            for zid, s in self._build_aggregated().items()
        }


class AggregatedTLState:
    __slots__ = ("zone_id", "state", "confidence", "light_type",
                 "green_grace", "green_until", "per_roi")

    def __init__(self, zone_id, state, confidence, light_type,
                 green_grace, green_until, per_roi):
        self.zone_id    = zone_id
        self.state      = state
        self.confidence = confidence
        self.light_type = light_type
        self.green_grace = green_grace   # True → пешеходу разрешено (с учётом типа + grace)
        self.green_until = green_until
        self.per_roi    = per_roi

    def is_green_or_grace(self) -> bool:
        """Совместимость со старым кодом."""
        return self.green_grace