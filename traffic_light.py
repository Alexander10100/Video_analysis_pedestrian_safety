"""
traffic_light.py — Определение цвета светофора по ROI кадра.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# HSV-диапазоны цветов светофора
_RED_RANGES = [
    ((0, 120, 80), (15, 255, 255)),
    ((155, 120, 80), (180, 255, 255)),
]

_YELLOW_RANGES = [
    ((16, 100, 80), (38, 255, 255)),
]

_GREEN_RANGES = [
    ((40, 80, 80), (95, 255, 255)),
]

_BODY_HUE_RANGES = [
    (45, 75),
]

# Публичные константы состояния
STATE_RED = "red"
STATE_GREEN = "green"
STATE_YELLOW = "yellow"
STATE_UNKNOWN = "unknown"

LIGHT_TYPE_PEDESTRIAN = "pedestrian"
LIGHT_TYPE_VEHICLE = "vehicle"

# Параметры
GREEN_GRACE_SECONDS = 5.0
DEFAULT_CACHE_FRAMES = 2
VOTE_WINDOW = 7
MIN_AREA_RATIO = 0.15
MIN_TOTAL_PIXELS = 10
MIN_ROI_DIM = 6
LAMP_CORE_THRESHOLD = 0.35

_STATE_PRIORITY = {
    STATE_RED: 4,
    STATE_YELLOW: 3,
    STATE_UNKNOWN: 2,
    STATE_GREEN: 1,
}


def pedestrian_allowed(raw_state: str, light_type: str) -> bool:
    if light_type == LIGHT_TYPE_VEHICLE:
        return raw_state in (STATE_RED, STATE_YELLOW)
    return raw_state == STATE_GREEN


def _mask_area(hsv: np.ndarray, lo: tuple, hi: tuple) -> int:
    mask = cv2.inRange(hsv,
                       np.array(lo, dtype=np.uint8),
                       np.array(hi, dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return int(cv2.countNonZero(mask))


def _sum_ranges(hsv: np.ndarray, ranges: list[tuple]) -> int:
    return sum(_mask_area(hsv, lo, hi) for lo, hi in ranges)


def _enhance_roi(roi: np.ndarray) -> np.ndarray:
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = cl.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _find_lamp_core_mask(hsv: np.ndarray,
                         body_hue_ranges: list[tuple[int, int]]) -> np.ndarray | None:
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    score = sat * val

    hue = hsv[:, :, 0]
    body_mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in body_hue_ranges:
        body_mask |= (hue >= lo) & (hue <= hi)

    score_fg = score.copy()
    score_fg[body_mask] = 0.0

    peak = float(score_fg.max())
    if peak < 1000:
        return None

    threshold = peak * LAMP_CORE_THRESHOLD
    lamp_core = (score_fg >= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lamp_core = cv2.morphologyEx(lamp_core, cv2.MORPH_CLOSE, kernel, iterations=1)

    if cv2.countNonZero(lamp_core) < MIN_TOTAL_PIXELS:
        return None

    return lamp_core


def classify_roi(roi: np.ndarray) -> tuple[str, float, dict]:
    empty_debug = {"red": 0, "yellow": 0, "green": 0, "total": 0, "method": "none"}

    if roi is None or roi.size == 0:
        return STATE_UNKNOWN, 0.0, empty_debug

    h, w = roi.shape[:2]
    if max(h, w) < MIN_ROI_DIM:
        return STATE_UNKNOWN, 0.0, empty_debug

    scale = min(1.0, 120 / max(h, w, 1))
    if scale < 1.0:
        roi = cv2.resize(roi,
                         (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)

    roi_enh = _enhance_roi(roi)
    hsv = cv2.cvtColor(roi_enh, cv2.COLOR_BGR2HSV)

    lamp_mask = _find_lamp_core_mask(hsv, _BODY_HUE_RANGES)

    if lamp_mask is not None:
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=lamp_mask)
        red_area = _sum_ranges(hsv_masked, _RED_RANGES)
        yellow_area = _sum_ranges(hsv_masked, _YELLOW_RANGES)
        green_area = _sum_ranges(hsv_masked, _GREEN_RANGES)
        total = red_area + yellow_area + green_area
        method = "lamp_core"
    else:
        red_area = _sum_ranges(hsv, _RED_RANGES)
        yellow_area = _sum_ranges(hsv, _YELLOW_RANGES)
        green_area = _sum_ranges(hsv, _GREEN_RANGES)
        total = red_area + yellow_area + green_area
        method = "full_roi"

    debug = {
        "red": red_area, "yellow": yellow_area,
        "green": green_area, "total": total,
        "method": method,
    }

    if total < MIN_TOTAL_PIXELS:
        return STATE_UNKNOWN, 0.1, debug

    best_area = max(red_area, yellow_area, green_area)
    ratio = best_area / total

    if ratio < MIN_AREA_RATIO:
        return STATE_UNKNOWN, 0.0, debug

    confidence = min(1.0, 0.5 + (ratio - MIN_AREA_RATIO) / (1.0 - MIN_AREA_RATIO) * 0.5)

    if best_area == red_area:
        return STATE_RED, round(confidence, 2), debug
    if best_area == green_area:
        return STATE_GREEN, round(confidence, 2), debug
    return STATE_YELLOW, round(confidence, 2), debug


def _aggregate_states(states: list[str]) -> str:
    if not states:
        return STATE_UNKNOWN
    return max(states, key=lambda s: _STATE_PRIORITY.get(s, 0))


@dataclass
class TrafficLightState:
    light_id: str
    state: str = STATE_UNKNOWN
    confidence: float = 0.0
    green_until: float = 0.0
    _history: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    _frame_cnt: int = 0

    def update(self, raw_state: str, raw_conf: float,
               light_type: str = LIGHT_TYPE_PEDESTRIAN):
        self._history.append(raw_state)
        counts = {s: self._history.count(s)
                  for s in (STATE_RED, STATE_GREEN, STATE_YELLOW, STATE_UNKNOWN)}
        best = max(counts, key=counts.__getitem__)
        self.confidence = counts[best] / len(self._history)
        self.state = best
        if pedestrian_allowed(best, light_type):
            self.green_until = time.time() + GREEN_GRACE_SECONDS

    def is_pedestrian_allowed_or_grace(self,
                                       light_type: str = LIGHT_TYPE_PEDESTRIAN) -> bool:
        if pedestrian_allowed(self.state, light_type):
            return True
        return time.time() < self.green_until


class AggregatedTLState:
    __slots__ = ("zone_id", "state", "confidence", "light_type",
                 "green_grace", "green_until", "per_roi")

    def __init__(self, zone_id, state, confidence, light_type,
                 green_grace, green_until, per_roi):
        self.zone_id = zone_id
        self.state = state
        self.confidence = confidence
        self.light_type = light_type
        self.green_grace = green_grace
        self.green_until = green_until
        self.per_roi = per_roi

    def is_green_or_grace(self) -> bool:
        return self.green_grace


class TrafficLightAnalyzer:
    def __init__(self, cache_frames: int = DEFAULT_CACHE_FRAMES):
        self._states: dict[str, list[TrafficLightState]] = {}
        self._zone_types: dict[str, str] = {}
        self._cache_frames = max(1, cache_frames)

    def set_cache_frames(self, n: int):
        self._cache_frames = max(1, n)

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
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            if all(isinstance(v, (int, float)) for v in raw):
                return [list(raw)]
        result = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                if all(isinstance(v, (int, float)) for v in item):
                    result.append(list(item))
        return result

    def process_frame(self, frame: np.ndarray,
                      crosswalk_zones: list) -> dict[str, AggregatedTLState]:
        fh, fw = frame.shape[:2]

        for zone in crosswalk_zones:
            if not getattr(zone, "has_light", False):
                continue
            roi_raw = getattr(zone, "traffic_light_roi", None)
            rois = self._parse_rois(roi_raw)
            if not rois:
                continue

            light_type = getattr(zone, "light_type", LIGHT_TYPE_PEDESTRIAN)
            self._zone_types[zone.id] = light_type
            states_list = self._get_or_create(zone.id, len(rois))

            for idx, (roi_def, st) in enumerate(zip(rois, states_list)):
                st._frame_cnt += 1
                if st._frame_cnt % self._cache_frames != 0:
                    continue

                x, y, w, h = roi_def
                x1 = int(max(0, x * fw))
                y1 = int(max(0, y * fh))
                x2 = int(min(fw, (x + w) * fw))
                y2 = int(min(fh, (y + h) * fh))

                if x2 <= x1 or y2 <= y1:
                    continue

                raw_state, raw_conf, _debug = classify_roi(frame[y1:y2, x1:x2])
                st.update(raw_state, raw_conf, light_type)

        return self._build_aggregated()

    def _build_aggregated(self) -> dict[str, AggregatedTLState]:
        result = {}
        for zone_id, states in self._states.items():
            raw_states = [s.state for s in states]
            agg_state = _aggregate_states(raw_states)
            avg_conf = (sum(s.confidence for s in states) / len(states)) if states else 0.0
            light_type = self._zone_types.get(zone_id, LIGHT_TYPE_PEDESTRIAN)
            any_allowed = any(s.is_pedestrian_allowed_or_grace(light_type) for s in states)

            result[zone_id] = AggregatedTLState(
                zone_id=zone_id,
                state=agg_state,
                confidence=avg_conf,
                light_type=light_type,
                green_grace=any_allowed,
                green_until=max((s.green_until for s in states), default=0.0),
                per_roi=[
                    {"state": s.state, "confidence": round(s.confidence, 2)}
                    for s in states
                ],
            )
        return result

    def get_state(self, zone_id: str) -> AggregatedTLState | None:
        return self._build_aggregated().get(zone_id)

    def get_all_states(self) -> dict[str, dict]:
        return {
            zid: {
                "state": s.state,
                "confidence": round(s.confidence, 2),
                "light_type": s.light_type,
                "green_grace": s.green_grace,
                "green_until": s.green_until,
                "per_roi": s.per_roi,
                "pedestrian_allowed": s.green_grace,
            }
            for zid, s in self._build_aggregated().items()
        }

    def reset(self):
        self._states.clear()
        self._zone_types.clear()