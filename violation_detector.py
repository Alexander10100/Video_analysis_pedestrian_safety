"""
violation_detector.py — Определение нарушений ПДД пешеходами.

Типы нарушений:
  ROAD_TRESPASS — человек идёт по проезжей части вне перехода
  RED_LIGHT     — человек на переходе при запрещающем сигнале (с учётом типа светофора)

Изменения vs оригинал:
  - Логика green_grace заменена на pedestrian_allowed из AggregatedTLState
    (учитывает light_type: pedestrian/vehicle и grace-период)
  - PersonViolation получил поля age_label / age_conf
  - draw_violations рисует маркер возраста рядом с рамкой
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from zone_manager import RoadZone, ZoneManager
from traffic_light import TrafficLightAnalyzer, STATE_UNKNOWN


# ── Утилита: фильтрация водителей/пассажиров в транспорте ────────────────────

def person_inside_vehicle(
    px1: int, py1: int, px2: int, py2: int,
    vehicle_boxes: list[tuple[int, int, int, int]],
    overlap_thresh: float = 0.45,
) -> bool:
    """
    Вернуть True если человек скорее всего находится внутри ТС.

    Берём нижние 60% bbox человека (туловище/торс) и смотрим какая доля
    этой области перекрывается с bbox автомобиля/автобуса/грузовика.
    Если >= overlap_thresh — это водитель или пассажир, не пешеход.

    overlap_thresh = 0.45: требуем почти половину нижней части внутри ТС.
    Пешеход, идущий рядом с машиной, не попадает под фильтр.

    Используется в stream_detect.py и offline_detect.py.
    YOLO классы ТС: 2=car, 3=motorcycle, 5=bus, 7=truck.
    """
    if not vehicle_boxes:
        return False

    ph = py2 - py1
    lower_y1 = py1 + int(ph * 0.40)   # нижние 60%
    lower_area = (px2 - px1) * (lower_y1 - py2).__abs__()
    if lower_area <= 0:
        return False

    for vx1, vy1, vx2, vy2 in vehicle_boxes:
        ix1 = max(px1, vx1)
        iy1 = max(lower_y1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        intersection = (ix2 - ix1) * (iy2 - iy1)
        if intersection / lower_area >= overlap_thresh:
            return True
    return False


# ── Цвета рамок (BGR) ─────────────────────────────────────────────────────────
COLOR_OK = (50, 205, 50)
COLOR_VIOLATION = (0, 0, 230)
COLOR_CROSSWALK = (0, 180, 255)
COLOR_WARNING = (0, 140, 255)

# ── Шрифт ─────────────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


@lru_cache(maxsize=8)
def _get_pil_font(size: int):
    if not _PIL_AVAILABLE:
        return None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _put_text_pil(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color_bgr: tuple[int, int, int],
    font_size: int = 14,
    bg_color_bgr: tuple[int, int, int] | None = None,
    padding: int = 3,
) -> np.ndarray:
    if not _PIL_AVAILABLE:
        r, g, b = color_bgr
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (b, g, r), 1, cv2.LINE_AA)
        return frame

    font = _get_pil_font(font_size)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy

    if bg_color_bgr is not None:
        br, bg, bb = bg_color_bgr
        draw.rectangle([x, y, x + tw + padding * 2, y + th + padding * 2],
                       fill=(bb, bg, br))

    r, g, b = color_bgr
    draw.text((x + padding, y + padding), text, font=font, fill=(b, g, r))
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


@dataclass
class PersonViolation:
    track_id: int
    violation: str    # "none" | "road_trespass" | "red_light"
    zone_label: str
    box: tuple  # нормализованные (nx1, ny1, nx2, ny2)
    color: tuple  # BGR
    note: str = ""
    conf: float = 0.0
    age_label: str = "adult"   # "adult" | "child" | "unknown"
    age_conf: float = 0.0


class ViolationDetector:
    """
    На каждом кадре принимает список нормализованных bbox
    и возвращает PersonViolation для каждого.
    """

    def __init__(self, zone_mgr: ZoneManager, tl_analyzer: TrafficLightAnalyzer):
        self._zmgr = zone_mgr
        self._tla = tl_analyzer

    def analyze(self, boxes: list[tuple]) -> list[PersonViolation]:
        """
        boxes: [(track_id, nx1, ny1, nx2, ny2, conf, age_label, age_conf), ...]
        """
        road_zones = self._zmgr.road_zones()
        crosswalk_zones = self._zmgr.crosswalk_zones()
        tl_states = self._tla.get_all_states()

        return [
            self._classify(*b, road_zones, crosswalk_zones, tl_states)
            for b in boxes
        ]

    def _classify(self, tid, nx1, ny1, nx2, ny2, conf,
                  age_label, age_conf,
                  road_zones, crosswalk_zones, tl_states) -> PersonViolation:

        foot_x = (nx1 + nx2) / 2
        foot_y = ny2

        # 1. Crosswalk-зоны (приоритет)
        for cw in crosswalk_zones:
            if not cw.contains_point(foot_x, foot_y):
                continue

            if not cw.has_light:
                return PersonViolation(
                    tid, "none", cw.label,
                    (nx1, ny1, nx2, ny2), COLOR_CROSSWALK,
                    "переход без светофора", conf,
                    age_label, age_conf,
                )

            st = tl_states.get(cw.id)
            if st is None or st["state"] == STATE_UNKNOWN:
                return PersonViolation(
                    tid, "none", cw.label,
                    (nx1, ny1, nx2, ny2), COLOR_WARNING,
                    "светофор: неизвестно", conf,
                    age_label, age_conf,
                )

            light_type = st.get("light_type", "pedestrian")
            allowed = st.get("pedestrian_allowed", st.get("green_grace", False))

            if allowed:
                # Описываем фактическое состояние светофора для пользователя
                if light_type == "vehicle":
                    note = f"авто:{st['state']} → пешеход ОК"
                else:
                    note = "зелёный" if st["state"] == "green" else "grace"
                return PersonViolation(
                    tid, "none", cw.label,
                    (nx1, ny1, nx2, ny2), COLOR_CROSSWALK,
                    note, conf, age_label, age_conf,
                )

            # Нарушение
            if light_type == "vehicle":
                note = f"авто:{st['state']} → СТОП"
            else:
                note = f"красный ({st['state']})"

            return PersonViolation(
                tid, "red_light", cw.label,
                (nx1, ny1, nx2, ny2), COLOR_VIOLATION,
                note, conf, age_label, age_conf,
            )

        # 2. Road-зоны
        for rz in road_zones:
            if rz.contains_point(foot_x, foot_y):
                return PersonViolation(
                    tid, "road_trespass", rz.label,
                    (nx1, ny1, nx2, ny2), COLOR_VIOLATION,
                    "на проезжей части", conf,
                    age_label, age_conf,
                )

        # 3. Вне зон
        return PersonViolation(
            tid, "none", "",
            (nx1, ny1, nx2, ny2), COLOR_OK, "",
            conf, age_label, age_conf,
        )


# ── Метки возраста ────────────────────────────────────────────────────────────
_AGE_LABELS = {
    "adult": ("ВЗР", (50, 205, 50)),
    "child": ("РЕБ", (0, 165, 255)),
    "unknown": ("?", (160, 160, 160)),
}


def draw_violations(
    frame: np.ndarray,
    violations: list[PersonViolation],
    fw: int,
    fh: int,
) -> tuple[np.ndarray, int]:
    vcount = 0

    label_map = {
        "road_trespass": "ДОРОГА",
        "red_light": "КРАСНЫЙ",
    }

    for pv in violations:
        nx1, ny1, nx2, ny2 = pv.box
        x1 = int(nx1 * fw)
        y1 = int(ny1 * fh)
        x2 = int(nx2 * fw)
        y2 = int(ny2 * fh)
        color = pv.color
        thickness = 3 if pv.violation != "none" else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)

        # Основная метка
        parts = [f"{pv.conf:.0%}"]
        if pv.violation != "none":
            parts.append(label_map.get(pv.violation, pv.violation))
            vcount += 1
        if pv.note:
            parts.append(pv.note)

        lbl = "  ".join(parts)
        font_sz = 13
        lbl_y = max(y1 - 2, 20)

        frame = _put_text_pil(
            frame, lbl,
            (x1, lbl_y - font_sz - 6),
            color_bgr=(240, 240, 240),
            font_size=font_sz,
            bg_color_bgr=color,
            padding=3,
        )

        # Метка возраста (правый верхний угол bbox)
        age_lbl, age_color = _AGE_LABELS.get(pv.age_label, _AGE_LABELS["unknown"])
        age_text = age_lbl
        if pv.age_conf > 0:
            age_text += f" {pv.age_conf:.0%}"
        frame = _put_text_pil(
            frame, age_text,
            (x2 - 55, lbl_y - font_sz - 6),
            color_bgr=(240, 240, 240),
            font_size=11,
            bg_color_bgr=age_color,
            padding=2,
        )

    return frame, vcount


# ── Цвета и подписи светофоров ────────────────────────────────────────────────
_TL_COLORS = {
    "red": (0, 0, 210),
    "green": (0, 190, 0),
    "yellow": (0, 190, 230),
    "unknown": (70, 70, 70),
    "grace": (0, 140, 70),
    # Для vehicle-светофора при красном (= разрешено пешеходам)
    "vehicle_allowed": (0, 190, 100),
}
_TL_LABELS = {
    "red": "КРАСНЫЙ",
    "green": "ЗЕЛЁНЫЙ",
    "yellow": "ЖЁЛТЫЙ",
    "unknown": "?",
    "grace": "GRACE",
    "vehicle_allowed": "АВТ.КР→МОЖНО",
}


def _tl_display_key(st: dict) -> str:
    """Вернуть ключ для цвета/метки с учётом типа светофора."""
    light_type = st.get("light_type", "pedestrian")
    state = st.get("state", "unknown")
    allowed = st.get("pedestrian_allowed", False)

    if light_type == "vehicle":
        if allowed:
            return "vehicle_allowed"
        # Не разрешено — green для машин = опасно
        return "green" if state == "green" else state

    # Пешеходный
    if st.get("green_grace") and state != "green":
        return "grace"
    return state


def draw_traffic_light_states(
    frame: np.ndarray,
    crosswalk_zones: list,
    tl_states: dict,
    fw: int,
    fh: int,
) -> np.ndarray:
    for zone in crosswalk_zones:
        if not zone.has_light or not zone.traffic_light_roi:
            continue
        st = tl_states.get(zone.id)
        if st is None:
            continue

        key = _tl_display_key(st)
        color = _TL_COLORS.get(key, _TL_COLORS["unknown"])
        label = _TL_LABELS.get(key, "?")

        # Добавляем пометку типа светофора
        light_type = st.get("light_type", "pedestrian")
        type_mark = " [А]" if light_type == "vehicle" else " [П]"

        conf_pct = f"{st.get('confidence', 0):.0%}"
        per_roi = st.get("per_roi", [])

        raw = zone.traffic_light_roi
        rois = [raw] if (raw and isinstance(raw[0], (int, float))) else list(raw or [])

        for roi_idx, roi_def in enumerate(rois):
            rx, ry, rw, rh = roi_def
            x1 = int(rx * fw)
            y1 = int(ry * fh)
            x2 = int((rx + rw) * fw)
            y2 = int((ry + rh) * fh)

            roi_color = color
            roi_label = label
            roi_conf = conf_pct

            if roi_idx < len(per_roi):
                roi_state = per_roi[roi_idx]["state"]
                # Для vehicle отдельно пересчитываем отображаемый ключ
                if light_type == "vehicle":
                    from traffic_light import pedestrian_allowed, LIGHT_TYPE_VEHICLE
                    roi_allowed = pedestrian_allowed(roi_state, LIGHT_TYPE_VEHICLE)
                    roi_key = "vehicle_allowed" if roi_allowed else roi_state
                else:
                    roi_key = roi_state
                roi_color = _TL_COLORS.get(roi_key, _TL_COLORS["unknown"])
                roi_label = _TL_LABELS.get(roi_key, "?")
                roi_conf = f"{per_roi[roi_idx]['confidence']:.0%}"

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), roi_color, -1)
            cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

            full_lbl = f"{roi_label}{type_mark}  {roi_conf}"
            lbl_y = max(y1 - 2, 22)
            frame = _put_text_pil(
                frame, full_lbl,
                (x1, lbl_y - 22),
                color_bgr=(230, 230, 230),
                font_size=13,
                bg_color_bgr=roi_color,
                padding=3,
            )
            name_lbl = f"{zone.label} #{roi_idx + 1}" if len(rois) > 1 else zone.label
            frame = _put_text_pil(
                frame, name_lbl,
                (x1 + 2, y2 - 18),
                color_bgr=(
                    min(255, roi_color[0] + 40),
                    min(255, roi_color[1] + 40),
                    min(255, roi_color[2] + 40),
                ),
                font_size=11,
            )

    return frame


def draw_zones(
    frame: np.ndarray,
    zones: list[RoadZone],
    tl_states: dict | None = None,
    alpha: float = 0.22,
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    tl_states = tl_states or {}

    for z in zones:
        if len(z.polygon) < 3:
            continue
        pts = np.array(
            [[int(x * fw), int(y * fh)] for x, y in z.polygon],
            dtype=np.int32,
        )

        if z.type == "crosswalk" and z.id in tl_states:
            st = tl_states[z.id]
            allowed = st.get("pedestrian_allowed", st.get("green_grace", False))
            if allowed:
                color_bgr = (0, 200, 0)
            elif st["state"] in ("red", "yellow", "green"):
                # green для machine = опасно
                color_bgr = (0, 0, 200)
            else:
                color_bgr = (0, 160, 200)
        else:
            r, g, b = z.color
            color_bgr = (b, g, r)

        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2)

        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        frame = _put_text_pil(
            frame, z.label,
            (cx - 30, cy - 10),
            color_bgr=color_bgr,
            font_size=14,
            bg_color_bgr=(10, 10, 10),
            padding=2,
        )

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame