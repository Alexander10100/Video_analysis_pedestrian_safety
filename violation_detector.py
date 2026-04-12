"""
violation_detector.py — Определение нарушений ПДД пешеходами.

Типы нарушений:
  ROAD_TRESPASS — человек идёт по проезжей части вне перехода
  RED_LIGHT     — человек на пешеходном переходе при красном (без grace-периода)

Координаты bbox нормализованы (0..1), координаты зон тоже нормализованы.
При отрисовке денормализуются обратно в пиксели через fw/fh.

Отрисовка текста через PIL — поддержка кириллицы и любых Unicode-символов.
Проверка вхождения человека в зону — строго по нижней середине bbox (ноги),
чтобы исключить ложные срабатывания у края зоны.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


# ── Цвета рамок (BGR) ─────────────────────────────────────────────────────────
COLOR_OK        = ( 50, 205,  50)   # зелёный  — норма, вне зон
COLOR_VIOLATION = (  0,   0, 230)   # красный  — нарушение
COLOR_CROSSWALK = (  0, 180, 255)   # оранжевый — переход разрешён
COLOR_WARNING   = (  0, 140, 255)   # жёлтый   — светофор неизвестен


# ── Шрифт для PIL ─────────────────────────────────────────────────────────────
# Ищем системный шрифт с поддержкой кириллицы
_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    # macOS
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


@lru_cache(maxsize=8)
def _get_pil_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    if not _PIL_AVAILABLE:
        return None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # fallback — встроенный bitmap-шрифт PIL (только ASCII, но не упадёт)
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
    """
    Рисует текст через PIL с поддержкой кириллицы.
    xy — левый верхний угол текстового блока (с учётом padding).
    Возвращает изменённый frame (в формате BGR).
    """
    if not _PIL_AVAILABLE:
        # Fallback на OpenCV — кириллица превратится в '?', но не упадёт
        r, g, b = color_bgr
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (b, g, r), 1, cv2.LINE_AA)
        return frame

    font = _get_pil_font(font_size)

    # Конвертируем BGR → RGB для PIL
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)

    # Размер текста
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    x, y = xy

    # Фон под текстом
    if bg_color_bgr is not None:
        br, bg, bb = bg_color_bgr
        draw.rectangle(
            [x, y, x + tw + padding * 2, y + th + padding * 2],
            fill=(bb, bg, br),
        )

    # Текст (PIL: RGB)
    r, g, b = color_bgr
    draw.text((x + padding, y + padding), text, font=font, fill=(b, g, r))

    # Конвертируем обратно BGR
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


@dataclass
class PersonViolation:
    track_id:   int
    violation:  str    # "none" | "road_trespass" | "red_light"
    zone_label: str
    box:        tuple  # нормализованные (nx1, ny1, nx2, ny2)
    color:      tuple  # цвет рамки BGR
    note:       str   = ""
    conf:       float = 0.0   # уверенность детектора 0..1
    age_label:  str = "adult"
    age_conf:   float = 0.0


class ViolationDetector:
    """
    На каждом кадре принимает список нормализованных bbox
    и возвращает PersonViolation для каждого.

    Проверка зон — строго по нижней середине bbox (ноги человека),
    что исключает ложные срабатывания когда человек стоит рядом с зоной.
    """

    def __init__(self, zone_mgr: ZoneManager, tl_analyzer: TrafficLightAnalyzer):
        self._zmgr = zone_mgr
        self._tla  = tl_analyzer

    def analyze(self, boxes: list[tuple]) -> list[PersonViolation]:
        """
        boxes: [(track_id, nx1, ny1, nx2, ny2, conf), ...]  — координаты 0..1
        """
        road_zones      = self._zmgr.road_zones()
        crosswalk_zones = self._zmgr.crosswalk_zones()
        tl_states       = self._tla.get_all_states()

        return [
            self._classify(tid, nx1, ny1, nx2, ny2, conf,
                           road_zones, crosswalk_zones, tl_states)
            for tid, nx1, ny1, nx2, ny2, conf in boxes
        ]

    def _classify(self, tid, nx1, ny1, nx2, ny2, conf,
                  road_zones, crosswalk_zones, tl_states) -> PersonViolation:

        # Точка «ноги» — нижняя середина bbox (единственная точка проверки)
        foot_x = (nx1 + nx2) / 2
        foot_y = ny2

        # 1. Проверяем crosswalk-зоны (приоритет над дорогой)
        for cw in crosswalk_zones:
            if not cw.contains_point(foot_x, foot_y):
                continue

            if not cw.has_light:
                return PersonViolation(
                    tid, "none", cw.label,
                    (nx1, ny1, nx2, ny2), COLOR_CROSSWALK,
                    "переход без светофора", conf,
                )

            st = tl_states.get(cw.id)
            if st is None or st["state"] == STATE_UNKNOWN:
                return PersonViolation(
                    tid, "none", cw.label,
                    (nx1, ny1, nx2, ny2), COLOR_WARNING,
                    "светофор: неизвестно", conf,
                )

            if st["green_grace"]:
                note = "зелёный" if st["state"] == "green" else "grace"
                return PersonViolation(
                    tid, "none", cw.label,
                    (nx1, ny1, nx2, ny2), COLOR_CROSSWALK, note, conf,
                )

            return PersonViolation(
                tid, "red_light", cw.label,
                (nx1, ny1, nx2, ny2), COLOR_VIOLATION,
                f"красный ({st['state']})", conf,
            )

        # 2. Проверяем road-зоны (только ноги)
        for rz in road_zones:
            if rz.contains_point(foot_x, foot_y):
                return PersonViolation(
                    tid, "road_trespass", rz.label,
                    (nx1, ny1, nx2, ny2), COLOR_VIOLATION,
                    "на проезжей части", conf,
                )

        # 3. Вне всех зон
        return PersonViolation(
            tid, "none", "",
            (nx1, ny1, nx2, ny2), COLOR_OK, "", conf,
        )


def draw_violations(
    frame: np.ndarray,
    violations: list[PersonViolation],
    fw: int,
    fh: int,
) -> tuple[np.ndarray, int]:
    """
    Рисует bbox на кадре с цветовой кодировкой нарушений.
    Метка: уверенность % + тип нарушения + заметка.
    Текст рендерится через PIL для корректного отображения кириллицы.
    Возвращает (annotated_frame, violation_count).
    """
    vcount = 0

    label_map = {
        "road_trespass": "ДОРОГА",
        "red_light":     "КРАСНЫЙ",
    }

    for pv in violations:
        nx1, ny1, nx2, ny2 = pv.box
        x1 = int(nx1 * fw);  y1 = int(ny1 * fh)
        x2 = int(nx2 * fw);  y2 = int(ny2 * fh)
        color     = pv.color
        thickness = 3 if pv.violation != "none" else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Точка «ноги» — визуальное подтверждение точки проверки
        foot_px = ((x1 + x2) // 2, y2)
        cv2.circle(frame, foot_px, 4, color, -1)

        # Метка
        parts = [f"{pv.conf:.0%}"]
        if pv.violation != "none":
            parts.append(label_map.get(pv.violation, pv.violation))
            vcount += 1
        if pv.note:
            parts.append(pv.note)

        lbl      = "  ".join(parts)
        font_sz  = 13
        lbl_y    = max(y1 - 2, 20)

        frame = _put_text_pil(
            frame, lbl,
            (x1, lbl_y - font_sz - 6),
            color_bgr=(240, 240, 240),
            font_size=font_sz,
            bg_color_bgr=color,
            padding=3,
        )

    return frame, vcount


# ── Цвета и подписи состояний светофора (BGR) ─────────────────────────────────
_TL_COLORS = {
    "red":     (  0,   0, 210),
    "green":   (  0, 190,   0),
    "yellow":  (  0, 190, 230),
    "unknown": ( 70,  70,  70),
    "grace":   (  0, 140,  70),
}
_TL_LABELS = {
    "red":     "КРАСНЫЙ",
    "green":   "ЗЕЛЁНЫЙ",
    "yellow":  "ЖЁЛТЫЙ",
    "unknown": "?",
    "grace":   "GRACE",
}


def draw_traffic_light_states(
    frame: np.ndarray,
    crosswalk_zones: list,
    tl_states: dict,
    fw: int,
    fh: int,
) -> np.ndarray:
    """
    Рисует рамку вокруг каждого ROI светофора (их может быть несколько)
    и цветную плашку с его состоянием прямо на кадре.

    crosswalk_zones — список RoadZone с type='crosswalk' и has_light=True.
    tl_states       — dict zone_id → {state, confidence, green_grace, green_until, per_roi}
    fw, fh          — размеры кадра в пикселях для денормализации.
    """
    from traffic_light import TrafficLightAnalyzer

    for zone in crosswalk_zones:
        if not zone.has_light or not zone.traffic_light_roi:
            continue

        st = tl_states.get(zone.id)
        if st is None:
            continue

        state_str = st["state"]
        is_grace  = st.get("green_grace", False) and state_str != "green"
        key       = "grace" if is_grace else state_str
        color     = _TL_COLORS.get(key, _TL_COLORS["unknown"])
        label     = _TL_LABELS.get(key, "?")
        conf_pct  = f"{st.get('confidence', 0):.0%}"
        per_roi   = st.get("per_roi", [])

        # Нормализуем к списку ROI
        raw = zone.traffic_light_roi
        if raw and isinstance(raw[0], (int, float)):
            rois = [raw]
        else:
            rois = list(raw)

        for roi_idx, roi_def in enumerate(rois):
            rx, ry, rw, rh = roi_def
            x1 = int(rx * fw)
            y1 = int(ry * fh)
            x2 = int((rx + rw) * fw)
            y2 = int((ry + rh) * fh)

            # Цвет конкретного ROI (если есть per_roi данные)
            if roi_idx < len(per_roi):
                roi_state = per_roi[roi_idx]["state"]
                roi_key   = "grace" if (is_grace and roi_state == "green") else roi_state
                roi_color = _TL_COLORS.get(roi_key, _TL_COLORS["unknown"])
                roi_label = _TL_LABELS.get(roi_key, "?")
                roi_conf  = f"{per_roi[roi_idx]['confidence']:.0%}"
            else:
                roi_color = color
                roi_label = label
                roi_conf  = conf_pct

            # Полупрозрачная заливка ROI
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), roi_color, -1)
            cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

            # Рамка ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

            # Плашка с состоянием над ROI (через PIL для кириллицы)
            full_lbl = f"{roi_label}  {roi_conf}"
            lbl_y    = max(y1 - 2, 22)
            frame = _put_text_pil(
                frame, full_lbl,
                (x1, lbl_y - 22),
                color_bgr=(230, 230, 230),
                font_size=13,
                bg_color_bgr=roi_color,
                padding=3,
            )

            # Имя зоны + номер ROI внутри блока снизу
            if len(rois) > 1:
                name_lbl = f"{zone.label} #{roi_idx + 1}"
            else:
                name_lbl = zone.label

            frame = _put_text_pil(
                frame, name_lbl,
                (x1 + 2, y2 - 18),
                color_bgr=(roi_color[0] + 40, roi_color[1] + 40, roi_color[2] + 40),
                font_size=11,
            )

    return frame


def draw_zones(
    frame: np.ndarray,
    zones: list[RoadZone],
    tl_states: dict | None = None,
    alpha: float = 0.22,
) -> np.ndarray:
    """
    Накладывает полупрозрачные зоны на кадр.
    Зоны хранятся в нормализованных координатах — масштабируем в пиксели здесь.
    Подписи зон рендерятся через PIL для корректной кириллицы.
    tl_states: dict zone_id → {"state": ..., "green_grace": bool}
    """
    fh, fw    = frame.shape[:2]
    overlay   = frame.copy()
    tl_states = tl_states or {}

    for z in zones:
        if len(z.polygon) < 3:
            continue

        # Денормализуем полигон
        pts = np.array(
            [[int(x * fw), int(y * fh)] for x, y in z.polygon],
            dtype=np.int32,
        )

        # Цвет заливки: для crosswalk зависит от светофора
        if z.type == "crosswalk" and z.id in tl_states:
            st = tl_states[z.id]
            if st["green_grace"]:
                color_bgr = (0, 200, 0)
            elif st["state"] in ("red", "yellow"):
                color_bgr = (0, 0, 200)
            else:
                color_bgr = (0, 160, 200)
        else:
            r, g, b   = z.color
            color_bgr = (b, g, r)

        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2)

        # Центр зоны
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))

        # Подпись через PIL — кириллица отображается корректно
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