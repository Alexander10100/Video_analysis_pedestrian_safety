"""
zone_manager.py — Управление зонами разметки.

Структура зоны (RoadZone):
  {
    "id":      "uuid4",
    "label":   "Дорога №1",
    "polygon": [[x,y], ...],          # нормализованные 0..1
    "type":    "road",                # road | crosswalk
    "color":   [R, G, B],
    # только для type=crosswalk:
    "traffic_light_roi": [x,y,w,h] | [[x,y,w,h], ...] | null,  # нормализованные 0..1
    "has_light": true | false,
    "light_type": "pedestrian" | "vehicle",
      # pedestrian — пешеходный светофор: зелёный = разрешено
      # vehicle    — автомобильный (навстречу пешеходам): красный для машин = разрешено пешеходам
  }
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from traffic_light import LIGHT_TYPE_PEDESTRIAN

ZONES_FILE = Path("zones.json")


@dataclass
class RoadZone:
    id: str
    label: str
    polygon: list[list[float]]
    type: str               # "road" | "crosswalk"
    color: list[int]         # [R, G, B]
    traffic_light_roi: Optional[object] = None
    has_light: bool = False
    parent_id: Optional[str] = None
    light_type: str = LIGHT_TYPE_PEDESTRIAN  # "pedestrian" | "vehicle"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "polygon": self.polygon,
            "type": self.type,
            "color": self.color,
            "traffic_light_roi": self.traffic_light_roi,
            "has_light": self.has_light,
            "parent_id": self.parent_id,
            "light_type": self.light_type,
        }

    @staticmethod
    def from_dict(d: dict) -> "RoadZone":
        return RoadZone(
            id=d["id"],
            label=d["label"],
            polygon=d["polygon"],
            type=d.get("type", "road"),
            color=d.get("color", [0, 120, 255]),
            traffic_light_roi=d.get("traffic_light_roi"),
            has_light=d.get("has_light", False),
            parent_id=d.get("parent_id"),
            light_type=d.get("light_type", LIGHT_TYPE_PEDESTRIAN),
        )

    def np_polygon(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.float32)

    def contains_point(self, x: float, y: float) -> bool:
        if len(self.polygon) < 3:
            return False
        return cv2.pointPolygonTest(self.np_polygon(), (float(x), float(y)), False) >= 0

    def contains_box(self, x1: float, y1: float, x2: float, y2: float,
                     mode: str = "feet") -> bool:
        if mode == "feet":
            return self.contains_point((x1 + x2) / 2, y2)
        if mode == "center":
            return self.contains_point((x1 + x2) / 2, (y1 + y2) / 2)
        pts = [
            ((x1 + x2) / 2, (y1 + y2) / 2),
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        ]
        return any(self.contains_point(px, py) for px, py in pts)


class ZoneManager:
    """Потокобезопасное хранилище зон с персистентностью."""

    def __init__(self, filepath: Path = ZONES_FILE):
        self._lock = threading.Lock()
        self._zones: dict[str, RoadZone] = {}
        self._file = filepath
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_zone(self, label: str, polygon: list, zone_type: str = "road",
                 color: list | None = None, has_light: bool = False,
                 traffic_light_roi=None,
                 parent_id: str | None = None,
                 light_type: str = LIGHT_TYPE_PEDESTRIAN) -> RoadZone:
        if color is None:
            color = [0, 80, 220] if zone_type == "road" else [220, 160, 0]
        z = RoadZone(
            id=str(uuid.uuid4()),
            label=label,
            polygon=polygon,
            type=zone_type,
            color=color,
            has_light=has_light,
            traffic_light_roi=traffic_light_roi,
            parent_id=parent_id,
            light_type=light_type,
        )
        with self._lock:
            self._zones[z.id] = z
        self._save()
        return z

    def update_zone(self, zone_id: str, **kwargs) -> bool:
        with self._lock:
            z = self._zones.get(zone_id)
            if z is None:
                return False
            for k, v in kwargs.items():
                if hasattr(z, k):
                    setattr(z, k, v)
        self._save()
        return True

    def delete_zone(self, zone_id: str) -> bool:
        with self._lock:
            if zone_id not in self._zones:
                return False
            children = [zid for zid, z in self._zones.items()
                        if z.parent_id == zone_id]
            for cid in children:
                del self._zones[cid]
            del self._zones[zone_id]
        self._save()
        return True

    def get_all(self) -> list[RoadZone]:
        with self._lock:
            return list(self._zones.values())

    def get(self, zone_id: str) -> RoadZone | None:
        with self._lock:
            return self._zones.get(zone_id)

    def road_zones(self) -> list[RoadZone]:
        with self._lock:
            return [z for z in self._zones.values() if z.type == "road"]

    def crosswalk_zones(self) -> list[RoadZone]:
        with self._lock:
            return [z for z in self._zones.values() if z.type == "crosswalk"]

    # ── Персистентность ───────────────────────────────────────────────────────

    def _save(self):
        with self._lock:
            data = [z.to_dict() for z in self._zones.values()]
        self._file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self):
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "cameras" in raw:
                print("[zones] Обнаружен формат cameras-dict. Ожидайте /set_camera.")
                return
            if isinstance(raw, list):
                with self._lock:
                    for d in raw:
                        z = RoadZone.from_dict(d)
                        self._zones[z.id] = z
                print(f"[zones] Загружено {len(self._zones)} зон (плоский формат)")
        except Exception as e:
            print(f"[zones] Ошибка загрузки: {e}")

    def reload_for_camera(self, camera_id: str) -> int:
        with self._lock:
            self._zones.clear()
        if not self._file.exists():
            return 0
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "cameras" in raw:
                cam_data = raw["cameras"].get(camera_id, {})
                zones_list = cam_data.get("zones", [])
            elif isinstance(raw, list):
                zones_list = raw
            else:
                zones_list = []
            with self._lock:
                for d in zones_list:
                    z = RoadZone.from_dict(d)
                    self._zones[z.id] = z
            n = len(self._zones)
            print(f"[zones] Камера '{camera_id}': загружено {n} зон")
            return n
        except Exception as e:
            print(f"[zones] Ошибка reload_for_camera('{camera_id}'): {e}")
            return 0

    @staticmethod
    def get_cameras_from_file(filepath: Path = ZONES_FILE) -> dict:
        if not filepath.exists():
            return {}
        try:
            raw = json.loads(filepath.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "cameras" in raw:
                return {
                    cid: {
                        "label": info.get("label", cid),
                        "zones_count": len(info.get("zones", [])),
                    }
                    for cid, info in raw["cameras"].items()
                }
        except Exception:
            pass
        return {}

    def to_json_list(self) -> list[dict]:
        with self._lock:
            return [z.to_dict() for z in self._zones.values()]