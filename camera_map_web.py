import json
import os
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "video"
ZONES_FILE = BASE_DIR / "zones.json"
MAP_CONFIG_FILE = BASE_DIR / "camera_points.json"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_map_config():
    raw = load_json(MAP_CONFIG_FILE, {})
    cameras = raw.get("cameras", {})
    if not isinstance(cameras, dict):
        cameras = {}

    map_state = raw.get("map", {})
    if not isinstance(map_state, dict):
        map_state = {}

    return {
        "map": {
            "center": map_state.get("center", [55.751244, 37.618423]),
            "zoom": map_state.get("zoom", 11),
        },
        "cameras": cameras,
    }


def save_map_config(config):
    save_json(MAP_CONFIG_FILE, config)


def load_zones_db():
    raw = load_json(ZONES_FILE, {})
    if isinstance(raw, dict) and "cameras" in raw and isinstance(raw["cameras"], dict):
        return raw
    return {"cameras": {}}


def save_zones_db(data):
    save_json(ZONES_FILE, data)


def discover_camera_ids():
    ids = set()
    if VIDEO_ROOT.exists():
        ids.update(p.name for p in VIDEO_ROOT.iterdir() if p.is_dir())
    ids.update(load_zones_db().get("cameras", {}).keys())
    ids.update(load_map_config().get("cameras", {}).keys())
    return sorted(ids)


def resolve_video_folder(camera_id: str, config: dict) -> Path | None:
    camera_cfg = config.get("cameras", {}).get(camera_id, {})
    configured = camera_cfg.get("video_folder")
    if configured:
        folder = Path(configured)
        if not folder.is_absolute():
            folder = BASE_DIR / configured
        if folder.is_dir():
            return folder

    default_folder = VIDEO_ROOT / camera_id
    if default_folder.is_dir():
        return default_folder
    return None


def list_videos(folder: Path | None):
    if folder is None or not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def first_video_for_camera(camera_id: str, config: dict) -> Path | None:
    videos = list_videos(resolve_video_folder(camera_id, config))
    return videos[0] if videos else None


def camera_summary(camera_id: str, config: dict, zones_db: dict):
    camera_cfg = config.get("cameras", {}).get(camera_id, {})
    cam_zones = zones_db.get("cameras", {}).get(camera_id, {})
    first_video = first_video_for_camera(camera_id, config)
    return {
        "camera_id": camera_id,
        "label": camera_cfg.get("label", camera_id),
        "lat": camera_cfg.get("lat"),
        "lon": camera_cfg.get("lon"),
        "video_folder": str(resolve_video_folder(camera_id, config) or ""),
        "first_video": first_video.name if first_video else None,
        "has_video": first_video is not None,
        "zones_count": len(cam_zones.get("zones", [])),
    }


def all_camera_summaries():
    config = load_map_config()
    zones_db = load_zones_db()
    return [camera_summary(camera_id, config, zones_db) for camera_id in discover_camera_ids()]


def extract_first_frame_bytes(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return encoded.tobytes()


def _clean_roi(value):
    """
    Нормализовать traffic_light_roi из запроса.
    Принимает:
      - null / None / False / пустой список → None (нет ROI)
      - [x, y, w, h]         → [[x, y, w, h]]   (один ROI, нормализуем к списку списков)
      - [[x,y,w,h], ...]     → [[x,y,w,h], ...]  (уже список ROI)
    Возвращает None или список списков [x, y, w, h] с нормализованными 0..1 значениями.
    """
    if value is None or value is False:
        return None
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None

    # Проверяем: это одиночный ROI [x,y,w,h] из чисел?
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        rois_raw = [value]
    else:
        # Список ROI
        rois_raw = value

    cleaned_rois = []
    for roi in rois_raw:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            continue
        try:
            x = float(roi[0])
            y = float(roi[1])
            w = float(roi[2])
            h = float(roi[3])
        except (TypeError, ValueError):
            continue
        # Значения должны быть нормализованы 0..1 и иметь ненулевую площадь
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        w = max(0.001, min(1.0 - x, w))
        h = max(0.001, min(1.0 - y, h))
        cleaned_rois.append([
            round(x, 6),
            round(y, 6),
            round(w, 6),
            round(h, 6),
        ])

    if not cleaned_rois:
        return None
    # Если один ROI — храним как список из одного элемента
    # (единообразно для traffic_light.py)
    return cleaned_rois


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template(
            "camera_map.html",
            yandex_maps_api_key=os.environ.get("YANDEX_MAPS_API_KEY", ""),
        )

    @app.route("/camera/<camera_id>")
    def camera_detail(camera_id: str):
        if camera_id not in discover_camera_ids():
            abort(404)
        return render_template("camera_detail.html", camera_id=camera_id)

    @app.route("/api/map/state")
    def api_map_state():
        config = load_map_config()
        return jsonify({
            "api_key_present": bool(os.environ.get("YANDEX_MAPS_API_KEY")),
            "map": config.get("map", {}),
            "cameras": all_camera_summaries(),
        })

    @app.route("/api/map/state", methods=["POST"])
    def api_map_state_save():
        payload = request.get_json(force=True)
        config = load_map_config()
        map_state = payload.get("map", {})
        if isinstance(map_state, dict):
            center = map_state.get("center")
            zoom = map_state.get("zoom")
            if isinstance(center, list) and len(center) == 2:
                try:
                    lat = float(center[0])
                    lon = float(center[1])
                    config["map"]["center"] = [lat, lon]
                except Exception:
                    pass
            if zoom is not None:
                try:
                    config["map"]["zoom"] = float(zoom)
                except Exception:
                    pass
        save_map_config(config)
        return jsonify({"ok": True, "map": config["map"]})

    @app.route("/api/camera/<camera_id>/position", methods=["POST"])
    def api_camera_position(camera_id: str):
        payload = request.get_json(force=True)
        try:
            lat = float(payload["lat"])
            lon = float(payload["lon"])
        except Exception:
            return jsonify({"ok": False, "error": "lat/lon are required"}), 400

        config = load_map_config()
        cam = config["cameras"].setdefault(camera_id, {"label": camera_id})
        cam["lat"] = lat
        cam["lon"] = lon
        if "label" not in cam:
            cam["label"] = camera_id
        save_map_config(config)
        return jsonify({"ok": True, "camera": camera_summary(camera_id, config, load_zones_db())})

    @app.route("/api/camera/<camera_id>")
    def api_camera_detail(camera_id: str):
        config = load_map_config()
        zones_db = load_zones_db()
        if camera_id not in discover_camera_ids():
            abort(404)
        first_video = first_video_for_camera(camera_id, config)
        return jsonify({
            "camera": camera_summary(camera_id, config, zones_db),
            "zones": zones_db.get("cameras", {}).get(camera_id, {}).get("zones", []),
            "frame_url": f"/camera/{camera_id}/frame",
            "has_frame": first_video is not None,
        })

    @app.route("/camera/<camera_id>/frame")
    def camera_frame(camera_id: str):
        config = load_map_config()
        video_path = first_video_for_camera(camera_id, config)
        if video_path is None:
            abort(404)
        frame_bytes = extract_first_frame_bytes(video_path)
        if frame_bytes is None:
            abort(404)
        return Response(frame_bytes, mimetype="image/jpeg")

    @app.route("/api/camera/<camera_id>/zones", methods=["POST"])
    def api_camera_zones_save(camera_id: str):
        payload = request.get_json(force=True)
        zones = payload.get("zones", [])
        if not isinstance(zones, list):
            return jsonify({"ok": False, "error": "zones must be a list"}), 400

        cleaned = []
        errors = []

        for zone_idx, zone in enumerate(zones):
            if not isinstance(zone, dict):
                errors.append(f"zone[{zone_idx}] не является объектом")
                continue

            # ── Полигон ───────────────────────────────────────────────────────
            polygon = zone.get("polygon", [])
            if not isinstance(polygon, list) or len(polygon) < 3:
                errors.append(f"zone[{zone_idx}] polygon < 3 точек")
                continue

            normalized = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    normalized = []
                    break
                try:
                    x = max(0.0, min(1.0, float(point[0])))
                    y = max(0.0, min(1.0, float(point[1])))
                except (TypeError, ValueError):
                    normalized = []
                    break
                normalized.append([round(x, 6), round(y, 6)])

            if len(normalized) < 3:
                errors.append(f"zone[{zone_idx}] некорректные координаты полигона")
                continue

            # ── Тип и цвет ────────────────────────────────────────────────────
            zone_type = str(zone.get("type") or "road")
            if zone_type not in ("road", "crosswalk"):
                zone_type = "road"

            color = zone.get("color")
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                color = [0, 80, 220] if zone_type == "road" else [220, 160, 0]
            else:
                try:
                    color = [max(0, min(255, int(c))) for c in color]
                except (TypeError, ValueError):
                    color = [0, 80, 220] if zone_type == "road" else [220, 160, 0]

            # ── traffic_light_roi — важный блок, не теряем ────────────────────
            tl_roi = _clean_roi(zone.get("traffic_light_roi"))

            # ── has_light: True только если есть валидный ROI ─────────────────
            has_light_raw = zone.get("has_light", False)
            has_light = bool(has_light_raw) and (tl_roi is not None)

            # ── light_type ────────────────────────────────────────────────────
            light_type = str(zone.get("light_type") or "pedestrian")
            if light_type not in ("pedestrian", "vehicle"):
                light_type = "pedestrian"

            # ── ID и метка ────────────────────────────────────────────────────
            zone_id = str(zone.get("id") or f"{camera_id}_{len(cleaned) + 1}")
            label = str(zone.get("label") or f"Зона {len(cleaned) + 1}")

            cleaned.append({
                "id": zone_id,
                "label": label,
                "polygon": normalized,
                "type": zone_type,
                "color": color,
                "traffic_light_roi": tl_roi,
                "has_light": has_light,
                "parent_id": zone.get("parent_id"),
                "light_type": light_type,
            })

        zones_db = load_zones_db()
        zones_db.setdefault("cameras", {})
        entry = zones_db["cameras"].setdefault(camera_id, {"label": camera_id, "zones": []})
        entry["label"] = entry.get("label") or camera_id
        entry["zones"] = cleaned
        save_zones_db(zones_db)

        response = {"ok": True, "zones_count": len(cleaned)}
        if errors:
            response["warnings"] = errors
        return jsonify(response)

    @app.route("/api/camera/<camera_id>/zones/debug", methods=["GET"])
    def api_camera_zones_debug(camera_id: str):
        """Отладочный эндпоинт: вернуть зоны как они хранятся в zones.json."""
        zones_db = load_zones_db()
        cam_data = zones_db.get("cameras", {}).get(camera_id, {})
        zones = cam_data.get("zones", [])
        result = []
        for z in zones:
            result.append({
                "id": z.get("id"),
                "label": z.get("label"),
                "type": z.get("type"),
                "has_light": z.get("has_light"),
                "light_type": z.get("light_type"),
                "traffic_light_roi": z.get("traffic_light_roi"),
                "polygon_points": len(z.get("polygon", [])),
            })
        return jsonify({"camera_id": camera_id, "zones": result, "total": len(result)})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5080, debug=True)