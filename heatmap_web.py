import json
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIRS = [BASE_DIR / "reports", BASE_DIR / "reports_batches"]
POSITIONS_FILE = BASE_DIR / "heatmap_camera_positions.json"
ASSETS_DIR = BASE_DIR / "heatmap_assets"
BACKGROUND_DIR = ASSETS_DIR / "backgrounds"
ZONES_FILE = BASE_DIR / "zones.json"
VIDEO_ROOT = BASE_DIR / "video"
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_positions_config():
    config = load_json(POSITIONS_FILE, {})
    positions = config.get("positions")
    if not isinstance(positions, dict):
        positions = {}

    background = config.get("background", {})
    if not isinstance(background, dict):
        background = {}
    ui = config.get("ui", {})
    if not isinstance(ui, dict):
        ui = {}

    return {
        "positions": positions,
        "background": {
            "filename": background.get("filename"),
            "original_name": background.get("original_name"),
            "updated_at": background.get("updated_at"),
        },
        "ui": {
            "positions_locked": bool(ui.get("positions_locked", False)),
        },
    }


def save_positions_config(config):
    save_json(POSITIONS_FILE, config)


def discover_report_files():
    report_files = []
    for report_dir in REPORT_DIRS:
        if not report_dir.exists():
            continue

        report_files.extend(report_dir.glob("**/report.json"))
        for json_file in report_dir.glob("**/*.json"):
            if json_file.name == "combined_report.json":
                continue
            if json_file.name == "report.json":
                continue
            report_files.append(json_file)

    unique_files = []
    seen = set()
    for file_path in report_files:
        resolved = str(file_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)
    return unique_files


def extract_report_summary(path: Path):
    payload = load_json(path, None)
    if not isinstance(payload, dict):
        return None

    camera_id = str(payload.get("camera_id") or "").strip()
    if not camera_id or camera_id == "all":
        return None

    total_violations = int(payload.get("total_violations") or 0)
    unique_persons = int(payload.get("unique_persons") or 0)
    summary = payload.get("summary", {})
    by_type = summary.get("by_type", {}) if isinstance(summary, dict) else {}
    by_age = summary.get("by_age", {}) if isinstance(summary, dict) else {}
    timeline = payload.get("timeline", [])
    raw_violations = payload.get("raw_violations", [])

    latest_ts = None
    if timeline and isinstance(timeline, list):
        last_point = timeline[-1]
        if isinstance(last_point, dict):
            latest_ts = last_point.get("timestamp")

    generated_at = payload.get("generated_at")
    report_ts = latest_ts or generated_at or ""

    return {
        "camera_id": camera_id,
        "total_violations": total_violations,
        "unique_persons": unique_persons,
        "by_type": by_type if isinstance(by_type, dict) else {},
        "by_age": by_age if isinstance(by_age, dict) else {},
        "raw_violations": raw_violations if isinstance(raw_violations, list) else [],
        "generated_at": generated_at,
        "report_ts": report_ts,
        "source_file": str(path.relative_to(BASE_DIR)),
    }


def choose_latest_report(summaries):
    def sort_key(item):
        raw_ts = item.get("report_ts") or ""
        try:
            dt = datetime.fromisoformat(raw_ts)
            return (1, dt)
        except Exception:
            return (0, raw_ts)

    return sorted(summaries, key=sort_key)[-1]


def collect_reports_by_camera():
    grouped = {}
    for report_file in discover_report_files():
        summary = extract_report_summary(report_file)
        if not summary:
            continue
        grouped.setdefault(summary["camera_id"], []).append(summary)

    result = {}
    for camera_id, items in grouped.items():
        latest = choose_latest_report(items)
        latest["sources"] = [item["source_file"] for item in items]
        latest["reports_found"] = len(items)
        result[camera_id] = latest
    return result


def collect_camera_ids():
    ids = set()

    positions = load_positions_config().get("positions", {})
    ids.update(positions.keys())

    zones = load_json(ZONES_FILE, {})
    cameras = zones.get("cameras", {}) if isinstance(zones, dict) else {}
    if isinstance(cameras, dict):
        ids.update(cameras.keys())

    ids.update(collect_reports_by_camera().keys())
    return sorted(ids)


def violation_sort_key(violation):
    raw_ts = violation.get("timestamp") if isinstance(violation, dict) else ""
    try:
        return datetime.fromisoformat(str(raw_ts))
    except Exception:
        return datetime.min


def build_violation_stats(report):
    by_type = report.get("by_type", {})
    if not isinstance(by_type, dict):
        by_type = {}

    total = int(report.get("total_violations") or 0)
    road_trespass = int(by_type.get("road_trespass") or 0)
    red_light = int(by_type.get("red_light") or 0)

    return {
        "total": total,
        "road_trespass": {
            "count": road_trespass,
            "percent": round((road_trespass / total) * 100, 1) if total else 0.0,
        },
        "red_light": {
            "count": red_light,
            "percent": round((red_light / total) * 100, 1) if total else 0.0,
        },
    }


def camera_violation_payload(camera_id: str):
    report = collect_reports_by_camera().get(camera_id, {})
    raw_violations = report.get("raw_violations", [])
    if not isinstance(raw_violations, list):
        raw_violations = []

    violations = []
    for violation in sorted(raw_violations, key=violation_sort_key):
        if not isinstance(violation, dict):
            continue
        violations.append({
            "timestamp": violation.get("timestamp"),
            "violation_type": violation.get("violation_type"),
        })

    return {
        "stats": build_violation_stats(report),
        "violations": violations,
        "source_file": report.get("source_file"),
        "generated_at": report.get("generated_at"),
    }


def resolve_video_folder(camera_id: str) -> Path | None:
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


def first_video_for_camera(camera_id: str) -> Path | None:
    videos = list_videos(resolve_video_folder(camera_id))
    return videos[0] if videos else None


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


def zones_for_camera(camera_id: str):
    zones_payload = load_json(ZONES_FILE, {})
    if not isinstance(zones_payload, dict):
        return []
    cameras = zones_payload.get("cameras", {})
    if not isinstance(cameras, dict):
        return []
    camera_entry = cameras.get(camera_id, {})
    if not isinstance(camera_entry, dict):
        return []
    zones = camera_entry.get("zones", [])
    if not isinstance(zones, list):
        return []
    return zones


def background_url(config):
    filename = config.get("background", {}).get("filename")
    if not filename:
        return None
    return f"/backgrounds/{filename}"


def build_heatmap_points():
    config = load_positions_config()
    positions = config.get("positions", {})
    reports = collect_reports_by_camera()

    points = []
    for camera_id in collect_camera_ids():
        pos = positions.get(camera_id)
        report = reports.get(camera_id, {})
        point = {
            "camera_id": camera_id,
            "label": camera_id,
            "x": None,
            "y": None,
            "total_violations": int(report.get("total_violations") or 0),
            "unique_persons": int(report.get("unique_persons") or 0),
            "generated_at": report.get("generated_at"),
            "source_file": report.get("source_file"),
            "sources": report.get("sources", []),
            "reports_found": int(report.get("reports_found") or 0),
            "by_type": report.get("by_type", {}),
        }
        if isinstance(pos, dict):
            try:
                point["x"] = float(pos.get("x"))
                point["y"] = float(pos.get("y"))
            except Exception:
                point["x"] = None
                point["y"] = None
        points.append(point)

    return {
        "background_url": background_url(config),
        "points": points,
    }


def create_app():
    app = Flask(__name__)
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/")
    def index():
        return render_template("heatmap.html")

    @app.route("/api/state")
    def api_state():
        config = load_positions_config()
        return jsonify({
            "background_url": background_url(config),
            "config": config,
            "camera_ids": collect_camera_ids(),
            "reports": collect_reports_by_camera(),
        })

    @app.route("/camera/<camera_id>")
    def camera_preview(camera_id: str):
        if camera_id not in collect_camera_ids():
            return render_template("index.html"), 404
        return render_template("heatmap_camera_preview.html", camera_id=camera_id)

    @app.route("/api/camera/<camera_id>/preview")
    def api_camera_preview(camera_id: str):
        if camera_id not in collect_camera_ids():
            return jsonify({"ok": False, "error": "camera not found"}), 404
        video_path = first_video_for_camera(camera_id)
        zones = zones_for_camera(camera_id)
        violations = camera_violation_payload(camera_id)
        return jsonify({
            "ok": True,
            "camera_id": camera_id,
            "has_frame": video_path is not None,
            "frame_url": f"/camera/{camera_id}/frame",
            "zones": zones,
            "violations": violations,
        })

    @app.route("/camera/<camera_id>/frame")
    def camera_frame(camera_id: str):
        video_path = first_video_for_camera(camera_id)
        if video_path is None:
            return jsonify({"ok": False, "error": "frame not found"}), 404
        frame_bytes = extract_first_frame_bytes(video_path)
        if frame_bytes is None:
            return jsonify({"ok": False, "error": "frame not found"}), 404
        return app.response_class(frame_bytes, mimetype="image/jpeg")

    @app.route("/api/heatmap")
    def api_heatmap():
        return jsonify(build_heatmap_points())

    @app.route("/api/positions", methods=["POST"])
    def api_positions():
        payload = request.get_json(force=True)
        positions = payload.get("positions", {})
        positions_locked = bool(payload.get("positions_locked", False))
        if not isinstance(positions, dict):
            return jsonify({"ok": False, "error": "positions must be an object"}), 400

        cleaned = {}
        for camera_id, coords in positions.items():
            if not isinstance(coords, dict):
                continue
            try:
                x = float(coords["x"])
                y = float(coords["y"])
            except Exception:
                continue
            cleaned[str(camera_id)] = {
                "x": max(0.0, min(1.0, x)),
                "y": max(0.0, min(1.0, y)),
            }

        config = load_positions_config()
        config["positions"] = cleaned
        config["ui"] = {"positions_locked": positions_locked}
        save_positions_config(config)
        return jsonify({"ok": True, "positions_saved": len(cleaned), "positions_locked": positions_locked})

    @app.route("/api/upload_background", methods=["POST"])
    def api_upload_background():
        file = request.files.get("background")
        if file is None or not file.filename:
            return jsonify({"ok": False, "error": "background file is required"}), 400

        original_name = secure_filename(file.filename)
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTS:
            return jsonify({"ok": False, "error": "unsupported image format"}), 400

        filename = f"background{suffix}"
        destination = BACKGROUND_DIR / filename
        file.save(destination)

        config = load_positions_config()
        config["background"] = {
            "filename": filename,
            "original_name": original_name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_positions_config(config)
        return jsonify({
            "ok": True,
            "background_url": background_url(config),
            "background": config["background"],
        })

    @app.route("/backgrounds/<path:filename>")
    def serve_background(filename):
        return send_from_directory(BACKGROUND_DIR, filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5055, debug=True)
