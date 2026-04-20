import json
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIRS = [BASE_DIR / "reports", BASE_DIR / "reports_batches"]
POSITIONS_FILE = BASE_DIR / "heatmap_camera_positions.json"
ASSETS_DIR = BASE_DIR / "heatmap_assets"
BACKGROUND_DIR = ASSETS_DIR / "backgrounds"
ZONES_FILE = BASE_DIR / "zones.json"
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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

    return {
        "positions": positions,
        "background": {
            "filename": background.get("filename"),
            "original_name": background.get("original_name"),
            "updated_at": background.get("updated_at"),
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

    @app.route("/api/heatmap")
    def api_heatmap():
        return jsonify(build_heatmap_points())

    @app.route("/api/positions", methods=["POST"])
    def api_positions():
        payload = request.get_json(force=True)
        positions = payload.get("positions", {})
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
        save_positions_config(config)
        return jsonify({"ok": True, "positions_saved": len(cleaned)})

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
