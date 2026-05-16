import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

ZONES_FILE = Path("zones.json")


def create_app(ctx):
    app = Flask(__name__)

    # ── MJPEG-стрим ──────────────────────────────────────────────────────────

    def _gen_frames():
        blank = None
        while True:
            try:
                frame = ctx._frame_queue.get(timeout=2.0)
            except queue.Empty:
                if blank is None:
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for stream...",
                                (160, 180), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (55, 55, 55), 2)
                frame = blank
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

    # ── Страницы ──────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        videos = []
        if ctx._source_folder:
            videos = sorted(
                p.name for p in Path(ctx._source_folder).iterdir()
                if p.suffix.lower() in ctx.VIDEO_EXTS
            )
        with ctx.state["lock"]:
            sz = ctx.state["model_size"]
            conf = ctx.state["conf"]
            imgsz = ctx.state["imgsz"]
            fpm = ctx.state["fpm"]
        return render_template(
            "index.html",
            model=sz.upper(),
            model_size=sz,
            conf=conf,
            imgsz=imgsz,
            fpm=fpm,
            source_url=ctx._source_url or "",
            source_label=(ctx._source_url or ctx._source_folder or ""),
            videos=videos,
            current_video=ctx._current_video or "",
        )

    @app.route("/video_feed")
    def video_feed():
        return Response(_gen_frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── Статистика ────────────────────────────────────────────────────────────

    @app.route("/stats")
    def stats_route():
        with ctx.state["lock"]:
            return jsonify({
                "persons": ctx.state["persons"],
                "fps": ctx.state["fps"],
                "dfps": ctx.state["dfps"],
                "ms": ctx.state["ms"],
                "frames": ctx.state["frames"],
                "frame_skip": ctx.state["frame_skip"],
                "source_type": ctx.state["source_type"],
                "ts": ctx.state["ts"],
                "model_size": ctx.state["model_size"],
                "model_loading": ctx.state["model_loading"],
                "violations": ctx.state.get("violations", 0),
                "camera_id": ctx.state.get("camera_id"),
                "adults": ctx.state.get("adults", 0),
                "children": ctx.state.get("children", 0),
                "age_calibrated": ctx.state.get("age_calibrated", False),
            })

    # ── Параметры детекции ────────────────────────────────────────────────────

    @app.route("/set_params", methods=["POST"])
    def set_params():
        data = request.get_json(force=True)
        with ctx.state["lock"]:
            if "conf" in data:
                ctx.state["conf"] = float(data["conf"])
            if "imgsz" in data:
                ctx.state["imgsz"] = int(data["imgsz"])
            if "fpm" in data:
                ctx.state["fpm"] = max(1, int(data["fpm"]))
        return jsonify({"ok": True})

    # ── Переключение модели ───────────────────────────────────────────────────

    @app.route("/set_model", methods=["POST"])
    def set_model_route():
        data = request.get_json(force=True)
        size = data.get("model", "m").strip().lower()
        if size not in {"n", "s", "m", "l", "x"}:
            return jsonify({"ok": False, "error": "invalid"}), 400
        with ctx.state["lock"]:
            if ctx.state["model_loading"] is not None:
                return jsonify({"ok": False, "error": "already loading"}), 409
            if ctx.state["model_size"] == size:
                return jsonify({"ok": True, "note": "already loaded"})
            ctx.state["model_loading"] = size
        threading.Thread(target=ctx._do_reload_model, args=(size,),
                         daemon=True, name=f"reload-{size}").start()
        return jsonify({"ok": True})

    # ── Переключение источника ────────────────────────────────────────────────

    @app.route("/set_source", methods=["POST"])
    def set_source():
        data = request.get_json(force=True)
        if "url" in data:
            ctx._source_url = data["url"].strip()
        if "video" in data:
            ctx._current_video = data["video"]
        ctx._restart_event.set()
        return jsonify({"ok": True})

    # ── Зоны разметки ─────────────────────────────────────────────────────────

    @app.route("/zones", methods=["GET"])
    def zones_get():
        if ZONES_FILE.exists():
            return ZONES_FILE.read_text(encoding="utf-8"), 200, {
                "Content-Type": "application/json; charset=utf-8"
            }
        return jsonify({"cameras": {}})

    @app.route("/zones", methods=["POST"])
    def zones_save():
        data = request.get_json(force=True)
        ZONES_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True})

    # ── Список камер ──────────────────────────────────────────────────────────

    @app.route("/cameras", methods=["GET"])
    def cameras_route():
        from zone_manager import ZoneManager
        cameras = ZoneManager.get_cameras_from_file(ZONES_FILE)

        # Дополняем информацией о наличии калибровки возраста
        from pathlib import Path as P
        calib_dir = P("calibrations")
        for cid, info in cameras.items():
            calib_path = calib_dir / f"{cid}.json"
            info["age_calibrated"] = calib_path.exists()

        return jsonify({"cameras": cameras})

    # ── Переключение активной камеры ──────────────────────────────────────────

    @app.route("/set_camera", methods=["POST"])
    def set_camera_route():
        data = request.get_json(force=True)
        camera_id = data.get("camera_id", "").strip()
        ctx.set_camera(camera_id)
        return jsonify({"ok": True, "camera_id": camera_id or None})

    # ── Состояния светофоров ──────────────────────────────────────────────────

    @app.route("/tl_states", methods=["GET"])
    def tl_states_route():
        analyzer = getattr(ctx, "tl_analyzer", None)
        if analyzer is None:
            return jsonify({})
        return jsonify(analyzer.get_all_states())

    # ── Включение/выключение детекции ─────────────────────────────────────────

    @app.route("/toggle_detect", methods=["POST"])
    def toggle_detect():
        data = request.get_json()
        enabled = data.get("enabled", True)
        with ctx.state["lock"]:
            ctx.state["detect_enabled"] = enabled
        return jsonify({"ok": True, "enabled": enabled})

    # ── Статус калибровки возраста для текущей камеры ────────────────────────

    @app.route("/age_calibration_status", methods=["GET"])
    def age_calibration_status():
        """Возвращает статус калибровки и список доступных видео для выбора."""
        with ctx.state["lock"]:
            camera_id = ctx.state.get("camera_id") or ""
            age_calibrated = ctx.state.get("age_calibrated", False)

        from pathlib import Path as P
        calib_dir = P("calibrations")
        calib_file = calib_dir / f"{camera_id}.json" if camera_id else None
        calib_info = {}

        if calib_file and calib_file.exists():
            try:
                data = json.loads(calib_file.read_text(encoding="utf-8"))
                refs = data.get("refs", {})
                samples = data.get("samples_count", {})
                calib_info = {
                    "ready_bands": len(refs),
                    "total_bands": 10,
                    "percent": int(len(refs) / 10 * 100),
                    "total_samples": sum(samples.values()),
                }
            except Exception:
                pass

        # Видео доступные для калибровки
        available_videos = []
        if ctx._source_folder:
            folder = P(ctx._source_folder)
            if folder.is_dir():
                available_videos = sorted(
                    p.name for p in folder.iterdir()
                    if p.suffix.lower() in ctx.VIDEO_EXTS
                )

        return jsonify({
            "camera_id": camera_id,
            "calibrated": age_calibrated,
            "calib_info": calib_info,
            "available_videos": available_videos,
        })

    return app