import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from datetime import datetime, timedelta
from collections import defaultdict
from flask import send_file
from report_generator import get_report_generator, REPORTS_DIR, get_collector

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
            sz    = ctx.state["model_size"]
            conf  = ctx.state["conf"]
            imgsz = ctx.state["imgsz"]
            fpm   = ctx.state["fpm"]
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
                "persons":        ctx.state["persons"],
                "fps":            ctx.state["fps"],
                "dfps":           ctx.state["dfps"],
                "ms":             ctx.state["ms"],
                "frames":         ctx.state["frames"],
                "frame_skip":     ctx.state["frame_skip"],
                "source_type":    ctx.state["source_type"],
                "ts":             ctx.state["ts"],
                "model_size":     ctx.state["model_size"],
                "model_loading":  ctx.state["model_loading"],
                "violations":     ctx.state.get("violations", 0),
                "camera_id":      ctx.state.get("camera_id"),
                "adults":         ctx.state.get("adults", 0),
                "children":       ctx.state.get("children", 0),
                "age_calibrated": ctx.state.get("age_calibrated", False),
            })

    # ── Параметры детекции ────────────────────────────────────────────────────

    @app.route("/set_params", methods=["POST"])
    def set_params():
        data = request.get_json(force=True)
        with ctx.state["lock"]:
            if "conf"  in data: ctx.state["conf"]  = float(data["conf"])
            if "imgsz" in data: ctx.state["imgsz"] = int(data["imgsz"])
            if "fpm"   in data: ctx.state["fpm"]   = max(1, int(data["fpm"]))
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
        if "url"   in data: ctx._source_url    = data["url"].strip()
        if "video" in data: ctx._current_video = data["video"]
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
        data      = request.get_json(force=True)
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
        data    = request.get_json()
        enabled = data.get("enabled", True)
        with ctx.state["lock"]:
            ctx.state["detect_enabled"] = enabled
        return jsonify({"ok": True, "enabled": enabled})

    # ── Статус калибровки возраста для текущей камеры ────────────────────────

    @app.route("/age_calibration_status", methods=["GET"])
    def age_calibration_status():
        """Возвращает статус калибровки и список доступных видео для выбора."""
        with ctx.state["lock"]:
            camera_id     = ctx.state.get("camera_id") or ""
            age_calibrated = ctx.state.get("age_calibrated", False)

        from pathlib import Path as P
        calib_dir  = P("calibrations")
        calib_file = calib_dir / f"{camera_id}.json" if camera_id else None
        calib_info = {}

        if calib_file and calib_file.exists():
            try:
                data = json.loads(calib_file.read_text(encoding="utf-8"))
                refs = data.get("refs", {})
                samples = data.get("samples_count", {})
                calib_info = {
                    "ready_bands":   len(refs),
                    "total_bands":   10,
                    "percent":       int(len(refs) / 10 * 100),
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
            "camera_id":     camera_id,
            "calibrated":    age_calibrated,
            "calib_info":    calib_info,
            "available_videos": available_videos,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # API для работы с отчетами о нарушениях
    # ──────────────────────────────────────────────────────────────────────────

    @app.route("/reports/generate", methods=["POST"])
    def generate_report():
        """
        Сгенерировать отчет о нарушениях.
        
        JSON body:
            {"interval": 30, "format": "json"}
            или
            {"intervals": [30, 60, 90], "format": "txt"}
        """
        from datetime import datetime
        from report_generator import get_report_generator
        
        data = request.get_json(force=True)
        
        camera_id = ""
        with ctx.state["lock"]:
            camera_id = ctx.state.get("camera_id") or ""
        
        generator = get_report_generator(camera_id)
        
        # Одиночный интервал
        if "interval" in data:
            minutes = int(data["interval"])
            fmt = data.get("format", "json")
            path, stats = generator.generate(duration_minutes=minutes, output_format=fmt)
            return jsonify({"ok": True, "file": str(path), "filename": path.name, "stats": stats})
        
        # Несколько интервалов
        if "intervals" in data:
            intervals = [int(i) for i in data["intervals"]]
            fmt = data.get("format", "json")
            results = generator.generate_interval_report(intervals=intervals, output_format=fmt)
            return jsonify({
                "ok": True, 
                "files": [{"path": str(p), "filename": p.name, "stats": s} for p, s in results]
            })
        
        return jsonify({"ok": False, "error": "No interval specified"}), 400

    # --------------------------------------------------------------------------

    @app.route("/reports/list", methods=["GET"])
    def list_reports():
        """Получить список всех сгенерированных отчетов."""
        from datetime import datetime
        from report_generator import REPORTS_DIR
        
        reports = []
        for ext in ["json", "txt", "csv"]:
            for path in REPORTS_DIR.glob(f"*.{ext}"):
                stat = path.stat()
                reports.append({
                    "filename": path.name,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                })
        
        reports.sort(key=lambda x: x["created"], reverse=True)
        return jsonify({"reports": reports})

    # --------------------------------------------------------------------------

    @app.route("/reports/download/<filename>", methods=["GET"])
    def download_report(filename: str):
        """Скачать файл отчета."""
        from flask import send_file
        from report_generator import REPORTS_DIR
        
       
        if ".." in filename or "/" in filename or "\\" in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = REPORTS_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "File not found"}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)

    # --------------------------------------------------------------------------

    @app.route("/reports/delete/<filename>", methods=["DELETE"])
    def delete_report(filename: str):
        """Удалить файл отчета."""
        from report_generator import REPORTS_DIR
        
        if ".." in filename or "/" in filename or "\\" in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = REPORTS_DIR / filename
        if filepath.exists():
            filepath.unlink()
            return jsonify({"ok": True, "deleted": filename})
        
        return jsonify({"error": "File not found"}), 404

    # --------------------------------------------------------------------------

    @app.route("/reports/stats/current", methods=["GET"])
    def current_stats():
        """Получить текущую статистику по нарушениям (последние 5 минут)."""
        from datetime import datetime, timedelta
        from collections import defaultdict
        from report_generator import get_collector
        
        collector = get_collector()
        camera_id = ""
        
        with ctx.state["lock"]:
            camera_id = ctx.state.get("camera_id") or ""
        
        now = datetime.now()
        recent_events = collector.get_events(start_time=now - timedelta(minutes=5))
        
        track_groups = {}
        for event in recent_events:
            if event.track_id not in track_groups:
                track_groups[event.track_id] = []
            track_groups[event.track_id].append(event)
        
        total_tracks = len(track_groups)
        by_type = defaultdict(int)
        by_age = defaultdict(int)
        
        for events in track_groups.values():
            first = events[0]
            by_type[first.violation_type] += 1
         
            if hasattr(first, 'age_label'):
                by_age[first.age_label] += 1
        
        return jsonify({
            "camera_id": camera_id,
            "period_seconds": 300,
            "total_violations": total_tracks,
            "total_events": len(recent_events),
            "by_type": dict(by_type),
            "by_age": {
                "adults": by_age.get("adult", 0),
                "children": by_age.get("child", 0),
            },
            "timestamp": datetime.now().isoformat(),
        })
    
    return app
