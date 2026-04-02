import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request


def create_app(ctx):
    app = Flask(__name__)

    def _gen_frames():
        blank = None
        while True:
            try:
                frame = ctx._frame_queue.get(timeout=2.0)
            except queue.Empty:
                if blank is None:
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Ожидание потока...",
                        (160, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (55, 55, 55),
                        2,
                    )
                frame = blank
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ret:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

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
        return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats_route():
        with ctx.state["lock"]:
            return jsonify(
                {
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
                }
            )

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
        threading.Thread(
            target=ctx._do_reload_model,
            args=(size,),
            daemon=True,
            name=f"reload-{size}",
        ).start()
        return jsonify({"ok": True})

    @app.route("/set_source", methods=["POST"])
    def set_source():
        data = request.get_json(force=True)
        if "url" in data:
            ctx._source_url = data["url"].strip()
        if "video" in data:
            ctx._current_video = data["video"]
        ctx._restart_event.set()
        return jsonify({"ok": True})

    return app
