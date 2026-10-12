# tests/test_stream_detect_web.py
import sys
from pathlib import Path
import json
from unittest.mock import MagicMock
import queue


sys.path.insert(0, str(Path(__file__).parent.parent))

from stream_detect_web import create_app


class TestStreamDetectWeb:
    """Тесты для stream_detect_web.py"""

    def test_create_app(self):
        """Создание Flask-приложения"""
        ctx = MagicMock()
        ctx.state = {"lock": MagicMock()}
        app = create_app(ctx)
        assert app is not None
        assert app.name == "stream_detect_web"

    def test_index_route(self):
        """Тест главной страницы"""
        ctx = MagicMock()
        ctx._source_folder = None
        ctx.VIDEO_EXTS = {".mp4", ".avi"}
        ctx.state = {"lock": MagicMock(), "model_size": "m", "conf": 0.45, "imgsz": 640, "fpm": 60}

        app = create_app(ctx)
        client = app.test_client()

        response = client.get("/")
        assert response.status_code == 200

    def test_video_feed_route(self):
        """Тест видео-потока"""
        ctx = MagicMock()
        ctx._frame_queue = MagicMock()
        ctx._frame_queue.get = MagicMock(side_effect=queue.Empty())
        ctx._frame_queue.empty = MagicMock(return_value=True)

        app = create_app(ctx)
        client = app.test_client()

        response = client.get("/video_feed")
        assert response.status_code == 200
        assert response.content_type == "multipart/x-mixed-replace; boundary=frame"

    def test_stats_route(self):
        """Тест статистики"""
        ctx = MagicMock()
        ctx.state = {
            "lock": MagicMock(),
            "persons": 5,
            "fps": 25.0,
            "dfps": 10.0,
            "ms": 15.0,
            "frames": 1000,
            "frame_skip": 2,
            "source_type": "test",
            "ts": 1234567890,
            "model_size": "m",
            "model_loading": None,
            "violations": 3,
            "camera_id": "cam1",
            "adults": 4,
            "children": 1,
            "age_calibrated": True,
        }

        app = create_app(ctx)
        client = app.test_client()

        response = client.get("/stats")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["persons"] == 5
        assert data["fps"] == 25.0
        assert data["violations"] == 3

    def test_set_params_route(self):
        """Тест изменения параметров"""
        ctx = MagicMock()
        ctx.state = {"lock": MagicMock(), "conf": 0.45, "imgsz": 640, "fpm": 60}

        app = create_app(ctx)
        client = app.test_client()

        response = client.post(
            "/set_params",
            json={"conf": 0.5, "imgsz": 800, "fpm": 30},
            content_type="application/json",
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["ok"] is True

    def test_set_model_route(self):
        """Тест переключения модели"""
        ctx = MagicMock()
        ctx.state = {"lock": MagicMock(), "model_size": "m", "model_loading": None}
        ctx._do_reload_model = MagicMock()

        app = create_app(ctx)
        client = app.test_client()

        response = client.post(
            "/set_model",
            json={"model": "l"},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_set_model_invalid(self):
        """Тест переключения модели (неверная модель)"""
        ctx = MagicMock()
        ctx.state = {"lock": MagicMock(), "model_size": "m", "model_loading": None}

        app = create_app(ctx)
        client = app.test_client()

        response = client.post(
            "/set_model",
            json={"model": "invalid"},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_set_source_route(self):
        """Тест переключения источника"""
        ctx = MagicMock()
        ctx._source_url = None
        ctx._restart_event = MagicMock()

        app = create_app(ctx)
        client = app.test_client()

        response = client.post(
            "/set_source",
            json={"url": "https://example.com/stream"},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_zones_get_route(self, tmp_path):
        """Тест получения зон"""
        import stream_detect_web
        original = stream_detect_web.ZONES_FILE
        stream_detect_web.ZONES_FILE = tmp_path / "zones.json"

        zones_data = {"cameras": {"test": {"zones": []}}}
        stream_detect_web.ZONES_FILE.write_text(json.dumps(zones_data), encoding="utf-8")

        ctx = MagicMock()
        ctx.state = {"lock": MagicMock()}

        try:
            app = create_app(ctx)
            client = app.test_client()

            response = client.get("/zones")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert "cameras" in data
        finally:
            stream_detect_web.ZONES_FILE = original

    def test_zones_save_route(self, tmp_path):
        """Тест сохранения зон"""
        import stream_detect_web
        original = stream_detect_web.ZONES_FILE
        stream_detect_web.ZONES_FILE = tmp_path / "zones.json"

        ctx = MagicMock()
        ctx.state = {"lock": MagicMock()}

        try:
            app = create_app(ctx)
            client = app.test_client()

            zones_data = {"cameras": {"new": {"zones": []}}}
            response = client.post(
                "/zones",
                json=zones_data,
                content_type="application/json",
            )
            assert response.status_code == 200

            assert stream_detect_web.ZONES_FILE.exists()
            data = json.loads(stream_detect_web.ZONES_FILE.read_text(encoding="utf-8"))
            assert "new" in data["cameras"]
        finally:
            stream_detect_web.ZONES_FILE = original

    def test_cameras_route(self, tmp_path):
        """Тест списка камер"""
        import stream_detect_web
        original_zones = stream_detect_web.ZONES_FILE
        stream_detect_web.ZONES_FILE = tmp_path / "zones.json"

        zones_data = {
            "cameras": {
                "cam1": {"label": "Camera 1", "zones": []},
                "cam2": {"label": "Camera 2", "zones": [{"id": "z1"}]},
            }
        }
        stream_detect_web.ZONES_FILE.write_text(json.dumps(zones_data), encoding="utf-8")

        ctx = MagicMock()
        ctx.state = {"lock": MagicMock()}

        try:
            app = create_app(ctx)
            client = app.test_client()

            response = client.get("/cameras")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert "cameras" in data
            assert "cam1" in data["cameras"]
            assert "cam2" in data["cameras"]
        finally:
            stream_detect_web.ZONES_FILE = original_zones

    def test_set_camera_route(self):
        """Тест переключения камеры"""
        ctx = MagicMock()
        ctx.set_camera = MagicMock()
        ctx.state = {"lock": MagicMock()}

        app = create_app(ctx)
        client = app.test_client()

        response = client.post(
            "/set_camera",
            json={"camera_id": "cam1"},
            content_type="application/json",
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["ok"] is True

    def test_tl_states_route(self):
        """Тест состояний светофоров"""
        ctx = MagicMock()
        ctx.tl_analyzer = MagicMock()
        ctx.tl_analyzer.get_all_states = MagicMock(return_value={"test": {"state": "green"}})
        ctx.state = {"lock": MagicMock()}

        app = create_app(ctx)
        client = app.test_client()

        response = client.get("/tl_states")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["test"]["state"] == "green"

    def test_toggle_detect_route(self):
        """Тест включения/выключения детекции"""
        ctx = MagicMock()
        ctx.state = {"lock": MagicMock(), "detect_enabled": True}

        app = create_app(ctx)
        client = app.test_client()

        response = client.post(
            "/toggle_detect",
            json={"enabled": False},
            content_type="application/json",
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["ok"] is True
        assert data["enabled"] is False

    def test_age_calibration_status_route(self):
        """Тест статуса калибровки возраста"""
        ctx = MagicMock()
        ctx.state = {"lock": MagicMock(), "camera_id": "cam1", "age_calibrated": True}
        ctx._source_folder = None

        app = create_app(ctx)
        client = app.test_client()

        response = client.get("/age_calibration_status")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["camera_id"] == "cam1"
        assert data["calibrated"] is True