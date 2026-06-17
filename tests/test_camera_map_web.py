import sys
from pathlib import Path
import json


sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_map_web import (
    load_json,
    save_json,
    load_map_config,
    save_map_config,
    load_zones_db,
    save_zones_db,
    discover_camera_ids,
    resolve_video_folder,
    list_videos,
    first_video_for_camera,
    camera_summary,
    _clean_roi,
    VIDEO_EXTS,
)


class TestCameraMapWeb:

    def test_load_json_existing(self, tmp_path):
        """Загрузка существующего JSON"""
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        data = load_json(file_path, default={})
        assert data["key"] == "value"

    def test_load_json_not_existing(self, tmp_path):
        """Загрузка несуществующего JSON"""
        file_path = tmp_path / "none.json"
        data = load_json(file_path, default={"default": True})
        assert data["default"] is True

    def test_save_json(self, tmp_path):
        """Сохранение JSON"""
        file_path = tmp_path / "test.json"
        payload = {"key": "value", "number": 42}

        save_json(file_path, payload)
        assert file_path.exists()

        data = json.loads(file_path.read_text(encoding="utf-8"))
        assert data["key"] == "value"
        assert data["number"] == 42

    def test_load_map_config_default(self, tmp_path):
        """Загрузка конфигурации карты по умолчанию"""
        import camera_map_web
        original = camera_map_web.MAP_CONFIG_FILE
        camera_map_web.MAP_CONFIG_FILE = tmp_path / "none.json"

        try:
            config = load_map_config()
            assert "map" in config
            assert "cameras" in config
            assert config["map"]["center"] == [55.751244, 37.618423]
            assert config["map"]["zoom"] == 11
        finally:
            camera_map_web.MAP_CONFIG_FILE = original

    def test_save_map_config(self, tmp_path):
        """Сохранение конфигурации карты"""
        import camera_map_web
        original = camera_map_web.MAP_CONFIG_FILE
        camera_map_web.MAP_CONFIG_FILE = tmp_path / "map.json"

        try:
            config = {
                "map": {"center": [55.0, 37.0], "zoom": 10},
                "cameras": {"cam1": {"label": "Camera 1", "lat": 55.0, "lon": 37.0}},
            }
            save_map_config(config)
            assert camera_map_web.MAP_CONFIG_FILE.exists()

            data = json.loads(camera_map_web.MAP_CONFIG_FILE.read_text(encoding="utf-8"))
            assert data["map"]["zoom"] == 10
        finally:
            camera_map_web.MAP_CONFIG_FILE = original

    def test_load_zones_db(self, tmp_path):
        """Загрузка базы зон"""
        import camera_map_web
        original = camera_map_web.ZONES_FILE
        camera_map_web.ZONES_FILE = tmp_path / "zones.json"

        zones_data = {"cameras": {"test": {"zones": []}}}
        camera_map_web.ZONES_FILE.write_text(json.dumps(zones_data), encoding="utf-8")

        try:
            data = load_zones_db()
            assert "cameras" in data
            assert "test" in data["cameras"]
        finally:
            camera_map_web.ZONES_FILE = original

    def test_load_zones_db_default(self, tmp_path):
        """Загрузка базы зон по умолчанию"""
        import camera_map_web
        original = camera_map_web.ZONES_FILE
        camera_map_web.ZONES_FILE = tmp_path / "none.json"

        try:
            data = load_zones_db()
            assert data == {"cameras": {}}
        finally:
            camera_map_web.ZONES_FILE = original

    def test_save_zones_db(self, tmp_path):
        """Сохранение базы зон"""
        import camera_map_web
        original = camera_map_web.ZONES_FILE
        camera_map_web.ZONES_FILE = tmp_path / "zones.json"

        try:
            data = {"cameras": {"new": {"zones": []}}}
            save_zones_db(data)
            assert camera_map_web.ZONES_FILE.exists()

            loaded = json.loads(camera_map_web.ZONES_FILE.read_text(encoding="utf-8"))
            assert "new" in loaded["cameras"]
        finally:
            camera_map_web.ZONES_FILE = original

    def test_discover_camera_ids(self, tmp_path):
        """Поиск ID камер"""
        import camera_map_web
        original_video = camera_map_web.VIDEO_ROOT
        original_zones = camera_map_web.ZONES_FILE
        original_map = camera_map_web.MAP_CONFIG_FILE

        video_root = tmp_path / "video"
        video_root.mkdir()

        (video_root / "cam1").mkdir()
        (video_root / "cam2").mkdir()

        camera_map_web.VIDEO_ROOT = video_root
        camera_map_web.ZONES_FILE = tmp_path / "zones.json"
        camera_map_web.MAP_CONFIG_FILE = tmp_path / "map.json"

        zones_data = {"cameras": {"cam3": {"zones": []}}}
        camera_map_web.ZONES_FILE.write_text(json.dumps(zones_data), encoding="utf-8")

        try:
            ids = discover_camera_ids()
            assert "cam1" in ids
            assert "cam2" in ids
            assert "cam3" in ids
        finally:
            camera_map_web.VIDEO_ROOT = original_video
            camera_map_web.ZONES_FILE = original_zones
            camera_map_web.MAP_CONFIG_FILE = original_map

    def test_resolve_video_folder_with_default(self, tmp_path):
        """Поиск папки с видео по умолчанию"""
        import camera_map_web
        original = camera_map_web.VIDEO_ROOT
        camera_map_web.VIDEO_ROOT = tmp_path / "video"

        video_folder = tmp_path / "video" / "cam1"
        video_folder.mkdir(parents=True)

        try:
            config = {"cameras": {}}
            folder = resolve_video_folder("cam1", config)
            assert folder == video_folder
        finally:
            camera_map_web.VIDEO_ROOT = original

    def test_resolve_video_folder_with_config(self, tmp_path):
        """Поиск папки с видео из конфигурации"""
        import camera_map_web
        original = camera_map_web.VIDEO_ROOT
        camera_map_web.VIDEO_ROOT = tmp_path / "video"

        config_folder = tmp_path / "custom_video" / "cam1"
        config_folder.mkdir(parents=True)

        config = {
            "cameras": {
                "cam1": {"video_folder": str(config_folder)}
            }
        }

        try:
            folder = resolve_video_folder("cam1", config)
            assert folder == config_folder
        finally:
            camera_map_web.VIDEO_ROOT = original

    def test_list_videos(self, tmp_path):
        """Список видеофайлов"""
        folder = tmp_path / "videos"
        folder.mkdir()

        for ext in [".mp4", ".avi", ".mkv", ".txt"]:
            (folder / f"video{ext}").touch()

        videos = list_videos(folder)
        assert len(videos) == 3
        assert all(v.suffix in VIDEO_EXTS for v in videos)

    def test_list_videos_empty(self, tmp_path):
        """Пустая папка с видео"""
        videos = list_videos(tmp_path)
        assert videos == []

    def test_first_video_for_camera(self, tmp_path):
        """Поиск первого видео для камеры"""
        import camera_map_web
        original = camera_map_web.VIDEO_ROOT
        camera_map_web.VIDEO_ROOT = tmp_path / "video"

        video_folder = tmp_path / "video" / "cam1"
        video_folder.mkdir(parents=True)

        video_file = video_folder / "test.mp4"
        video_file.touch()

        try:
            config = {"cameras": {}}
            video = first_video_for_camera("cam1", config)
            assert video == video_file
        finally:
            camera_map_web.VIDEO_ROOT = original

    def test_first_video_for_camera_not_found(self, tmp_path):
        """Поиск первого видео для камеры (нет видео)"""
        import camera_map_web
        original = camera_map_web.VIDEO_ROOT
        camera_map_web.VIDEO_ROOT = tmp_path / "video"

        video_folder = tmp_path / "video" / "cam1"
        video_folder.mkdir(parents=True)

        try:
            config = {"cameras": {}}
            video = first_video_for_camera("cam1", config)
            assert video is None
        finally:
            camera_map_web.VIDEO_ROOT = original

    def test_camera_summary(self, tmp_path):
        """Краткая информация о камере"""
        import camera_map_web
        original = camera_map_web.VIDEO_ROOT
        camera_map_web.VIDEO_ROOT = tmp_path / "video"

        video_folder = tmp_path / "video" / "cam1"
        video_folder.mkdir(parents=True)
        (video_folder / "test.mp4").touch()

        config = {
            "cameras": {
                "cam1": {"label": "Camera 1", "lat": 55.0, "lon": 37.0}
            }
        }
        zones_db = {"cameras": {"cam1": {"zones": [{"id": "z1"}]}}}

        try:
            summary = camera_summary("cam1", config, zones_db)
            assert summary["camera_id"] == "cam1"
            assert summary["label"] == "Camera 1"
            assert summary["lat"] == 55.0
            assert summary["lon"] == 37.0
            assert summary["has_video"] is True
            assert summary["zones_count"] == 1
        finally:
            camera_map_web.VIDEO_ROOT = original

    def test_clean_roi_none(self):
        """Очистка ROI: None"""
        result = _clean_roi(None)
        assert result is None

    def test_clean_roi_empty_list(self):
        """Очистка ROI: пустой список"""
        result = _clean_roi([])
        assert result is None

    def test_clean_roi_single(self):
        """Очистка ROI: одиночный ROI"""
        result = _clean_roi([0.1, 0.2, 0.3, 0.4])
        assert result == [[0.1, 0.2, 0.3, 0.4]]

    def test_clean_roi_multiple(self):
        """Очистка ROI: несколько ROI"""
        rois = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        result = _clean_roi(rois)
        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[1]) == 4

    def test_clean_roi_invalid(self):
        """Очистка ROI: некорректные значения"""
        result = _clean_roi([1.5, 1.5, 1.5, 1.5])  # > 1.0
        assert result is None

    def test_video_extensions(self):
        """Проверка поддерживаемых расширений"""
        assert ".mp4" in VIDEO_EXTS
        assert ".avi" in VIDEO_EXTS
        assert ".mov" in VIDEO_EXTS
        assert ".mkv" in VIDEO_EXTS
