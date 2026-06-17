import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_folder_offline import (
    list_video_files,
    collect_video_folders,
    load_zones_db,
    ensure_camera_entry,
    camera_has_zones,
    build_report_data,
    summarize_processing_stats,
    write_json,
    make_violation_logger,
    merge_combined_reports,
    load_json_if_exists,
    COOLDOWN_SECONDS,
)


class TestBatchFolderOffline:

    def test_list_video_files(self, tmp_path):
        """Список видеофайлов в папке"""
        for ext in [".mp4", ".avi", ".mkv", ".txt"]:
            (tmp_path / f"test{ext}").touch()

        videos = list_video_files(tmp_path)
        assert len(videos) == 3
        assert all(p.suffix in [".mp4", ".avi", ".mkv"] for p in videos)

    def test_list_video_files_empty(self, tmp_path):
        """Пустая папка"""
        videos = list_video_files(tmp_path)
        assert videos == []

    def test_collect_video_folders(self, tmp_path):
        """Сбор папок с видео"""
        folder1 = tmp_path / "cam1"
        folder2 = tmp_path / "cam2"
        folder3 = tmp_path / "empty"

        folder1.mkdir()
        folder2.mkdir()
        folder3.mkdir()

        (folder1 / "video.mp4").touch()
        (folder2 / "video.avi").touch()

        class Args:
            folders = None
            root = str(tmp_path)

        args = Args()
        folders = collect_video_folders(args)

        assert len(folders) == 2
        assert folder1 in folders
        assert folder2 in folders
        assert folder3 not in folders

    def test_load_zones_db(self, tmp_path):
        """Загрузка базы зон"""
        zones_file = tmp_path / "zones.json"
        zones_file.write_text(
            json.dumps({"cameras": {"test": {"zones": []}}}),
            encoding="utf-8"
        )

        import batch_folder_offline
        original = batch_folder_offline.ZONES_FILE
        batch_folder_offline.ZONES_FILE = zones_file

        try:
            data = load_zones_db()
            assert "cameras" in data
            assert "test" in data["cameras"]
        finally:
            batch_folder_offline.ZONES_FILE = original

    def test_ensure_camera_entry(self, tmp_path):
        """Создание записи камеры"""
        zones_file = tmp_path / "zones.json"
        zones_file.write_text(json.dumps({"cameras": {}}), encoding="utf-8")

        import batch_folder_offline
        original = batch_folder_offline.ZONES_FILE
        batch_folder_offline.ZONES_FILE = zones_file

        try:
            ensure_camera_entry("new_camera")
            data = json.loads(zones_file.read_text(encoding="utf-8"))
            assert "new_camera" in data["cameras"]
        finally:
            batch_folder_offline.ZONES_FILE = original

    def test_camera_has_zones(self, tmp_path):
        """Проверка наличия зон у камеры"""
        zones_file = tmp_path / "zones.json"
        zones_file.write_text(
            json.dumps({
                "cameras": {
                    "cam1": {"zones": [{"id": "z1"}]},
                    "cam2": {"zones": []},
                }
            }),
            encoding="utf-8"
        )

        import batch_folder_offline
        original = batch_folder_offline.ZONES_FILE
        batch_folder_offline.ZONES_FILE = zones_file

        try:
            assert camera_has_zones("cam1") is True
            assert camera_has_zones("cam2") is False
            assert camera_has_zones("cam3") is False
        finally:
            batch_folder_offline.ZONES_FILE = original

    def test_build_report_data(self):
        """Построение отчёта из нарушений"""
        now = datetime.now()

        violations = [
            {
                "timestamp": now.isoformat(),
                "track_id": 1,
                "camera_id": "cam1",
                "violation_type": "red_light",
                "zone_label": "crosswalk",
                "age_label": "adult",
                "person_conf": 0.85,
            },
            {
                "timestamp": (now - timedelta(minutes=5)).isoformat(),
                "track_id": 2,
                "camera_id": "cam1",
                "violation_type": "road_trespass",
                "zone_label": "road",
                "age_label": "child",
                "person_conf": 0.75,
            },
            {
                "timestamp": (now - timedelta(minutes=10)).isoformat(),
                "track_id": 1,
                "camera_id": "cam1",
                "violation_type": "red_light",
                "zone_label": "crosswalk",
                "age_label": "adult",
                "person_conf": 0.90,
            },
        ]

        report = build_report_data(violations, "cam1")

        assert report["camera_id"] == "cam1"
        assert report["total_violations"] == 3
        assert report["unique_persons"] == 2
        assert report["summary"]["by_type"]["red_light"] == 2
        assert report["summary"]["by_type"]["road_trespass"] == 1
        assert report["confidence"]["average"] == pytest.approx(0.833, 0.01)
        assert report["confidence"]["min"] == 0.75
        assert report["confidence"]["max"] == 0.90

    def test_summarize_processing_stats(self):
        """Суммирование статистики обработки"""
        items = [
            {
                "total_frames": 100,
                "detected_frames": 50,
                "total_persons": 10,
                "total_violations": 3,
                "total_adults": 8,
                "total_children": 2,
                "inference_ms_sum": 1000.0,
                "elapsed_sec": 10.0,
            },
            {
                "total_frames": 200,
                "detected_frames": 100,
                "total_persons": 20,
                "total_violations": 5,
                "total_adults": 15,
                "total_children": 5,
                "inference_ms_sum": 2000.0,
                "elapsed_sec": 20.0,
            },
        ]

        summary = summarize_processing_stats(items)
        assert summary["total_frames"] == 300
        assert summary["total_violations"] == 8
        assert summary["total_persons"] == 30
        assert summary["inference_ms_sum"] == 3000.0

    def test_write_json(self, tmp_path):
        """Запись JSON файла"""
        output = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        write_json(output, data)
        assert output.exists()

        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_make_violation_logger(self):
        """Создание логгера нарушений"""
        sink = []
        logger = make_violation_logger("cam1", 1000.0, sink)

        class MockViolation:
            def __init__(self, violation, track_id, zone_label, note, conf, age_label, age_conf):
                self.violation = violation
                self.track_id = track_id
                self.zone_label = zone_label
                self.note = note
                self.conf = conf
                self.age_label = age_label
                self.age_conf = age_conf

        violations = [
            MockViolation("red_light", 1, "crosswalk", "", 0.85, "adult", 0.90),
            MockViolation("none", 2, "", "", 0.80, "adult", 0.85),
        ]

        logger(violations, 5.0)
        assert len(sink) == 1
        assert sink[0]["track_id"] == 1
        assert sink[0]["violation_type"] == "red_light"

    def test_merge_combined_reports(self):
        """Объединение отчётов"""
        existing = {
            "raw_violations": [
                {"track_id": 1, "violation_type": "red_light"},
            ],
            "folders": {"cam1": {"total_violations": 1}},
            "videos": [{"name": "video1.mp4"}],
            "processing_stats": {"total_frames": 100},
        }

        new_data = {
            "raw_violations": [
                {"track_id": 2, "violation_type": "road_trespass"},
            ],
            "folders": {"cam2": {"total_violations": 2}},
            "videos": [{"name": "video2.mp4"}],
            "processing_stats": {"total_frames": 200},
        }

        merged = merge_combined_reports(existing, new_data)
        assert merged["folders_count"] == 2
        assert merged["videos_count"] == 2
        assert len(merged["raw_violations"]) == 2
        assert merged["processing_stats"]["total_frames"] == 300

    def test_load_json_if_exists(self, tmp_path):
        """Загрузка JSON если существует"""
        output = tmp_path / "exists.json"
        output.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        data = load_json_if_exists(output)
        assert data is not None
        assert data["key"] == "value"

        non_existent = tmp_path / "none.json"
        data = load_json_if_exists(non_existent)
        assert data is None

    def test_cooldown_seconds_constant(self):
        """Проверка константы cooldown"""
        assert COOLDOWN_SECONDS == 10.0
