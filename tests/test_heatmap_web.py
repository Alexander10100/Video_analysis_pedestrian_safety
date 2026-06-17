import sys
from pathlib import Path
import json
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from heatmap_web import (
    load_json,
    save_json,
    load_positions_config,
    save_positions_config,
    discover_report_files,
    extract_report_summary,
    choose_latest_report,
    collect_reports_by_camera,
    collect_camera_ids,
    violation_sort_key,
    build_violation_stats,
    camera_violation_payload,
    zones_for_camera,
    background_url,
    ALLOWED_IMAGE_EXTS,
    VIDEO_EXTS,
)


class TestHeatmapWeb:

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
        payload = {"key": "value"}

        save_json(file_path, payload)
        assert file_path.exists()

        data = json.loads(file_path.read_text(encoding="utf-8"))
        assert data["key"] == "value"

    def test_load_positions_config_default(self, tmp_path):
        """Загрузка конфигурации позиций по умолчанию"""
        import heatmap_web
        original = heatmap_web.POSITIONS_FILE
        heatmap_web.POSITIONS_FILE = tmp_path / "none.json"

        try:
            config = load_positions_config()
            assert "positions" in config
            assert "background" in config
            assert "ui" in config
        finally:
            heatmap_web.POSITIONS_FILE = original

    def test_save_positions_config(self, tmp_path):
        """Сохранение конфигурации позиций"""
        import heatmap_web
        original = heatmap_web.POSITIONS_FILE
        heatmap_web.POSITIONS_FILE = tmp_path / "positions.json"

        try:
            config = {
                "positions": {"cam1": {"x": 0.5, "y": 0.5}},
                "background": {"filename": "bg.jpg"},
                "ui": {"positions_locked": True},
            }
            save_positions_config(config)
            assert heatmap_web.POSITIONS_FILE.exists()

            data = json.loads(heatmap_web.POSITIONS_FILE.read_text(encoding="utf-8"))
            assert "cam1" in data["positions"]
        finally:
            heatmap_web.POSITIONS_FILE = original

    def test_discover_report_files_no_reports(self, tmp_path):
        """Поиск отчётов (нет отчётов)"""
        import heatmap_web
        original = heatmap_web.REPORT_DIRS
        heatmap_web.REPORT_DIRS = [tmp_path / "reports"]

        try:
            files = discover_report_files()
            assert files == []
        finally:
            heatmap_web.REPORT_DIRS = original

    def test_discover_report_files_with_reports(self, tmp_path):
        """Поиск отчётов (есть отчёты)"""
        import heatmap_web
        original = heatmap_web.REPORT_DIRS
        heatmap_web.REPORT_DIRS = [tmp_path / "reports"]

        report_dir = tmp_path / "reports" / "cam1"
        report_dir.mkdir(parents=True)

        report_file = report_dir / "report.json"
        report_file.write_text(json.dumps({"camera_id": "cam1"}), encoding="utf-8")

        try:
            files = discover_report_files()
            assert len(files) == 1
            assert files[0].name == "report.json"
        finally:
            heatmap_web.REPORT_DIRS = original

    def test_extract_report_summary_valid(self, tmp_path):
        """Извлечение сводки из отчёта"""
        import heatmap_web
        original_base = heatmap_web.BASE_DIR

        heatmap_web.BASE_DIR = tmp_path

        try:
            report_data = {
                "camera_id": "cam1",
                "total_violations": 10,
                "unique_persons": 5,
                "summary": {
                    "by_type": {"red_light": 6, "road_trespass": 4},
                    "by_age": {"adult": 8, "child": 2},
                },
                "raw_violations": [{"timestamp": "2026-01-01T00:00:00"}],
                "generated_at": "2026-01-01T00:00:00",
            }

            report_file = tmp_path / "report.json"
            report_file.write_text(json.dumps(report_data), encoding="utf-8")

            summary = extract_report_summary(report_file)
            assert summary is not None
            assert summary["camera_id"] == "cam1"
            assert summary["total_violations"] == 10
            assert summary["unique_persons"] == 5
        finally:
            heatmap_web.BASE_DIR = original_base

    def test_extract_report_summary_invalid(self, tmp_path):
        """Извлечение сводки из некорректного отчёта"""
        report_file = tmp_path / "invalid.json"
        report_file.write_text("invalid json", encoding="utf-8")

        summary = extract_report_summary(report_file)
        assert summary is None

    def test_choose_latest_report(self):
        """Выбор последнего отчёта"""
        reports = [
            {"report_ts": "2026-01-01T00:00:00", "data": "old"},
            {"report_ts": "2026-01-02T00:00:00", "data": "new"},
        ]

        latest = choose_latest_report(reports)
        assert latest["data"] == "new"

    def test_choose_latest_report_empty(self):
        """Выбор последнего отчёта (пустой список)"""
        with pytest.raises(IndexError):
            choose_latest_report([])

    def test_collect_reports_by_camera(self, tmp_path):
        """Сбор отчётов по камерам"""
        import heatmap_web
        original_base = heatmap_web.BASE_DIR
        original_dirs = heatmap_web.REPORT_DIRS

        heatmap_web.BASE_DIR = tmp_path
        heatmap_web.REPORT_DIRS = [tmp_path / "reports"]

        try:
            report_dir = tmp_path / "reports" / "cam1"
            report_dir.mkdir(parents=True)

            report_data = {
                "camera_id": "cam1",
                "total_violations": 10,
                "unique_persons": 5,
                "summary": {"by_type": {}, "by_age": {}},
                "raw_violations": [],
                "generated_at": "2026-01-01T00:00:00",
            }
            (report_dir / "report.json").write_text(json.dumps(report_data), encoding="utf-8")

            result = collect_reports_by_camera()
            assert "cam1" in result
            assert result["cam1"]["total_violations"] == 10
        finally:
            heatmap_web.BASE_DIR = original_base
            heatmap_web.REPORT_DIRS = original_dirs

    def test_collect_camera_ids(self, tmp_path):
        """Сбор ID камер"""
        import heatmap_web
        original_video = heatmap_web.VIDEO_ROOT
        original_zones = heatmap_web.ZONES_FILE
        original_positions = heatmap_web.POSITIONS_FILE

        # Добавляем камеры через zones.json и positions.json
        zones_data = {"cameras": {"cam1": {"zones": []}, "cam2": {"zones": []}}}
        positions_data = {"positions": {"cam3": {"x": 0.5, "y": 0.5}, "cam4": {"x": 0.6, "y": 0.6}}}

        heatmap_web.VIDEO_ROOT = tmp_path / "video"
        heatmap_web.ZONES_FILE = tmp_path / "zones.json"
        heatmap_web.POSITIONS_FILE = tmp_path / "positions.json"

        heatmap_web.ZONES_FILE.write_text(json.dumps(zones_data), encoding="utf-8")
        heatmap_web.POSITIONS_FILE.write_text(json.dumps(positions_data), encoding="utf-8")

        try:
            ids = collect_camera_ids()
            # Проверяем камеры из zones.json и positions.json
            assert "cam1" in ids, "cam1 должна быть обнаружена через zones.json"
            assert "cam2" in ids, "cam2 должна быть обнаружена через zones.json"
            assert "cam3" in ids, "cam3 должна быть обнаружена через positions.json"
            assert "cam4" in ids, "cam4 должна быть обнаружена через positions.json"
            assert len(ids) >= 4, f"Ожидалось минимум 4 камеры, получено {len(ids)}: {ids}"
        finally:
            heatmap_web.VIDEO_ROOT = original_video
            heatmap_web.ZONES_FILE = original_zones
            heatmap_web.POSITIONS_FILE = original_positions

    def test_violation_sort_key(self):
        """Сортировка нарушений по времени"""
        violation1 = {"timestamp": "2026-01-01T00:00:00"}
        violation2 = {"timestamp": "2026-01-02T00:00:00"}

        sorted_violations = sorted(
            [violation1, violation2],
            key=violation_sort_key
        )
        assert sorted_violations[0]["timestamp"] == "2026-01-01T00:00:00"

    def test_violation_sort_key_invalid(self):
        """Сортировка нарушений с некорректной датой"""
        violation = {"timestamp": "invalid"}
        dt = violation_sort_key(violation)
        assert dt == datetime.min

    def test_build_violation_stats(self):
        """Построение статистики нарушений"""
        report = {
            "total_violations": 10,
            "by_type": {"road_trespass": 4, "red_light": 6},
        }

        stats = build_violation_stats(report)
        assert stats["total"] == 10
        assert stats["road_trespass"]["count"] == 4
        assert stats["road_trespass"]["percent"] == 40.0
        assert stats["red_light"]["count"] == 6
        assert stats["red_light"]["percent"] == 60.0

    def test_build_violation_stats_empty(self):
        """Построение статистики без нарушений"""
        report = {"total_violations": 0, "by_type": {}}

        stats = build_violation_stats(report)
        assert stats["total"] == 0
        assert stats["road_trespass"]["percent"] == 0.0

    def test_camera_violation_payload(self, tmp_path):
        """Получение данных о нарушениях для камеры"""
        import heatmap_web
        original_base = heatmap_web.BASE_DIR
        original_dirs = heatmap_web.REPORT_DIRS

        heatmap_web.BASE_DIR = tmp_path
        heatmap_web.REPORT_DIRS = [tmp_path / "reports"]

        try:
            report_dir = tmp_path / "reports" / "cam1"
            report_dir.mkdir(parents=True)

            report_data = {
                "camera_id": "cam1",
                "total_violations": 2,
                "by_type": {"red_light": 2},
                "raw_violations": [
                    {"timestamp": "2026-01-01T00:00:00", "violation_type": "red_light"},
                    {"timestamp": "2026-01-01T00:01:00", "violation_type": "red_light"},
                ],
                "generated_at": "2026-01-01T00:00:00",
            }
            (report_dir / "report.json").write_text(json.dumps(report_data), encoding="utf-8")

            payload = camera_violation_payload("cam1")
            assert payload["stats"]["total"] == 2
            assert len(payload["violations"]) == 2
        finally:
            heatmap_web.BASE_DIR = original_base
            heatmap_web.REPORT_DIRS = original_dirs

    def test_zones_for_camera(self, tmp_path):
        """Получение зон для камеры"""
        import heatmap_web
        original = heatmap_web.ZONES_FILE
        heatmap_web.ZONES_FILE = tmp_path / "zones.json"

        zones_data = {
            "cameras": {
                "cam1": {
                    "zones": [
                        {"id": "z1", "label": "Zone 1", "type": "road"},
                        {"id": "z2", "label": "Zone 2", "type": "crosswalk"},
                    ]
                }
            }
        }
        heatmap_web.ZONES_FILE.write_text(json.dumps(zones_data), encoding="utf-8")

        try:
            zones = zones_for_camera("cam1")
            assert len(zones) == 2
            assert zones[0]["label"] == "Zone 1"
        finally:
            heatmap_web.ZONES_FILE = original

    def test_zones_for_camera_empty(self, tmp_path):
        """Получение зон для камеры (нет зон)"""
        import heatmap_web
        original = heatmap_web.ZONES_FILE
        heatmap_web.ZONES_FILE = tmp_path / "zones.json"

        heatmap_web.ZONES_FILE.write_text(json.dumps({"cameras": {}}), encoding="utf-8")

        try:
            zones = zones_for_camera("cam1")
            assert zones == []
        finally:
            heatmap_web.ZONES_FILE = original

    def test_background_url(self):
        """Получение URL фона"""
        config = {"background": {"filename": "bg.jpg"}}
        url = background_url(config)
        assert url == "/backgrounds/bg.jpg"

    def test_background_url_empty(self):
        """Получение URL фона (нет фона)"""
        config = {"background": {}}
        url = background_url(config)
        assert url is None

    def test_video_extensions(self):
        """Проверка поддерживаемых расширений"""
        assert ".mp4" in VIDEO_EXTS
        assert ".avi" in VIDEO_EXTS
        assert ".mov" in VIDEO_EXTS
        assert ".mkv" in VIDEO_EXTS

    def test_allowed_image_exts(self):
        """Проверка поддерживаемых расширений изображений"""
        assert ".png" in ALLOWED_IMAGE_EXTS
        assert ".jpg" in ALLOWED_IMAGE_EXTS
        assert ".jpeg" in ALLOWED_IMAGE_EXTS
