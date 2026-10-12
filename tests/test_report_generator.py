import json

# Добавляем корень проекта в PYTHONPATH
from pedestrian_safety.pipelines.report_generator import generate_report, load_violations


def test_load_violations_no_file():
    """Проверка поведения при отсутствии лог-файла"""
    violations = load_violations(camera_id=None)
    assert isinstance(violations, list)


def test_generate_empty_report(tmp_path):
    """Генерация пустого отчёта"""
    output_file = tmp_path / "test_empty.json"

    generate_report([], str(output_file), camera_id="test_cam")

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert data["total_violations"] == 0
    assert data["unique_persons"] == 0
    assert data["camera_id"] == "test_cam"
    assert "summary" in data


class TestReportHelpers:
    def test_parse_since_hours_and_minutes(self):
        from datetime import datetime, timedelta

        from pedestrian_safety.pipelines.report_generator import parse_since

        since = datetime.fromisoformat(parse_since("2h"))
        assert abs((datetime.now() - since) - timedelta(hours=2)) < timedelta(seconds=5)
        since = datetime.fromisoformat(parse_since("30m"))
        assert abs((datetime.now() - since) - timedelta(minutes=30)) < timedelta(seconds=5)

    def test_parse_since_rejects_garbage(self):
        import pytest

        from pedestrian_safety.pipelines.report_generator import parse_since

        for bad in ("24", "h", "xxh", "10d"):
            with pytest.raises(ValueError):
                parse_since(bad)

    def test_report_written_once_with_same_schema_for_empty(self, tmp_path):
        """Пустой и непустой отчёт имеют одинаковый набор ключей."""
        from pedestrian_safety.pipelines.report_generator import generate_report

        empty = generate_report([], tmp_path / "empty.json", "cam1")
        full = generate_report(
            [
                {
                    "timestamp": "2026-04-13T09:00:10",
                    "track_id": 1,
                    "camera_id": "cam1",
                    "violation_type": "red_light",
                    "zone_label": "Переход",
                    "age_label": "child",
                    "person_conf": 0.9,
                }
            ],
            tmp_path / "full.json",
            "cam1",
        )
        assert set(empty) == set(full)
        assert full["summary"]["by_age"] == {"child": 1}
        assert (
            json.loads((tmp_path / "full.json").read_text(encoding="utf-8"))["total_violations"]
            == 1
        )
