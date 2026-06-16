import json
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from report_generator import generate_report, load_violations


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
