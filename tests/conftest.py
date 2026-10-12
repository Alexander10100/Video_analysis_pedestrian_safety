# tests/conftest.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_frame():
    """Создаёт тестовый кадр 720x1280"""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Тестовый bbox (x1, y1, x2, y2)"""
    return (100, 200, 200, 500)  # высота 300px


@pytest.fixture
def temp_calibration(tmp_path):
    """Создаёт временный файл калибровки"""
    import age_classifier
    original_dir = age_classifier.CALIBRATIONS_DIR

    calib_dir = tmp_path / "calibrations"
    calib_dir.mkdir()

    calib_file = calib_dir / "test_cam.json"
    calib_file.write_text(
        json.dumps({
            "frame_height": 720,
            "refs": {"5": 180.0, "6": 185.0},
            "samples_count": {"5": 50, "6": 45}
        }),
        encoding="utf-8"
    )

    age_classifier.CALIBRATIONS_DIR = calib_dir
    yield calib_dir
    age_classifier.CALIBRATIONS_DIR = original_dir
