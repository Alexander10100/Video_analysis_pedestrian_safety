import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibrate_camera import (
    AutoCalibrator,
    CALIBRATIONS_DIR,
    SUPPORTED_EXTENSIONS,
    _progress_bar,
)


class TestAutoCalibrator:
    """Тесты для AutoCalibrator"""

    def test_init(self):
        """Создание калибратора"""
        cal = AutoCalibrator(frame_height=720)
        assert cal.frame_height == 720
        assert cal.band_h == 720 / cal.N_BANDS
        assert cal._samples == {}
        assert cal._refs == {}

    def test_update(self):
        """Обновление калибратора"""
        cal = AutoCalibrator(frame_height=720)

        cal.update(100, 100, 200, 300)
        cal.update(100, 100, 200, 320)

        band = cal._get_band(300)
        assert band in cal._samples
        assert len(cal._samples[band]) == 2

    def test_calibrate(self):
        """Калибровка"""
        cal = AutoCalibrator(frame_height=720)

        band = 5
        for i in range(20):
            y2 = int((band + 0.5) * cal.band_h)
            cal.update(100, 100, 200, y2)

        updated = cal.calibrate()
        assert updated >= 1
        assert band in cal._refs

    def test_classify_without_calibration(self):
        """Классификация без калибровки"""
        cal = AutoCalibrator(frame_height=720)
        label, conf = cal.classify(100, 100, 200, 300)
        assert label == "unknown"
        assert conf == 0.0

    def test_classify_with_calibration(self):
        """Классификация с калибровкой"""
        cal = AutoCalibrator(frame_height=720)
        cal._refs[5] = 200.0

        label, conf = cal.classify(100, 100, 200, 300)
        assert label in ("adult", "unknown")

        label, conf = cal.classify(100, 100, 180, 200)
        assert label == "child"

    def test_to_dict(self):
        """Преобразование в словарь"""
        cal = AutoCalibrator(frame_height=720)
        cal._refs[5] = 200.0
        cal._samples[5] = [180, 200, 220]

        data = cal.to_dict()
        assert data["frame_height"] == 720
        assert data["refs"]["5"] == 200.0
        assert data["samples_count"]["5"] == 3

    def test_from_dict(self):
        """Восстановление из словаря"""
        data = {
            "frame_height": 720,
            "refs": {"5": 200.0, "6": 210.0},
            "samples_count": {"5": 10, "6": 15},
        }

        cal = AutoCalibrator.from_dict(data)
        assert cal.frame_height == 720
        assert cal._refs[5] == 200.0
        assert cal._refs[6] == 210.0

    def test_status(self):
        """Статус калибровки"""
        cal = AutoCalibrator(frame_height=720)
        cal._refs[5] = 200.0

        status = cal.status()
        assert status["ready_bands"] == 1
        assert status["total_bands"] == cal.N_BANDS
        assert status["percent"] == int(1 / cal.N_BANDS * 100)

    def test_get_band(self):
        """Определение зоны по Y-координате"""
        cal = AutoCalibrator(frame_height=720)

        band = cal._get_band(700)
        assert band == cal.N_BANDS - 1

        band = cal._get_band(50)
        assert band == 0

        band = cal._get_band(360)
        assert band == cal.N_BANDS // 2

    def test_get_ref(self):
        """Поиск эталона"""
        cal = AutoCalibrator(frame_height=720)
        cal._refs[5] = 200.0
        cal._refs[7] = 220.0

        ref = cal._get_ref(5)
        assert ref == 200.0

        ref = cal._get_ref(6)
        assert ref == 200.0 or ref == 220.0

        ref = cal._get_ref(0)
        assert ref is None or isinstance(ref, float)


class TestCalibrateCameraUtils:
    """Тесты для утилит calibrate_camera.py"""

    def test_supported_extensions(self):
        """Проверка поддерживаемых расширений"""
        assert ".mp4" in SUPPORTED_EXTENSIONS
        assert ".avi" in SUPPORTED_EXTENSIONS
        assert ".mov" in SUPPORTED_EXTENSIONS
        assert ".mkv" in SUPPORTED_EXTENSIONS

    def test_progress_bar(self):
        """Тест прогресс-бара"""
        bar = _progress_bar(50, 100)
        assert "[" in bar
        assert "]" in bar

        bar = _progress_bar(0, 100)
        assert "0.0%" in bar or "0%" in bar

    def test_calibrations_dir(self):
        """Проверка директории калибровок"""
        assert CALIBRATIONS_DIR.name == "calibrations"
        assert CALIBRATIONS_DIR.is_dir() or not CALIBRATIONS_DIR.exists()
