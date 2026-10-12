# tests/test_offline_detect.py
import sys
from pathlib import Path
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from offline_detect import (
    _progress_bar,
    _format_eta,
    _make_output_path,
    _configure_runtime,
    _resolve_device,
    _use_half_for_device,
    _iou,
    _merge_tile_boxes,
    _draw_legend_offline,
    VIDEO_EXTS,
    OUTPUT_FOURCC,
    OUTPUT_EXT,
)


class TestOfflineDetect:
    """Тесты для offline_detect.py"""

    def test_progress_bar(self):
        """Тест прогресс-бара"""
        bar = _progress_bar(50, 100)
        assert "[" in bar
        assert "]" in bar
        assert "50.0%" in bar or "50%" in bar

    def test_progress_bar_zero_total(self):
        """Прогресс-бар с нулевым total"""
        bar = _progress_bar(0, 0)
        assert "?" in bar

    def test_format_eta(self):
        """Форматирование ETA"""
        eta = _format_eta(30)
        assert "30s" in eta

        eta = _format_eta(90)
        assert "1m30s" in eta or "1m30" in eta

        eta = _format_eta(3661)
        assert "h" in eta

    def test_make_output_path(self):
        """Создание пути выходного файла"""
        input_path = Path("test.mp4")

        result = _make_output_path(input_path, None)
        assert result.suffix == OUTPUT_EXT
        assert "test_annotated" in str(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _make_output_path(input_path, tmpdir)
            assert result.parent == Path(tmpdir)

    def test_configure_runtime(self):
        """Настройка runtime"""
        torch = _configure_runtime()
        assert torch is None or hasattr(torch, "cuda")

    def test_resolve_device(self):
        """Разрешение устройства"""
        device = _resolve_device("auto")
        assert device in ("cpu", "cuda:0")

        device = _resolve_device("cpu")
        assert device == "cpu"

        device = _resolve_device("0")
        assert device == "cuda:0"

    def test_use_half_for_device(self):
        """Проверка использования half precision"""
        assert _use_half_for_device("cpu") is False

        result = _use_half_for_device("cuda:0")
        if torch := _configure_runtime():
            if torch.cuda.is_available():
                assert result is True

    def test_iou(self):
        """Расчёт IoU"""
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)

        iou = _iou(a, b)
        assert iou == 25.0 / 175.0

        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        iou = _iou(a, b)
        assert iou == 0.0

        a = (0, 0, 10, 10)
        b = (0, 0, 10, 10)
        iou = _iou(a, b)
        assert iou == 1.0

    def test_video_extensions(self):
        """Проверка поддерживаемых расширений видео"""
        assert ".mp4" in VIDEO_EXTS
        assert ".avi" in VIDEO_EXTS
        assert ".mkv" in VIDEO_EXTS
        assert ".mov" in VIDEO_EXTS
        assert ".ts" in VIDEO_EXTS
        assert ".webm" in VIDEO_EXTS
        assert ".m4v" in VIDEO_EXTS

    def test_output_constants(self):
        """Проверка констант вывода"""
        assert OUTPUT_FOURCC == "mp4v"
        assert OUTPUT_EXT == ".mp4"

    def test_merge_tile_boxes_empty(self):
        """Объединение тайловых боксов с пустым списком"""
        norm_boxes = []
        tile_raw = []
        result = _merge_tile_boxes(norm_boxes, tile_raw, 100, 100)
        assert result == []

    def test_merge_tile_boxes_with_existing(self):
        """Объединение тайловых боксов с существующими"""
        norm_boxes = [
            (1, 0.1, 0.1, 0.2, 0.2, 0.9, "adult", 0.8)
        ]
        tile_raw = [
            {"x1": 15, "y1": 15, "x2": 25, "y2": 25, "conf": 0.8},
            {"x1": 80, "y1": 80, "x2": 90, "y2": 90, "conf": 0.7},
        ]
        result = _merge_tile_boxes(norm_boxes, tile_raw, 100, 100, iou_thresh=0.3)
        assert len(result) >= 1
        found_new = any(b["x1"] == 80 and b["y1"] == 80 for b in result)
        assert found_new is True

    def test_draw_legend_offline(self):
        """Рисование легенды (проверка что не падает)"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = _draw_legend_offline(
            frame,
            persons=5,
            ms=10.5,
            violations=2,
            adults=3,
            children=2,
            model_sz="m",
            camera_id="test_cam",
            age_calibrated=True,
            frame_idx=100,
            total_frames=1000,
            detect_every=3,
            tile_enabled=True,
            tile_extra=1,
        )

        assert result is not None
        assert result.shape == (720, 1280, 3)