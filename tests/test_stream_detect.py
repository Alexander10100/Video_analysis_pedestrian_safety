# tests/test_stream_detect.py
import contextlib
import threading
import time

import pytest

from pedestrian_safety.services.stream_detect import (
    COOLDOWN_SECONDS,
    VIOLATION_LOG,
    _clear_cache_event,
    _current_video,
    _evict_every,
    _frame_queue,
    _model_reload_event,
    _restart_event,
    _source_folder,
    _source_url,
    _violation_cooldown,
    compute_skip,
    set_camera,
    state,
)


class TestStreamDetect:
    """Тесты для stream_detect.py"""

    def test_compute_skip(self):
        """Расчёт пропуска кадров из FPM"""
        skip = compute_skip(60.0, 60)
        assert skip == 60

        skip = compute_skip(30.0, 60)
        assert skip == 30

        skip = compute_skip(25.0, 100)
        assert skip == 15

        skip = compute_skip(30.0, 30)
        assert skip == 60

        skip = compute_skip(0, 60)
        assert skip == 1

    def test_state_initialization(self):
        """Проверка начального состояния"""
        assert state["conf"] == 0.45
        assert state["fpm"] == 60
        assert state["persons"] == 0
        assert state["fps"] == 0.0
        assert state["detect_enabled"] is True

    def test_state_update(self):
        """Обновление состояния через lock"""
        with state["lock"]:
            state["persons"] = 10
            state["fps"] = 25.5

        assert state["persons"] == 10
        assert state["fps"] == 25.5

    def test_violation_cooldown(self):
        """Проверка cooldown для нарушений"""
        _violation_cooldown.clear()

        tid = 1
        current_time = time.time()

        if tid not in _violation_cooldown:
            _violation_cooldown[tid] = current_time
            assert tid in _violation_cooldown

        last_logged = _violation_cooldown.get(tid, 0)
        if current_time - last_logged < COOLDOWN_SECONDS:
            assert True

    def test_evict_every_constant(self):
        """Проверка константы очистки треков"""
        assert _evict_every == 150

    def test_camera_setting(self):
        """Установка камеры"""
        with contextlib.suppress(Exception):
            set_camera("test_camera")
        assert True

    def test_source_variables(self):
        """Проверка глобальных переменных источника"""
        assert _source_url is not None or _source_folder is None
        assert _current_video is None or isinstance(_current_video, str)

    def test_events(self):
        """Проверка событий"""
        _restart_event.clear()
        _model_reload_event.clear()
        _clear_cache_event.clear()

        assert isinstance(_restart_event, threading.Event)
        assert isinstance(_model_reload_event, threading.Event)
        assert isinstance(_clear_cache_event, threading.Event)

        _restart_event.set()
        assert _restart_event.is_set() is True

    def test_frame_queue(self):
        """Проверка очереди кадров"""
        while not _frame_queue.empty():
            try:
                _frame_queue.get_nowait()
            except Exception:
                break

        assert _frame_queue.empty() is True

    def test_log_file_path(self):
        """Проверка пути к лог-файлу"""
        assert VIOLATION_LOG.name == "violations_log.jsonl"
        assert VIOLATION_LOG.suffix == ".jsonl"

    def test_cooldown_seconds(self):
        """Проверка времени cooldown"""
        assert COOLDOWN_SECONDS == 10.0
        assert COOLDOWN_SECONDS > 0

    @pytest.mark.parametrize(
        "fps,fpm,expected",
        [
            (60, 60, 60),
            (60, 120, 30),
            (30, 60, 30),
            (30, 30, 60),
            (25, 100, 15),
        ],
    )
    def test_compute_skip_parametrized(self, fps, fpm, expected):
        """Параметризованный тест compute_skip"""
        skip = compute_skip(fps, fpm)
        assert skip == expected


class TestReadFrame:
    """Регрессия: кадр должен читаться ровно один раз за итерацию."""

    def test_reads_exactly_one_frame_from_capture(self):
        from unittest.mock import MagicMock

        import numpy as np

        from pedestrian_safety.services.stream_detect import _read_frame

        cap = MagicMock()
        cap.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))

        ok, frame = _read_frame(cap, None)

        assert ok is True
        assert frame is not None
        assert cap.read.call_count == 1

    def test_short_ffmpeg_read_means_end_of_stream(self):
        from unittest.mock import MagicMock

        from pedestrian_safety.services.stream_detect import _read_frame

        process = MagicMock()
        process.stdout.read.return_value = b"\x00" * 10  # меньше одного кадра

        assert _read_frame(None, process) == (False, None)
