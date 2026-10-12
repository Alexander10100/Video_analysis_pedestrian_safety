# tests/test_traffic_light.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_light import (
    STATE_RED,
    STATE_GREEN,
    STATE_UNKNOWN,
    LIGHT_TYPE_PEDESTRIAN,
    LIGHT_TYPE_VEHICLE,
    pedestrian_allowed,
    classify_roi,
    TrafficLightAnalyzer,
)


class TestPedestrianAllowed:
    """Тесты для функции pedestrian_allowed"""

    def test_pedestrian_light_green(self):
        """Пешеходный светофор: зелёный = разрешено"""
        assert pedestrian_allowed(STATE_GREEN, LIGHT_TYPE_PEDESTRIAN) is True

    def test_pedestrian_light_red(self):
        """Пешеходный светофор: красный = запрещено"""
        assert pedestrian_allowed(STATE_RED, LIGHT_TYPE_PEDESTRIAN) is False

    def test_vehicle_light_red(self):
        """Автомобильный светофор: красный = пешеходам разрешено"""
        assert pedestrian_allowed(STATE_RED, LIGHT_TYPE_VEHICLE) is True

    def test_vehicle_light_green(self):
        """Автомобильный светофор: зелёный = пешеходам запрещено"""
        assert pedestrian_allowed(STATE_GREEN, LIGHT_TYPE_VEHICLE) is False


class TestClassifyRoi:
    """Тесты для classify_roi"""

    def test_empty_roi(self):
        """Пустой ROI возвращает UNKNOWN"""
        state, conf, debug = classify_roi(None)
        assert state == STATE_UNKNOWN
        assert conf == 0.0

    def test_small_roi(self):
        """Слишком маленький ROI возвращает UNKNOWN"""
        small_roi = np.zeros((3, 3, 3), dtype=np.uint8)
        state, conf, debug = classify_roi(small_roi)
        assert state == STATE_UNKNOWN

    def test_roi_with_red(self):
        """ROI с красным цветом"""
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        roi[:, :, 2] = 255
        state, conf, debug = classify_roi(roi)
        assert state in (STATE_RED, STATE_UNKNOWN)


class TestTrafficLightAnalyzer:
    """Тесты для TrafficLightAnalyzer"""

    def test_init(self):
        """Создание анализатора"""
        analyzer = TrafficLightAnalyzer()
        assert analyzer._cache_frames == 2
        assert analyzer._states == {}
        assert analyzer._zone_types == {}

    def test_set_cache_frames(self):
        """Изменение кэша кадров"""
        analyzer = TrafficLightAnalyzer()
        analyzer.set_cache_frames(5)
        assert analyzer._cache_frames == 5

    def test_reset(self):
        """Сброс состояния"""
        analyzer = TrafficLightAnalyzer()
        analyzer._states["test"] = []
        analyzer._zone_types["test"] = LIGHT_TYPE_PEDESTRIAN
        analyzer.reset()
        assert analyzer._states == {}
        assert analyzer._zone_types == {}

    def test_parse_rois(self):
        """Разбор ROI"""
        analyzer = TrafficLightAnalyzer()

        assert analyzer._parse_rois(None) == []
        assert analyzer._parse_rois([]) == []

        roi = [0.1, 0.2, 0.3, 0.4]
        result = analyzer._parse_rois(roi)
        assert result == [roi]

        rois = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        result = analyzer._parse_rois(rois)
        assert result == rois