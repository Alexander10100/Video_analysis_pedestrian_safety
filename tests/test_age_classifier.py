# tests/test_age_classifier.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from age_classifier import AgeClassifier, AgeTracker, BboxEMA


class TestAgeClassifier:
    """Тесты для AgeClassifier"""

    def test_uncalibrated_returns_adult(self):
        """Без калибровки всегда возвращает adult"""
        clf = AgeClassifier(frame_height=720)
        assert not clf.is_calibrated()

        label, conf = clf.classify(100, 300, 200, 500)
        assert label == "adult"
        assert conf == 0.0

    def test_calibrated_classification(self):
        """Классификация с калибровкой"""
        clf = AgeClassifier(frame_height=720)
        clf._calibrated = True
        clf._refs = {5: 180.0}  # эталон в зоне 5

        # Взрослый (высота ~200px, ratio 1.11)
        label, conf = clf.classify(100, 300, 200, 500)
        assert label == "adult"
        assert conf > 0.5

        # Ребёнок (высота ~100px, ratio 0.55)
        label, conf = clf.classify(100, 400, 180, 500)
        assert label == "child"
        assert conf > 0.5

    def test_band_fallback(self):
        """Поиск эталона в соседних зонах"""
        clf = AgeClassifier(frame_height=720)
        clf._calibrated = True
        clf._refs = {3: 170.0, 7: 190.0}

        # Зона 5 — нет эталона, должен найти в зоне 3 или 7
        label, conf = clf.classify(100, 400, 200, 520)  # y2=520 -> зона ~7
        assert label in ("adult", "child")


class TestAgeTracker:
    """Тесты для AgeTracker с асимметричными порогами"""

    def test_single_frame(self):
        """Один кадр — возвращает raw_label"""
        tracker = AgeTracker(window=5, min_votes=3)

        label, conf = tracker.update(1, "adult", 0.9)
        assert label == "adult"
        assert conf == 0.45  # warmup_scale=0.5 * 0.9

    def test_adult_to_child_flip(self):
        """Переход adult→child при достаточном количестве голосов"""
        tracker = AgeTracker(window=5, min_votes=3, flip_to_child=0.55)

        # 3 кадра adult
        for _ in range(3):
            tracker.update(1, "adult", 0.9)

        # 3 кадра child (должно переключиться)
        for _ in range(3):
            label, conf = tracker.update(1, "child", 0.9)

        assert label == "child"

    def test_evict_old_tracks(self):
        """Очистка мёртвых треков"""
        tracker = AgeTracker()

        tracker.update(1, "adult", 0.9)
        tracker.update(2, "child", 0.9)

        assert len(tracker) == 2

        tracker.evict({1})  # трек 2 удаляем
        assert len(tracker) == 1

    def test_reset(self):
        """Сброс состояния"""
        tracker = AgeTracker()
        tracker.update(1, "adult", 0.9)
        assert len(tracker) == 1

        tracker.reset()
        assert len(tracker) == 0


class TestBboxEMA:
    """Тесты для EMA-сглаживания bbox"""

    def test_first_frame_no_smoothing(self):
        """Первый кадр — возвращает исходные координаты"""
        ema = BboxEMA(alpha=0.35)

        x1, y1, x2, y2 = ema.smooth(1, 100, 200, 300, 400)
        assert (x1, y1, x2, y2) == (100, 200, 300, 400)

    def test_smoothing_effect(self):
        """Проверка эффекта сглаживания"""
        ema = BboxEMA(alpha=0.5)

        # Первый кадр
        ema.smooth(1, 100, 200, 300, 400)

        # Второй кадр с резким скачком
        x1, y1, x2, y2 = ema.smooth(1, 150, 250, 350, 450)

        # Ожидаем среднее: (150+100)/2 = 125, итд
        assert x1 == 125
        assert y1 == 225
        assert x2 == 325
        assert y2 == 425

    def test_evict(self):
        """Очистка мёртвых треков"""
        ema = BboxEMA()

        ema.smooth(1, 100, 200, 300, 400)
        ema.smooth(2, 100, 200, 300, 400)

        assert len(ema) == 2

        evicted = ema.evict({1})
        assert evicted == 1
        assert len(ema) == 1

    def test_reset(self):
        """Сброс состояния"""
        ema = BboxEMA()
        ema.smooth(1, 100, 200, 300, 400)
        assert len(ema) == 1

        ema.reset()
        assert len(ema) == 0
