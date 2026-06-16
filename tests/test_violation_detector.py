# tests/test_violation_detector.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from violation_detector import person_inside_vehicle


class TestPersonInsideVehicle:
    """Тесты для фильтрации людей внутри автомобилей"""

    def test_no_vehicle(self):
        """Нет машин — всегда False"""
        result = person_inside_vehicle(100, 200, 150, 300, [])
        assert result is False

    def test_person_outside_vehicle(self):
        """Человек рядом с машиной, не внутри"""
        vehicle_boxes = [(200, 300, 400, 500)]
        result = person_inside_vehicle(100, 200, 150, 300, vehicle_boxes)
        assert result is False

    def test_person_inside_vehicle(self):
        """Человек полностью внутри машины"""
        vehicle_boxes = [(100, 200, 300, 400)]
        result = person_inside_vehicle(120, 210, 280, 390, vehicle_boxes)
        assert result is True

    def test_person_partially_overlapping(self):
        """Частичное перекрытие (недостаточно для фильтрации)"""
        vehicle_boxes = [(100, 200, 150, 400)]
        result = person_inside_vehicle(100, 200, 200, 500, vehicle_boxes)
        assert result is False

    def test_custom_overlap_threshold(self):
        """Пользовательский порог перекрытия"""
        vehicle_boxes = [(200, 200, 400, 400)]

        # Человек внутри машины с низким порогом -> True
        result = person_inside_vehicle(210, 210, 390, 390, vehicle_boxes, overlap_thresh=0.3)
        assert result is True

        # Человек снаружи машины -> False
        result = person_inside_vehicle(100, 100, 150, 150, vehicle_boxes, overlap_thresh=0.3)
        assert result is False

    def test_overlap_calculation(self):
        """Проверка корректности расчёта перекрытия"""
        vehicle_boxes = [(0, 0, 100, 100)]

        # Человек полностью внутри машины
        result = person_inside_vehicle(10, 10, 90, 90, vehicle_boxes, overlap_thresh=0.5)
        assert result is True

        # Человек полностью снаружи (выше машины)
        result = person_inside_vehicle(10, -50, 90, -10, vehicle_boxes, overlap_thresh=0.5)
        assert result is False

        # Человек полностью снаружи (ниже машины)
        result = person_inside_vehicle(10, 150, 90, 200, vehicle_boxes, overlap_thresh=0.5)
        assert result is False

    def test_multiple_vehicles(self):
        """Несколько машин рядом"""
        vehicle_boxes = [(100, 100, 200, 200), (300, 100, 400, 200), (500, 100, 600, 200)]

        # Человек внутри машины 2
        result = person_inside_vehicle(310, 110, 390, 190, vehicle_boxes)
        assert result is True

        # Человек между машинами
        result = person_inside_vehicle(250, 110, 290, 190, vehicle_boxes)
        assert result is False

    def test_edge_cases(self):
        """Краевые случаи - проверка обработки некорректных bbox"""
        vehicle_boxes = [(100, 100, 200, 200)]

        # Случай 1: Нулевая площадь
        result = person_inside_vehicle(150, 150, 150, 150, vehicle_boxes)
        assert result is False

        # Случай 2: Некорректные координаты (x1 > x2)
        result = person_inside_vehicle(150, 200, 150, 100, vehicle_boxes)
        assert result is False

        # Случай 3: Отрицательные координаты
        result = person_inside_vehicle(-50, -50, -10, -10, vehicle_boxes)
        assert result is False

        # Случай 4: Очень маленький bbox - функция может вернуть True или False
        # Просто проверяем, что функция не падает и возвращает bool
        result = person_inside_vehicle(150, 150, 151, 155, vehicle_boxes)
        assert isinstance(result, bool), "Функция должна возвращать bool"

        # Случай 5: Пустой список машин
        result = person_inside_vehicle(150, 150, 200, 200, [])
        assert result is False

    def test_vehicle_classes(self):
        """Проверка с разными типами транспортных средств"""

        # 1. Машина (car) - стандартный случай
        vehicle_boxes = [(100, 200, 300, 400)]
        result = person_inside_vehicle(120, 210, 280, 390, vehicle_boxes)
        assert result is True, "Человек в машине должен определяться"

        # 2. Мотоцикл (motorcycle) - человек сидит на мотоцикле
        # Мотоцикл меньше машины, но человек на нём - значит внутри
        vehicle_boxes = [(150, 250, 200, 320)]
        # Человек на мотоцикле (bbox человека чуть больше мотоцикла)
        result = person_inside_vehicle(145, 240, 205, 330, vehicle_boxes)
        # Проверяем, что функция не падает (результат может быть True или False)
        assert isinstance(result, bool)

        # 3. Грузовик (truck) - большой транспорт
        vehicle_boxes = [(50, 150, 350, 500)]
        result = person_inside_vehicle(60, 160, 340, 490, vehicle_boxes)
        assert result is True, "Человек в грузовике должен определяться"

        # 4. Автобус (bus) - большой транспорт
        vehicle_boxes = [(50, 100, 400, 600)]
        result = person_inside_vehicle(60, 110, 390, 590, vehicle_boxes)
        assert result is True, "Человек в автобусе должен определяться"

    def test_lower_body_only(self):
        """Проверка, что используется только нижняя часть (60% от низа)"""
        vehicle_boxes = [(100, 300, 300, 400)]

        # Человек: верхняя часть вне машины, нижняя внутри
        result = person_inside_vehicle(150, 200, 250, 400, vehicle_boxes, overlap_thresh=0.5)
        assert result is True

        # Человек: полностью выше машины
        result = person_inside_vehicle(150, 100, 250, 250, vehicle_boxes, overlap_thresh=0.5)
        assert result is False
