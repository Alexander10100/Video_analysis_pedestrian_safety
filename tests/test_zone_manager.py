# tests/test_zone_manager.py
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from zone_manager import ZoneManager, RoadZone


class TestRoadZone:
    """Тесты для RoadZone"""

    def test_create_zone(self):
        """Создание зоны"""
        zone = RoadZone(
            id="test_id",
            label="Test Zone",
            polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
            type="road",
            color=[0, 80, 220]
        )
        assert zone.id == "test_id"
        assert zone.label == "Test Zone"
        assert len(zone.polygon) == 4
        assert zone.type == "road"

    def test_to_dict(self):
        """Преобразование в словарь"""
        zone = RoadZone(
            id="test_id",
            label="Test Zone",
            polygon=[[0.1, 0.1], [0.2, 0.1]],
            type="crosswalk",
            color=[220, 160, 0],
            has_light=True,
            light_type="pedestrian"
        )
        data = zone.to_dict()
        assert data["id"] == "test_id"
        assert data["label"] == "Test Zone"
        assert data["type"] == "crosswalk"
        assert data["has_light"] is True

    def test_contains_point(self):
        """Проверка принадлежности точки зоне"""
        zone = RoadZone(
            id="test",
            label="Test",
            polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
            type="road",
            color=[0, 80, 220]
        )
        assert zone.contains_point(0.15, 0.15) is True
        assert zone.contains_point(0.5, 0.5) is False


class TestZoneManager:
    """Тесты для ZoneManager"""

    def test_init(self):
        """Создание менеджера"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cameras": {}}')
            temp_file = Path(f.name)

        mgr = ZoneManager(filepath=temp_file)
        assert isinstance(mgr._zones, dict)

        temp_file.unlink()

    def test_add_zone(self):
        """Добавление зоны"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cameras": {}}')
            temp_file = Path(f.name)

        mgr = ZoneManager(filepath=temp_file)
        zone = mgr.add_zone(
            label="Test Zone",
            polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
            zone_type="road"
        )
        assert zone.id in mgr._zones
        assert len(mgr._zones) == 1

        temp_file.unlink()

    def test_get_all(self):
        """Получение всех зон"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cameras": {}}')
            temp_file = Path(f.name)

        mgr = ZoneManager(filepath=temp_file)
        mgr.add_zone("Zone 1", [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
        mgr.add_zone("Zone 2", [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4]])
        zones = mgr.get_all()
        assert len(zones) == 2

        temp_file.unlink()

    def test_delete_zone(self):
        """Удаление зоны"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cameras": {}}')
            temp_file = Path(f.name)

        mgr = ZoneManager(filepath=temp_file)
        zone = mgr.add_zone("Test", [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
        assert len(mgr._zones) == 1

        mgr.delete_zone(zone.id)
        assert len(mgr._zones) == 0

        temp_file.unlink()

    def test_road_zones(self):
        """Фильтрация дорожных зон"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cameras": {}}')
            temp_file = Path(f.name)

        mgr = ZoneManager(filepath=temp_file)
        mgr.add_zone("Road", [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], zone_type="road")
        mgr.add_zone("Crosswalk", [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4]], zone_type="crosswalk")

        roads = mgr.road_zones()
        assert len(roads) == 1
        assert roads[0].type == "road"

        temp_file.unlink()

    def test_crosswalk_zones(self):
        """Фильтрация пешеходных переходов"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cameras": {}}')
            temp_file = Path(f.name)

        mgr = ZoneManager(filepath=temp_file)
        mgr.add_zone("Road", [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], zone_type="road")
        mgr.add_zone("Crosswalk", [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4]], zone_type="crosswalk")

        crosswalks = mgr.crosswalk_zones()
        assert len(crosswalks) == 1
        assert crosswalks[0].type == "crosswalk"

        temp_file.unlink()