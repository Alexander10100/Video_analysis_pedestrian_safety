"""Smoke-тест: все модули пакета импортируются (ловит битые импорты после рефакторинга)."""

import importlib

import pytest

MODULES = [
    "pedestrian_safety",
    "pedestrian_safety.config",
    "pedestrian_safety.logging_config",
    "pedestrian_safety.core.age_classifier",
    "pedestrian_safety.core.traffic_light",
    "pedestrian_safety.core.violation_detector",
    "pedestrian_safety.core.zone_manager",
    "pedestrian_safety.services.stream_detect",
    "pedestrian_safety.services.stream_detect_web",
    "pedestrian_safety.services.heatmap_web",
    "pedestrian_safety.services.camera_map_web",
    "pedestrian_safety.pipelines.offline_detect",
    "pedestrian_safety.pipelines.batch_folder_offline",
    "pedestrian_safety.pipelines.report_generator",
    "pedestrian_safety.pipelines.calibrate_camera",
]


@pytest.mark.parametrize("module", MODULES)
def test_import_module(module):
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize(
    ("module", "func"),
    [
        ("pedestrian_safety.services.stream_detect", "main"),
        ("pedestrian_safety.services.heatmap_web", "main"),
        ("pedestrian_safety.services.camera_map_web", "main"),
        ("pedestrian_safety.pipelines.offline_detect", "main"),
        ("pedestrian_safety.pipelines.batch_folder_offline", "main"),
        ("pedestrian_safety.pipelines.report_generator", "main"),
        ("pedestrian_safety.pipelines.calibrate_camera", "main"),
    ],
)
def test_entry_points_exist(module, func):
    """Каждая консольная команда из pyproject.toml указывает на существующую функцию."""
    assert callable(getattr(importlib.import_module(module), func))
