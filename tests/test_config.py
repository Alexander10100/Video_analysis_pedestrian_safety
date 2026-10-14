"""Тесты централизованной конфигурации."""

from pathlib import Path

import pytest

from pedestrian_safety import config


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_defaults_use_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in ("PSAFETY_DATA_DIR", "PSAFETY_LOG_LEVEL", "PSAFETY_LOG_JSON", "FFMPEG_PATH"):
        monkeypatch.delenv(var, raising=False)

    s = config.get_settings()

    assert s.data_dir == tmp_path.resolve()
    assert s.zones_file == tmp_path.resolve() / "zones.json"
    assert s.calibrations_dir.name == "calibrations"
    assert s.log_level == "INFO"
    assert s.log_json is False


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PSAFETY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PSAFETY_LOG_LEVEL", "debug")
    monkeypatch.setenv("PSAFETY_LOG_JSON", "1")
    monkeypatch.setenv("FFMPEG_PATH", "/opt/ffmpeg/bin/ffmpeg")

    s = config.get_settings()

    assert s.data_dir == tmp_path.resolve()
    assert s.violation_log == tmp_path.resolve() / "violations_log.jsonl"
    assert s.log_level == "DEBUG"
    assert s.log_json is True
    assert s.ffmpeg_path == "/opt/ffmpeg/bin/ffmpeg"


def test_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PSAFETY_LOG_LEVEL=ERROR\nYANDEX_MAPS_API_KEY='abc'\n")
    monkeypatch.setenv("PSAFETY_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("YANDEX_MAPS_API_KEY", raising=False)

    s = config.get_settings()

    assert s.log_level == "WARNING"  # переменная окружения важнее .env
    assert s.yandex_maps_api_key == "abc"


def test_ffmpeg_is_string_not_path(monkeypatch):
    """Регрессия: на Linux FFMPEG_PATH был str, а код вызывал .exists()."""
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    s = config.get_settings()
    assert isinstance(s.ffmpeg_path, str)
    assert Path(s.ffmpeg_path).name.startswith("ffmpeg")
