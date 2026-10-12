"""Централизованная конфигурация приложения.

Все пути к данным и параметры окружения собраны здесь, а не разбросаны
по модулям. Значения читаются из переменных окружения (и из файла ``.env``,
если он есть в рабочей директории), что соответствует принципам 12-factor app:
одна и та же сборка работает локально, в Docker и в CI.

Переменные окружения
--------------------
``PSAFETY_DATA_DIR``   корневая папка с данными (zones.json, calibrations/, video/ …).
                       По умолчанию — текущая рабочая директория.
``PSAFETY_LOG_LEVEL``  уровень логирования (DEBUG/INFO/WARNING/ERROR). По умолчанию INFO.
``PSAFETY_LOG_JSON``   ``1`` — писать логи в JSON (удобно для Docker/ELK).
``FFMPEG_PATH``        путь к бинарнику ffmpeg. По умолчанию ищется в ``PATH``,
                       на Windows дополнительно ``<data_dir>/ffmpeg/bin/ffmpeg.exe``.
``YANDEX_MAPS_API_KEY`` ключ API Яндекс.Карт для сервиса карты камер.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"})
IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
YOLO_SIZES: tuple[str, ...] = ("n", "s", "m", "l", "x")


def _load_dotenv(path: Path) -> None:
    """Минимальный парсер ``.env``: ``KEY=VALUE``, без перезаписи уже заданных переменных."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _resolve_ffmpeg(data_dir: Path) -> str:
    explicit = os.environ.get("FFMPEG_PATH")
    if explicit:
        return explicit
    if sys.platform == "win32":
        bundled = data_dir / "ffmpeg" / "bin" / "ffmpeg.exe"
        if bundled.exists():
            return str(bundled)
    return shutil.which("ffmpeg") or "ffmpeg"


@dataclass(frozen=True)
class Settings:
    """Неизменяемый набор настроек, общий для всех сервисов."""

    data_dir: Path
    log_level: str
    log_json: bool
    ffmpeg_path: str
    yandex_maps_api_key: str

    # ── Производные пути ──────────────────────────────────────────────────
    @property
    def zones_file(self) -> Path:
        return self.data_dir / "zones.json"

    @property
    def calibrations_dir(self) -> Path:
        return self.data_dir / "calibrations"

    @property
    def video_root(self) -> Path:
        return self.data_dir / "video"

    @property
    def violation_log(self) -> Path:
        return self.data_dir / "violations_log.jsonl"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def reports_batches_dir(self) -> Path:
        return self.data_dir / "reports_batches"

    @property
    def heatmap_assets_dir(self) -> Path:
        return self.data_dir / "heatmap_assets"

    @property
    def heatmap_positions_file(self) -> Path:
        return self.data_dir / "heatmap_camera_positions.json"

    @property
    def camera_points_file(self) -> Path:
        return self.data_dir / "camera_points.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Прочитать настройки один раз за процесс."""
    _load_dotenv(Path.cwd() / ".env")
    data_dir = Path(os.environ.get("PSAFETY_DATA_DIR", Path.cwd())).resolve()
    return Settings(
        data_dir=data_dir,
        log_level=os.environ.get("PSAFETY_LOG_LEVEL", "INFO").upper(),
        log_json=os.environ.get("PSAFETY_LOG_JSON", "0") == "1",
        ffmpeg_path=_resolve_ffmpeg(data_dir),
        yandex_maps_api_key=os.environ.get("YANDEX_MAPS_API_KEY", ""),
    )
