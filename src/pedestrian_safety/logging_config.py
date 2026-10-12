"""Единая настройка логирования вместо разрозненных ``print``.

Использование в модуле::

    import logging
    logger = logging.getLogger(__name__)

В точке входа (``main``) один раз вызвать :func:`setup_logging`.
"""

from __future__ import annotations

import json
import logging
import sys

from pedestrian_safety.config import get_settings

_TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    """Структурированные логи: одна JSON-строка на событие."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    """Сконфигурировать корневой логгер (идемпотентно)."""
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or settings.log_level)

    # Шумные сторонние библиотеки
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
