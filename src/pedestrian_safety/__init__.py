"""Система видеоаналитики безопасности пешеходов.

Пакет разбит на три слоя:

* :mod:`pedestrian_safety.core`      — доменная логика (зоны, светофоры,
  нарушения, классификация возраста). Не зависит от Flask и YOLO.
* :mod:`pedestrian_safety.services`  — долгоживущие веб-сервисы
  (стрим-детекция, тепловая карта, карта камер).
* :mod:`pedestrian_safety.pipelines` — пакетные/CLI-сценарии (офлайн-детекция,
  отчёты, калибровка).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pedestrian-safety")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников)
    __version__ = "0.0.0"

__all__ = ["__version__"]
