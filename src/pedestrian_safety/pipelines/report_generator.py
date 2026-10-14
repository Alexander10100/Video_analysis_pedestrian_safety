"""
report_generator.py — Генерация JSON-отчётов по журналу нарушений.

Читает ``violations_log.jsonl`` (его пишет ``psafety-stream``) и формирует
сводку: число нарушений, уникальные нарушители, разбивка по типам, зонам,
возрасту и поминутная временная шкала.

Запуск::

    # Все нарушения за последние 24 часа для cam1
    psafety-report --camera cam1 --last 24h --output report_cam1_today.json

    # За конкретный час
    psafety-report --camera cam1 --since "2026-04-13T09:00:00" --until "2026-04-13T10:00:00"

    # Все камеры за всё время
    psafety-report --output full_report.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pedestrian_safety.config import get_settings
from pedestrian_safety.logging_config import setup_logging

logger = logging.getLogger(__name__)

LOG_FILE: Path = get_settings().violation_log


def load_violations(
    camera_id: str | None = None, since: str | None = None, until: str | None = None
) -> list:
    """Загружает нарушения из лог-файла с фильтрами."""
    if not LOG_FILE.exists():
        logger.warning(f"Лог нарушений не найден: {LOG_FILE}")
        logger.info("       Запустите stream_detect.py, чтобы начать запись нарушений.")
        return []

    violations = []
    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None

    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                v = json.loads(line)
                # Фильтр по камере
                if camera_id and v.get("camera_id") != camera_id:
                    continue
                # Фильтр по времени
                ts = datetime.fromisoformat(v["timestamp"])
                if since_dt and ts < since_dt:
                    continue
                if until_dt and ts > until_dt:
                    continue

                violations.append(v)
            except Exception:
                continue  # пропускаем повреждённые строки

    logger.info(f"Загружено {len(violations)} нарушений")
    return violations


def build_report_data(violations: list[dict], camera_id: str) -> dict[str, Any]:
    """Собрать сводный отчёт по списку записей журнала нарушений.

    Единственная реализация агрегации: используется и CLI ``psafety-report``,
    и пакетным пайплайном ``psafety-batch`` (раньше логика была продублирована).
    Повреждённые записи (без timestamp и т.п.) не роняют построение отчёта.
    """
    by_type: defaultdict[str, int] = defaultdict(int)
    by_zone: defaultdict[str, int] = defaultdict(int)
    by_age: defaultdict[str, int] = defaultdict(int)
    timeline: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    unique_tracks: set[int] = set()
    confidence_stats: list[float] = []

    for violation in violations:
        track_id = violation.get("track_id")
        if track_id is not None:
            unique_tracks.add(track_id)

        violation_type = violation.get("violation_type", "unknown")
        zone_label = violation.get("zone_label", "")
        age_label = violation.get("age_label", "unknown")

        by_type[violation_type] += 1
        by_zone[zone_label] += 1
        by_age[age_label] += 1

        if "person_conf" in violation:
            confidence_stats.append(float(violation["person_conf"]))

        try:
            minute = (
                datetime.fromisoformat(violation["timestamp"])
                .replace(second=0, microsecond=0)
                .isoformat()
            )
            timeline[minute][violation_type] += 1
        except (KeyError, TypeError, ValueError):
            logger.debug("Пропущена запись без корректного timestamp: %s", violation)

    avg_conf = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0.0
    min_conf = min(confidence_stats) if confidence_stats else 0.0
    max_conf = max(confidence_stats) if confidence_stats else 0.0

    return {
        "generated_at": datetime.now().isoformat(),
        "camera_id": camera_id,
        "total_violations": len(violations),
        "unique_persons": len(unique_tracks),
        "confidence": {
            "average": round(avg_conf, 3),
            "min": round(min_conf, 3),
            "max": round(max_conf, 3),
        },
        "summary": {
            "by_type": dict(by_type),
            "by_zone": dict(by_zone),
            "by_age": dict(by_age),
        },
        "timeline": [
            {
                "timestamp": minute,
                "total": sum(counts.values()),
                "by_type": dict(counts),
            }
            for minute, counts in sorted(timeline.items())
        ],
        "raw_violations": violations,
    }


def write_report(report: dict[str, Any], output: str | Path) -> Path:
    """Сохранить отчёт в JSON (создаёт родительские папки)."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def generate_report(
    violations: list[dict], output: str | Path, camera_id: str | None = None
) -> dict[str, Any]:
    """Построить отчёт, сохранить его в ``output`` и вывести краткую сводку в лог."""
    report = build_report_data(violations, camera_id or "all")
    write_report(report, output)

    if not violations:
        logger.info("За указанный период нарушений не найдено.")
        return report

    logger.info(f"Отчёт сохранён → {output}")
    logger.info(f"   Камера                  : {camera_id or 'Все камеры'}")
    logger.info(f"   Всего записей нарушений : {report['total_violations']}")
    logger.info(f"   Уникальных нарушителей  : {report['unique_persons']}")
    logger.info(f"   Средняя уверенность     : {report['confidence']['average']:.1%}")
    logger.info("   Типы нарушений:")
    by_type = report["summary"]["by_type"]
    for vtype, cnt in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"     • {vtype:15} : {cnt}")
    logger.info("   По возрасту:")
    for age, cnt in report["summary"]["by_age"].items():
        logger.info(f"     • {age:15} : {cnt}")
    return report


def parse_since(last: str) -> str:
    """Преобразовать ``24h`` / ``30m`` в ISO-время начала периода."""
    unit, amount = last[-1:], last[:-1]
    if unit not in {"h", "m"} or not amount.isdigit():
        raise ValueError("Параметр --last должен быть вида 24h или 30m")
    delta = timedelta(hours=int(amount)) if unit == "h" else timedelta(minutes=int(amount))
    return (datetime.now() - delta).isoformat()


def main(argv: list[str] | None = None) -> int:
    """Точка входа: ``psafety-report``."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Генератор отчётов по нарушениям ПДД")
    parser.add_argument(
        "--camera", type=str, default=None, help="ID камеры (например: cam1, cam_03)"
    )
    parser.add_argument(
        "--since", type=str, help="Начало периода в формате ISO (2026-04-13T09:00:00)"
    )
    parser.add_argument("--until", type=str, help="Конец периода в формате ISO")
    parser.add_argument("--last", type=str, help="Последние N часов или минут: 24h или 60m")
    parser.add_argument(
        "--output", type=str, default="report.json", help="Путь к выходному JSON-файлу"
    )
    args = parser.parse_args(argv)

    if args.last:
        try:
            args.since = parse_since(args.last)
        except ValueError as exc:
            logger.error(str(exc))
            return 1

    violations = load_violations(args.camera, args.since, args.until)
    generate_report(violations, args.output, args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
