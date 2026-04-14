"""

Запуск:
    # Все нарушения за последние 24 часа для cam1
python report_generator.py --camera cam1 --last 24h --output report_cam1_today.json

# За конкретный час
python report_generator.py --camera cam1 --since "2026-04-13T09:00:00" --until "2026-04-13T10:00:00"

# Все камеры за всё время
python report_generator.py --output full_report.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

LOG_FILE = Path("violations_log.jsonl")


def load_violations(camera_id: str | None = None,
                    since: str | None = None,
                    until: str | None = None) -> list:
    """Загружает нарушения из лог-файла с фильтрами."""
    if not LOG_FILE.exists():
        print(f"[WARN] Лог нарушений не найден: {LOG_FILE}")
        print("       Запустите stream_detect.py, чтобы начать запись нарушений.")
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

    print(f"[INFO] Загружено {len(violations)} нарушений")
    return violations


def generate_report(violations: list, output: str, camera_id: str | None = None):
    """Генерирует JSON-отчёт """
    if not violations:
        print("[INFO] За указанный период нарушений не найдено.")
        empty_report = {
            "generated_at": datetime.now().isoformat(),
            "camera_id": camera_id or "all",
            "total_violations": 0,
            "unique_persons": 0,
            "confidence": {"average": 0.0, "min": 0.0, "max": 0.0},
            "summary": {"by_type": {}, "by_zone": {}, "by_age": {}},
            "timeline": [],
            "raw_violations": []
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(empty_report, f, ensure_ascii=False, indent=2)
        return

    by_type = defaultdict(int)
    by_zone = defaultdict(int)
    by_age = defaultdict(int)
    unique_tracks = set()          #для подсчёта уникальных людей
    confidence_stats = []

    timeline = defaultdict(lambda: defaultdict(int))

    for v in violations:
        tid = v.get("track_id")
        if tid is not None:
            unique_tracks.add(tid)

        vt = v["violation_type"]
        by_type[vt] += 1
        by_zone[v["zone_label"]] += 1
        by_age[v.get("age_label", "unknown")] += 1

        if "person_conf" in v:
            confidence_stats.append(v["person_conf"])

        # timeline по минутам
        dt = datetime.fromisoformat(v["timestamp"])
        minute = dt.replace(second=0, microsecond=0).isoformat()
        timeline[minute][vt] += 1

    # Статистика по уверенности
    avg_conf = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0.0
    min_conf = min(confidence_stats) if confidence_stats else 0.0
    max_conf = max(confidence_stats) if confidence_stats else 0.0

    report = {
        "generated_at": datetime.now().isoformat(),
        "camera_id": camera_id or "all",
        "total_violations": len(violations),
        "unique_persons": len(unique_tracks),           
        "confidence": {
            "average": round(avg_conf, 3),
            "min": round(min_conf, 3),
            "max": round(max_conf, 3)
        },
        "summary": {
            "by_type": dict(by_type),
            "by_zone": dict(by_zone),
            "by_age": dict(by_age),
        },
        "timeline": [
            {
                "timestamp": ts,
                "total": sum(counts.values()),
                "by_type": dict(counts)
            }
            for ts, counts in sorted(timeline.items())
        ],
        "raw_violations": violations
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Вывод в консоль
    print(f"\n[✓] Отчёт сохранён → {output}")
    print(f"   Камера                  : {camera_id or 'Все камеры'}")
    print(f"   Всего записей нарушений : {report['total_violations']}")
    print(f"   Уникальных нарушителей  : {report['unique_persons']}")
    print(f"   Средняя уверенность     : {report['confidence']['average']:.1%}")
    print(f"   Типы нарушений:")
    for t, cnt in sorted(report["summary"]["by_type"].items(), key=lambda x: x[1], reverse=True):
        print(f"     • {t:15} : {cnt}")

    # Сохраняем отчёт
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


    print(f"\n[✓] Отчёт успешно сохранён → {output}")
    print(f"   Камера          : {camera_id or 'Все камеры'}")
    print(f"   Всего нарушений : {report['total_violations']}")
    print(f"   Типы нарушений:")
    for t, cnt in report["summary"]["by_type"].items():
        print(f"     • {t:15} : {cnt}")
    print(f"   По возрасту:")
    for age, cnt in report["summary"]["by_age"].items():
        print(f"     • {age:15} : {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор отчётов по нарушениям ПДД")
    parser.add_argument("--camera", type=str, default=None,
                        help="ID камеры (например: cam1, cam_03)")
    parser.add_argument("--since", type=str,
                        help="Начало периода в формате ISO (2026-04-13T09:00:00)")
    parser.add_argument("--until", type=str,
                        help="Конец периода в формате ISO")
    parser.add_argument("--last", type=str,
                        help="Последние N часов или минут: 24h или 60m")
    parser.add_argument("--output", type=str, default="report.json",
                        help="Путь к выходному JSON-файлу")

    args = parser.parse_args()

    
    if args.last:
        try:
            if args.last.endswith("h"):
                hours = int(args.last[:-1])
                args.since = (datetime.now() - timedelta(hours=hours)).isoformat()
            elif args.last.endswith("m"):
                minutes = int(args.last[:-1])
                args.since = (datetime.now() - timedelta(minutes=minutes)).isoformat()
            else:
                print("[ERROR] Параметр --last должен заканчиваться на 'h' или 'm' (пример: 24h или 30m)")
                exit(1)
        except ValueError:
            print("[ERROR] Некорректное значение для --last")
            exit(1)

    # Загружаем и генерируем отчёт
    violations = load_violations(args.camera, args.since, args.until)
    generate_report(violations, args.output, args.camera)
