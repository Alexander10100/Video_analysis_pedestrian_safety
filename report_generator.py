from __future__ import annotations

import json
import csv
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import threading


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


@dataclass
class ViolationEvent:
    timestamp: datetime
    track_id: int
    violation_type: str      # "road_trespass" | "red_light"
    zone_label: str
    age_label: str           # "adult" | "child"
    confidence: float
    frame_number: int
    bbox: tuple              # нормализованные координаты (nx1, ny1, nx2, ny2)
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class ViolationCollector:

    def __init__(self, max_events: int = 10000):
        self._lock = threading.Lock()
        self._events: list[ViolationEvent] = []
        self._max_events = max_events
        self._frame_counter = 0

    def set_frame_number(self, frame_num: int):
        self._frame_counter = frame_num

    def add_violation(
        self,
        track_id: int,
        violation_type: str,
        zone_label: str,
        age_label: str,
        confidence: float,
        bbox: tuple,
        note: str = "",
    ):
    
        event = ViolationEvent(
            timestamp=datetime.now(),
            track_id=track_id,
            violation_type=violation_type,
            zone_label=zone_label,
            age_label=age_label,
            confidence=confidence,
            frame_number=self._frame_counter,
            bbox=bbox,
            note=note,
        )

        with self._lock:
            self._events.append(event)
            # Ограничиваем размер
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    def get_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ViolationEvent]:
       
        with self._lock:
            events = self._events.copy()

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def clear_old(self, before_time: datetime):
        with self._lock:
            self._events = [e for e in self._events if e.timestamp >= before_time]

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class ReportGenerator:

    def __init__(self, collector: ViolationCollector, camera_id: str = ""):
        self.collector = collector
        self.camera_id = camera_id

    def generate(
        self,
        duration_minutes: int = 30,
        output_format: str = "json",
        output_path: Path | None = None,
        include_summary: bool = True,
    ) -> tuple[Path, dict]:
       
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)

        events = self.collector.get_events(start_time, end_time)

        # Группировка по трекам, чтобы не считать одно нарушение многократно
        track_groups = self._group_by_track(events)

        # Статистика
        stats = self._calculate_stats(track_groups, duration_minutes)

        # Имя файла
        if output_path is None:
            timestamp = end_time.strftime("%Y%m%d_%H%M%S")
            filename = f"violations_{self.camera_id}_{duration_minutes}min_{timestamp}.{output_format}"
            output_path = REPORTS_DIR / filename
        else:
            output_path = Path(output_path)

        # Генерируем файл
        if output_format == "json":
            self._write_json(output_path, track_groups, stats, start_time, end_time, include_summary)
        elif output_format == "txt":
            self._write_txt(output_path, track_groups, stats, start_time, end_time, include_summary)
        elif output_format == "csv":
            self._write_csv(output_path, track_groups)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        return output_path, stats

    def _group_by_track(self, events: list[ViolationEvent]) -> dict[int, list[ViolationEvent]]:
        groups = defaultdict(list)
        for event in events:
            groups[event.track_id].append(event)
        return dict(groups)

    def _calculate_stats(self, track_groups: dict, duration_minutes: int) -> dict:
        total_violations = 0
        by_type = defaultdict(int)
        by_zone = defaultdict(int)
        by_age = defaultdict(int)
        violation_details = []

        for track_id, events in track_groups.items():
            if not events:
                continue

            first = events[0]
            total_violations += 1

            by_type[first.violation_type] += 1
            by_zone[first.zone_label] += 1
            by_age[first.age_label] += 1

            violation_details.append({
                "track_id": track_id,
                "type": first.violation_type,
                "zone": first.zone_label,
                "age": first.age_label,
                "first_seen": first.timestamp,
                "last_seen": events[-1].timestamp,
                "duration_sec": (events[-1].timestamp - first.timestamp).total_seconds(),
                "frames_count": len(events),
            })

        return {
            "period_minutes": duration_minutes,
            "total_tracks": total_violations,
            "total_events": sum(len(e) for e in track_groups.values()),
            "by_violation_type": dict(by_type),
            "by_zone": dict(by_zone),
            "by_age": {
                "adults": by_age.get("adult", 0),
                "children": by_age.get("child", 0),
                "unknown": by_age.get("unknown", 0),
            },
            "violations": violation_details,
        }

    def _write_json(
        self,
        path: Path,
        track_groups: dict,
        stats: dict,
        start_time: datetime,
        end_time: datetime,
        include_summary: bool,
    ):
        """Записать отчет в JSON."""
        output = {
            "report_info": {
                "camera_id": self.camera_id,
                "generated_at": datetime.now().isoformat(),
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_minutes": stats["period_minutes"],
                },
            },
            "summary": stats if include_summary else None,
            "violations": {
                str(track_id): [e.to_dict() for e in events]
                for track_id, events in track_groups.items()
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _write_txt(
        self,
        path: Path,
        track_groups: dict,
        stats: dict,
        start_time: datetime,
        end_time: datetime,
        include_summary: bool,
    ):
        """Записать отчет в TXT"""
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О НАРУШЕНИЯХ ПДД\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Камера:          {self.camera_id or 'Не указана'}\n")
            f.write(f"Сгенерирован:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Период:          {start_time.strftime('%H:%M:%S')} — {end_time.strftime('%H:%M:%S')}\n")
            f.write(f"Интервал:        {stats['period_minutes']} мин.\n\n")

            if include_summary:
                f.write("-" * 40 + "\n")
                f.write("СВОДКА\n")
                f.write("-" * 40 + "\n")
                f.write(f"Всего нарушителей:     {stats['total_tracks']}\n")
                f.write(f"  Из них детей:        {stats['by_age']['children']}\n")
                f.write(f"  Из них взрослых:     {stats['by_age']['adults']}\n\n")

                f.write("По типам нарушений:\n")
                for vtype, count in stats['by_violation_type'].items():
                    label = "Переход на красный" if vtype == "red_light" else "Хождение по дороге"
                    f.write(f"  {label:<20} {count}\n")

                f.write("\nПо зонам:\n")
                for zone, count in stats['by_zone'].items():
                    f.write(f"  {zone:<20} {count}\n")
                f.write("\n")

            f.write("-" * 40 + "\n")
            f.write("ДЕТАЛИЗАЦИЯ ПО НАРУШИТЕЛЯМ\n")
            f.write("-" * 40 + "\n\n")

            for track_id, events in sorted(track_groups.items()):
                first = events[0]
                last = events[-1]
                duration = (last.timestamp - first.timestamp).total_seconds()

                vtype_label = "КРАСНЫЙ СВЕТ" if first.violation_type == "red_light" else "ПРОЕЗЖАЯ ЧАСТЬ"
                age_label = "Ребенок" if first.age_label == "child" else "Взрослый"

                f.write(f"[Нарушитель #{track_id}]\n")
                f.write(f"  Тип:          {vtype_label}\n")
                f.write(f"  Возраст:      {age_label}\n")
                f.write(f"  Зона:         {first.zone_label}\n")
                f.write(f"  Первое появление: {first.timestamp.strftime('%H:%M:%S')}\n")
                f.write(f"  Последнее появление: {last.timestamp.strftime('%H:%M:%S')}\n")
                f.write(f"  Длительность: {duration:.1f} сек ({len(events)} кадров)\n")
                if first.note:
                    f.write(f"  Примечание:   {first.note}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write(f"Всего записей о нарушениях: {stats['total_events']}\n")
            f.write("=" * 80 + "\n")

    def _write_csv(self, path: Path, track_groups: dict):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "track_id", "violation_type", "zone", "age",
                "first_seen", "last_seen", "duration_sec",
                "frames", "confidence", "note"
            ])

            for track_id, events in track_groups.items():
                first = events[0]
                last = events[-1]
                duration = (last.timestamp - first.timestamp).total_seconds()

                writer.writerow([
                    track_id,
                    first.violation_type,
                    first.zone_label,
                    first.age_label,
                    first.timestamp.isoformat(),
                    last.timestamp.isoformat(),
                    f"{duration:.1f}",
                    len(events),
                    f"{first.confidence:.2f}",
                    first.note,
                ])

    def generate_interval_report(
        self,
        intervals: list[int] = [30, 60, 90],
        output_format: str = "json",
    ) -> list[tuple[Path, dict]]:
        
        results = []
        for minutes in intervals:
            path, stats = self.generate(
                duration_minutes=minutes,
                output_format=output_format,
            )
            results.append((path, stats))
            print(f"[Report] Сгенерирован отчет за {minutes} мин: {path}")
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Интеграция с stream_detect.py
# ──────────────────────────────────────────────────────────────────────────────

_violation_collector: ViolationCollector | None = None


def get_collector() -> ViolationCollector:
    global _violation_collector
    if _violation_collector is None:
        _violation_collector = ViolationCollector()
    return _violation_collector


def get_report_generator(camera_id: str = "") -> ReportGenerator:
    return ReportGenerator(get_collector(), camera_id)