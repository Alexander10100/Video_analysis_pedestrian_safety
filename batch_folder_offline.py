from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from age_classifier import AgeTracker, BboxEMA
from offline_detect import (
    VIDEO_EXTS,
    _build_pipeline,
    _print_stats,
    load_yolo,
    process_video,
)
from traffic_light import TrafficLightAnalyzer
from violation_detector import ViolationDetector

ZONES_FILE = Path("zones.json")
COOLDOWN_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline by folders: annotate zones on the first video of each folder, "
            "then process all videos, save annotated outputs, and build per-folder plus combined reports."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folders", nargs="+", help="List of video folders to process")
    group.add_argument(
        "--root",
        help="Root folder; each subfolder with videos will be treated as a separate camera",
    )

    parser.add_argument("--annotated-root", default="annotated_batches")
    parser.add_argument("--reports-root", default="reports_batches")
    parser.add_argument("--combined-report-name", default="combined_report.json")
    parser.add_argument(
        "--replace-combined",
        action="store_true",
        help="Replace the existing combined report instead of merging into it",
    )
    parser.add_argument("--model", default="m", choices=["n", "s", "m", "l", "x"])
    parser.add_argument(
        "--device",
        default="auto",
        help="YOLO device: auto, cpu, or CUDA device index such as 0",
    )
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--preload-video",
        action="store_true",
        help="Load each video into RAM before processing",
    )
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=64,
        help="Frames buffered for asynchronous output video writing; 0 disables async writing",
    )
    parser.add_argument(
        "--reader-queue-size",
        type=int,
        default=0,
        help="Frames buffered by background video reader; 0 disables async reading",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=16,
        help="Frames per YOLO batch when video is preloaded and detect-every is 1",
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Do not render/save annotated mp4; fastest mode for reports only",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Do not open the web UI for manual zone annotation",
    )
    parser.add_argument(
        "--force-annotation",
        action="store_true",
        help="Open the web UI even if zones for the folder already exist",
    )
    return parser.parse_args()


def list_video_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def collect_video_folders(args: argparse.Namespace) -> list[Path]:
    if args.folders:
        candidates = [Path(p) for p in args.folders]
    else:
        root = Path(args.root)
        candidates = [p for p in root.iterdir() if p.is_dir()]

    folders: list[Path] = []
    for folder in candidates:
        if folder.is_dir() and list_video_files(folder):
            folders.append(folder)
    return sorted(folders)


def load_zones_db() -> dict:
    if not ZONES_FILE.exists():
        return {"cameras": {}}
    try:
        raw = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"cameras": {}}
    if isinstance(raw, dict) and "cameras" in raw:
        return raw
    return {"cameras": {}}


def ensure_camera_entry(camera_id: str) -> None:
    data = load_zones_db()
    cameras = data.setdefault("cameras", {})
    if camera_id not in cameras:
        cameras[camera_id] = {"label": camera_id, "zones": []}
        ZONES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def camera_has_zones(camera_id: str) -> bool:
    data = load_zones_db()
    cam = data.get("cameras", {}).get(camera_id, {})
    return bool(cam.get("zones"))


def prompt_annotation(folder: Path, camera_id: str, port: int) -> None:
    ensure_camera_entry(camera_id)
    cmd = [
        sys.executable,
        "stream_detect.py",
        "--folder",
        str(folder),
        "--port",
        str(port),
        "--camera",
        camera_id,
    ]
    proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent)
    try:
        print()
        print(f"[ANNOTATE] Folder: {folder}")
        print(f"[ANNOTATE] Camera ID: {camera_id}")
        print(f"[ANNOTATE] Open http://localhost:{port}")
        print("[ANNOTATE] Check the first video, adjust zones and save them in the UI.")
        input("[ANNOTATE] Press Enter here when markup for this folder is finished...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def build_report_data(violations: list[dict], camera_id: str) -> dict:
    by_type = defaultdict(int)
    by_zone = defaultdict(int)
    by_age = defaultdict(int)
    timeline = defaultdict(lambda: defaultdict(int))
    unique_tracks = set()
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
        except Exception:
            pass

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


def summarize_processing_stats(items: list[dict]) -> dict:
    total = {
        "total_frames": 0,
        "detected_frames": 0,
        "total_persons": 0,
        "total_violations": 0,
        "total_adults": 0,
        "total_children": 0,
        "inference_ms_sum": 0.0,
        "elapsed_sec": 0.0,
    }
    for item in items:
        for key in total:
            total[key] += item.get(key, 0)
    return total


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_violation_logger(camera_id: str, video_started_at: float, sink: list[dict]):
    cooldown: dict[int, float] = {}

    def _logger(violations: list, video_second: float) -> None:
        for item in violations:
            if item.violation == "none" or item.track_id < 0:
                continue
            last_logged = cooldown.get(item.track_id, -1e9)
            if video_second - last_logged < COOLDOWN_SECONDS:
                continue
            cooldown[item.track_id] = video_second
            sink.append(
                {
                    "timestamp": datetime.fromtimestamp(video_started_at + video_second).isoformat(),
                    "track_id": item.track_id,
                    "camera_id": camera_id,
                    "violation_type": item.violation,
                    "zone_label": item.zone_label,
                    "note": item.note or "",
                    "age_label": item.age_label,
                    "person_conf": float(item.conf),
                    "age_conf": float(item.age_conf),
                }
            )

    return _logger


def process_folder_job(
    folder_str: str,
    annotated_root_str: str,
    reports_root_str: str,
    model_size: str,
    device: str,
    conf: float,
    imgsz: int,
    detect_every: int,
    max_frames: int,
    preload_video: bool,
    writer_queue_size: int,
    inference_batch_size: int,
    write_annotated: bool,
    reader_queue_size: int,
) -> dict:
    folder = Path(folder_str)
    annotated_root = Path(annotated_root_str)
    reports_root = Path(reports_root_str)

    camera_id = folder.name
    videos = list_video_files(folder)
    if not videos:
        return {"camera_id": camera_id, "skipped": True}

    print(f"[START] {camera_id}: {len(videos)} videos")

    model = load_yolo(model_size, device=device)
    zone_mgr, _, _ = _build_pipeline(camera_id)
    age_tracker = AgeTracker(
        window=15,
        min_votes=5,
        flip_threshold=0.70,
        warmup_scale=0.50,
    )
    bbox_ema = BboxEMA(alpha=0.35)
    age_clf = None

    out_dir = annotated_root / camera_id
    report_dir = reports_root / camera_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    total_stat = {
        "total_frames": 0,
        "detected_frames": 0,
        "total_persons": 0,
        "total_violations": 0,
        "total_adults": 0,
        "total_children": 0,
        "inference_ms_sum": 0.0,
        "elapsed_sec": 0.0,
    }
    folder_violations: list[dict] = []
    video_reports: list[dict] = []

    for video_path in videos:
        output_path = out_dir / f"{video_path.stem}.mp4"
        tl_analyzer = TrafficLightAnalyzer()
        viol_det = ViolationDetector(zone_mgr, tl_analyzer)
        video_started_at = time.time()

        stat = process_video(
            input_path=video_path,
            output_path=output_path,
            model=model,
            zone_mgr=zone_mgr,
            tl_analyzer=tl_analyzer,
            viol_det=viol_det,
            age_clf=age_clf,
            age_tracker=age_tracker,
            bbox_ema=bbox_ema,
            camera_id=camera_id,
            conf=conf,
            imgsz=imgsz,
            detect_every=detect_every,
            max_frames=max_frames,
            violation_callback=make_violation_logger(camera_id, video_started_at, folder_violations),
            preload_video=preload_video,
            writer_queue_size=writer_queue_size,
            inference_batch_size=inference_batch_size,
            write_annotated=write_annotated,
            reader_queue_size=reader_queue_size,
        )

        if not stat:
            continue

        _print_stats(stat, video_path, output_path)
        video_reports.append(
            {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "annotated_path": str(output_path) if write_annotated else None,
                "processing_stats": stat,
            }
        )
        for key in total_stat:
            total_stat[key] += stat.get(key, 0)

    report_data = build_report_data(folder_violations, camera_id)
    report_data["processing_stats"] = total_stat
    report_data["source_folder"] = str(folder)
    report_data["annotated_folder"] = str(out_dir)

    write_json(report_dir / "report.json", report_data)
    (report_dir / "violations.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in folder_violations),
        encoding="utf-8",
    )

    print(f"[DONE] {camera_id}: violations={report_data['total_violations']}")

    return {
        "camera_id": camera_id,
        "skipped": False,
        "report": report_data,
        "report_path": str(report_dir / "report.json"),
        "video_reports": video_reports,
    }


def run_jobs(
    folders: list[Path],
    annotated_root: Path,
    reports_root: Path,
    model_size: str,
    device: str,
    conf: float,
    imgsz: int,
    detect_every: int,
    max_frames: int,
    preload_video: bool,
    writer_queue_size: int,
    inference_batch_size: int,
    write_annotated: bool,
    reader_queue_size: int,
    workers: int,
) -> list[dict]:
    jobs = [
        (
            str(folder),
            str(annotated_root),
            str(reports_root),
            model_size,
            device,
            conf,
            imgsz,
            detect_every,
            max_frames,
            preload_video,
            writer_queue_size,
            inference_batch_size,
            write_annotated,
            reader_queue_size,
        )
        for folder in folders
    ]

    if workers <= 1:
        return [process_folder_job(*job) for job in jobs]

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_folder_job, *job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def build_combined_report(results: list[dict]) -> dict:
    all_violations: list[dict] = []
    by_folder: dict[str, dict] = {}
    all_videos: list[dict] = []

    for item in results:
        if item.get("skipped") or "report" not in item:
            continue
        report = item["report"]
        camera_id = item["camera_id"]
        by_folder[camera_id] = {
            "report_path": item["report_path"],
            "total_violations": report["total_violations"],
            "unique_persons": report["unique_persons"],
            "summary": report["summary"],
            "processing_stats": report.get("processing_stats", {}),
        }
        all_violations.extend(report.get("raw_violations", []))
        all_videos.extend(item.get("video_reports", []))

    combined = build_report_data(all_violations, "all")
    combined["generated_at"] = datetime.now().isoformat()
    combined["folders_count"] = len(by_folder)
    combined["folders"] = by_folder
    combined["videos_count"] = len(all_videos)
    combined["videos"] = all_videos
    combined["processing_stats"] = summarize_processing_stats(
        [
            folder_info.get("processing_stats", {})
            for folder_info in by_folder.values()
        ]
    )
    return combined


def merge_combined_reports(existing: dict, new_data: dict) -> dict:
    existing_violations = existing.get("raw_violations", [])
    new_violations = new_data.get("raw_violations", [])
    merged_violations = existing_violations + new_violations

    merged = build_report_data(merged_violations, "all")
    merged["generated_at"] = datetime.now().isoformat()

    merged_folders = {}
    merged_folders.update(existing.get("folders", {}))
    merged_folders.update(new_data.get("folders", {}))
    merged["folders"] = merged_folders
    merged["folders_count"] = len(merged_folders)

    merged_videos = existing.get("videos", []) + new_data.get("videos", [])
    merged["videos"] = merged_videos
    merged["videos_count"] = len(merged_videos)

    merged["processing_stats"] = summarize_processing_stats(
        [
            existing.get("processing_stats", {}),
            new_data.get("processing_stats", {}),
        ]
    )
    return merged


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    folders = collect_video_folders(args)
    if not folders:
        print("[ERROR] No folders with video files were found.")
        sys.exit(1)

    annotated_root = Path(args.annotated_root)
    reports_root = Path(args.reports_root)
    detect_every = max(1, args.detect_every)
    workers = max(1, min(args.workers, len(folders)))

    print(f"[INFO] Folders found: {len(folders)}")
    for folder in folders:
        print(f"  - {folder}")

    for folder in folders:
        camera_id = folder.name
        need_annotation = args.force_annotation or not camera_has_zones(camera_id)
        if need_annotation and not args.skip_annotation:
            prompt_annotation(folder, camera_id, args.port)

    results = run_jobs(
        folders=folders,
        annotated_root=annotated_root,
        reports_root=reports_root,
        model_size=args.model,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        detect_every=detect_every,
        max_frames=args.max_frames,
        preload_video=args.preload_video,
        writer_queue_size=args.writer_queue_size,
        inference_batch_size=args.inference_batch_size,
        write_annotated=not args.no_annotated_video,
        reader_queue_size=args.reader_queue_size,
        workers=workers,
    )

    combined_report = build_combined_report(results)
    combined_report_path = reports_root / args.combined_report_name
    if not args.replace_combined:
        existing_report = load_json_if_exists(combined_report_path)
        if existing_report:
            combined_report = merge_combined_reports(existing_report, combined_report)
    write_json(combined_report_path, combined_report)

    print(f"[OK] Batch processing finished. Workers: {workers}")
    print(f"[OK] Combined report: {combined_report_path}")


if __name__ == "__main__":
    main()
