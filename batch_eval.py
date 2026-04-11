# ─────────────────────────────────────────────
#  tools/batch_eval.py
#  Batch evaluation on UCF-Crime dataset
#  Runs pipeline on N videos per class,
#  collects metrics, saves a report.
#
#  Usage:
#    python tools/batch_eval.py --folder data/videos --per-class 5
#    python tools/batch_eval.py --folder data/videos/Robbery --per-class 10
# ─────────────────────────────────────────────

import os
import sys
import json
import time
import argparse
from pathlib    import Path
from datetime   import datetime
from loguru     import logger
from rich.console import Console
from rich.table   import Table
from rich.progress import track

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings       import DATA_DIR, OUTPUT_DIR
from storage.database      import init_db, get_event_counts_by_level, get_recent_events
from storage.audit_trail   import export_events_csv, export_audit_csv
from pipeline.detector     import Detector
from pipeline.face_reid    import FaceReID
from pipeline.behaviour_engine  import BehaviourEngine
from pipeline.context_generator import build_context
from agents.worker_agent   import WorkerAgent
from agents.manager_agent  import ManagerAgent
from alerts.mqtt_publisher import MQTTPublisher
from storage.database      import insert_event, insert_tracked_object, audit_log

console = Console()

# UCF-Crime class names
UCF_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccident", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "Normal"
]


def discover_videos(folder: str, per_class: int = 5) -> list[tuple[str, str]]:
    """
    Returns [(video_path, class_label), ...]
    Detects class from folder name or parent folder.
    """
    folder  = Path(folder)
    results = []

    for cls in UCF_CLASSES:
        class_dir = folder / cls
        if class_dir.exists():
            videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
            for v in sorted(videos)[:per_class]:
                results.append((str(v), cls))

    # If no class subfolders, treat all mp4s in folder as one "unknown" class
    if not results:
        videos = list(folder.rglob("*.mp4")) + list(folder.rglob("*.avi"))
        for v in sorted(videos)[:per_class]:
            # Try to guess class from any parent folder
            cls = "Unknown"
            for part in v.parts:
                if part in UCF_CLASSES:
                    cls = part
                    break
            results.append((str(v), cls))

    return results


def process_single_video_headless(
    video_path: str,
    source_class: str,
    detector:  Detector,
    face_reid: FaceReID,
    worker:    WorkerAgent,
    manager:   ManagerAgent,
    mqtt:      MQTTPublisher,
) -> dict:
    """
    Run full pipeline on one video (no display).
    Returns per-video metrics dict.
    """
    import cv2
    video_name = Path(video_path).name
    label      = f"{source_class}/{video_name}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}")
        return {"video": label, "status": "failed", "events": 0}

    behaviour = BehaviourEngine(source_video=label)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 25
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    event_count  = 0
    frame_count  = 0
    t_start      = time.time()
    level_counts = {"RED": 0, "ORANGE": 0, "YELLOW": 0, "GREEN": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        persons, _ = detector.process_frame(frame)
        if not persons:
            continue

        for p in persons:
            insert_tracked_object({**p.to_dict(), "source_video": label})

        face_results: dict = {}
        for person in persons:
            fc = detector.get_face_crop(frame, person)
            if fc is not None:
                face_results[person.track_id] = face_reid.process(
                    frame, fc, person.track_id, label
                )

        events = behaviour.update(persons, face_results, h, w)

        for event in events:
            event_count += 1
            state   = behaviour.get_state(event.track_id)
            context = build_context(event, state,
                                    video_meta={"class": source_class, "fps": fps})
            description = worker.describe(context)
            verdict     = manager.classify(context, description)
            final_level = verdict.get("final_alert_level", event.alert_level)
            level_counts[final_level] = level_counts.get(final_level, 0) + 1

            insert_event({
                "event_id":       event.event_id,
                "timestamp":      event.timestamp,
                "event_type":     event.event_type,
                "alert_level":    final_level,
                "track_id":       event.track_id,
                "face_hash":      event.face_hash,
                "zone_id":        event.zone_id,
                "description":    description,
                "manager_verdict": json.dumps(verdict),
                "raw_json":       context,
                "source_video":   label,
            })

    cap.release()
    elapsed = time.time() - t_start

    return {
        "video":        label,
        "class":        source_class,
        "frames":       frame_count,
        "events":       event_count,
        "elapsed_s":    round(elapsed, 1),
        "fps_achieved": round(frame_count / max(elapsed, 0.1), 1),
        "status":       "ok",
        **level_counts,
    }


def run_batch(folder: str, per_class: int = 5, save_report: bool = True):
    """Main batch evaluation entry point."""
    init_db()

    videos = discover_videos(folder, per_class)
    if not videos:
        console.print(f"[red]No videos found in {folder}[/red]")
        console.print("Run: python data/sample_fetch.py --instructions")
        return

    console.print(f"\n[bold cyan]Batch Evaluation[/bold cyan]")
    console.print(f"  Videos found : {len(videos)}")
    console.print(f"  Folder       : {folder}")
    console.print(f"  Per class    : {per_class}\n")

    # Init shared components (reuse across videos for speed)
    detector  = Detector()
    face_reid = FaceReID()
    worker    = WorkerAgent()
    manager   = ManagerAgent()
    mqtt      = MQTTPublisher()

    results  = []
    all_start = time.time()

    for video_path, cls in track(videos, description="Processing videos..."):
        logger.info(f"  → {cls}/{Path(video_path).name}")
        audit_log("batch_video", f"{cls}/{Path(video_path).name}")
        metrics = process_single_video_headless(
            video_path, cls,
            detector, face_reid, worker, manager, mqtt
        )
        results.append(metrics)
        console.print(
            f"    [green]✓[/green] {Path(video_path).name[:35]:<35} | "
            f"events={metrics['events']:>3} | "
            f"🔴{metrics.get('RED',0)} 🟠{metrics.get('ORANGE',0)} "
            f"🟡{metrics.get('YELLOW',0)} | "
            f"{metrics['fps_achieved']}fps"
        )

    mqtt.disconnect()
    total_elapsed = time.time() - all_start

    # ── Summary table ──────────────────────────
    table = Table(title="Batch Evaluation Summary", style="bold")
    table.add_column("Class",    style="cyan",  no_wrap=True)
    table.add_column("Videos",  justify="right")
    table.add_column("Events",  justify="right")
    table.add_column("🔴 RED",  justify="right", style="red")
    table.add_column("🟠 ORA",  justify="right", style="yellow")
    table.add_column("🟡 YEL",  justify="right", style="yellow")
    table.add_column("Avg FPS", justify="right")

    # Aggregate by class
    class_stats: dict = {}
    for r in results:
        c = r["class"]
        if c not in class_stats:
            class_stats[c] = {"videos": 0, "events": 0, "RED": 0,
                              "ORANGE": 0, "YELLOW": 0, "fps": []}
        class_stats[c]["videos"] += 1
        class_stats[c]["events"] += r["events"]
        class_stats[c]["RED"]    += r.get("RED", 0)
        class_stats[c]["ORANGE"] += r.get("ORANGE", 0)
        class_stats[c]["YELLOW"] += r.get("YELLOW", 0)
        class_stats[c]["fps"].append(r["fps_achieved"])

    for cls, s in sorted(class_stats.items()):
        avg_fps = round(sum(s["fps"]) / len(s["fps"]), 1)
        table.add_row(
            cls, str(s["videos"]), str(s["events"]),
            str(s["RED"]), str(s["ORANGE"]), str(s["YELLOW"]),
            str(avg_fps)
        )

    console.print(table)
    console.print(
        f"\n  Total time : {total_elapsed:.1f}s | "
        f"Videos : {len(videos)} | "
        f"Total events : {sum(r['events'] for r in results)}"
    )

    # ── Save report ───────────────────────────
    if save_report:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(OUTPUT_DIR) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # JSON metrics
        json_path = report_dir / f"batch_metrics_{ts}.json"
        with open(json_path, "w") as f:
            json.dump({
                "run_timestamp":  ts,
                "folder":         folder,
                "per_class":      per_class,
                "total_elapsed_s": round(total_elapsed, 1),
                "videos":         results,
                "class_summary":  {
                    k: {**v, "fps": round(sum(v["fps"])/len(v["fps"]), 1)}
                    for k, v in class_stats.items()
                }
            }, f, indent=2)

        # CSV exports
        events_csv = export_events_csv(str(report_dir / f"events_{ts}.csv"))
        audit_csv  = export_audit_csv(str(report_dir / f"audit_{ts}.csv"))

        console.print(f"\n  [bold]Reports saved:[/bold]")
        console.print(f"    JSON    : {json_path}")
        console.print(f"    Events  : {events_csv}")
        console.print(f"    Audit   : {audit_csv}")

    console.print(f"\n  [bold cyan]Dashboard:[/bold cyan] streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation on UCF-Crime dataset")
    parser.add_argument("--folder",    type=str, default="data/videos",
                        help="Folder containing class subfolders (e.g. data/videos/)")
    parser.add_argument("--per-class", type=int, default=5,
                        help="Max videos per class to process (default: 5)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip saving the JSON/CSV report")
    args = parser.parse_args()

    run_batch(args.folder, args.per_class, save_report=not args.no_report)
