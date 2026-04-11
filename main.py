# ─────────────────────────────────────────────
#  main.py
#  Entry point — runs the full pipeline:
#  Video → Detect → Track → Behaviour → Face Re-ID
#       → Context JSON → Worker Agent → Manager Agent
#       → MQTT Alert → SQLite Storage
# ─────────────────────────────────────────────

import cv2
import time
import json
import argparse
import threading
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table  import Table
from rich        import print as rprint

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings        import DATA_DIR, OUTPUT_DIR, LOG_DIR, AlertLevel
from pipeline.detector      import Detector
from pipeline.face_reid     import FaceReID
from pipeline.behaviour_engine import BehaviourEngine
from pipeline.context_generator import build_context, context_to_prompt
from agents.worker_agent    import WorkerAgent
from agents.manager_agent   import ManagerAgent
from alerts.mqtt_publisher  import MQTTPublisher
from storage.database       import (
    init_db, insert_event, insert_tracked_object,
    audit_log, get_event_counts_by_level
)

console = Console()


# ══════════════════════════════════════════════
#  Setup
# ══════════════════════════════════════════════

def setup():
    """Create dirs, init DB, configure logger."""
    for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR,
              os.path.join(OUTPUT_DIR, "faces"),
              os.path.join(OUTPUT_DIR, "events")]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Loguru config
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(
        os.path.join(LOG_DIR, "security_{time:YYYY-MM-DD}.log"),
        rotation="1 day", retention="7 days", level="DEBUG"
    )

    init_db()
    audit_log("system_start", "Pipeline started")
    logger.success("Smart Security System initialised")


# ══════════════════════════════════════════════
#  Process a single video
# ══════════════════════════════════════════════

def process_video(
    video_path:   str,
    show_display: bool = True,
    save_output:  bool = False,
):
    video_name = Path(video_path).name
    logger.info(f"Processing: {video_name}")
    audit_log("video_start", video_name)

    # ── Init components ───────────────────────
    detector    = Detector()
    face_reid   = FaceReID()
    behaviour   = BehaviourEngine(source_video=video_name)
    worker      = WorkerAgent()
    manager     = ManagerAgent()
    mqtt        = MQTTPublisher()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"  Resolution: {w}x{h} @ {fps:.1f}fps | {total} frames")

    writer = None
    if save_output:
        out_path = os.path.join(OUTPUT_DIR, "events", f"annotated_{video_name}")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    event_count = 0
    t_start     = time.time()

    # ── Frame loop ────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ── Step 1: Detect + Track ─────────────
        persons, annotated_frame = detector.process_frame(frame)
        if not persons:
            if writer:
                writer.write(annotated_frame)
            if show_display:
                cv2.imshow("Smart Security", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # ── Step 2: Store tracked objects ──────
        for p in persons:
            insert_tracked_object({**p.to_dict(), "source_video": video_name})

        # ── Step 3: Face Re-ID ─────────────────
        face_results: dict[int, dict] = {}
        for person in persons:
            face_crop = detector.get_face_crop(frame, person)
            if face_crop is not None:
                reid_result = face_reid.process(
                    annotated_frame, face_crop,
                    person.track_id, video_name
                )
                face_results[person.track_id] = reid_result
                # Use blurred frame for display
                if reid_result.get("blurred_frame") is not None:
                    annotated_frame = reid_result["blurred_frame"]

        # ── Step 4: Behaviour analysis ─────────
        events = behaviour.update(persons, face_results, h, w)

        # ── Step 5: Process each event ─────────
        for event in events:
            event_count += 1
            state   = behaviour.get_state(event.track_id)
            context = build_context(event, state, video_meta={"fps": fps, "resolution": f"{w}x{h}"})

            # Worker agent: JSON → natural language
            description = worker.describe(context)

            # Manager agent: classification + verdict
            verdict = manager.classify(context, description)
            final_level = verdict.get("final_alert_level", event.alert_level)

            # Store in DB
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
                "source_video":   video_name,
            })

            # Mark face as warned if yellow
            if final_level == AlertLevel.YELLOW and event.face_hash:
                face_reid.mark_warned(event.face_hash)

            # Publish via MQTT
            mqtt.publish_event(event.to_dict(), verdict, description)

            # Overlay alert on frame
            annotated_frame = _draw_alert(annotated_frame, final_level, description)

        # Stats overlay
        elapsed = time.time() - t_start
        fps_actual = frame_count / max(elapsed, 0.1)
        cv2.putText(annotated_frame,
                    f"Events: {event_count} | FPS: {fps_actual:.1f} | Faces: {face_reid.cache_size()}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer:
            writer.write(annotated_frame)

        if show_display:
            cv2.imshow("Smart Security — Press Q to quit", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit.")
                break

    # ── Cleanup ───────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    mqtt.disconnect()

    elapsed = time.time() - t_start
    counts  = get_event_counts_by_level()
    audit_log("video_complete", f"{video_name} | {event_count} events | {elapsed:.1f}s")

    _print_summary(video_name, frame_count, event_count, elapsed, counts)


def _draw_alert(frame, level: str, description: str):
    """Draw a coloured alert banner on the frame."""
    colors = {
        AlertLevel.RED:    (0, 0, 220),
        AlertLevel.ORANGE: (0, 140, 255),
        AlertLevel.YELLOW: (0, 210, 255),
        AlertLevel.GREEN:  (0, 180, 0),
    }
    color = colors.get(level, (200, 200, 200))
    h, w  = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 45), (w, h - 20), color, -1)
    text  = f"[{level}] {description[:80]}"
    cv2.putText(frame, text, (5, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _print_summary(name, frames, events, elapsed, counts):
    table = Table(title=f"✅ Pipeline Complete — {name}", style="bold")
    table.add_column("Metric",    style="cyan")
    table.add_column("Value",     style="white")
    table.add_row("Frames processed", str(frames))
    table.add_row("Total events",     str(events))
    table.add_row("Time elapsed",     f"{elapsed:.1f}s")
    table.add_row("🔴 RED",           str(counts.get("RED", 0)))
    table.add_row("🟠 ORANGE",        str(counts.get("ORANGE", 0)))
    table.add_row("🟡 YELLOW",        str(counts.get("YELLOW", 0)))
    table.add_row("🟢 GREEN",         str(counts.get("GREEN", 0)))
    console.print(table)
    console.print(f"\n📊 Dashboard: [bold cyan]streamlit run dashboard/app.py[/bold cyan]")


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical SLM Smart Security System"
    )
    parser.add_argument("--video",    type=str, help="Path to video file (.mp4/.avi)")
    parser.add_argument("--folder",   type=str, help="Process all videos in a folder")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no OpenCV window)")
    parser.add_argument("--save",     action="store_true",   help="Save annotated output video")
    parser.add_argument("--placeholder", action="store_true",help="Generate + run a placeholder test video")
    args = parser.parse_args()

    setup()

    if args.placeholder:
        from data.videos.data.sample_fetch import create_placeholder_video
        test_vid = os.path.join(DATA_DIR, "test_placeholder.mp4")
        create_placeholder_video(test_vid)
        process_video(test_vid, show_display=not args.no_display, save_output=args.save)

    elif args.video:
        if not Path(args.video).exists():
            logger.error(f"File not found: {args.video}")
            sys.exit(1)
        process_video(args.video, show_display=not args.no_display, save_output=args.save)

    elif args.folder:
        folder = Path(args.folder)
        videos = list(folder.rglob("*.mp4")) + list(folder.rglob("*.avi"))
        if not videos:
            logger.error(f"No mp4/avi files found in {args.folder}")
            sys.exit(1)
        logger.info(f"Found {len(videos)} videos to process.")
        for v in videos:
            process_video(str(v), show_display=not args.no_display, save_output=args.save)

    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("  python main.py --placeholder            # generates & runs test video")
        print("  python main.py --video data/videos/x.mp4")
        print("  python main.py --folder data/videos/Robbery/")
        print("  streamlit run dashboard/app.py          # open dashboard")


if __name__ == "__main__":
    main()
