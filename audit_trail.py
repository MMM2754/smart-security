# ─────────────────────────────────────────────
#  storage/audit_trail.py
#  Human-readable audit trail export + reporting
# ─────────────────────────────────────────────

import csv
import json
import time
from datetime import datetime
from pathlib  import Path
from loguru   import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import get_audit_trail, get_recent_events, get_event_counts_by_level


def export_audit_csv(output_path: str = None) -> str:
    """Export full audit trail to a CSV file."""
    if output_path is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/audit_trail_{ts}.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trail = get_audit_trail(limit=10000)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_iso", "action", "details", "source"])
        writer.writeheader()
        for row in trail:
            writer.writerow({
                "timestamp_iso": datetime.fromtimestamp(row["timestamp"]).isoformat(),
                "action":        row["action"],
                "details":       row["details"] or "",
                "source":        row["source"],
            })

    logger.info(f"Audit trail exported: {output_path} ({len(trail)} entries)")
    return output_path


def export_events_csv(output_path: str = None) -> str:
    """Export all security events to CSV."""
    if output_path is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/events_{ts}.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    events = get_recent_events(limit=10000)

    fieldnames = [
        "event_id", "timestamp_iso", "event_type", "alert_level",
        "track_id", "face_hash", "zone_id", "description",
        "manager_verdict_summary", "source_video"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            # Summarise verdict JSON to a short string
            verdict_summary = ""
            if ev.get("manager_verdict"):
                try:
                    v = json.loads(ev["manager_verdict"])
                    verdict_summary = (
                        f"{v.get('event_category','?')} | "
                        f"action={v.get('recommended_action','?')} | "
                        f"conf={v.get('confidence','?')}"
                    )
                except Exception:
                    verdict_summary = str(ev["manager_verdict"])[:80]

            writer.writerow({
                "event_id":               ev.get("event_id", ""),
                "timestamp_iso":          datetime.fromtimestamp(ev["timestamp"]).isoformat(),
                "event_type":             ev.get("event_type", ""),
                "alert_level":            ev.get("alert_level", ""),
                "track_id":               ev.get("track_id", ""),
                "face_hash":              (ev.get("face_hash") or "")[:12],
                "zone_id":                ev.get("zone_id", ""),
                "description":            ev.get("description", ""),
                "manager_verdict_summary": verdict_summary,
                "source_video":           ev.get("source_video", ""),
            })

    logger.info(f"Events exported: {output_path} ({len(events)} events)")
    return output_path


def print_session_report():
    """Print a summary report to console."""
    counts = get_event_counts_by_level()
    events = get_recent_events(limit=1000)
    total  = sum(counts.values())

    # Event type breakdown
    type_counts: dict[str, int] = {}
    for ev in events:
        t = ev.get("event_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n" + "═" * 50)
    print("  SECURITY SESSION REPORT")
    print("═" * 50)
    print(f"  Total events : {total}")
    print(f"  🔴 RED       : {counts.get('RED', 0)}")
    print(f"  🟠 ORANGE    : {counts.get('ORANGE', 0)}")
    print(f"  🟡 YELLOW    : {counts.get('YELLOW', 0)}")
    print(f"  🟢 GREEN     : {counts.get('GREEN', 0)}")
    print("─" * 50)
    print("  By Event Type:")
    for etype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:<28} {cnt}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    print_session_report()
    a = export_audit_csv()
    e = export_events_csv()
    print(f"Audit CSV  : {a}")
    print(f"Events CSV : {e}")
