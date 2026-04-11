# ─────────────────────────────────────────────
#  pipeline/context_generator.py
#  Assembles the final structured JSON context
#  that gets fed to the Worker SLM agent
# ─────────────────────────────────────────────

import time
import json
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.behaviour_engine import BehaviourEvent, ObjectState


def build_context(
    event:        BehaviourEvent,
    state:        ObjectState | None,
    video_meta:   dict = None,
    extra:        dict = None,
) -> dict:
    """
    Assembles the full JSON context passed to the Worker agent.

    Structure mirrors the diagram:
      behavioural features → JSON metadata → three-tier agent pipeline
    """

    # ── Subject block ──────────────────────────
    subject = {
        "track_id":      event.track_id,
        "face_hash":     event.face_hash,
        "is_known_face": event.face_hash is not None,
    }

    if state:
        subject.update({
            "dwell_seconds":    round(state.time_in_zone(), 1),
            "speed_px_s":       round(state.current_speed_px_s(), 1),
            "trajectory_length": len(state.position_history),
            "first_seen_ago_s": round(time.time() - state.first_seen, 1),
        })

    # ── Location block ─────────────────────────
    location = {
        "zone_id":    event.zone_id,
        "zone_name":  _zone_name(event.zone_id),
    }

    # ── Video metadata block ───────────────────
    if video_meta is None:
        video_meta = {}
    video_block = {
        "source":     event.source_video or "unknown",
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.timestamp)),
        **video_meta,
    }

    # ── Full context object ────────────────────
    context = {
        "schema_version": "1.0",
        "event_id":       event.event_id,
        "event_type":     event.event_type,
        "alert_level":    event.alert_level,
        "subject":        subject,
        "location":       location,
        "video":          video_block,
        "details":        event.details,
    }

    if extra:
        context["extra"] = extra

    return context


def context_to_prompt(context: dict) -> str:
    """
    Format the context dict into a clean JSON string
    suitable for the Worker agent prompt.
    """
    return json.dumps(context, indent=2, ensure_ascii=False)


def _zone_name(zone_id: str | None) -> str:
    names = {
        "Z1": "Entry / Exit",
        "Z2": "Open Area",
        "Z3": "Restricted Zone",
        "Z4": "Perimeter",
    }
    return names.get(zone_id, "Unknown Zone") if zone_id else "No Zone"
