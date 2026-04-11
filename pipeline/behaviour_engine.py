# ─────────────────────────────────────────────
#  pipeline/behaviour_engine.py
#  Tracks per-object state across frames and
#  fires behavioural events when thresholds hit
# ─────────────────────────────────────────────

import time
import math
import uuid
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    LOITER_SECONDS, RUNNING_SPEED_PX_S, CROWD_DENSITY_COUNT,
    PERIMETER_MARGIN_PX, ABANDONED_FRAMES, FRAME_SKIP, AlertLevel
)
from pipeline.detector import TrackedPerson


# ══════════════════════════════════════════════
#  Per-object state
# ══════════════════════════════════════════════

@dataclass
class ObjectState:
    track_id:       int
    first_seen:     float = field(default_factory=time.time)
    last_seen:      float = field(default_factory=time.time)
    zone_id:        str   = None
    zone_entry_time: float = None    # when they entered current zone
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    face_hash:      str   = None
    is_repeat_face: bool  = False
    events_fired:   set   = field(default_factory=set)

    def update(self, person: "TrackedPerson"):
        self.last_seen = time.time()
        pos = (person.cx, person.cy, time.time())
        self.position_history.append(pos)

        # Zone change?
        if person.zone_id != self.zone_id:
            self.zone_id         = person.zone_id
            self.zone_entry_time = time.time()

    def time_in_zone(self) -> float:
        if self.zone_entry_time is None:
            return 0.0
        return time.time() - self.zone_entry_time

    def current_speed_px_s(self) -> float:
        """Estimate speed from last two positions."""
        if len(self.position_history) < 2:
            return 0.0
        x1, y1, t1 = self.position_history[-2]
        x2, y2, t2 = self.position_history[-1]
        dt = t2 - t1
        if dt < 1e-6:
            return 0.0
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist / dt

    def trajectory_points(self) -> list[tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.position_history]


# ══════════════════════════════════════════════
#  Behaviour event
# ══════════════════════════════════════════════

@dataclass
class BehaviourEvent:
    event_id:    str
    timestamp:   float
    event_type:  str
    alert_level: str
    track_id:    int
    face_hash:   str | None
    zone_id:     str | None
    details:     dict
    source_video: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id":    self.event_id,
            "timestamp":   self.timestamp,
            "event_type":  self.event_type,
            "alert_level": self.alert_level,
            "track_id":    self.track_id,
            "face_hash":   self.face_hash,
            "zone_id":     self.zone_id,
            "source_video": self.source_video,
            **self.details,
        }

    def to_context_json(self) -> dict:
        """Full structured context for the SLM worker agent."""
        return {
            "event_id":    self.event_id,
            "timestamp_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
            ),
            "alert_level": self.alert_level,
            "event_type":  self.event_type,
            "subject": {
                "track_id":     self.track_id,
                "face_hash":    self.face_hash,
                "is_known_face": self.face_hash is not None,
            },
            "location": {
                "zone_id": self.zone_id,
            },
            "source_video": self.source_video,
            "details":     self.details,
        }


# ══════════════════════════════════════════════
#  Behaviour Engine
# ══════════════════════════════════════════════

class BehaviourEngine:

    def __init__(self, source_video: str = ""):
        self.states: dict[int, ObjectState] = {}
        self.source_video  = source_video
        self.frame_number  = 0
        self._stale_limit  = 5.0   # seconds before removing unseen track
        self.zones_config  = None  # injected by main pipeline

    def _new_event(self, event_type: str, alert_level: str,
                   state: ObjectState, details: dict) -> BehaviourEvent:
        return BehaviourEvent(
            event_id     = str(uuid.uuid4())[:12],
            timestamp    = time.time(),
            event_type   = event_type,
            alert_level  = alert_level,
            track_id     = state.track_id,
            face_hash    = state.face_hash,
            zone_id      = state.zone_id,
            details      = details,
            source_video = self.source_video,
        )

    def update(self,
               persons: list[TrackedPerson],
               face_results: dict[int, dict],
               frame_h: int,
               frame_w: int) -> list[BehaviourEvent]:
        """
        Called every processed frame.
        persons      — list of TrackedPerson from detector
        face_results — {track_id: face_reid_result_dict}
        Returns list of BehaviourEvent fired this frame.
        """
        self.frame_number += 1
        now    = time.time()
        events = []
        active_ids = set()

        for person in persons:
            tid = person.track_id
            active_ids.add(tid)

            # ── Init or update state ───────────
            if tid not in self.states:
                self.states[tid] = ObjectState(track_id=tid)
            state = self.states[tid]
            state.update(person)

            # Attach face info if available
            if tid in face_results and face_results[tid]["face_found"]:
                fr = face_results[tid]
                state.face_hash    = fr["face_hash"]
                state.is_repeat_face = fr["is_repeat"]

            # ══ Check behaviours ══════════════

            # 1. REPEAT FACE → 🟡 YELLOW
            if (state.is_repeat_face
                    and state.face_hash
                    and f"repeat_{state.face_hash}" not in state.events_fired):
                state.events_fired.add(f"repeat_{state.face_hash}")
                fr = face_results.get(tid, {})
                events.append(self._new_event(
                    "repeat_face", AlertLevel.YELLOW, state,
                    {
                        "face_hash":    state.face_hash,
                        "seen_count":   fr.get("seen_count", 2),
                        "similarity":   fr.get("similarity"),
                        "message":      "Previously seen individual detected again.",
                    }
                ))

            # 2. LOITERING → 🟠 ORANGE
            loiter_threshold = LOITER_SECONDS
            if state.zone_id:
                # Check for per-zone override
                pass  # loaded in main.py if needed

            if (state.time_in_zone() > loiter_threshold
                    and f"loiter_{state.zone_id}" not in state.events_fired):
                state.events_fired.add(f"loiter_{state.zone_id}")
                events.append(self._new_event(
                    "loitering", AlertLevel.ORANGE, state,
                    {
                        "duration_seconds": round(state.time_in_zone(), 1),
                        "zone_id":          state.zone_id,
                        "message":          f"Person loitering in {state.zone_id} for "
                                            f"{state.time_in_zone():.0f}s.",
                    }
                ))

            # 3. RUNNING → 🟠 ORANGE
            speed = state.current_speed_px_s()
            if (speed > RUNNING_SPEED_PX_S
                    and "running" not in state.events_fired):
                state.events_fired.add("running")
                events.append(self._new_event(
                    "running", AlertLevel.ORANGE, state,
                    {
                        "speed_px_s": round(speed, 1),
                        "message":    f"Person running at {speed:.0f} px/s.",
                    }
                ))
            elif speed <= RUNNING_SPEED_PX_S * 0.6:
                # Reset so it can fire again if they run again
                state.events_fired.discard("running")

            # 4. RESTRICTED ZONE ENTRY → 🔴 RED
            if (state.zone_id == "Z3"
                    and f"restricted_entry_{state.zone_id}" not in state.events_fired):
                state.events_fired.add(f"restricted_entry_{state.zone_id}")
                events.append(self._new_event(
                    "restricted_zone_entry", AlertLevel.RED, state,
                    {
                        "zone_id": state.zone_id,
                        "message": "Person entered restricted zone.",
                    }
                ))

            # 5. PERIMETER BREACH → 🟠 ORANGE
            near_edge = (
                person.cx < PERIMETER_MARGIN_PX or
                person.cx > frame_w - PERIMETER_MARGIN_PX or
                person.cy < PERIMETER_MARGIN_PX or
                person.cy > frame_h - PERIMETER_MARGIN_PX
            )
            if (near_edge and "perimeter" not in state.events_fired):
                state.events_fired.add("perimeter")
                events.append(self._new_event(
                    "perimeter_breach", AlertLevel.ORANGE, state,
                    {
                        "position": (person.cx, person.cy),
                        "message":  "Person detected near frame perimeter.",
                    }
                ))

        # 6. CROWD SURGE → 🔴 RED (zone-level, not per-person)
        zone_counts: dict[str, int] = defaultdict(int)
        for p in persons:
            if p.zone_id:
                zone_counts[p.zone_id] += 1
        for zone_id, count in zone_counts.items():
            if count >= CROWD_DENSITY_COUNT:
                # Create a synthetic "crowd" event once per burst
                crowd_key = f"crowd_{zone_id}_{self.frame_number // 30}"
                events.append(BehaviourEvent(
                    event_id     = str(uuid.uuid4())[:12],
                    timestamp    = now,
                    event_type   = "crowd_surge",
                    alert_level  = AlertLevel.RED,
                    track_id     = -1,
                    face_hash    = None,
                    zone_id      = zone_id,
                    source_video = self.source_video,
                    details      = {
                        "person_count": count,
                        "zone_id":      zone_id,
                        "message":      f"Crowd surge detected in {zone_id}: {count} people.",
                    }
                ))

        # ── Prune stale tracks ─────────────────
        for tid in list(self.states.keys()):
            if tid not in active_ids:
                if now - self.states[tid].last_seen > self._stale_limit:
                    del self.states[tid]

        return events

    def get_state(self, track_id: int) -> ObjectState | None:
        return self.states.get(track_id)

    def active_count(self) -> int:
        return len(self.states)

    def summary(self) -> dict:
        return {
            "active_tracks": self.active_count(),
            "frame_number":  self.frame_number,
        }
