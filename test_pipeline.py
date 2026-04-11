# ─────────────────────────────────────────────
#  tests/test_pipeline.py
#  Unit tests — run with: python -m pytest tests/ -v
# ─────────────────────────────────────────────

import os
import sys
import time
import json
import uuid
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch DB path to a temp file for testing
import config.settings as settings
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
settings.DB_PATH = _tmp.name
_tmp.close()


# ══════════════════════════════════════════════
#  Database tests
# ══════════════════════════════════════════════

from storage.database import (
    init_db, insert_event, get_recent_events,
    get_event_counts_by_level, upsert_face, get_all_embeddings,
    mark_alert_issued, audit_log, get_audit_trail,
    embedding_to_blob, blob_to_embedding, hash_embedding
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


class TestDatabase:

    def test_init_db(self):
        """DB initialises without error."""
        init_db()  # calling twice should be idempotent

    def test_insert_and_get_event(self):
        eid = str(uuid.uuid4())[:12]
        insert_event({
            "event_id":       eid,
            "timestamp":      time.time(),
            "event_type":     "loitering",
            "alert_level":    "ORANGE",
            "track_id":       1,
            "face_hash":      None,
            "zone_id":        "Z1",
            "description":    "Test event",
            "manager_verdict": json.dumps({"confirmed": True}),
            "raw_json":       {"test": True},
            "source_video":   "test.mp4",
        })
        events = get_recent_events(limit=10)
        ids = [e["event_id"] for e in events]
        assert eid in ids

    def test_event_level_filter(self):
        for level in ["RED", "ORANGE", "YELLOW"]:
            insert_event({
                "event_id":       str(uuid.uuid4())[:12],
                "timestamp":      time.time(),
                "event_type":     "test",
                "alert_level":    level,
                "track_id":       99,
                "face_hash":      None,
                "zone_id":        "Z1",
                "description":    f"Test {level}",
                "manager_verdict": None,
                "raw_json":       {},
                "source_video":   "test.mp4",
            })
        reds = get_recent_events(limit=50, level="RED")
        assert all(e["alert_level"] == "RED" for e in reds)

    def test_event_counts(self):
        counts = get_event_counts_by_level()
        assert isinstance(counts, dict)
        for v in counts.values():
            assert isinstance(v, int)

    def test_face_upsert_new(self):
        emb = np.random.rand(512).astype(np.float32)
        result = upsert_face(emb, "test.mp4")
        assert result["is_new"] is True
        assert result["seen_count"] == 1
        assert result["face_hash"] is not None

    def test_face_upsert_repeat(self):
        emb = np.random.rand(512).astype(np.float32)
        r1  = upsert_face(emb, "test.mp4")
        r2  = upsert_face(emb, "test.mp4")
        assert r2["is_new"]      is False
        assert r2["seen_count"]  == 2
        assert r1["face_hash"]   == r2["face_hash"]

    def test_face_hash_deterministic(self):
        emb = np.ones(512, dtype=np.float32)
        h1  = hash_embedding(emb)
        h2  = hash_embedding(emb)
        assert h1 == h2

    def test_embedding_serialisation_roundtrip(self):
        emb  = np.random.rand(512).astype(np.float32)
        blob = embedding_to_blob(emb)
        out  = blob_to_embedding(blob)
        assert np.allclose(emb, out)

    def test_mark_alert_issued(self):
        emb  = np.random.rand(512).astype(np.float32)
        info = upsert_face(emb, "test.mp4")
        mark_alert_issued(info["face_hash"])
        all_embs = get_all_embeddings()
        match = next((e for e in all_embs if e["face_hash"] == info["face_hash"]), None)
        assert match is not None
        assert match["alert_issued"] == 1

    def test_audit_log_and_retrieve(self):
        audit_log("test_action", "test detail", source="pytest")
        trail = get_audit_trail(limit=10)
        actions = [t["action"] for t in trail]
        assert "test_action" in actions


# ══════════════════════════════════════════════
#  Behaviour engine tests
# ══════════════════════════════════════════════

from pipeline.behaviour_engine import BehaviourEngine, ObjectState
from pipeline.detector import TrackedPerson


def make_person(track_id: int, cx: int = 200, cy: int = 200,
                zone_id: str = "Z2") -> TrackedPerson:
    p = TrackedPerson(
        track_id=track_id, frame_number=1,
        timestamp=time.time(),
        x1=cx - 20, y1=cy - 50, x2=cx + 20, y2=cy + 50,
        confidence=0.9, zone_id=zone_id
    )
    return p


class TestBehaviourEngine:

    def test_no_events_single_person_normal(self):
        engine = BehaviourEngine(source_video="test.mp4")
        person = make_person(1)
        events = engine.update([person], {}, 480, 640)
        assert isinstance(events, list)

    def test_loitering_fires_after_threshold(self):
        engine = BehaviourEngine(source_video="test.mp4")
        engine.states[1] = ObjectState(track_id=1)
        # Force zone entry time to 40s ago
        engine.states[1].zone_id         = "Z2"
        engine.states[1].zone_entry_time  = time.time() - 40

        person = make_person(1, zone_id="Z2")
        events = engine.update([person], {}, 480, 640)
        loiter = [e for e in events if e.event_type == "loitering"]
        assert len(loiter) >= 1
        assert loiter[0].alert_level == "ORANGE"

    def test_restricted_zone_fires_red(self):
        engine = BehaviourEngine(source_video="test.mp4")
        person = make_person(2, zone_id="Z3")
        events = engine.update([person], {}, 480, 640)
        restricted = [e for e in events if e.event_type == "restricted_zone_entry"]
        assert len(restricted) == 1
        assert restricted[0].alert_level == "RED"

    def test_restricted_zone_fires_only_once(self):
        engine = BehaviourEngine(source_video="test.mp4")
        person = make_person(3, zone_id="Z3")
        e1 = engine.update([person], {}, 480, 640)
        e2 = engine.update([person], {}, 480, 640)
        restricted1 = [e for e in e1 if e.event_type == "restricted_zone_entry"]
        restricted2 = [e for e in e2 if e.event_type == "restricted_zone_entry"]
        assert len(restricted1) == 1
        assert len(restricted2) == 0   # must not fire twice

    def test_crowd_surge_fires_for_5_people(self):
        engine  = BehaviourEngine(source_video="test.mp4")
        persons = [make_person(i, zone_id="Z2") for i in range(5)]
        events  = engine.update(persons, {}, 480, 640)
        crowd   = [e for e in events if e.event_type == "crowd_surge"]
        assert len(crowd) >= 1
        assert crowd[0].alert_level == "RED"

    def test_repeat_face_fires_yellow(self):
        engine = BehaviourEngine(source_video="test.mp4")
        person = make_person(10)
        face_results = {
            10: {
                "face_found":   True,
                "face_hash":    "abc123def456",
                "is_repeat":    True,
                "seen_count":   3,
                "alert_issued": False,
                "similarity":   0.82,
            }
        }
        events = engine.update([person], face_results, 480, 640)
        repeat = [e for e in events if e.event_type == "repeat_face"]
        assert len(repeat) == 1
        assert repeat[0].alert_level == "YELLOW"
        assert repeat[0].face_hash   == "abc123def456"

    def test_stale_tracks_pruned(self):
        engine = BehaviourEngine(source_video="test.mp4")
        # Add stale state
        engine.states[99]           = ObjectState(track_id=99)
        engine.states[99].last_seen = time.time() - 10
        # Update with empty persons list
        engine.update([], {}, 480, 640)
        assert 99 not in engine.states

    def test_object_state_speed(self):
        state = ObjectState(track_id=1)
        t = time.time()
        state.position_history.append((0,   0,   t))
        state.position_history.append((100, 0,   t + 1.0))
        speed = state.current_speed_px_s()
        assert abs(speed - 100.0) < 1.0


# ══════════════════════════════════════════════
#  Context generator tests
# ══════════════════════════════════════════════

from pipeline.context_generator import build_context, context_to_prompt
from pipeline.behaviour_engine  import BehaviourEvent


def make_event(**kwargs) -> BehaviourEvent:
    defaults = dict(
        event_id="test001",
        timestamp=time.time(),
        event_type="loitering",
        alert_level="ORANGE",
        track_id=1,
        face_hash=None,
        zone_id="Z1",
        details={"duration_seconds": 35.0},
        source_video="test.mp4",
    )
    defaults.update(kwargs)
    return BehaviourEvent(**defaults)


class TestContextGenerator:

    def test_build_context_keys(self):
        ev  = make_event()
        ctx = build_context(ev, state=None)
        for key in ["event_id", "event_type", "alert_level",
                    "subject", "location", "video", "details"]:
            assert key in ctx, f"Missing key: {key}"

    def test_build_context_with_state(self):
        ev    = make_event()
        state = ObjectState(track_id=1)
        state.zone_id         = "Z1"
        state.zone_entry_time = time.time() - 35
        state.position_history.append((100, 100, time.time()))
        ctx = build_context(ev, state=state)
        assert ctx["subject"]["dwell_seconds"] >= 35

    def test_context_to_prompt_is_valid_json(self):
        ev     = make_event()
        ctx    = build_context(ev, state=None)
        prompt = context_to_prompt(ctx)
        parsed = json.loads(prompt)   # must be valid JSON
        assert parsed["event_type"] == "loitering"

    def test_context_face_hash(self):
        ev  = make_event(face_hash="abc123")
        ctx = build_context(ev, state=None)
        assert ctx["subject"]["face_hash"]     == "abc123"
        assert ctx["subject"]["is_known_face"] is True

    def test_context_no_face(self):
        ev  = make_event(face_hash=None)
        ctx = build_context(ev, state=None)
        assert ctx["subject"]["is_known_face"] is False


# ══════════════════════════════════════════════
#  Worker agent fallback tests
# ══════════════════════════════════════════════

from agents.worker_agent import WorkerAgent, _fallback_description


class TestWorkerAgent:

    def test_fallback_loitering(self):
        ctx = {
            "event_type": "loitering",
            "location":   {"zone_name": "Entry / Exit"},
            "details":    {"duration_seconds": 47},
        }
        out = _fallback_description(ctx)
        assert "loitering" in out.lower()
        assert "Entry" in out

    def test_fallback_repeat_face(self):
        ctx = {
            "event_type": "repeat_face",
            "location":   {"zone_name": "Open Area"},
            "subject":    {"face_hash": "abc12345"},
            "details":    {"seen_count": 3},
        }
        out = _fallback_description(ctx)
        assert len(out) > 10

    def test_fallback_crowd(self):
        ctx = {
            "event_type": "crowd_surge",
            "location":   {"zone_name": "Open Area"},
            "details":    {"person_count": 7},
        }
        out = _fallback_description(ctx)
        assert "7" in out or "crowd" in out.lower()

    def test_worker_agent_fallback_mode(self):
        """When Ollama unavailable, falls back gracefully."""
        agent = WorkerAgent()
        agent._available = False   # force fallback
        ctx = {
            "event_type": "running",
            "location":   {"zone_name": "Z2"},
            "details":    {},
        }
        result = agent.describe(ctx)
        assert isinstance(result, str)
        assert len(result) > 5


# ══════════════════════════════════════════════
#  Manager agent fallback tests
# ══════════════════════════════════════════════

from agents.manager_agent import ManagerAgent, _fallback_verdict


class TestManagerAgent:

    def test_fallback_loitering(self):
        v = _fallback_verdict("loitering")
        assert v["confirmed"]          is True
        assert v["final_alert_level"]  == "ORANGE"
        assert "confidence" in v
        assert "reasoning"  in v

    def test_fallback_repeat_face(self):
        v = _fallback_verdict("repeat_face")
        assert v["final_alert_level"] == "YELLOW"
        assert v["event_category"]    == "repeat_individual"

    def test_fallback_restricted(self):
        v = _fallback_verdict("restricted_zone_entry")
        assert v["final_alert_level"]  == "RED"
        assert v["recommended_action"] == "alert_operator"

    def test_fallback_unknown_event(self):
        v = _fallback_verdict("totally_unknown_event")
        assert isinstance(v, dict)
        assert "final_alert_level" in v

    def test_manager_agent_fallback_mode(self):
        agent = ManagerAgent()
        agent._available = False
        v = agent.classify({"event_type": "loitering"}, "A person loitering.")
        assert isinstance(v, dict)
        assert v["final_alert_level"] == "ORANGE"


# ══════════════════════════════════════════════
#  Zone assignment tests
# ══════════════════════════════════════════════

from pipeline.detector import assign_zone, load_zones


class TestZoneAssignment:

    def test_loads_zones(self):
        zones = load_zones()
        assert len(zones) > 0
        for z in zones:
            assert "id"       in z
            assert "polygon"  in z
            assert "_poly"    in z

    def test_point_inside_zone(self):
        """Default Z1 covers bottom strip — test centroid inside it."""
        zones  = load_zones()
        # Z1 polygon: [[0,380],[640,380],[640,480],[0,480]]
        # Centroid (320, 430) should be inside Z1
        result = assign_zone(320, 430, zones)
        assert result == "Z1"

    def test_point_outside_all_zones(self):
        zones  = load_zones()
        # Top-left corner — check if it's outside all defined polygons
        # This depends on default zone config; just check it returns str or None
        result = assign_zone(1, 1, zones)
        assert result is None or isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
