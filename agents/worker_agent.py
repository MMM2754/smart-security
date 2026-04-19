# ─────────────────────────────────────────────
#  agents/worker_agent.py
#  Tier 1: JSON → natural language description
#  Uses Phi-3-mini via Ollama (local, no cloud)
# ─────────────────────────────────────────────

import json
import time
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OLLAMA_HOST, WORKER_MODEL, OLLAMA_TIMEOUT

try:
    import ollama as ollama_client
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Worker agent will use rule-based fallback.")


# ══════════════════════════════════════════════
#  System prompt for Worker agent
# ══════════════════════════════════════════════

WORKER_SYSTEM_PROMPT = """You are a security surveillance assistant. 
You receive structured JSON data about an event detected by a camera system.
Your job is to write ONE clear, concise sentence (max 30 words) describing what happened.

Rules:
- Be factual and specific. Use the data given.
- Do NOT invent details not present in the JSON.
- Do NOT mention technical terms like track_id, bbox, or pixel values.
- Refer to people as "a person" or "an individual".
- If face_hash is present, say "a previously seen individual".
- Keep it professional, like a real security log entry.
- Output ONLY the single sentence. No preamble, no explanation.

Example output:
"A previously seen individual has been loitering near the entry zone for over 45 seconds."
"""

# ══════════════════════════════════════════════
#  Rule-based fallback (when Ollama unavailable)
# ══════════════════════════════════════════════

_FALLBACK_TEMPLATES = {
    "repeat_face": (
        "A previously seen individual (ID: {face_hash}) has been detected again "
        "in {zone_name} — {seen_count} total sightings."
    ),
    "loitering": (
        "An individual has been loitering in {zone_name} for {duration_seconds} seconds."
    ),
    "running": (
        "An individual was observed running in {zone_name}."
    ),
    "restricted_zone_entry": (
        "An individual has entered the restricted zone: {zone_name}."
    ),
    "perimeter_breach": (
        "An individual was detected near the frame perimeter."
    ),
    "crowd_surge": (
        "A crowd of {person_count} people has gathered in {zone_name}."
    ),
    "default": (
        "Suspicious activity detected: {event_type} in {zone_name}."
    ),
}

def _fallback_description(context: dict) -> str:
    event_type = context.get("event_type", "unknown")
    template   = _FALLBACK_TEMPLATES.get(event_type, _FALLBACK_TEMPLATES["default"])
    zone_name  = context.get("location", {}).get("zone_name", "unknown zone")
    details    = context.get("details", {})
    face_hash  = (context.get("subject", {}).get("face_hash") or "")[:8]

    try:
        return template.format(
            zone_name        = zone_name,
            event_type       = event_type,
            face_hash        = face_hash,
            seen_count       = details.get("seen_count", "?"),
            duration_seconds = details.get("duration_seconds", "?"),
            person_count     = details.get("person_count", "?"),
        )
    except KeyError:
        return f"Security event detected: {event_type} in {zone_name}."


# ══════════════════════════════════════════════
#  Worker Agent class
# ══════════════════════════════════════════════

class WorkerAgent:
    """
    Tier 1 agent: converts structured JSON event context
    into a natural language description sentence.
    """

    def __init__(self):
        self._available = _OLLAMA_AVAILABLE
        if self._available:
            self._check_model()

    def _check_model(self):
        try:
            client = ollama_client.Client(host=OLLAMA_HOST)
            models = client.list()
            names  = [m.model for m in models.models]
            if not any(WORKER_MODEL in n for n in names):
                logger.warning(
                    f"Model '{WORKER_MODEL}' not found in Ollama. "
                    f"Run: ollama pull {WORKER_MODEL}"
                )
                self._available = False
            else:
                logger.success(f"Worker agent ready: {WORKER_MODEL}")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}. Using fallback.")
            self._available = False

    def describe(self, context: dict) -> str:
        """
        Generate a natural language description of the event.
        Falls back to template if Ollama unavailable.
        """
        if not self._available:
            return _fallback_description(context)

        prompt = (
            "Here is the security event JSON:\n\n"
            + json.dumps(context, indent=2)
            + "\n\nWrite the single-sentence description now:"
        )

        try:
            client   = ollama_client.Client(host=OLLAMA_HOST)
            response = client.chat(
                model   = WORKER_MODEL,
                messages = [
                    {"role": "system",  "content": WORKER_SYSTEM_PROMPT},
                    {"role": "user",    "content": prompt},
                ],
                options  = {"temperature": 0.1, "num_predict": 80},
            )
            description = response["message"]["content"].strip()
            # Clean up any quotes the model might add
            description = description.strip('"').strip("'")
            logger.debug(f"Worker agent output: {description}")
            return description

        except Exception as e:
            logger.warning(f"Worker agent LLM call failed: {e}. Using fallback.")
            return _fallback_description(context)


# ══════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════

if __name__ == "__main__":
    test_context = {
        "event_id":   "abc123",
        "event_type": "loitering",
        "alert_level": "ORANGE",
        "subject": {
            "track_id":      42,
            "face_hash":     None,
            "is_known_face": False,
            "dwell_seconds": 47.3,
            "speed_px_s":    5.1,
        },
        "location": {
            "zone_id":   "Z1",
            "zone_name": "Entry / Exit",
        },
        "video": {
            "source": "test_video.mp4",
            "timestamp": "2024-01-15T14:32:00Z",
        },
        "details": {
            "duration_seconds": 47.3,
            "message": "Person loitering in Z1 for 47s.",
        },
    }

    agent = WorkerAgent()
    result = agent.describe(test_context)
    print(f"\nWorker agent output:\n  {result}")
