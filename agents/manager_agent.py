# ─────────────────────────────────────────────
#  agents/manager_agent.py
#  Tier 2: Contextual reasoning & classification
#  Validates Worker output + assigns final verdict
# ─────────────────────────────────────────────

import json
import time
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OLLAMA_HOST, MANAGER_MODEL, OLLAMA_TIMEOUT, AlertLevel

try:
    import ollama as ollama_client
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


# ══════════════════════════════════════════════
#  System prompt for Manager agent
# ══════════════════════════════════════════════

MANAGER_SYSTEM_PROMPT = """You are a senior AI security analyst reviewing events flagged by a surveillance system.

You receive:
1. A structured JSON event context
2. A natural language description written by a junior analyst

Your job is to return a JSON verdict ONLY — no other text — with exactly these fields:
{
  "confirmed": true or false,
  "final_alert_level": "GREEN" or "YELLOW" or "ORANGE" or "RED",
  "event_category": "normal" | "suspicious" | "critical" | "repeat_individual",
  "recommended_action": "monitor" | "investigate" | "alert_operator" | "emergency_response",
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explaining your decision"
}

Alert level rules:
- GREEN:  Normal, no concern
- YELLOW: Repeat individual detected, needs monitoring
- ORANGE: Suspicious — loitering, running, perimeter breach, shoplifting, vandalism
- RED:    Critical — fighting, shooting, robbery, assault, restricted zone breach, explosion

Output ONLY valid JSON. No preamble, no markdown, no explanation outside the JSON.
"""

# ══════════════════════════════════════════════
#  Rule-based fallback verdicts
# ══════════════════════════════════════════════

_FALLBACK_VERDICTS = {
    "repeat_face": {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.YELLOW,
        "event_category":     "repeat_individual",
        "recommended_action": "monitor",
        "confidence":         0.85,
        "reasoning":          "Previously seen individual detected — monitoring recommended.",
    },
    "loitering": {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.ORANGE,
        "event_category":     "suspicious",
        "recommended_action": "investigate",
        "confidence":         0.75,
        "reasoning":          "Extended dwell time in zone indicates loitering.",
    },
    "running": {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.ORANGE,
        "event_category":     "suspicious",
        "recommended_action": "investigate",
        "confidence":         0.70,
        "reasoning":          "Running detected — context unclear, investigation needed.",
    },
    "restricted_zone_entry": {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.RED,
        "event_category":     "critical",
        "recommended_action": "alert_operator",
        "confidence":         0.95,
        "reasoning":          "Unauthorised entry into restricted zone.",
    },
    "perimeter_breach": {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.ORANGE,
        "event_category":     "suspicious",
        "recommended_action": "investigate",
        "confidence":         0.65,
        "reasoning":          "Activity detected near frame perimeter.",
    },
    "crowd_surge": {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.RED,
        "event_category":     "critical",
        "recommended_action": "alert_operator",
        "confidence":         0.90,
        "reasoning":          "High crowd density may indicate a developing incident.",
    },
}

def _fallback_verdict(event_type: str) -> dict:
    return _FALLBACK_VERDICTS.get(event_type, {
        "confirmed":          True,
        "final_alert_level":  AlertLevel.ORANGE,
        "event_category":     "suspicious",
        "recommended_action": "investigate",
        "confidence":         0.60,
        "reasoning":          f"Unclassified event: {event_type}.",
    })


# ══════════════════════════════════════════════
#  Manager Agent class
# ══════════════════════════════════════════════

class ManagerAgent:
    """
    Tier 2 agent: validates and classifies the event,
    produces a final structured verdict.
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
            if not any(MANAGER_MODEL in n for n in names):
                logger.warning(
                    f"Manager model '{MANAGER_MODEL}' not found. Using fallback."
                )
                self._available = False
            else:
                logger.success(f"Manager agent ready: {MANAGER_MODEL}")
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}. Manager using fallback.")
            self._available = False

    def classify(self, context: dict, worker_description: str) -> dict:
        """
        Classify the event and return a verdict dict.
        Falls back to rule-based logic if Ollama unavailable.
        """
        if not self._available:
            return _fallback_verdict(context.get("event_type", "unknown"))

        prompt = (
            "Event context JSON:\n"
            + json.dumps(context, indent=2)
            + f"\n\nJunior analyst description:\n\"{worker_description}\""
            + "\n\nReturn your JSON verdict now:"
        )

        try:
            client   = ollama_client.Client(host=OLLAMA_HOST)
            response = client.chat(
                model    = MANAGER_MODEL,
                messages = [
                    {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options  = {"temperature": 0.0, "num_predict": 200},
            )
            raw = response["message"]["content"].strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            verdict = json.loads(raw)
            logger.debug(
                f"Manager verdict: {verdict.get('final_alert_level')} | "
                f"{verdict.get('event_category')} | "
                f"conf={verdict.get('confidence')}"
            )
            return verdict

        except json.JSONDecodeError as e:
            logger.warning(f"Manager agent returned invalid JSON: {e}. Using fallback.")
            return _fallback_verdict(context.get("event_type", "unknown"))
        except Exception as e:
            logger.warning(f"Manager agent LLM call failed: {e}. Using fallback.")
            return _fallback_verdict(context.get("event_type", "unknown"))


# ══════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════

if __name__ == "__main__":
    test_context = {
        "event_id":    "abc123",
        "event_type":  "loitering",
        "alert_level": "ORANGE",
        "subject": {
            "track_id":      42,
            "face_hash":     "a3f2b891",
            "is_known_face": True,
            "dwell_seconds": 47.3,
        },
        "location": {"zone_id": "Z1", "zone_name": "Entry / Exit"},
        "details":  {"duration_seconds": 47.3},
    }
    worker_desc = "A previously seen individual has been loitering near the entry zone for over 45 seconds."

    agent   = ManagerAgent()
    verdict = agent.classify(test_context, worker_desc)
    print(f"\nManager verdict:\n{json.dumps(verdict, indent=2)}")
