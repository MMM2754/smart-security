# ─────────────────────────────────────────────
#  config/settings.py
#  Central config — edit this file to tune behaviour
# ─────────────────────────────────────────────

import os

# ── Paths ─────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
FACES_DIR       = os.path.join(OUTPUT_DIR, "faces")      # blurred face crops (for UI only)
EVENTS_DIR      = os.path.join(OUTPUT_DIR, "events")     # JSON event snapshots
DB_PATH         = os.path.join(BASE_DIR, "storage", "security.db")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
ZONES_PATH      = os.path.join(BASE_DIR, "config", "zones.json")

# ── YOLO ──────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"          # nano = fastest on CPU
YOLO_CONF       = 0.50                   # detection confidence threshold (higher = fewer false positives, faster)
YOLO_CLASSES    = [0]                    # 0 = person only (keeps it fast)
YOLO_IMG_SIZE   = 640
FRAME_SKIP      = 5                      # process every Nth frame (increase for faster processing)

# ── ByteTrack (built into Ultralytics) ────────
TRACK_PERSIST   = True
TRACK_CONF      = 0.35

# ── Behaviour Thresholds ──────────────────────
LOITER_SECONDS          = 30     # person in same zone > 30s = loitering
RUNNING_SPEED_PX_S      = 120    # pixels/second threshold = running
CROWD_DENSITY_COUNT     = 2      # >= 2 people in one zone = crowd surge (for testing)
PERIMETER_MARGIN_PX     = 30     # px from frame edge = perimeter zone
ABANDONED_FRAMES        = 90     # object stationary for N frames = abandoned

# ── Face Re-ID ────────────────────────────────
FACE_SIMILARITY_THRESHOLD = 0.55   # cosine similarity — tune if too many false positives
FACE_MODEL              = "buffalo_sc"  # lightest InsightFace model, CPU-safe
STORE_FACE_CROPS        = True         # save blurred crop to output/faces/ for UI
FACE_BLUR_STRENGTH      = 35           # kernel size for blur (must be odd)
MAX_FACE_DB_SIZE        = 10000        # max embeddings to keep in SQLite

# ── SLM / Ollama ──────────────────────────────
OLLAMA_HOST             = "http://localhost:11434"
WORKER_MODEL            = "phi3:mini"   # Worker Agent — JSON → natural language
MANAGER_MODEL           = "phi3:mini"   # Manager Agent — classification & severity
OLLAMA_TIMEOUT          = 15            # seconds before timeout (reduced for faster failure)

# ── MQTT ──────────────────────────────────────
MQTT_BROKER             = "localhost"
MQTT_PORT               = 1883
MQTT_TOPIC_ALERTS       = "security/alerts"
MQTT_TOPIC_WARNINGS     = "security/warnings"
MQTT_TOPIC_STATUS       = "security/status"
MQTT_CLIENT_ID          = "smart_security_system"

# ── Alert Levels ──────────────────────────────
class AlertLevel:
    GREEN   = "GREEN"    # normal
    YELLOW  = "YELLOW"   # repeat face / mild loitering
    ORANGE  = "ORANGE"   # suspicious behaviour
    RED     = "RED"      # critical — violence, weapons, robbery

# ── UCF-Crime class mapping → alert level ─────
UCF_CRIME_ALERT_MAP = {
    "Abuse":        AlertLevel.RED,
    "Arrest":       AlertLevel.ORANGE,
    "Arson":        AlertLevel.RED,
    "Assault":      AlertLevel.RED,
    "Burglary":     AlertLevel.RED,
    "Explosion":    AlertLevel.RED,
    "Fighting":     AlertLevel.RED,
    "RoadAccident": AlertLevel.ORANGE,
    "Robbery":      AlertLevel.RED,
    "Shooting":     AlertLevel.RED,
    "Shoplifting":  AlertLevel.ORANGE,
    "Stealing":     AlertLevel.ORANGE,
    "Vandalism":    AlertLevel.ORANGE,
    "Normal":       AlertLevel.GREEN,
}

# ── Dashboard ─────────────────────────────────
DASHBOARD_PORT          = 8501
DASHBOARD_REFRESH_MS    = 1000   # auto-refresh interval

# ── Privacy ───────────────────────────────────
STORE_RAW_FACES         = False   # NEVER store raw face images
ANONYMISE_LOGS          = True    # replace face IDs with hashes in logs
