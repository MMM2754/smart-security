# Hierarchical SLM Agents for Privacy-Preserving Smart Security System

> **Final Year CSE Major Project**
> Edge surveillance using YOLOv8 + ByteTrack + Phi-3-mini agents — fully offline, CPU-only.

---

## Architecture

```
Video (.mp4)
    │
    ▼
┌─────────────────────────────────────┐
│  01  Data Acquisition & Perception  │
│  YOLOv8n · ByteTrack                │
│  → Bounding boxes, track IDs, zones │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  02  Object Tracking & Temporal     │
│  Per-object state · trajectory      │
│  Speed · dwell time · zone history  │
│  Face Re-ID (InsightFace, CPU)      │
│  → 🟡 YELLOW if face seen again     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  03  Structured Context + SLM       │
│  Behaviour JSON                     │
│  Worker Agent  → natural language   │
│  Manager Agent → verdict + level    │
│  (Phi-3-mini via Ollama, local)     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  04  Decision Layer                 │
│  MQTT alerts · SQLite audit trail   │
│  Streamlit dashboard                │
│  Edge-ready (CPU-only)              │
└─────────────────────────────────────┘
```

## Alert Levels

| Level  | Trigger                                      |
|--------|----------------------------------------------|
| 🟢 GREEN  | Normal activity                           |
| 🟡 YELLOW | **Same face seen again** (repeat individual) |
| 🟠 ORANGE | Loitering · Running · Perimeter breach    |
| 🔴 RED    | Fighting · Shooting · Restricted zone     |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Phi-3-mini (one time)
```bash
bash setup_ollama.sh
```

### 3. Get test videos
```bash
# Option A — generate a synthetic placeholder
python data/sample_fetch.py --placeholder

# Option B — use your own mp4 files
cp your_video.mp4 data/videos/

# Option C — check UCF-Crime download instructions
python data/sample_fetch.py --instructions
```

### 4. Define zones (optional — defaults work fine)
```bash
python tools/zone_drawer.py data/videos/your_video.mp4
```

### 5. Run the pipeline
```bash
# With display window
python main.py --video data/videos/your_video.mp4

# Headless (server mode)
python main.py --video data/videos/your_video.mp4 --no-display

# Run entire folder
python main.py --folder data/videos/Robbery/

# Quick test with placeholder video
python main.py --placeholder
```

### 6. Open the dashboard
```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

---

## Project Structure

```
smart_security/
├── config/
│   ├── settings.py          # All config — edit here
│   └── zones.json           # Zone polygon definitions
│
├── pipeline/
│   ├── detector.py          # YOLOv8 + ByteTrack
│   ├── face_reid.py         # InsightFace re-ID (privacy-safe)
│   ├── behaviour_engine.py  # Loitering, running, crowd, etc.
│   └── context_generator.py # Builds JSON for SLM agents
│
├── agents/
│   ├── worker_agent.py      # JSON → natural language (Phi-3)
│   └── manager_agent.py     # Classification + verdict (Phi-3)
│
├── alerts/
│   └── mqtt_publisher.py    # Real-time MQTT alerts
│
├── storage/
│   └── database.py          # SQLite — events, faces, audit
│
├── dashboard/
│   └── app.py               # Streamlit UI
│
├── tools/
│   └── zone_drawer.py       # Interactive zone editor
│
├── data/
│   ├── videos/              # Put your .mp4 files here
│   └── sample_fetch.py      # Dataset download helper
│
├── main.py                  # Entry point
├── requirements.txt
├── setup_ollama.sh          # Phi-3-mini setup
└── README.md
```

---

## Dataset: UCF-Crime

- **13 crime categories** — Robbery, Shooting, Assault, Fighting, Stealing, etc.
- **1,900 real CCTV videos** (~129 hours)
- Download: https://www.crcv.ucf.edu/projects/real-world/
- Kaggle mirror: `kaggle datasets download odins0n/ucf-crime-dataset`

Recommended classes for this project:
`Robbery · Shooting · Assault · Fighting · Stealing · Shoplifting`

---

## Privacy Design

- ✅ Face embeddings stored (512-dim vector), **never raw face images**
- ✅ All face crops blurred before display
- ✅ IDs anonymised in logs (SHA256 hash prefix)
- ✅ Fully offline — no cloud API calls
- ✅ SQLite local storage only

---

## Behaviours Detected

| Behaviour            | Alert | Trigger                              |
|----------------------|-------|--------------------------------------|
| Repeat face          | 🟡    | Same person seen again               |
| Loitering            | 🟠    | >30s in same zone                    |
| Running              | 🟠    | Speed > threshold px/s               |
| Restricted zone      | 🔴    | Entry into Z3                        |
| Perimeter breach     | 🟠    | Near frame edge                      |
| Crowd surge          | 🔴    | ≥5 people in one zone                |

---

## Tuning Config

Edit `config/settings.py`:
```python
LOITER_SECONDS      = 30     # seconds before loitering fires
RUNNING_SPEED_PX_S  = 120    # px/s threshold
CROWD_DENSITY_COUNT = 5      # people count for crowd alert
FACE_SIMILARITY_THRESHOLD = 0.55  # cosine sim for face match
FRAME_SKIP          = 3      # process every Nth frame (CPU tuning)
```

---

## Tech Stack

| Component  | Technology                     |
|------------|--------------------------------|
| Detection  | YOLOv8n (Ultralytics)          |
| Tracking   | ByteTrack (built-in)           |
| Face Re-ID | InsightFace buffalo_sc (CPU)   |
| SLM        | Phi-3-mini via Ollama          |
| Alerts     | MQTT (paho-mqtt)               |
| Storage    | SQLite                         |
| Dashboard  | Streamlit + Plotly             |
| Language   | Python 3.10+                   |

---

## Requirements

- Python 3.10+
- CPU laptop (no GPU required)
- ~4GB RAM minimum
- ~3GB disk (for Phi-3-mini model)
- Ollama installed (for SLM agents)
