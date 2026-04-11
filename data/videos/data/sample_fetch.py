# ─────────────────────────────────────────────
#  data/sample_fetch.py
#  Downloads sample videos from UCF-Crime dataset
#  Uses the official UCF download page
# ─────────────────────────────────────────────

import os
import sys
import json
import requests
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(__file__).parent / "videos"

# UCF-Crime official dataset download page
UCF_OFFICIAL_URL = "https://www.crcv.ucf.edu/projects/real-world/"
UCF_KAGGLE_SLUG  = "odins0n/ucf-crime-dataset"

# Categories relevant to your project (chain snatching, shooting, etc.)
PRIORITY_CLASSES = [
    "Robbery",      # closest to chain snatching
    "Shooting",
    "Assault",
    "Fighting",
    "Stealing",
    "Shoplifting",
    "Vandalism",
    "Burglary",
]

def print_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         UCF-Crime Dataset — Download Instructions            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Option 1 — Official UCF download (recommended)             ║
║  ─────────────────────────────────────────────              ║
║  1. Go to: https://www.crcv.ucf.edu/projects/real-world/    ║
║  2. Fill the request form (free academic access)            ║
║  3. Download the zip (~5GB full / use partial classes)      ║
║  4. Extract to: data/videos/                                ║
║                                                              ║
║  Option 2 — Kaggle (faster, no form needed)                 ║
║  ─────────────────────────────────────────────              ║
║  pip install kaggle                                          ║
║  kaggle datasets download odins0n/ucf-crime-dataset          ║
║  unzip ucf-crime-dataset.zip -d data/videos/                ║
║                                                              ║
║  Option 3 — Use your own CCTV videos (mp4)                  ║
║  ─────────────────────────────────────────────              ║
║  Just copy your .mp4 files to data/videos/                  ║
║  Then run: python main.py --video data/videos/yourfile.mp4  ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Priority classes for your project:                          ║
║    Robbery, Shooting, Assault, Fighting, Stealing            ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_videos() -> dict:
    """Scan data/videos/ and report what's available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    mp4_files = list(DATA_DIR.rglob("*.mp4")) + list(DATA_DIR.rglob("*.avi"))

    for f in mp4_files:
        # Try to guess class from folder name
        parts = f.parts
        cls   = "Unknown"
        for part in parts:
            if part in PRIORITY_CLASSES or part in [
                "Abuse", "Arrest", "Arson", "Explosion",
                "RoadAccident", "Normal"
            ]:
                cls = part
                break
        found.setdefault(cls, []).append(str(f))

    return found

def download_via_kaggle(classes: list[str] = None, videos_per_class: int = 5):
    """
    Attempt to download a subset via Kaggle API.
    Requires: pip install kaggle + ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        logger.info("Kaggle API found. Attempting download...")
        os.makedirs(str(DATA_DIR), exist_ok=True)
        kaggle.api.dataset_download_files(
            UCF_KAGGLE_SLUG,
            path=str(DATA_DIR),
            unzip=True,
        )
        logger.success(f"Downloaded to {DATA_DIR}")
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        print_instructions()

def create_placeholder_video(output_path: str, duration_s: int = 10, fps: int = 25):
    """
    Create a synthetic test video with moving boxes (no real dataset needed for unit tests).
    """
    import cv2
    import numpy as np
    import random

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w     = 480, 640
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    n_frames = duration_s * fps

    # Simulate 3 "people" as moving rectangles
    people = [
        {"x": random.randint(50, 200),  "y": random.randint(100, 300), "vx": 2,  "vy": 1},
        {"x": random.randint(300, 500), "y": random.randint(100, 300), "vx": -1, "vy": 2},
        {"x": random.randint(100, 400), "y": random.randint(50, 200),  "vx": 1,  "vy": -1},
    ]

    for frame_i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :] = [20, 20, 40]
        cv2.putText(frame, f"TEST VIDEO — Frame {frame_i}/{n_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        for p in people:
            # Move
            p["x"] = max(20, min(w - 60, p["x"] + p["vx"]))
            p["y"] = max(20, min(h - 120, p["y"] + p["vy"]))
            # Bounce
            if p["x"] <= 20 or p["x"] >= w - 60:
                p["vx"] *= -1
            if p["y"] <= 20 or p["y"] >= h - 120:
                p["vy"] *= -1
            # Draw "person" as rectangle
            cv2.rectangle(frame, (p["x"], p["y"]),
                          (p["x"] + 40, p["y"] + 100),
                          (0, 200, 100), -1)

        writer.write(frame)

    writer.release()
    logger.success(f"Placeholder test video saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UCF-Crime dataset helper")
    parser.add_argument("--check",       action="store_true", help="Check what videos are already downloaded")
    parser.add_argument("--kaggle",      action="store_true", help="Download via Kaggle API")
    parser.add_argument("--placeholder", action="store_true", help="Generate a synthetic test video")
    parser.add_argument("--instructions",action="store_true", help="Show download instructions")
    args = parser.parse_args()

    if args.check:
        found = check_videos()
        if found:
            print(f"\n✅ Found {sum(len(v) for v in found.values())} videos:\n")
            for cls, files in sorted(found.items()):
                print(f"  {cls}: {len(files)} videos")
        else:
            print("\n❌ No videos found in data/videos/")
            print_instructions()

    elif args.kaggle:
        download_via_kaggle()

    elif args.placeholder:
        out = str(DATA_DIR / "test_placeholder.mp4")
        create_placeholder_video(out)
        print(f"\nTest with: python main.py --video {out}")

    else:
        print_instructions()
        found = check_videos()
        if found:
            print(f"Currently available: {sum(len(v) for v in found.values())} videos")
        else:
            print("No videos in data/videos/ yet.")
