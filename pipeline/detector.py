# ─────────────────────────────────────────────
#  pipeline/detector.py
#  Phase 1: YOLOv8n detection + ByteTrack tracking
#           + zone assignment per tracked object
# ─────────────────────────────────────────────

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL, YOLO_CONF, YOLO_CLASSES, YOLO_IMG_SIZE,
    FRAME_SKIP, TRACK_PERSIST, TRACK_CONF, ZONES_PATH,
    PERIMETER_MARGIN_PX
)


# ══════════════════════════════════════════════
#  Zone helpers
# ══════════════════════════════════════════════

def load_zones(path: str = ZONES_PATH) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    zones = []
    for z in data["zones"]:
        poly = np.array(z["polygon"], dtype=np.int32)
        zones.append({**z, "_poly": poly})
    return zones


def point_in_zone(cx: int, cy: int, zone: dict) -> bool:
    """Check if centroid (cx, cy) falls inside a zone polygon."""
    return cv2.pointPolygonTest(zone["_poly"], (float(cx), float(cy)), False) >= 0


def assign_zone(cx: int, cy: int, zones: list[dict]) -> str | None:
#def assign_zone(cx: int, cy: int, zones: list[dict]) -> Optional[str]:
    """Return the first matching zone ID, or None."""
    for zone in zones:
        if point_in_zone(cx, cy, zone):
            return zone["id"]
    return None


def draw_zones(frame: np.ndarray, zones: list[dict]) -> np.ndarray:
    """Draw zone overlays on frame."""
    overlay = frame.copy()
    for zone in zones:
        color  = tuple(zone.get("color", [100, 100, 255]))
        poly   = zone["_poly"]
        cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [poly], color=(*color, 40))
        # Label
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(overlay, zone["name"], (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


# ══════════════════════════════════════════════
#  Detection result dataclass
# ══════════════════════════════════════════════

class TrackedPerson:
    __slots__ = ["track_id", "frame_number", "timestamp",
                 "x1", "y1", "x2", "y2", "confidence",
                 "cx", "cy", "zone_id", "bbox_wh"]

    def __init__(self, track_id, frame_number, timestamp,
                 x1, y1, x2, y2, confidence, zone_id):
        self.track_id    = int(track_id)
        self.frame_number = frame_number
        self.timestamp   = timestamp
        self.x1, self.y1 = float(x1), float(y1)
        self.x2, self.y2 = float(x2), float(y2)
        self.confidence  = float(confidence)
        self.cx          = int((x1 + x2) / 2)
        self.cy          = int((y1 + y2) / 2)
        self.zone_id     = zone_id
        self.bbox_wh     = (int(x2 - x1), int(y2 - y1))

    def to_dict(self) -> dict:
        return {
            "track_id":    self.track_id,
            "frame_number":self.frame_number,
            "timestamp":   self.timestamp,
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "confidence":  self.confidence,
            "cx": self.cx, "cy": self.cy,
            "zone_id":     self.zone_id,
            "bbox_wh":     self.bbox_wh,
        }


# ══════════════════════════════════════════════
#  Main Detector class
# ══════════════════════════════════════════════

class Detector:
    def __init__(self):
        logger.info(f"Loading YOLO model: {YOLO_MODEL}")
        self.model  = YOLO(YOLO_MODEL)
        self.zones  = load_zones()
        self.frame_count = 0
        logger.success(f"Detector ready. Zones loaded: {[z['name'] for z in self.zones]}")

    def process_frame(self, frame: np.ndarray) -> tuple[list[TrackedPerson], np.ndarray]:
        """
        Run detection + tracking on a single frame.
        Returns: (list of TrackedPerson, annotated frame)
        """
        self.frame_count += 1

        # Skip frames for CPU relief
        if self.frame_count % FRAME_SKIP != 0:
            return [], frame

        h, w = frame.shape[:2]
        timestamp = time.time()

        # Run YOLOv8 + ByteTrack
        results = self.model.track(
            frame,
            persist=TRACK_PERSIST,
            conf=YOLO_CONF,
            classes=YOLO_CLASSES,
            imgsz=YOLO_IMG_SIZE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_persons: list[TrackedPerson] = []
        annotated = draw_zones(frame.copy(), self.zones)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id        = int(box.id[0].item())
                conf            = float(box.conf[0].item())
                cx, cy          = int((x1 + x2) / 2), int((y1 + y2) / 2)
                zone_id         = assign_zone(cx, cy, self.zones)

                person = TrackedPerson(
                    track_id, self.frame_count, timestamp,
                    x1, y1, x2, y2, conf, zone_id
                )
                tracked_persons.append(person)

                # ── Draw bounding box ──────────────
                color = self._zone_color(zone_id)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2)
                label = f"ID:{track_id}"
                if zone_id:
                    label += f" [{zone_id}]"
                cv2.putText(annotated, label,
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # ── Draw centroid dot ──────────────
                cv2.circle(annotated, (cx, cy), 4, color, -1)

        # Frame counter overlay
        cv2.putText(annotated, f"Frame: {self.frame_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Tracked: {len(tracked_persons)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)

        return tracked_persons, annotated

    def _zone_color(self, zone_id: str | None) -> tuple:
        """Return BGR color for a zone ID."""
        if not zone_id:
            return (200, 200, 200)
        for z in self.zones:
            if z["id"] == zone_id:
                r, g, b = z.get("color", [200, 200, 200])
                return (b, g, r)   # cv2 uses BGR
        return (200, 200, 200)

    def crop_person(self, frame: np.ndarray, person: TrackedPerson,
                    padding: int = 10) -> np.ndarray | None:
        """Extract a padded crop of the person from the frame."""
        h, w = frame.shape[:2]
        x1 = max(0, int(person.x1) - padding)
        y1 = max(0, int(person.y1) - padding)
        x2 = min(w, int(person.x2) + padding)
        y2 = min(h, int(person.y2) + padding)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        return frame[y1:y2, x1:x2].copy()

    def get_face_crop(self, frame: np.ndarray, person: TrackedPerson) -> np.ndarray | None:
        """
        Try to crop just the upper 35% of the bounding box as a face region.
        InsightFace will verify if it's actually a face.
        """
        h_box = int(person.y2 - person.y1)
        face_y2 = int(person.y1 + h_box * 0.40)   # top 40% = likely head
        x1 = max(0, int(person.x1))
        y1 = max(0, int(person.y1))
        x2 = min(frame.shape[1], int(person.x2))

        if x2 - x1 < 15 or face_y2 - y1 < 15:
            return None
        return frame[y1:face_y2, x1:x2].copy()


# ══════════════════════════════════════════════
#  Standalone test
# ══════════════════════════════════════════════

def run_on_video(video_path: str, show: bool = True, save_path: str = None):
    """Test the detector on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {Path(video_path).name}  {w}x{h} @ {fps:.1f}fps")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps / FRAME_SKIP, (w, h))

    detector = Detector()
    total_persons = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons, annotated = detector.process_frame(frame)
        total_persons += len(persons)

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("Smart Security — Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.success(f"Done. Total detections across all frames: {total_persons}")


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/test.mp4"
    run_on_video(video)
