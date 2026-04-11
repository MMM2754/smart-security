# ─────────────────────────────────────────────
#  pipeline/face_reid.py
#  Privacy-preserving face re-identification
#  • Embeddings stored, never raw face images
#  • Blurred crops only for dashboard display
#  • Returns YELLOW warning if face seen before
# ─────────────────────────────────────────────

import cv2
import numpy as np
import time
from pathlib import Path
from loguru import logger
from scipy.spatial.distance import cosine

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    FACE_MODEL, FACE_SIMILARITY_THRESHOLD,
    FACE_BLUR_STRENGTH, STORE_FACE_CROPS,
    FACES_DIR, STORE_RAW_FACES
)
from storage.database import get_all_embeddings, upsert_face, mark_alert_issued


# ══════════════════════════════════════════════
#  InsightFace loader (lazy — only loads once)
# ══════════════════════════════════════════════

_face_app = None

def _get_face_app():
    global _face_app
    if _face_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(
                name=FACE_MODEL,
                providers=["CPUExecutionProvider"]   # CPU only
            )
            _face_app.prepare(ctx_id=-1, det_size=(320, 320))  # small det for speed
            logger.success(f"InsightFace loaded: {FACE_MODEL}")
        except Exception as e:
            logger.warning(f"InsightFace unavailable: {e}. Face re-ID disabled.")
            _face_app = None
    return _face_app


# ══════════════════════════════════════════════
#  Blur helper (privacy)
# ══════════════════════════════════════════════

def blur_face_region(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Blur a rectangular region in-place.
    bbox = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    k = FACE_BLUR_STRENGTH
    if k % 2 == 0:
        k += 1
    region = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return frame


# ══════════════════════════════════════════════
#  Core Re-ID logic
# ══════════════════════════════════════════════

def find_best_match(query_embedding: np.ndarray, stored: list[dict]) -> dict | None:
    """
    Cosine similarity search against stored embeddings.
    Returns the best match dict if above threshold, else None.
    """
    if not stored:
        return None

    best_sim  = -1.0
    best_entry = None

    for entry in stored:
        sim = 1.0 - cosine(query_embedding, entry["embedding"])
        if sim > best_sim:
            best_sim   = sim
            best_entry = entry

    if best_sim >= FACE_SIMILARITY_THRESHOLD:
        return {**best_entry, "similarity": best_sim}
    return None


class FaceReID:
    def __init__(self):
        self._app     = None
        self._cache   = []     # in-memory cache of embeddings for this session
        self._loaded  = False
        Path(FACES_DIR).mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self):
        if not self._loaded:
            self._app    = _get_face_app()
            self._cache  = get_all_embeddings()   # load from DB into RAM
            self._loaded = True
            logger.info(f"Face re-ID cache loaded: {len(self._cache)} known faces")

    def process(self,
                frame: np.ndarray,
                person_crop: np.ndarray,
                track_id: int,
                source_video: str = "") -> dict:
        """
        Extract face from person crop → compare to DB → return result dict.

        Returns:
        {
            "face_found":    bool,
            "face_hash":     str | None,
            "is_repeat":     bool,        ← True = 🟡 YELLOW warning
            "seen_count":    int,
            "alert_issued":  bool,
            "similarity":    float | None,
            "blurred_frame": np.ndarray,  ← frame with face blurred
        }
        """
        self._ensure_loaded()
        result = {
            "face_found":   False,
            "face_hash":    None,
            "is_repeat":    False,
            "seen_count":   0,
            "alert_issued": False,
            "similarity":   None,
            "blurred_frame": frame.copy(),
        }

        if self._app is None or person_crop is None:
            return result

        # Resize crop to reasonable size for detection
        crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        try:
            faces = self._app.get(crop_rgb)
        except Exception as e:
            logger.debug(f"Face detection error on track {track_id}: {e}")
            return result

        if not faces:
            return result

        # Use the highest-confidence face
        face = max(faces, key=lambda f: f.det_score)
        if face.det_score < 0.40:    # low confidence — skip
            return result

        result["face_found"] = True
        embedding = face.embedding   # 512-dim float32 vector

        # Blur face in the frame for display (privacy)
        if hasattr(face, "bbox"):
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            result["blurred_frame"] = blur_face_region(
                result["blurred_frame"], (fx1, fy1, fx2, fy2)
            )

        # Check against known faces
        match = find_best_match(embedding, self._cache)

        if match:
            # ── Known face ──────────────────────
            face_info = upsert_face(embedding, source_video)
            result.update({
                "face_hash":    face_info["face_hash"],
                "is_repeat":    True,
                "seen_count":   face_info["seen_count"],
                "alert_issued": face_info["alert_issued"],
                "similarity":   match["similarity"],
            })
            # Update cache entry
            for c in self._cache:
                if c["face_hash"] == face_info["face_hash"]:
                    c["seen_count"] = face_info["seen_count"]
                    break

            logger.info(
                f"🟡 Repeat face detected | track={track_id} | "
                f"hash={face_info['face_hash']} | "
                f"seen={face_info['seen_count']}x | "
                f"sim={match['similarity']:.3f}"
            )
        else:
            # ── New face ─────────────────────────
            face_info = upsert_face(embedding, source_video)
            new_entry = {
                "face_hash":    face_info["face_hash"],
                "embedding":    embedding,
                "seen_count":   1,
                "alert_issued": False,
            }
            self._cache.append(new_entry)
            result.update({
                "face_hash":    face_info["face_hash"],
                "is_repeat":    False,
                "seen_count":   1,
                "alert_issued": False,
            })
            logger.debug(f"New face registered: {face_info['face_hash']} | track={track_id}")

        # Save blurred crop to disk (optional, for dashboard)
        if STORE_FACE_CROPS and not STORE_RAW_FACES:
            self._save_blurred_crop(person_crop, face, result["face_hash"], track_id)

        return result

    def _save_blurred_crop(self, crop: np.ndarray, face, face_hash: str, track_id: int):
        """Save a blurred version of the face crop for dashboard display."""
        try:
            blurred = crop.copy()
            if hasattr(face, "bbox"):
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                blurred = blur_face_region(blurred, (fx1, fy1, fx2, fy2))
            save_path = Path(FACES_DIR) / f"{face_hash}.jpg"
            if not save_path.exists():   # only save once
                cv2.imwrite(str(save_path), blurred)
        except Exception as e:
            logger.debug(f"Could not save face crop: {e}")

    def mark_warned(self, face_hash: str):
        """Mark that a yellow warning has been issued for this face."""
        mark_alert_issued(face_hash)
        for c in self._cache:
            if c["face_hash"] == face_hash:
                c["alert_issued"] = True
                break

    def cache_size(self) -> int:
        return len(self._cache)
