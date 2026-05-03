"""Card recognition via perceptual hashing.

Pipeline:
  1. Preprocess the captured image with OpenCV (grayscale, CLAHE, resize).
  2. Compute a 64-bit pHash with imagehash.
  3. Linear scan against the in-memory hash table loaded from SQLite.
  4. Return the closest match if it's within HASH_MATCH_THRESHOLD.

The hash table (~240 KB for 30 k cards) fits comfortably in memory and
makes linear Hamming-distance search fast enough (< 5 ms on a modern CPU).
"""

import sqlite3
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import imagehash

from config import HASH_MATCH_THRESHOLD, resolve_db_path

logger = logging.getLogger(__name__)

# Lazy-loaded in-memory table: list of (name, hash_as_int)
_db: Optional[list] = None

# Standard normalised size for hashing — matches what db_builder uses
_HASH_W, _HASH_H = 256, 358


def _load_db() -> None:
    global _db
    if _db is not None:
        return

    db_path = resolve_db_path()
    if not db_path.exists():
        logger.warning("Hash database not found. Run db_builder.py first.")
        _db = []
        return

    logger.info("Loading hash database from %s", db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name, hash_int FROM cards WHERE hash_int IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    # Convert signed SQLite integers back to unsigned for XOR comparison
    _db = [(name, h & 0xFFFFFFFFFFFFFFFF) for name, h in rows]
    logger.info("Loaded %d card hashes from database", len(_db))


def reload_db() -> None:
    """Force a fresh load from disk (e.g. after db_builder runs)."""
    global _db
    _db = None
    _load_db()


def db_is_ready() -> bool:
    _load_db()
    return bool(_db)


def _preprocess(img: Image.Image) -> Image.Image:
    """Convert to grayscale, enhance local contrast, resize to standard dims."""
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    # CLAHE improves matching under variable screen brightness/gamma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    arr = cv2.resize(arr, (_HASH_W, _HASH_H), interpolation=cv2.INTER_AREA)
    return Image.fromarray(arr)


def compute_hash(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(_preprocess(img))


def find_best_match(img: Image.Image) -> Optional[Tuple[str, int]]:
    """Return (card_name, hamming_distance) for the closest match, or None."""
    _load_db()
    if not _db:
        return None

    query_int = int(str(compute_hash(img)), 16)

    best_name: Optional[str] = None
    best_dist = HASH_MATCH_THRESHOLD + 1

    for name, hash_int in _db:
        dist = bin(query_int ^ hash_int).count("1")
        if dist < best_dist:
            best_dist = dist
            best_name = name
            if dist == 0:
                break  # perfect match; stop early

    if best_name and best_dist <= HASH_MATCH_THRESHOLD:
        logger.info("Recognised '%s' (Hamming distance %d)", best_name, best_dist)
        return best_name, best_dist

    logger.warning("No match within threshold (best distance %d)", best_dist)
    return None
