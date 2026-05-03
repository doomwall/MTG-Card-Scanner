"""Card recognition via perceptual hashing.

Pipeline:
  1. Build several preprocessed variants of the capture (see _variants).
  2. Compute a 64-bit pHash for each variant.
  3. Linear scan against the in-memory hash table from SQLite.
  4. Return the closest match found across all variants.

Multiple variants help because we don't know up-front whether the webcam
image needs denoising, sharpening, or just contrast normalisation.
"""

import sqlite3
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import imagehash

from config import HASH_MATCH_THRESHOLD, resolve_db_path

logger = logging.getLogger(__name__)

_db: Optional[list] = None
_HASH_W, _HASH_H = 256, 358

# How much the hue hash contributes to the combined score.
# 0.0 = ignore colour entirely  /  1.0 = weight equal to grayscale.
HUE_WEIGHT = 0.5

# Unsharp-mask kernel — enhances edges without over-amplifying noise
_SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)


# ── database ──────────────────────────────────────────────────────────────────

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
            "SELECT name, hash_int, hue_hash_int FROM cards WHERE hash_int IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    # Load both hashes; hue_hash_int may be NULL for old DB rows
    _db = [
        (name,
         gray & 0xFFFFFFFFFFFFFFFF,
         (hue & 0xFFFFFFFFFFFFFFFF) if hue is not None else None)
        for name, gray, hue in rows
    ]
    logger.info("Loaded %d card hashes from database", len(_db))


def reload_db() -> None:
    global _db
    _db = None
    _load_db()


def db_is_ready() -> bool:
    _load_db()
    return bool(_db)


# ── preprocessing ─────────────────────────────────────────────────────────────

def _to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _apply_clahe(arr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(arr)


def _resize(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (_HASH_W, _HASH_H), interpolation=cv2.INTER_AREA)


def _variants(img: Image.Image) -> List[Image.Image]:
    """Return several preprocessed versions of the capture to try.

    v1  Plain CLAHE — works well for clean screen captures.
    v2  Denoise → CLAHE — removes webcam sensor noise first.
    v3  Denoise → sharpen → CLAHE — recovers detail lost to webcam blur.
    v4  Bilateral filter → CLAHE — edge-preserving smooth for low-light shots.
    """
    gray = _to_gray(img)
    out = []

    # v1: plain CLAHE
    v1 = _apply_clahe(gray.copy())
    out.append(Image.fromarray(_resize(v1)))

    # v2: fast denoise → CLAHE
    v2 = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    v2 = _apply_clahe(v2)
    out.append(Image.fromarray(_resize(v2)))

    # v3: denoise → sharpen → CLAHE
    v3 = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    v3 = cv2.filter2D(v3, -1, _SHARPEN_KERNEL)
    v3 = np.clip(v3, 0, 255).astype(np.uint8)
    v3 = _apply_clahe(v3, clip=3.0)
    out.append(Image.fromarray(_resize(v3)))

    # v4: bilateral (preserves edges, smooths flat areas) → CLAHE
    v4 = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    v4 = _apply_clahe(v4)
    out.append(Image.fromarray(_resize(v4)))

    return out


# ── hue hash ─────────────────────────────────────────────────────────────────

def _compute_hue_hash(img: Image.Image) -> int:
    """pHash of the hue channel — captures frame/background colour."""
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue_img = Image.fromarray(hsv[:, :, 0]).resize((_HASH_W, _HASH_H))
    return int(str(imagehash.phash(hue_img)), 16)


# ── matching ──────────────────────────────────────────────────────────────────

def find_best_match(img: Image.Image) -> Optional[Tuple[str, int]]:
    """Return (card_name, hamming_distance) for the closest match, or None.

    Matching uses two pHashes per card:
      gray_dist  — structural / artwork similarity (from multiple preprocessings)
      hue_dist   — frame/background colour similarity

    Combined score = gray_dist + HUE_WEIGHT * hue_dist
    Cards with similar art but a wrong frame colour are pushed down the ranking.
    """
    _load_db()
    if not _db:
        return None

    query_grays = [int(str(imagehash.phash(v)), 16) for v in _variants(img)]
    query_hue   = _compute_hue_hash(img)

    best_name:  Optional[str] = None
    best_score: float         = float("inf")
    best_dist:  int           = HASH_MATCH_THRESHOLD + 1

    for name, gray_int, hue_int in _db:
        gray_dist = min(bin(q ^ gray_int).count("1") for q in query_grays)

        # Only score candidates that could possibly be within threshold
        if gray_dist > HASH_MATCH_THRESHOLD:
            continue

        if hue_int is not None:
            hue_dist = bin(query_hue ^ hue_int).count("1")
        else:
            hue_dist = 0  # old DB row without hue hash — no colour adjustment

        score = gray_dist + HUE_WEIGHT * hue_dist

        if score < best_score:
            best_score = score
            best_name  = name
            best_dist  = gray_dist

    if best_name and best_dist <= HASH_MATCH_THRESHOLD:
        logger.info(
            "Recognised '%s'  (gray_dist=%d  score=%.1f)",
            best_name, best_dist, best_score,
        )
        return best_name, best_dist

    logger.warning("No match within threshold")
    return None
