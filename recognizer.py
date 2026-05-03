"""Card recognition — pHash with OCR fallback.

Phase 1 — perceptual hashing:
  Tries four preprocessed variants of the capture (CLAHE, denoise,
  denoise+sharpen, bilateral) against the SQLite hash database.
  Returns immediately if a match is found within HASH_MATCH_THRESHOLD.

Phase 2 — OCR fallback (only reached when pHash finds nothing):
  Crops the card name-bar, sharpens it, and reads the text with EasyOCR.
  The result is passed to Scryfall's fuzzy-name endpoint which handles
  minor OCR errors.
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
_HASH_W, _HASH_H    = 256, 358
HUE_WEIGHT          = 0.5

_SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)

# EasyOCR reader — lazy-initialised on first OCR attempt
_ocr_reader = None


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


# ── preprocessing helpers ─────────────────────────────────────────────────────

def _to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _apply_clahe(arr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(arr)


def _resize(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (_HASH_W, _HASH_H), interpolation=cv2.INTER_AREA)


def _variants(img: Image.Image) -> List[Image.Image]:
    """Four preprocessed greyscale versions tried against the hash DB."""
    gray = _to_gray(img)

    v1 = _apply_clahe(gray.copy())

    v2 = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    v2 = _apply_clahe(v2)

    v3 = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    v3 = np.clip(cv2.filter2D(v3, -1, _SHARPEN_KERNEL), 0, 255).astype(np.uint8)
    v3 = _apply_clahe(v3, clip=3.0)

    v4 = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    v4 = _apply_clahe(v4)

    return [Image.fromarray(_resize(a)) for a in (v1, v2, v3, v4)]


def _masked_hue(arr: np.ndarray) -> np.ndarray:
    """Hue channel with achromatic pixels zeroed (they carry no colour info)."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].copy()
    hue[hsv[:, :, 1] < 30] = 0
    return hue


def _compute_hue_hash(img: Image.Image) -> int:
    arr = np.array(img.convert("RGB"))
    return int(str(imagehash.phash(
        Image.fromarray(_masked_hue(arr)).resize((_HASH_W, _HASH_H))
    )), 16)


# ── debug helper ──────────────────────────────────────────────────────────────

def get_preprocessing_steps(img: Image.Image) -> List[tuple]:
    """Return (label, image, selectable) tuples for the debug grid window."""
    gray  = _to_gray(img)
    steps: List[tuple] = []

    steps.append(("Original",             img.convert("RGB"),        False))
    steps.append(("Greyscale",            Image.fromarray(gray),     False))

    v1 = _apply_clahe(gray.copy())
    steps.append(("v1 — CLAHE",           Image.fromarray(_resize(v1)),  True))

    v2 = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    steps.append(("v2 — Denoise",         Image.fromarray(v2),           False))
    steps.append(("v2 — Denoise + CLAHE", Image.fromarray(_resize(_apply_clahe(v2))), True))

    v3 = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    v3s = np.clip(cv2.filter2D(v3, -1, _SHARPEN_KERNEL), 0, 255).astype(np.uint8)
    steps.append(("v3 — Denoise + Sharpen", Image.fromarray(v3s),        False))
    steps.append(("v3 — + CLAHE",           Image.fromarray(_resize(_apply_clahe(v3s, clip=3.0))), True))

    v4 = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    steps.append(("v4 — Bilateral",          Image.fromarray(v4),        False))
    steps.append(("v4 — Bilateral + CLAHE",  Image.fromarray(_resize(_apply_clahe(v4))), True))

    steps.append(("Hue (masked)",
                  Image.fromarray(_masked_hue(np.array(img.convert("RGB")))),
                  False))

    return steps


# ── phase 1: perceptual hash ──────────────────────────────────────────────────

def _phash_match(
    img: Image.Image,
    force_gray: Optional[Image.Image] = None,
) -> Optional[Tuple[str, int]]:
    _load_db()
    if not _db:
        return None

    query_grays = (
        [int(str(imagehash.phash(force_gray)), 16)]
        if force_gray is not None
        else [int(str(imagehash.phash(v)), 16) for v in _variants(img)]
    )
    query_hue = _compute_hue_hash(img)

    best_name:  Optional[str] = None
    best_score: float         = float("inf")
    best_dist:  int           = HASH_MATCH_THRESHOLD + 1

    for name, gray_int, hue_int in _db:
        gray_dist = min(bin(q ^ gray_int).count("1") for q in query_grays)
        if gray_dist > HASH_MATCH_THRESHOLD:
            continue

        hue_dist = bin(query_hue ^ hue_int).count("1") if hue_int is not None else 0
        score    = gray_dist + HUE_WEIGHT * hue_dist

        if score < best_score:
            best_score = score
            best_name  = name
            best_dist  = gray_dist

    if best_name and best_dist <= HASH_MATCH_THRESHOLD:
        logger.info("pHash match: '%s' (dist=%d score=%.1f)", best_name, best_dist, best_score)
        return best_name, best_dist

    return None


# ── phase 2: OCR fallback ─────────────────────────────────────────────────────

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Loading EasyOCR model (first OCR attempt may be slow)…")
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR ready")
    return _ocr_reader


def _ocr_match(img: Image.Image) -> Optional[Tuple[str, int]]:
    w, h = img.size

    # Crop name bar — top strip, left of the mana cost symbols
    name_bar = img.crop((int(w * 0.04), int(h * 0.03),
                         int(w * 0.85), int(h * 0.11)))

    # Scale up and sharpen so OCR sees larger, crisper text
    name_bar = name_bar.resize((name_bar.width * 4, name_bar.height * 4), Image.LANCZOS)
    arr = cv2.cvtColor(np.array(name_bar), cv2.COLOR_RGB2GRAY)
    arr = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    results = _get_ocr_reader().readtext(arr, detail=1)
    if not results:
        logger.warning("OCR: no text found")
        return None

    best       = max(results, key=lambda r: r[2])
    card_name  = best[1].strip()
    confidence = best[2]

    if not card_name:
        return None

    pseudo_dist = int((1.0 - confidence) * 64)
    logger.info("OCR match: '%s' (confidence=%.2f)", card_name, confidence)
    return card_name, pseudo_dist


# ── public interface ──────────────────────────────────────────────────────────

def find_best_match(
    img: Image.Image,
    force_gray: Optional[Image.Image] = None,
) -> Optional[Tuple[str, int]]:
    """Try pHash first; fall back to OCR if no hash match is found."""

    result = _phash_match(img, force_gray)
    if result is not None:
        return result

    logger.info("pHash found no match — trying OCR fallback")
    return _ocr_match(img)