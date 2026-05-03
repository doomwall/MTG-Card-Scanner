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

# ── OCR reader (lazy-initialised on first use) ────────────────────────────────
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Loading EasyOCR model (first run may take a moment)…")
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR ready")
    return _ocr_reader

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
    return True  # OCR mode — no hash DB needed


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


# ── hue helpers ───────────────────────────────────────────────────────────────

def _masked_hue(arr: np.ndarray) -> np.ndarray:
    """Return the hue channel with achromatic pixels zeroed out.

    Pixels with low saturation have no meaningful hue (white, black, grey
    all return random hue noise).  Setting them to 0 keeps the image clean
    and stops undefined values from polluting the hash.
    """
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].copy()
    hue[hsv[:, :, 1] < 30] = 0   # silence achromatic pixels
    return hue


def _compute_hue_hash(img: Image.Image) -> int:
    """pHash of the saturation-masked hue channel."""
    arr = np.array(img.convert("RGB"))
    hue_img = Image.fromarray(_masked_hue(arr)).resize((_HASH_W, _HASH_H))
    return int(str(imagehash.phash(hue_img)), 16)


# ── debug helper ─────────────────────────────────────────────────────────────

def get_preprocessing_steps(img: Image.Image) -> List[tuple]:
    """Return (label, image, selectable) tuples for every pipeline stage.

    selectable=True marks the four final variants the user can click to
    force a specific preprocessing for the hash comparison.
    """
    gray = _to_gray(img)
    steps: List[tuple] = []

    steps.append(("Original",   img.convert("RGB"),        False))
    steps.append(("Greyscale",  Image.fromarray(gray),     False))

    v1 = _apply_clahe(gray.copy())
    steps.append(("v1 — CLAHE",              Image.fromarray(_resize(v1)),  True))

    v2 = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    steps.append(("v2 — Denoise",            Image.fromarray(v2),           False))
    steps.append(("v2 — Denoise + CLAHE",    Image.fromarray(_resize(_apply_clahe(v2))), True))

    v3 = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    v3s = np.clip(cv2.filter2D(v3, -1, _SHARPEN_KERNEL), 0, 255).astype(np.uint8)
    steps.append(("v3 — Denoise + Sharpen",  Image.fromarray(v3s),          False))
    steps.append(("v3 — + CLAHE",            Image.fromarray(_resize(_apply_clahe(v3s, clip=3.0))), True))

    v4 = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    steps.append(("v4 — Bilateral",          Image.fromarray(v4),           False))
    steps.append(("v4 — Bilateral + CLAHE",  Image.fromarray(_resize(_apply_clahe(v4))), True))

    steps.append(("Hue (masked)",
                  Image.fromarray(_masked_hue(np.array(img.convert("RGB")))),
                  False))

    return steps


# ── matching ──────────────────────────────────────────────────────────────────

def find_best_match(
    img: Image.Image,
    force_gray: Optional[Image.Image] = None,  # unused in OCR mode, kept for compat
) -> Optional[Tuple[str, int]]:
    """Read the card name from the name-bar region using OCR."""
    w, h = img.size

    # Crop the name bar — top strip of the card, left of the mana cost
    name_bar = img.crop((
        int(w * 0.04), int(h * 0.03),
        int(w * 0.85), int(h * 0.11),
    ))

    # Scale up so OCR sees larger text
    scale = 4
    name_bar = name_bar.resize(
        (name_bar.width * scale, name_bar.height * scale),
        Image.LANCZOS,
    )

    # High-contrast greyscale helps OCR
    arr = cv2.cvtColor(np.array(name_bar), cv2.COLOR_RGB2GRAY)
    arr = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )

    reader  = _get_ocr_reader()
    results = reader.readtext(arr, detail=1)

    if not results:
        logger.warning("OCR returned no text")
        return None

    # Pick the result with the highest confidence
    best      = max(results, key=lambda r: r[2])
    card_name = best[1].strip()
    confidence = best[2]           # 0.0–1.0

    if not card_name:
        return None

    # Map confidence to a pseudo-distance (0 = perfect, 64 = worst)
    pseudo_dist = int((1.0 - confidence) * 64)
    logger.info("OCR: '%s'  (confidence=%.2f)", card_name, confidence)
    return card_name, pseudo_dist
