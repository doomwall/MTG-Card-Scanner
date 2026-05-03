"""Card recognition via perceptual hashing with colour re-ranking.

Pipeline:
  1. Build several preprocessed variants of the capture (see _variants).
  2. Compute a 64-bit pHash for each variant.
  3. Linear scan against the in-memory hash table from SQLite.
     Collect the top-N candidates within a wide search radius.
  4. Sample the card's frame regions (name bar + text box) to detect the
     approximate frame colour (light / dark / red / blue / green / gold).
  5. Apply a colour penalty to candidates whose Scryfall colours field
     contradicts the detected frame colour, then return the best scorer.

Step 5 corrects the common failure mode where two cards share similar
artwork but have different frame colours (e.g. white vs. red card).
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
_TOP_N = 5          # candidates to colour-check after pHash scan
_WIDE  = 2          # multiplier on threshold for the initial wide search

_SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)

# Maps a single Scryfall colour letter to the frame colour category we detect
_SCRYFALL_TO_FRAME = {
    "W": "light",
    "U": "blue",
    "B": "dark",
    "R": "red",
    "G": "green",
}

# Pairs of frame colours that can never belong to the same card frame
_INCOMPATIBLE = {
    frozenset({"light", "dark"}),
    frozenset({"light", "red"}),
    frozenset({"light", "blue"}),
    frozenset({"light", "green"}),
    frozenset({"red",   "blue"}),
    frozenset({"red",   "green"}),
    frozenset({"red",   "dark"}),
    frozenset({"blue",  "green"}),
    frozenset({"blue",  "dark"}),
    frozenset({"green", "dark"}),
}


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
            "SELECT name, hash_int FROM cards WHERE hash_int IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    _db = [(name, h & 0xFFFFFFFFFFFFFFFF) for name, h in rows]
    logger.info("Loaded %d card hashes from database", len(_db))


def reload_db() -> None:
    global _db
    _db = None
    _load_db()


def db_is_ready() -> bool:
    _load_db()
    return bool(_db)


# ── image preprocessing ───────────────────────────────────────────────────────

def _to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _apply_clahe(arr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(arr)


def _resize(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (_HASH_W, _HASH_H), interpolation=cv2.INTER_AREA)


def _variants(img: Image.Image) -> List[Image.Image]:
    """Several preprocessed grayscale versions to try against the hash DB."""
    gray = _to_gray(img)
    out  = []

    v1 = _apply_clahe(gray.copy())
    out.append(Image.fromarray(_resize(v1)))

    v2 = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    v2 = _apply_clahe(v2)
    out.append(Image.fromarray(_resize(v2)))

    v3 = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    v3 = cv2.filter2D(v3, -1, _SHARPEN_KERNEL)
    v3 = np.clip(v3, 0, 255).astype(np.uint8)
    v3 = _apply_clahe(v3, clip=3.0)
    out.append(Image.fromarray(_resize(v3)))

    v4 = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    v4 = _apply_clahe(v4)
    out.append(Image.fromarray(_resize(v4)))

    return out


# ── frame colour detection ────────────────────────────────────────────────────

def _detect_frame_color(img: Image.Image) -> str:
    """Sample the name-bar and text-box regions and return a colour label.

    Returns one of: 'light', 'dark', 'red', 'blue', 'green', 'gold',
    or 'unknown' when the image is too small or ambiguous.

    These regions sit outside the card art so they reflect the frame
    colour rather than the artwork.
    """
    w, h = img.size
    if w < 30 or h < 30:
        return "unknown"

    # Name bar:  top ~3-10 % of card height (below the very top border)
    # Text box:  ~63-87 % of card height
    name_bar = np.array(
        img.crop((int(w * 0.08), int(h * 0.03), int(w * 0.92), int(h * 0.10)))
           .resize((60, 10))
    )
    text_box = np.array(
        img.crop((int(w * 0.06), int(h * 0.63), int(w * 0.94), int(h * 0.87)))
           .resize((60, 30))
    )

    sample = np.vstack([name_bar, text_box])
    hsv    = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)

    sat = hsv[:, :, 1].flatten().astype(float)
    val = hsv[:, :, 2].flatten().astype(float)
    hue = hsv[:, :, 0].flatten().astype(float)

    avg_sat = sat.mean()
    avg_val = val.mean()

    # Low saturation → achromatic frame (white or black)
    if avg_sat < 45:
        detected = "light" if avg_val > 150 else "dark"
        logger.debug("Frame colour: %s  (sat=%.0f val=%.0f)", detected, avg_sat, avg_val)
        return detected

    # Chromatic: look at the hue of saturated pixels only
    mask           = sat > 60
    saturated_hue  = hue[mask]
    if saturated_hue.size < 10:
        return "unknown"

    avg_h = saturated_hue.mean()

    # OpenCV hue: 0-180 (half the 0-360 circle)
    if avg_h < 15 or avg_h > 165:
        detected = "red"
    elif avg_h < 30:
        detected = "gold"    # orange-gold = multicolour cards
    elif avg_h < 85:
        detected = "green"
    elif avg_h < 130:
        detected = "blue"
    else:
        detected = "dark"    # purple/indigo often accompanies black frames

    logger.debug("Frame colour: %s  (hue=%.0f sat=%.0f)", detected, avg_h, avg_sat)
    return detected


def _color_penalty(detected: str, card_name: str) -> int:
    """Return a Hamming-distance penalty based on colour compatibility.

    +10  clear mismatch (e.g. white frame detected, but card is red)
     -2  colour match (small bonus to prefer correct-colour candidates)
      0  uncertain or colourless — no adjustment
    """
    if detected == "unknown":
        return 0

    # Import here to avoid a module-level circular dependency risk
    from scryfall import get_card_by_exact_name
    card = get_card_by_exact_name(card_name)
    if not card or card.get("object") == "error":
        return 0

    colors = card.get("colors", [])

    if len(colors) == 0:
        return 0          # colourless/artifact frames vary widely — skip check
    elif len(colors) > 1:
        expected = "gold"
    else:
        expected = _SCRYFALL_TO_FRAME.get(colors[0], "unknown")

    if expected == "unknown":
        return 0

    if detected == expected:
        return -2

    if frozenset({detected, expected}) in _INCOMPATIBLE:
        logger.debug(
            "Colour mismatch for '%s': detected=%s expected=%s",
            card_name, detected, expected,
        )
        return 10

    return 0


# ── matching ──────────────────────────────────────────────────────────────────

def find_best_match(img: Image.Image) -> Optional[Tuple[str, int]]:
    """Return (card_name, hamming_distance) for the best match, or None."""
    _load_db()
    if not _db:
        return None

    query_ints = [int(str(imagehash.phash(v)), 16) for v in _variants(img)]

    # ── phase 1: collect top-N candidates within a wide radius ────────────
    wide_threshold = HASH_MATCH_THRESHOLD * _WIDE
    candidates: List[Tuple[int, str]] = []  # (distance, name)

    for name, hash_int in _db:
        dist = min(bin(q ^ hash_int).count("1") for q in query_ints)
        if dist <= wide_threshold:
            candidates.append((dist, name))

    if not candidates:
        logger.warning("No candidates within 2× threshold")
        return None

    candidates.sort()
    top = candidates[:_TOP_N]

    # ── phase 2: colour re-ranking ────────────────────────────────────────
    detected_color = _detect_frame_color(img)
    logger.debug("Detected frame colour: %s", detected_color)

    best_name:  Optional[str] = None
    best_score: int           = wide_threshold + 1
    best_dist:  int           = wide_threshold + 1

    for dist, name in top:
        penalty = _color_penalty(detected_color, name)
        score   = dist + penalty
        logger.debug("  candidate '%s'  dist=%d  penalty=%d  score=%d",
                     name, dist, penalty, score)
        if score < best_score:
            best_score = score
            best_name  = name
            best_dist  = dist

    # Final gate: the raw pHash distance must still be within the threshold
    if best_name and best_dist <= HASH_MATCH_THRESHOLD:
        logger.info(
            "Recognised '%s'  (pHash dist=%d  frame=%s)",
            best_name, best_dist, detected_color,
        )
        return best_name, best_dist

    logger.warning("No match within threshold (best dist=%d)", best_dist)
    return None