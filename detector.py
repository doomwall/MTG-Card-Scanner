"""Card boundary detection using OpenCV contour analysis.

Tries multiple blur levels, Canny thresholds, and a CLAHE-enhanced pass
so card edges are found even in noisy or low-contrast webcam images.
Falls back to minAreaRect when approxPolyDP won't simplify a contour to
exactly 4 points.
"""

import logging
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_CARD_RATIO   = 63.0 / 88.0   # ≈ 0.716  (short / long side)
_RATIO_TOL    = 0.25           # ±25 % — generous to absorb perspective skew
_MIN_AREA_PCT = 0.02           # contour must cover at least 2 % of image area
_DEDUP_PX     = 40             # centres closer than this belong to the same card

# Preprocessing combinations to sweep
_BLURS   = [(3, 3), (5, 5), (9, 9)]
_CANNIES = [(20, 60), (50, 150), (100, 200)]


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return [top-left, top-right, bottom-right, bottom-left]."""
    pts  = pts.astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray:
    """Best-effort 4-point quad.

    Tries approxPolyDP with several epsilon values; if none yield exactly
    4 corners, falls back to the rotated minimum-area rectangle.
    """
    peri = cv2.arcLength(cnt, True)
    for eps in [0.01, 0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(cnt)
    return cv2.boxPoints(rect)


def _is_card_ratio(cnt: np.ndarray) -> bool:
    _, (w, h), _ = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return False
    ratio = min(w, h) / max(w, h)
    return abs(ratio - _CARD_RATIO) <= _RATIO_TOL


def detect_cards(img: Image.Image) -> List[np.ndarray]:
    """Return a list of (4, 2) float32 arrays, one per detected card."""
    arr      = np.array(img.convert("RGB"))
    gray     = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    img_area = img.width * img.height

    found:   List[np.ndarray] = []
    centers: List[tuple]      = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for blur_k in _BLURS:
        blurred  = cv2.GaussianBlur(gray, blur_k, 0)
        enhanced = clahe.apply(blurred)   # CLAHE pass helps low-contrast images

        for source in (blurred, enhanced):
            for lo, hi in _CANNIES:
                edges  = cv2.Canny(source, lo, hi)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                edges  = cv2.dilate(edges, kernel, iterations=2)

                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    if cv2.contourArea(cnt) < img_area * _MIN_AREA_PCT:
                        continue
                    if not _is_card_ratio(cnt):
                        continue

                    quad = _order_points(_quad_from_contour(cnt))
                    cx, cy = quad.mean(axis=0)

                    if any(abs(cx - ex) < _DEDUP_PX and abs(cy - ey) < _DEDUP_PX
                           for ex, ey in centers):
                        continue

                    centers.append((cx, cy))
                    found.append(quad)
                    logger.debug(
                        "Card found: center=(%.0f,%.0f) blur=%s canny=(%d,%d)",
                        cx, cy, blur_k, lo, hi,
                    )

    logger.info("detect_cards: %d card(s) found", len(found))
    return found


def extract_card(img: Image.Image, quad: np.ndarray,
                 out_w: int = 256, out_h: int = 358) -> Image.Image:
    """Perspective-warp the quad region into a clean upright card image."""
    src = _order_points(quad.astype(np.float32))
    dst = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32
    )
    M   = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(np.array(img.convert("RGB")), M, (out_w, out_h))
    return Image.fromarray(out)