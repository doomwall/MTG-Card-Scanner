"""Card boundary detection using OpenCV contour analysis.

Looks for quadrilaterals with an MTG-card aspect ratio inside a captured
image.  Returns the raw 4-point contours so the caller can either draw
them as overlays or warp them into a perspective-corrected crop.
"""

import logging
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MTG card: 63 mm × 88 mm  →  ratio ≈ 0.716  (portrait)
_CARD_RATIO   = 63.0 / 88.0
_RATIO_TOL    = 0.18   # ±18 % to absorb perspective skew
_MIN_AREA_PCT = 0.03   # contour must be at least 3 % of image area


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]     # smallest x+y  → top-left
    rect[2] = pts[np.argmax(s)]     # largest  x+y  → bottom-right
    rect[1] = pts[np.argmin(diff)]  # smallest x-y  → top-right
    rect[3] = pts[np.argmax(diff)]  # largest  x-y  → bottom-left
    return rect


def detect_cards(img: Image.Image) -> List[np.ndarray]:
    """Return a list of (4, 2) arrays, each the four corners of one card."""
    arr  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Blur reduces noise, making edges cleaner
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny with automatic thresholds via Otsu
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, otsu_thresh * 0.5, otsu_thresh)

    # Close small gaps in card borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.width * img.height
    cards: List[np.ndarray] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * _MIN_AREA_PCT:
            continue

        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)

        # Use the minimum-area rotated rect to get a reliable aspect ratio
        # even when the card is rotated or lightly skewed
        _, (w, h), _ = cv2.minAreaRect(approx)
        if w == 0 or h == 0:
            continue
        ratio = min(w, h) / max(w, h)

        # Accept portrait or landscape orientation
        portrait  = abs(ratio - _CARD_RATIO) <= _RATIO_TOL
        landscape = abs(ratio - (1.0 - _CARD_RATIO)) <= _RATIO_TOL
        if not (portrait or landscape):
            continue

        cards.append(_order_points(pts))
        logger.debug("Card detected: area=%.0f ratio=%.3f", area, ratio)

    logger.info("Detected %d card(s)", len(cards))
    return cards


def extract_card(img: Image.Image, quad: np.ndarray,
                 out_w: int = 256, out_h: int = 358) -> Image.Image:
    """Perspective-warp the quadrilateral region to a clean upright card."""
    src = _order_points(quad.astype(np.float32))
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
                   dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(np.array(img.convert("RGB")), M, (out_w, out_h))
    return Image.fromarray(arr)
