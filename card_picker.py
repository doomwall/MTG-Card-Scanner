"""Card-picker overlay.

Shows the captured image with detected card outlines drawn on it.
The user clicks inside a highlighted card to select it.
Returns a perspective-corrected PIL Image of the chosen card, or None.

Special cases handled automatically:
  • 0 cards detected → returns None  (caller falls back to full capture)
  • 1 card  detected → returns it immediately, no click needed
  • 2+ cards         → shows the picker window
"""

import tkinter as tk
from typing import List, Optional
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from detector import detect_cards, extract_card

logger = logging.getLogger(__name__)

_HINT      = "Click the card you want to identify  •  ESC to cancel"
_OUTLINE   = (0, 255, 65)      # green
_HIGHLIGHT = (255, 200, 0)     # yellow on hover
_MAX_W     = 900               # max display width (scaled down for large captures)
_BORDER    = 3                 # outline thickness in pixels


def _scale_image(img: Image.Image) -> tuple[Image.Image, float]:
    """Scale image down if wider than _MAX_W; return (scaled_img, scale)."""
    if img.width <= _MAX_W:
        return img, 1.0
    scale = _MAX_W / img.width
    new_h = int(img.height * scale)
    return img.resize((_MAX_W, new_h), Image.LANCZOS), scale


def _draw_outlines(img: Image.Image, quads: List[np.ndarray],
                   highlight: Optional[int] = None) -> Image.Image:
    """Return a copy of img with card quads drawn on it."""
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    for i, quad in enumerate(quads):
        colour = _HIGHLIGHT if i == highlight else _OUTLINE
        pts    = [tuple(p) for p in quad.astype(int)]
        draw.polygon(pts, outline=colour)
        # Draw border a few pixels thick by slightly shrinking/expanding
        for t in range(1, _BORDER + 1):
            scaled = _shrink_quad(quad, t)
            draw.polygon([tuple(p) for p in scaled.astype(int)], outline=colour)
    return out


def _shrink_quad(quad: np.ndarray, px: int) -> np.ndarray:
    """Inset a quad by px pixels toward its centroid."""
    cx, cy = quad.mean(axis=0)
    dirs   = np.array([[cx - p[0], cy - p[1]] for p in quad], dtype=float)
    norms  = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1, norms)
    return quad + (dirs / norms * px).astype(np.float32)


def _point_in_quad(x: int, y: int, quad: np.ndarray) -> bool:
    pt  = (float(x), float(y))
    cnt = quad.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.pointPolygonTest(cnt, pt, False) >= 0


def pick_card(parent: tk.Tk, img: Image.Image) -> Optional[Image.Image]:
    """Detect cards in img and let the user pick one.  Returns extracted card."""

    quads = detect_cards(img)

    if not quads:
        logger.info("No cards detected — using full capture")
        return None

    if len(quads) == 1:
        logger.info("Single card detected — auto-selecting")
        return extract_card(img, quads[0])

    # ── multiple cards: show picker ───────────────────────────────────────
    logger.info("%d cards detected — showing picker", len(quads))

    display, scale = _scale_image(img)
    scaled_quads   = [q * scale for q in quads]

    result:   list = [None]
    hovered:  list = [-1]
    _tk_img:  list = [None]

    win = tk.Toplevel(parent)
    win.title("Select a card")
    win.attributes("-topmost", True)
    win.resizable(False, False)

    # Hint label
    tk.Label(
        win, text=_HINT,
        bg="#1e1e2e", fg="white",
        font=("Segoe UI", 10),
    ).pack(fill="x", padx=0, pady=(0, 0))

    canvas = tk.Canvas(
        win,
        width=display.width,
        height=display.height,
        highlightthickness=0,
        cursor="hand2",
    )
    canvas.pack()

    def _refresh(hi: int = -1) -> None:
        frame        = _draw_outlines(display, scaled_quads, highlight=hi)
        _tk_img[0]   = ImageTk.PhotoImage(frame)
        canvas.itemconfig(bg_item, image=_tk_img[0])

    _tk_img[0] = ImageTk.PhotoImage(_draw_outlines(display, scaled_quads))
    bg_item    = canvas.create_image(0, 0, image=_tk_img[0], anchor="nw")

    def on_move(event: tk.Event) -> None:
        for i, q in enumerate(scaled_quads):
            if _point_in_quad(event.x, event.y, q):
                if hovered[0] != i:
                    hovered[0] = i
                    _refresh(i)
                return
        if hovered[0] != -1:
            hovered[0] = -1
            _refresh()

    def on_click(event: tk.Event) -> None:
        for i, q in enumerate(scaled_quads):
            if _point_in_quad(event.x, event.y, q):
                result[0] = extract_card(img, quads[i])
                win.destroy()
                return

    def on_escape(_event: tk.Event) -> None:
        win.destroy()

    canvas.bind("<Motion>",        on_move)
    canvas.bind("<ButtonPress-1>", on_click)
    win.bind("<Escape>",           on_escape)

    parent.wait_window(win)
    return result[0]
