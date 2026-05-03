"""Fullscreen region-selection overlay.

Creates a semi-transparent Toplevel window where the user can drag a
rectangle to mark a card on screen.  Must be called from the main
tkinter thread.  Returns (x1, y1, x2, y2) in physical screen pixels,
or None if the user pressed Escape or dragged a region that was too small.
"""

import tkinter as tk
from typing import Optional, Tuple
import logging

from config import MIN_SELECTION_SIZE

logger = logging.getLogger(__name__)

# Colours used in the overlay
_BG = "#0d0d0d"
_RECT_OUTLINE = "#00ff41"   # bright green selection border
_RECT_FILL = "#00ff4115"    # near-transparent fill
_LABEL_FG = "white"
_HINT_TEXT = "Drag to select a card  •  ESC to cancel"


def select_region(parent: tk.Tk) -> Optional[Tuple[int, int, int, int]]:
    """Show the fullscreen selection overlay and block until the user
    finishes (or cancels).  Returns screen-pixel coordinates."""

    result: list = [None]
    start: list = [0, 0]
    rect_id: list = [None]

    win = tk.Toplevel(parent)
    win.attributes("-fullscreen", True)
    win.attributes("-topmost", True)
    win.attributes("-alpha", 0.35)
    win.overrideredirect(True)
    win.configure(bg=_BG, cursor="crosshair")

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()

    canvas = tk.Canvas(win, bg=_BG, cursor="crosshair", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    canvas.create_text(
        sw // 2, 28,
        text=_HINT_TEXT,
        fill=_LABEL_FG,
        font=("Segoe UI", 13, "bold"),
    )

    # ── event handlers ────────────────────────────────────────────────────

    def on_press(event: tk.Event) -> None:
        start[0], start[1] = event.x, event.y
        if rect_id[0]:
            canvas.delete(rect_id[0])
        rect_id[0] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=_RECT_OUTLINE,
            fill=_RECT_FILL,
            width=2,
        )

    def on_drag(event: tk.Event) -> None:
        if rect_id[0]:
            canvas.coords(rect_id[0], start[0], start[1], event.x, event.y)

    def on_release(event: tk.Event) -> None:
        x1 = min(start[0], event.x)
        y1 = min(start[1], event.y)
        x2 = max(start[0], event.x)
        y2 = max(start[1], event.y)
        if x2 - x1 >= MIN_SELECTION_SIZE and y2 - y1 >= MIN_SELECTION_SIZE:
            result[0] = (x1, y1, x2, y2)
            logger.debug("Region selected: (%d,%d)→(%d,%d)", x1, y1, x2, y2)
        else:
            logger.debug("Selection too small, discarded")
        win.destroy()

    def on_escape(_event: tk.Event) -> None:
        logger.debug("Selection cancelled by user")
        win.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    win.bind("<Escape>", on_escape)
    win.focus_force()

    # Block this call until the window is destroyed while still processing
    # events (safe because we're on the main tkinter thread).
    parent.wait_window(win)

    return result[0]
