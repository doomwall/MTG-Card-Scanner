"""Fullscreen region-selection overlay.

Takes a screenshot of the screen, displays it as a frozen background,
then lets the user drag a selection rectangle over it at full opacity.
The selection border and size label are always crisp regardless of the
underlying content.

Must be called from the main tkinter thread.
Returns (x1, y1, x2, y2) in physical screen pixels, or None if cancelled.
"""

import tkinter as tk
from typing import Optional, Tuple
import logging

import mss
from PIL import Image, ImageTk

from config import MIN_SELECTION_SIZE

logger = logging.getLogger(__name__)

_HINT    = "Drag to select a card  •  ESC to cancel"
_OUTLINE = "#00ff41"   # bright green selection border
_DIM     = "gray25"    # stipple pattern for the dark overlay (~25% opacity)
_CLEAR   = "gray75"    # stipple for inside the selection (lighter dimming)


def select_region(parent: tk.Tk) -> Optional[Tuple[int, int, int, int]]:
    result: list = [None]
    start:  list = [0, 0]

    # ── 1. capture the screen before the overlay window appears ──────────
    with mss.mss() as sct:
        monitor = sct.monitors[1]   # primary monitor
        shot = sct.grab(monitor)
        bg = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    # ── 2. build fullscreen window at full opacity ────────────────────────
    win = tk.Toplevel(parent)
    win.attributes("-fullscreen", True)
    win.attributes("-topmost", True)
    win.overrideredirect(True)
    win.configure(cursor="crosshair")

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()

    canvas = tk.Canvas(win, cursor="crosshair", highlightthickness=0, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)

    # Keep a reference so the image isn't garbage-collected
    _tk_bg = ImageTk.PhotoImage(bg)
    canvas.create_image(0, 0, image=_tk_bg, anchor="nw", tags="bg")

    # Dark stipple over the whole screen
    canvas.create_rectangle(
        0, 0, sw, sh,
        fill="black", stipple=_DIM, outline="", tags="overlay"
    )

    # Instruction text
    canvas.create_text(
        sw // 2, 28,
        text=_HINT,
        fill="white",
        font=("Segoe UI", 13, "bold"),
        tags="hint",
    )

    # Placeholder canvas items for the live selection
    inside = canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple=_CLEAR, outline="")
    border = canvas.create_rectangle(0, 0, 0, 0, outline=_OUTLINE, width=2, fill="")
    sizelabel = canvas.create_text(0, 0, text="", fill=_OUTLINE, font=("Segoe UI", 9, "bold"))

    # ── event handlers ────────────────────────────────────────────────────

    def on_press(event: tk.Event) -> None:
        start[0], start[1] = event.x, event.y
        # Collapse selection to a point
        for item in (inside, border, sizelabel):
            canvas.coords(item, event.x, event.y, event.x, event.y)

    def on_drag(event: tk.Event) -> None:
        x1 = min(start[0], event.x)
        y1 = min(start[1], event.y)
        x2 = max(start[0], event.x)
        y2 = max(start[1], event.y)
        canvas.coords(inside, x1, y1, x2, y2)
        canvas.coords(border, x1, y1, x2, y2)
        # Size label just above the top-left corner, or below if near top edge
        lx = x1 + 4
        ly = y1 - 14 if y1 > 20 else y2 + 14
        canvas.coords(sizelabel, lx, ly)
        canvas.itemconfig(sizelabel, text=f"{x2 - x1} × {y2 - y1} px")

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

    parent.wait_window(win)
    return result[0]