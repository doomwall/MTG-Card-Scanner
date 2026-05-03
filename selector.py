"""Fullscreen region-selection overlay.

Takes a screenshot, shows a pre-dimmed version as the background, then
reveals the original (undimmed) screenshot inside the drag rectangle so
the card stands out clearly from the rest of the screen.

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
_OUTLINE = "#00ff41"
_DIM     = 0.45   # how dark the overlay is (0 = black, 1 = original)


def select_region(
    parent: tk.Tk,
) -> Optional[Tuple[Tuple[int, int, int, int], "Image.Image"]]:
    result: list = [None]
    start:  list = [0, 0]

    # ── 1. grab screen before the overlay appears ─────────────────────────
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        bg_orig = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    # Pre-compute dimmed version once (blend with black at DIM factor)
    black = Image.new("RGB", bg_orig.size, (0, 0, 0))
    bg_dim = Image.blend(bg_orig, black, alpha=1.0 - _DIM)

    # ── 2. fullscreen window ──────────────────────────────────────────────
    win = tk.Toplevel(parent)
    win.attributes("-fullscreen", True)
    win.attributes("-topmost", True)
    win.overrideredirect(True)
    win.configure(cursor="crosshair")

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()

    canvas = tk.Canvas(win, cursor="crosshair", highlightthickness=0, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)

    # Dimmed background (kept as attribute to prevent GC)
    _tk_dim = ImageTk.PhotoImage(bg_dim)
    canvas.create_image(0, 0, image=_tk_dim, anchor="nw")

    # Selection reveal image — starts as 1×1 placeholder
    _tk_sel = [ImageTk.PhotoImage(Image.new("RGB", (1, 1)))]
    sel_img = canvas.create_image(0, 0, image=_tk_sel[0], anchor="nw")

    # Selection border and size label drawn on top
    border = canvas.create_rectangle(0, 0, 0, 0, outline=_OUTLINE, width=2, fill="")
    sizelabel = canvas.create_text(
        0, 0, text="", anchor="nw",
        fill=_OUTLINE, font=("Segoe UI", 9, "bold"),
    )

    # Hint text
    canvas.create_text(
        sw // 2, 28, text=_HINT,
        fill="white", font=("Segoe UI", 13, "bold"),
    )

    # ── event handlers ────────────────────────────────────────────────────

    def on_press(event: tk.Event) -> None:
        start[0], start[1] = event.x, event.y

    def on_drag(event: tk.Event) -> None:
        x1 = min(start[0], event.x)
        y1 = min(start[1], event.y)
        x2 = max(start[0], event.x)
        y2 = max(start[1], event.y)

        if x2 > x1 and y2 > y1:
            # Reveal original screenshot inside the selection
            crop = bg_orig.crop((x1, y1, x2, y2))
            _tk_sel[0] = ImageTk.PhotoImage(crop)
            canvas.coords(sel_img, x1, y1)
            canvas.itemconfig(sel_img, image=_tk_sel[0])

        canvas.coords(border, x1, y1, x2, y2)

        lx = x1 + 4
        ly = y1 - 16 if y1 > 22 else y2 + 4
        canvas.coords(sizelabel, lx, ly)
        canvas.itemconfig(sizelabel, text=f"{x2 - x1} × {y2 - y1} px")

    def on_release(event: tk.Event) -> None:
        x1 = min(start[0], event.x)
        y1 = min(start[1], event.y)
        x2 = max(start[0], event.x)
        y2 = max(start[1], event.y)
        if x2 - x1 >= MIN_SELECTION_SIZE and y2 - y1 >= MIN_SELECTION_SIZE:
            # Return the region AND the screenshot already in memory —
            # the caller crops from this instead of taking a second screenshot.
            result[0] = ((x1, y1, x2, y2), bg_orig)
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