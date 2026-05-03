"""Debug window — shows every preprocessing stage as a thumbnail grid.

Called from main.py when debug mode is active.  Non-blocking (no
wait_window) so the card-info popup can appear alongside it.
"""

import tkinter as tk
from typing import List, Tuple

from PIL import Image, ImageTk

_THUMB_W = 150
_THUMB_H = 210   # preserves 256:358 ratio
_COLS    = 4
_BG      = "#1e1e2e"
_FG      = "#cdd6f4"
_ACCENT  = "#89b4fa"


def show_preprocessing_steps(
    parent: tk.Tk,
    steps: List[Tuple[str, Image.Image]],
) -> None:
    """Open a non-blocking window showing each (label, image) pair as a grid."""

    win = tk.Toplevel(parent)
    win.title("Preprocessing Debug")
    win.attributes("-topmost", True)
    win.configure(bg=_BG)
    win.resizable(False, False)

    tk.Label(
        win,
        text="Preprocessing pipeline  —  what the recogniser actually sees",
        bg=_BG, fg=_ACCENT,
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=_COLS, padx=10, pady=(10, 4))

    # Keep PhotoImage references alive for the lifetime of the window
    refs: List[ImageTk.PhotoImage] = []

    for idx, (label, img) in enumerate(steps):
        grid_row = (idx // _COLS) + 1   # +1 for the header label
        grid_col =  idx % _COLS

        # Normalise to RGB so greyscale single-channel images display properly
        thumb = img.convert("RGB").resize((_THUMB_W, _THUMB_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        refs.append(photo)

        cell = tk.Frame(win, bg=_BG)
        cell.grid(row=grid_row, column=grid_col, padx=6, pady=6)

        tk.Label(cell, image=photo, bg=_BG, relief="flat").pack()
        tk.Label(
            cell,
            text=label,
            bg=_BG, fg=_FG,
            font=("Segoe UI", 8),
            wraplength=_THUMB_W,
            justify="center",
        ).pack(pady=(2, 0))

    tk.Button(
        win, text="Close",
        command=win.destroy,
        bg="#45475a", fg=_FG,
        font=("Segoe UI", 9),
        relief="flat", padx=10, pady=3,
        cursor="hand2",
    ).grid(row=999, column=0, columnspan=_COLS, pady=(4, 10))

    win.bind("<Escape>", lambda _: win.destroy())

    # Attach refs so they aren't garbage-collected
    win._photo_refs = refs

    # Position to the left of centre so it doesn't cover the card popup
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w  = win.winfo_reqwidth()
    h  = win.winfo_reqheight()
    win.geometry(f"+{max(0, sw // 2 - w - 20)}+{max(0, (sh - h) // 2)}")
