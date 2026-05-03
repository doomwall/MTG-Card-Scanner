"""Debug window — preprocessing pipeline visualiser and variant selector.

Shows every preprocessing stage as a thumbnail grid.
Selectable variants (the four final forms used for hashing) have a green
border and "Click to use" label.  Clicking one returns that image so the
caller can force it as the sole hash input instead of trying all variants.

Returns Optional[Image.Image]:
  • None   — user closed without choosing; caller uses all variants as normal
  • Image  — the chosen preprocessed image; caller passes it as force_gray
"""

import tkinter as tk
from typing import List, Optional, Tuple

from PIL import Image, ImageTk

_THUMB_W = 150
_THUMB_H = 210       # preserves 256:358 ratio
_COLS    = 4
_BG      = "#1e1e2e"
_FG      = "#cdd6f4"
_ACCENT  = "#89b4fa"
_SEL_BG  = "#1e3a1e"   # dark green tint for selectable cells
_SEL_BD  = "#00ff41"   # green border on selectable cells
_HOV_BD  = "#f9e2af"   # yellow border on hover


def show_debug_window(
    parent: tk.Tk,
    steps: List[Tuple[str, Image.Image, bool]],
) -> Optional[Image.Image]:
    """Modal debug window.  Returns selected variant image or None."""

    result: list = [None]

    win = tk.Toplevel(parent)
    win.title("Debug — click a green variant to use it for matching")
    win.attributes("-topmost", True)
    win.configure(bg=_BG)
    win.resizable(False, False)

    tk.Label(
        win,
        text="Preprocessing pipeline",
        bg=_BG, fg=_ACCENT,
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=_COLS, padx=10, pady=(10, 0))

    tk.Label(
        win,
        text="Green = final hash variant  •  click one to force it  •  close to use all",
        bg=_BG, fg="#6c7086",
        font=("Segoe UI", 8),
    ).grid(row=1, column=0, columnspan=_COLS, padx=10, pady=(0, 6))

    refs: List[ImageTk.PhotoImage] = []

    for idx, (label, img, selectable) in enumerate(steps):
        grid_row = (idx // _COLS) + 2
        grid_col =  idx % _COLS

        thumb = img.convert("RGB").resize((_THUMB_W, _THUMB_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        refs.append(photo)

        cell_bg = _SEL_BG if selectable else _BG
        cell = tk.Frame(
            win, bg=cell_bg,
            highlightthickness=2,
            highlightbackground=_SEL_BD if selectable else _BG,
        )
        cell.grid(row=grid_row, column=grid_col, padx=6, pady=6)

        img_lbl = tk.Label(cell, image=photo, bg=cell_bg)
        img_lbl.pack()

        caption = label
        if selectable:
            caption += "\n▶ click to use"

        tk.Label(
            cell,
            text=caption,
            bg=cell_bg,
            fg=_SEL_BD if selectable else _FG,
            font=("Segoe UI", 8),
            wraplength=_THUMB_W,
            justify="center",
        ).pack(pady=(2, 4))

        if selectable:
            captured_img = img   # capture loop variable

            def on_enter(e, c=cell):
                c.configure(highlightbackground=_HOV_BD)

            def on_leave(e, c=cell, s=selectable):
                c.configure(highlightbackground=_SEL_BD if s else _BG)

            def on_click(e, chosen=captured_img):
                result[0] = chosen
                win.destroy()

            for widget in (cell, img_lbl):
                widget.configure(cursor="hand2")
                widget.bind("<Enter>",          on_enter)
                widget.bind("<Leave>",          on_leave)
                widget.bind("<ButtonPress-1>",  on_click)

    tk.Button(
        win, text="Close (use all variants)",
        command=win.destroy,
        bg="#45475a", fg=_FG,
        font=("Segoe UI", 9),
        relief="flat", padx=10, pady=4,
        cursor="hand2",
    ).grid(row=999, column=0, columnspan=_COLS, pady=(4, 12))

    win.bind("<Escape>", lambda _: win.destroy())
    win._photo_refs = refs

    # Centre the window
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w  = win.winfo_reqwidth()
    h  = win.winfo_reqheight()
    win.geometry(f"+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}")

    parent.wait_window(win)
    return result[0]
