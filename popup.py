"""Card-info popup window.

Opens a non-blocking Toplevel that stays on top and shows the card details
retrieved from Scryfall.  The user closes it with Escape or the Close button.
"""

import tkinter as tk
from typing import Any, Dict, Optional
import logging

from scryfall import format_mana_cost, get_usd_price

logger = logging.getLogger(__name__)

# ── palette (Catppuccin Mocha-inspired) ──────────────────────────────────────
_BG       = "#1e1e2e"
_SURFACE  = "#313244"
_OVERLAY  = "#45475a"
_FG       = "#cdd6f4"
_BLUE     = "#89b4fa"
_YELLOW   = "#f9e2af"
_GREEN    = "#a6e3a1"
_MUTED    = "#6c7086"
_ITALIC   = "#b4befe"


def _sep(parent: tk.Frame) -> None:
    tk.Frame(parent, height=1, bg=_OVERLAY).pack(fill="x", pady=5)


def show_card_popup(
    parent: tk.Tk,
    card: Dict[str, Any],
    confidence: Optional[int] = None,
) -> None:
    """Open a non-blocking popup.  Returns immediately; user closes it."""

    win = tk.Toplevel(parent)
    win.title("MTG Card Scanner")
    win.attributes("-topmost", True)
    win.resizable(False, False)
    win.configure(bg=_BG)

    f = tk.Frame(win, bg=_BG, padx=18, pady=14)
    f.pack(fill=tk.BOTH, expand=True)

    # ── card name ─────────────────────────────────────────────────────────
    name = card.get("name", "Unknown Card")
    tk.Label(
        f, text=name,
        bg=_BG, fg=_BLUE,
        font=("Segoe UI", 14, "bold"),
        wraplength=360, justify="left",
    ).pack(anchor="w")

    # ── mana cost ─────────────────────────────────────────────────────────
    mana_raw = card.get("mana_cost", "")
    if not mana_raw and card.get("card_faces"):
        mana_raw = card["card_faces"][0].get("mana_cost", "")
    mana = format_mana_cost(mana_raw)
    if mana:
        tk.Label(
            f, text=f"Mana: {mana}",
            bg=_BG, fg=_YELLOW,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

    # ── type line ─────────────────────────────────────────────────────────
    type_line = card.get("type_line", "")
    if type_line:
        tk.Label(
            f, text=type_line,
            bg=_BG, fg=_ITALIC,
            font=("Segoe UI", 10, "italic"),
        ).pack(anchor="w", pady=(2, 0))

    _sep(f)

    # ── oracle text ───────────────────────────────────────────────────────
    oracle = card.get("oracle_text", "")
    if not oracle and card.get("card_faces"):
        oracle = "\n\n".join(
            face.get("oracle_text", "") for face in card["card_faces"]
        ).strip()
    if oracle:
        box = tk.Frame(f, bg=_SURFACE, padx=8, pady=6)
        box.pack(fill="x", pady=(0, 6))
        tk.Label(
            box, text=oracle,
            bg=_SURFACE, fg=_FG,
            font=("Segoe UI", 9),
            wraplength=352, justify="left",
        ).pack(anchor="w")

    # ── power/toughness or loyalty ────────────────────────────────────────
    if card.get("power") and card.get("toughness"):
        tk.Label(
            f, text=f"{card['power']}/{card['toughness']}",
            bg=_BG, fg=_FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="e")
    elif card.get("loyalty"):
        tk.Label(
            f, text=f"Loyalty: {card['loyalty']}",
            bg=_BG, fg=_FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="e")

    # ── set info ──────────────────────────────────────────────────────────
    set_name = card.get("set_name", "")
    set_code = card.get("set", "").upper()
    if set_name:
        tk.Label(
            f, text=f"{set_name} ({set_code})",
            bg=_BG, fg=_MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

    # ── price ─────────────────────────────────────────────────────────────
    price_str = get_usd_price(card)
    if price_str and price_str != "N/A":
        tk.Label(
            f, text=f"Price: {price_str}",
            bg=_BG, fg=_GREEN,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(4, 0))

    # ── match confidence ──────────────────────────────────────────────────
    if confidence is not None:
        pct = max(0, round(100 - confidence * 100 / 64))
        tk.Label(
            f, text=f"Match confidence: {pct}%",
            bg=_BG, fg=_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 0))

    _sep(f)

    # ── attribution (required by Scryfall) ───────────────────────────────
    tk.Label(
        f, text="Card data provided by Scryfall  •  scryfall.com",
        bg=_BG, fg=_MUTED,
        font=("Segoe UI", 8),
    ).pack(anchor="e")

    # ── close button ──────────────────────────────────────────────────────
    tk.Button(
        f, text="Close",
        command=win.destroy,
        bg=_OVERLAY, fg=_FG,
        font=("Segoe UI", 9),
        relief="flat", padx=10, pady=3,
        cursor="hand2", activebackground=_SURFACE,
    ).pack(anchor="e", pady=(6, 0))

    win.bind("<Escape>", lambda _e: win.destroy())

    # Position: right side of screen, vertically centered
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w = win.winfo_reqwidth()
    h = win.winfo_reqheight()
    win.geometry(f"+{sw - w - 40}+{max(0, (sh - h) // 2)}")
    win.focus_force()
