"""MTG Card Scanner — main entry point.

Architecture
------------
* A hidden Tk root runs the main event loop on the main thread.
* The keyboard library fires the hotkey callback on a background thread.
* The callback enqueues a task; the root's 50 ms poll dispatches it on the
  main thread so all Tk windows are created there (required on Windows).
* A small status Toplevel is shown so the user knows the app is running.

Scan pipeline (all on main thread)
-----------------------------------
  1. select_region()   — fullscreen overlay, user drags a rectangle
  2. capture_region()  — mss grabs that rectangle
  3. find_best_match() — perceptual hash search in SQLite
  4. get_card_by_*()   — Scryfall API (disk-cached)
  5. show_card_popup() — non-blocking Toplevel with card details
"""

import ctypes
import logging
import queue
import sys
import tkinter as tk
from tkinter import messagebox
import threading

# DPI awareness must be set before any Tk window is created
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from config import APP_NAME, APP_VERSION, CACHE_DIR, DATA_DIR, DB_PATH, HOTKEY
from hotkey_listener import HotkeyListener
from selector import select_region
from capture import capture_region
from recognizer import find_best_match, db_is_ready, get_preprocessing_steps
from scryfall import get_card_by_exact_name, get_card_by_fuzzy_name
from card_picker import pick_card
from popup import show_card_popup
from debug_view import show_preprocessing_steps

# ── logging ───────────────────────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "scanner.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── app ───────────────────────────────────────────────────────────────────────

class MTGScannerApp:
    def __init__(self) -> None:
        self._task_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._scanning = threading.Event()

        # Root must exist before any tk variables are created
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title(APP_NAME)

        self._debug_mode = tk.BooleanVar(value=False)

        self._build_status_window()
        self._poll_queue()

    # ── status window ─────────────────────────────────────────────────────

    def _build_status_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title(APP_NAME)
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", self._on_quit)
        win.configure(bg="#1e1e2e")

        tk.Label(
            win,
            text=f"{APP_NAME} v{APP_VERSION}",
            bg="#1e1e2e", fg="#89b4fa",
            font=("Segoe UI", 11, "bold"),
        ).pack(padx=16, pady=(12, 2))

        db_ok = DB_PATH.exists()
        hotkey_label = HOTKEY.strip("<>").upper()
        status_text = (
            f"Press  {hotkey_label}  to scan a card"
            if db_ok
            else f"Press  {hotkey_label}  to scan  •  DB not built yet"
        )
        self._status_var = tk.StringVar(value=status_text)
        tk.Label(
            win,
            textvariable=self._status_var,
            bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 9),
        ).pack(padx=16, pady=(2, 4))

        if not db_ok:
            tk.Label(
                win,
                text="Run:  python db_builder.py --limit 500",
                bg="#1e1e2e", fg="#f9e2af",
                font=("Consolas", 8),
            ).pack(padx=16, pady=(0, 6))

        # ── button row ────────────────────────────────────────────────────
        btn_row = tk.Frame(win, bg="#1e1e2e")
        btn_row.pack(pady=(0, 8))

        tk.Button(
            btn_row,
            text=f"Scan  ({HOTKEY.strip('<>').upper()})",
            command=self.enqueue_scan,
            bg="#89b4fa", fg="#1e1e2e",
            font=("Segoe UI", 9, "bold"),
            relief="flat", padx=12, pady=4,
            cursor="hand2",
        ).pack(side="left", padx=(16, 6))

        tk.Button(
            btn_row,
            text="Quit",
            command=self._on_quit,
            bg="#45475a", fg="#cdd6f4",
            font=("Segoe UI", 9),
            relief="flat", padx=10, pady=4,
            cursor="hand2",
        ).pack(side="left", padx=(0, 16))

        # ── debug toggle ──────────────────────────────────────────────────
        tk.Checkbutton(
            win,
            text="Show preprocessing debug",
            variable=self._debug_mode,
            bg="#1e1e2e", fg="#6c7086",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            font=("Segoe UI", 8),
            cursor="hand2",
        ).pack(pady=(0, 8))

    # ── queue polling (main thread) ───────────────────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                task = self._task_queue.get_nowait()
                task()
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    # ── hotkey callback (background thread → enqueue) ─────────────────────

    def enqueue_scan(self) -> None:
        if not self._scanning.is_set():
            self._task_queue.put(self._run_scan)

    # ── scan pipeline (main thread) ───────────────────────────────────────

    def _run_scan(self) -> None:
        if self._scanning.is_set():
            return
        self._scanning.set()
        self._status_var.set("Scanning…")
        try:
            self._pipeline()
        except Exception as exc:
            logger.exception("Unexpected error in scan pipeline")
            messagebox.showerror(APP_NAME, f"Unexpected error:\n{exc}", parent=self.root)
        finally:
            self._scanning.clear()
            self._status_var.set(f"Press  {HOTKEY.strip('<>').upper()}  to scan a card")

    def _pipeline(self) -> None:
        # 1. Region selection — also returns the frozen screenshot taken
        #    before the overlay appeared, so no second grab is needed.
        selection = select_region(self.root)
        if selection is None:
            logger.info("Selection cancelled")
            return

        (x1, y1, x2, y2), screenshot = selection
        logger.info("Selected region (%d,%d)→(%d,%d)", x1, y1, x2, y2)

        # 2. Crop from the already-captured screenshot (same frame the user
        #    saw in the overlay — cards can't have moved between the two).
        raw_img = screenshot.crop((x1, y1, x2, y2))

        # 3. Card detection — let user pick one if multiple are visible.
        #    Falls back to full capture when no card outline is found.
        card_img = pick_card(self.root, raw_img)
        if card_img is None:
            card_img = raw_img   # no outline detected; use full capture as before

        # 4. Debug window (optional)
        if self._debug_mode.get():
            steps = get_preprocessing_steps(card_img)
            show_preprocessing_steps(self.root, steps)

        # 5. Card recognition
        if not db_is_ready():
            messagebox.showwarning(
                APP_NAME,
                "The hash database has not been built yet.\n\n"
                "Run  python db_builder.py --limit 500  to get started,\n"
                "or  python db_builder.py  for the full database.",
                parent=self.root,
            )
            return

        match = find_best_match(card_img)
        if match is None:
            messagebox.showwarning(
                APP_NAME,
                "No card recognised.\n\n"
                "Tips:\n"
                "• Select just the card, not surrounding content.\n"
                "• The hash database may not include this card yet.\n"
                "• Try running  python db_builder.py  to rebuild.",
                parent=self.root,
            )
            return

        card_name, distance = match
        logger.info("Recognised '%s' (Hamming distance %d)", card_name, distance)

        # 4. Scryfall lookup
        card_data = get_card_by_exact_name(card_name)
        if card_data is None or card_data.get("object") == "error":
            card_data = get_card_by_fuzzy_name(card_name)

        if card_data is None or card_data.get("object") == "error":
            messagebox.showerror(
                APP_NAME,
                f"Could not fetch Scryfall data for '{card_name}'.\n"
                "Check your internet connection and try again.",
                parent=self.root,
            )
            return

        # 5. Show popup (non-blocking)
        show_card_popup(self.root, card_data, confidence=distance)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def _on_quit(self) -> None:
        logger.info("Shutting down")
        self.root.quit()

    def run(self) -> None:
        listener = HotkeyListener(HOTKEY, self.enqueue_scan)
        listener.start()
        logger.info("%s v%s started. Press %s to scan.", APP_NAME, APP_VERSION, HOTKEY.strip("<>").upper())
        try:
            self.root.mainloop()
        finally:
            listener.stop()
            logger.info("Stopped.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MTGScannerApp().run()
