import threading
import logging
from typing import Callable

from pynput import keyboard

logger = logging.getLogger(__name__)


class HotkeyListener:
    """Listens for a global hotkey and fires a callback on each press.

    The callback is guarded by a lock so rapid keypresses don't stack up
    while a scan is already in progress.
    """

    def __init__(self, hotkey: str, callback: Callable[[], None]) -> None:
        self._hotkey = hotkey
        self._callback = callback
        self._active = False
        self._lock = threading.Lock()
        self._listener = None

    def start(self) -> None:
        hotkeys = {self._hotkey: self._on_hotkey}
        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._listener.start()
        logger.info("Hotkey listener active: press %s to scan", self._hotkey.upper())

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    def _on_hotkey(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True
        try:
            self._callback()
        finally:
            with self._lock:
                self._active = False
