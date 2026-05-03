"""Screen region capture using mss."""

from PIL import Image
import mss


def capture_region(x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """Capture (x1, y1)→(x2, y2) from the screen and return an RGB PIL Image."""
    with mss.mss() as sct:
        monitor = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
        shot = sct.grab(monitor)
        # mss returns BGRA; convert to RGB
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return img
