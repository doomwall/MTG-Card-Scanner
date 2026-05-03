import sys
from pathlib import Path

APP_NAME = "MTG Card Scanner"
APP_VERSION = "1.0.0"

DATA_DIR = Path.home() / ".mtg_scanner"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "cards.db"


def _app_dir() -> Path:
    """Directory where the script or frozen exe lives."""
    if getattr(sys, "frozen", False):
        # PyInstaller unpacks to sys._MEIPASS; the exe itself is sys.executable
        return Path(sys.executable).parent
    return Path(__file__).parent


def resolve_db_path() -> Path:
    """Return the database path to use, in priority order:
    1. User data dir (~/.mtg_scanner/cards.db)  — updated by db_builder
    2. Next to the exe/script (cards.db)         — shipped with the release
    3. PyInstaller bundle (_MEIPASS/cards.db)    — embedded in the exe
    """
    if DB_PATH.exists():
        return DB_PATH
    beside_exe = _app_dir() / "cards.db"
    if beside_exe.exists():
        return beside_exe
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "cards.db"
    if bundled.exists():
        return bundled
    return DB_PATH  # doesn't exist yet; callers handle the missing-db case

HOTKEY = "<f9>"

# Scryfall API requires a descriptive User-Agent per their guidelines
USER_AGENT = (
    f"MTGCardScanner/{APP_VERSION} "
    "(https://github.com/user/mtg-card-scanner; educational-tool)"
)

SCRYFALL_BASE_URL = "https://api.scryfall.com"

# Scryfall guidelines: no more than 10 requests/second; 100ms is safe
API_RATE_LIMIT_DELAY = 0.1

# Minimum drag size (pixels) to register as a selection
MIN_SELECTION_SIZE = 20

# Maximum perceptual hash Hamming distance (0–64) to count as a match.
# 15 = good for clean screen captures; raise to 20–22 for webcam quality.
HASH_MATCH_THRESHOLD = 22
