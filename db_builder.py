"""Hash database builder.

Downloads (or reads a local) Scryfall bulk JSON, then downloads each card's
'small' image, computes a perceptual hash, and stores everything in SQLite.

Both oracle_cards and default_cards formats are supported.
default_cards (~114 k entries, one per printing) gives better coverage
because different printings often have different artwork.

Usage
-----
    # Use a local bulk JSON you already have (recommended):
    python db_builder.py --input default-cards-20260502210903.json

    # Smoke-test with a small slice:
    python db_builder.py --input default-cards-20260502210903.json --limit 200

    # Download fresh bulk data from Scryfall and build:
    python db_builder.py

    # Re-download bulk JSON even if cached:
    python db_builder.py --force-bulk

    # Store card names only, skip downloading images:
    python db_builder.py --input default-cards-20260502210903.json --skip-images

Storage: ~600 MB for full image set (114 k× ~5 KB 'small' images).
Runtime: 2–4 hours for a full build on a home connection; use --limit first.
"""

import argparse
import io
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
import imagehash
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    API_RATE_LIMIT_DELAY,
    DATA_DIR,
    DB_PATH,
    SCRYFALL_BASE_URL,
    USER_AGENT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BULK_JSON_PATH = DATA_DIR / "oracle_cards.json"

# Must match recognizer.py exactly so hashes are comparable
_HASH_W, _HASH_H = 256, 358


# ── database ──────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cards (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scryfall_id     TEXT    UNIQUE NOT NULL,
            oracle_id       TEXT,
            name            TEXT    NOT NULL,
            set_code        TEXT,
            image_uri       TEXT,
            hash_int        INTEGER,
            image_hash      TEXT,
            hue_hash_int    INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_hash     ON cards(hash_int);
        CREATE INDEX IF NOT EXISTS idx_hue_hash ON cards(hue_hash_int);
        CREATE INDEX IF NOT EXISTS idx_name     ON cards(name COLLATE NOCASE);
    """)
    # Migrate existing DBs that pre-date the hue hash column
    try:
        conn.execute("ALTER TABLE cards ADD COLUMN hue_hash_int INTEGER")
        logger.info("DB migrated: added hue_hash_int column")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


# ── bulk data ─────────────────────────────────────────────────────────────────

def _load_cards_json(path: Path) -> list:
    """Read a Scryfall bulk JSON file.  Uses errors='replace' to survive the
    rare malformed byte that some bulk exports contain."""
    logger.info("Loading %s (%.0f MB)…", path.name, path.stat().st_size / 1_048_576)
    with open(path, encoding="utf-8", errors="replace") as fh:
        cards = json.load(fh)
    logger.info("Loaded %d card entries", len(cards))
    return cards


def _download_bulk_json(force: bool = False) -> Path:
    if BULK_JSON_PATH.exists() and not force:
        logger.info("Bulk data already cached at %s", BULK_JSON_PATH)
        return BULK_JSON_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}

    logger.info("Fetching bulk-data catalogue from Scryfall…")
    resp = requests.get(f"{SCRYFALL_BASE_URL}/bulk-data", headers=headers, timeout=15)
    resp.raise_for_status()

    # Prefer default_cards (one entry per printing = better art coverage)
    # Fall back to oracle_cards if not available
    bulk_types = {b["type"]: b for b in resp.json().get("data", [])}
    entry = bulk_types.get("default_cards") or bulk_types.get("oracle_cards")
    if entry is None:
        raise RuntimeError("No usable bulk-data entry found in Scryfall response")

    download_url = entry["download_uri"]
    size_mb = entry.get("size", 0) / 1_048_576
    logger.info("Downloading %s (%.1f MB)…", entry["type"], size_mb)

    resp = requests.get(download_url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(BULK_JSON_PATH, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=Path(BULK_JSON_PATH).name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65_536):
            fh.write(chunk)
            bar.update(len(chunk))

    logger.info("Saved to %s", BULK_JSON_PATH)
    return BULK_JSON_PATH


# ── image hashing ─────────────────────────────────────────────────────────────

def _to_signed(h: int) -> int:
    """Convert unsigned 64-bit pHash to signed for SQLite storage."""
    return h - 2**64 if h >= 2**63 else h


def _hashes_from_url(
    url: str, session: requests.Session
) -> tuple[Optional[int], Optional[int]]:
    """Download image and return (gray_hash, hue_hash) as signed 64-bit ints."""
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))

        # ── grayscale pHash (structure / artwork) ─────────────────────────
        gray = img.convert("L").resize((_HASH_W, _HASH_H), Image.LANCZOS)
        gray_hash = _to_signed(int(str(imagehash.phash(gray)), 16))

        # ── hue-channel pHash (frame / background colour) ─────────────────
        arr = np.array(img.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hue_img = Image.fromarray(hsv[:, :, 0]).resize((_HASH_W, _HASH_H), Image.LANCZOS)
        hue_hash = _to_signed(int(str(imagehash.phash(hue_img)), 16))

        return gray_hash, hue_hash
    except Exception as exc:
        logger.debug("Image hash failed (%s): %s", url, exc)
        return None, None


# ── main builder ──────────────────────────────────────────────────────────────

def build(
    input_path: Optional[Path] = None,
    limit: Optional[int] = None,
    force_bulk: bool = False,
    skip_images: bool = False,
    set_filter: Optional[str] = None,
) -> None:
    if input_path is not None:
        bulk_path = input_path
    else:
        bulk_path = _download_bulk_json(force=force_bulk)

    cards = _load_cards_json(bulk_path)

    if set_filter:
        set_code = set_filter.lower()
        cards = [c for c in cards if c.get("set", "").lower() == set_code]
        if not cards:
            logger.error("No cards found for set '%s'. Check the set code.", set_filter)
            return
        logger.info("Filtered to set '%s': %d cards", set_filter.upper(), len(cards))

    if limit:
        cards = cards[:limit]
        logger.info("Applying --limit: %d cards", limit)
    else:
        logger.info("Processing %d cards", len(cards))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    already_hashed: set = {
        row[0]
        for row in conn.execute(
            "SELECT scryfall_id FROM cards "
            "WHERE hash_int IS NOT NULL AND hue_hash_int IS NOT NULL"
        )
    }

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    inserted = hashed = skipped = 0
    commit_every = 200

    with tqdm(total=len(cards), desc="Building DB", unit="card") as bar:
        for card in cards:
            bar.update(1)

            sid = card.get("id")
            name = card.get("name", "").strip()
            if not sid or not name:
                continue

            # Pick image URI: prefer 'small', fall back through sizes
            uris = card.get("image_uris") or {}
            if not uris and card.get("card_faces"):
                uris = card["card_faces"][0].get("image_uris") or {}
            image_uri = (
                uris.get("small")
                or uris.get("normal")
                or uris.get("large")
                or uris.get("png")
            )

            # Upsert metadata row
            conn.execute(
                """
                INSERT INTO cards (scryfall_id, oracle_id, name, set_code, image_uri)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scryfall_id) DO UPDATE SET
                    name      = excluded.name,
                    image_uri = excluded.image_uri
                """,
                (sid, card.get("oracle_id"), name, card.get("set"), image_uri),
            )
            inserted += 1

            if skip_images or not image_uri or sid in already_hashed:
                skipped += 1
                if inserted % commit_every == 0:
                    conn.commit()
                continue

            gray_hash, hue_hash = _hashes_from_url(image_uri, session)
            if gray_hash is not None:
                conn.execute(
                    "UPDATE cards SET hash_int = ?, image_hash = ?, hue_hash_int = ? "
                    "WHERE scryfall_id = ?",
                    (gray_hash, format(gray_hash & 0xFFFFFFFFFFFFFFFF, "016x"),
                     hue_hash, sid),
                )
                already_hashed.add(sid)
                hashed += 1

            if inserted % commit_every == 0:
                conn.commit()
                time.sleep(API_RATE_LIMIT_DELAY)

    conn.commit()
    conn.close()

    logger.info(
        "Done. %d cards inserted, %d images hashed, %d skipped.",
        inserted, hashed, skipped,
    )
    logger.info("Database: %s", DB_PATH)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build MTG Card Scanner perceptual-hash database"
    )
    parser.add_argument(
        "--input", metavar="FILE",
        help="Path to a local Scryfall bulk JSON file (default_cards or oracle_cards). "
             "If omitted, the file is downloaded from Scryfall.",
    )
    parser.add_argument(
        "--set", metavar="CODE",
        help="Only process cards from this set (e.g. --set dsk, --set m21). "
             "Case-insensitive. Combine with --limit for an even smaller slice.",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N",
        help="Cap the number of cards after any --set filter is applied",
    )
    parser.add_argument(
        "--force-bulk", action="store_true",
        help="Re-download bulk JSON even if cached (ignored when --input is given)",
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Store card names only; do not download images or compute hashes",
    )
    args = parser.parse_args()
    build(
        input_path=Path(args.input) if args.input else None,
        limit=args.limit,
        force_bulk=args.force_bulk,
        skip_images=args.skip_images,
        set_filter=args.set,
    )
