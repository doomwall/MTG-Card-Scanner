"""Scryfall API client.

Rules followed per https://scryfall.com/docs/api:
  - Descriptive User-Agent on every request.
  - At least 100 ms between requests (API_RATE_LIMIT_DELAY).
  - Responses cached to disk; identical requests never hit the network twice.
  - Attribution shown in the UI (handled by popup.py).
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from config import API_RATE_LIMIT_DELAY, CACHE_DIR, SCRYFALL_BASE_URL, USER_AGENT

logger = logging.getLogger(__name__)

_last_request_at: float = 0.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _get_cached(key: str) -> Optional[Dict]:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _set_cached(key: str, data: Dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < API_RATE_LIMIT_DELAY:
        time.sleep(API_RATE_LIMIT_DELAY - elapsed)
    _last_request_at = time.monotonic()


def _get(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    cache_key = endpoint + json.dumps(params or {}, sort_keys=True)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    _rate_limit()
    url = f"{SCRYFALL_BASE_URL}{endpoint}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data: Dict = resp.json()
    except Exception as exc:
        logger.error("Scryfall request failed (%s %s): %s", endpoint, params, exc)
        return None

    # Only cache non-error responses
    if data.get("object") != "error":
        _set_cached(cache_key, data)

    return data


# ── public API ────────────────────────────────────────────────────────────────

def get_card_by_exact_name(name: str) -> Optional[Dict[str, Any]]:
    return _get("/cards/named", {"exact": name})


def get_card_by_fuzzy_name(name: str) -> Optional[Dict[str, Any]]:
    return _get("/cards/named", {"fuzzy": name})


def get_bulk_data_info() -> Optional[Dict[str, Any]]:
    return _get("/bulk-data")


# ── formatting helpers used by popup.py ──────────────────────────────────────

def format_mana_cost(raw: str) -> str:
    """Turn '{2}{W}{U}' into '2 W U'."""
    return raw.replace("{", "").replace("}", " ").strip() if raw else ""


def get_usd_price(card: Dict[str, Any]) -> str:
    prices = card.get("prices", {})
    usd = prices.get("usd")
    foil = prices.get("usd_foil")
    parts = []
    if usd:
        parts.append(f"${usd}")
    if foil:
        parts.append(f"${foil} foil")
    return "  /  ".join(parts) if parts else "N/A"
