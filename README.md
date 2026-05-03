# MTG Card Scanner

A Windows desktop tool for identifying Magic: The Gathering cards directly from your screen. Highly inaccurate! Press a hotkey, drag a selection over a card, and get the card name, mana cost, type, oracle text, and current price from Scryfall — instantly.

---

## How it works

1. Press **F9** (or click the **Scan** button in the status window)
2. Drag a rectangle over the card on your screen
3. The app attempts to identify the card using two methods in sequence:
   - **Perceptual hashing** — compares the captured image against a local database of card image hashes
   - **OCR fallback** — if hashing finds no match, reads the card name directly from the name-bar region using EasyOCR
4. Card details are fetched from the Scryfall API and displayed in a popup

---

## Recognition pipeline

### Phase 1 — Perceptual hashing
The captured image is preprocessed through four variants before hashing:

| Variant | Processing |
|---------|-----------|
| v1 | CLAHE contrast enhancement |
| v2 | Denoise → CLAHE |
| v3 | Denoise → Sharpen → CLAHE |
| v4 | Bilateral filter → CLAHE |

Each variant produces a 64-bit pHash. The best distance across all four is compared against the database. A secondary **hue-channel hash** (with achromatic pixels masked) helps distinguish cards with similar artwork but different frame colours (e.g. white vs. red cards).

### Phase 2 — OCR fallback
If no hash match is found within the threshold, EasyOCR reads the card name from the top portion of the captured image. The result is passed to Scryfall's fuzzy-name endpoint, which tolerates minor OCR errors.

---

## Installation

**Requirements:** Python 3.10+, Windows

```powershell
# Clone the repo
git clone https://github.com/user/MTG-Card-Scanner.git
cd MTG-Card-Scanner

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First-time PowerShell setup
If you get a script execution error activating the venv:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Building the hash database

The hash database is required for Phase 1 recognition. Download `default_cards.json` from [Scryfall Bulk Data](https://scryfall.com/docs/api/bulk-data) and run:

```powershell
# Test with a single set first (~2 minutes)
python db_builder.py --input default_cards.json --set dsk

# Full build — all 114k cards (~2-4 hours, ~600 MB of card images)
python db_builder.py --input default_cards.json
```

The database is saved to `%USERPROFILE%\.mtg_scanner\cards.db`. The build is incremental — if interrupted, re-run the same command to resume.

A pre-built `cards.db` placed in the project folder will be used automatically without needing to run the builder.

**Database lookup order:**
1. `%USERPROFILE%\.mtg_scanner\cards.db` (user-built or updated)
2. `cards.db` next to the script / exe (shipped with a release)

---

## Running

```powershell
python main.py
```

A small status window appears in the taskbar. Press **F9** or click **Scan** to begin.

---

## Debug mode

Tick **"Show preprocessing debug"** in the status window before scanning. After capturing a region, a grid window shows every preprocessing stage. Click any **green-bordered** variant to force that specific image to be used for hashing instead of trying all four automatically. Useful for diagnosing why a particular card is not being recognised.

---

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — tkinter app, hotkey wiring, scan pipeline |
| `config.py` | Constants: hotkey, paths, thresholds |
| `selector.py` | Fullscreen region-selection overlay with frozen screenshot background |
| `capture.py` | Screen region capture via mss |
| `detector.py` | OpenCV card boundary detection and perspective correction |
| `card_picker.py` | UI for selecting one card when multiple are detected |
| `recognizer.py` | pHash matching + OCR fallback |
| `scryfall.py` | Scryfall API client with disk caching and rate limiting |
| `popup.py` | Card info popup window |
| `db_builder.py` | CLI tool to build the perceptual hash database |
| `debug_view.py` | Debug grid window showing preprocessing pipeline stages |
| `hotkey_listener.py` | Global hotkey via pynput |

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `HOTKEY` | `<f9>` | Global hotkey (pynput format) |
| `HASH_MATCH_THRESHOLD` | `15` | Max Hamming distance for a valid pHash match (0–64). Raise to 20–22 for lower-quality images |
| `HUE_WEIGHT` | `0.5` | Colour channel contribution to the combined hash score |

---

## Scryfall attribution

Card data is provided by the [Scryfall API](https://scryfall.com/docs/api). All requests include a descriptive `User-Agent` header and are rate-limited to 100 ms between calls per Scryfall's guidelines. Attribution is shown in every card popup.

---

## Known limitations

- Recognition works best on **clear screen captures** (MTG Arena, MTGO, Moxfield, Scryfall website). Blurry or compressed webcam footage degrades both hash and OCR accuracy significantly.
- The OCR fallback requires the card name bar to be legible. The EasyOCR model (~100 MB) is downloaded on first use.
- Card detection (`detector.py`) works best when the card has a visible border against the background. It falls back to using the full selection if no card outline is found.

---

## Packaging as a Windows exe

```powershell
pip install pyinstaller
pyinstaller --windowed --onefile --add-data "cards.db;." main.py
```

The resulting exe in `dist/` will include the hash database and run without a console window.
