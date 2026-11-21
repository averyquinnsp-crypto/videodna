"""On-screen text extraction via the tesseract CLI (v2.0.4 item 6).

System package only — no pip dependency (no pytesseract / easyocr /
rapidocr / torch). The binary is `tesseract-ocr` from apt (5.3.4 on
Ubuntu 24.04, installed 2026-11-21) plus `tesseract-ocr-hin` for
Devanagari overlays; both are optional — when the binary is absent the
extraction returns {} and every shot card keeps stating
remake.OCR_TEXT_STATUS ("text not extracted (no OCR available)").

Design (measured on the test videos, not assumed):
- One mid frame per shot, decoded in ONE streaming pass at 1280px
  wide, fed to tesseract as PNG bytes on stdin (no temp files).
- PSM 11 (sparse text) with `-l eng+hin`: PSM 3 returned nothing on
  U2 while PSM 11 found its overlay text; PSM 6/7/8/12 did no better.
- Word boxes come free from the same run (TSV output) and are stored
  normalised to 0-1000 relative to the OCR frame.
- QUALITY GATE (the important part): tesseract is confidently wrong
  on busy graphics — on a speech2 frame, garbage single characters
  ("Y", "|", "S", "20", "!") scored confidence 60-90. Confidence
  alone cannot separate garbage from text. A word is kept only if:
    * TSV confidence >= 70, AND
    * it is an alphabetic word of >= 3 letters (Latin or Devanagari,
      punctuation stripped), OR a digit token immediately following a
      kept word ("Text 1", "Text 4").
  Measured effect: U2 keeps "Text 1" (shot 3 mid) and "Text 4 Text 4"
  (shot 3, 1fps sweep) and drops everything else; speech2 keeps
  nothing (its fragments are 1-2 chars or conf < 70). Strays can
  still slip through ("hws" conf 50.6 / "hens" conf 55.4 / "कैट"
  conf 50.7 were dropped by the 70 floor, but a confident 3-letter
  misread remains possible) — the gate is tuned on these videos,
  not proven generally. A kept word is what the engine read; a shot
  with no kept word reports "no text detected", never invented text.
- KNOWN GAP, stated plainly: U2's giant BLOOD text was NOT extracted
  — 10s of frames at 1fps across PSMs 3/6/7/8/11/12 plus
  grayscale/autocontrast/upscale preprocessing never read it (best
  fragment: "BL : ; 0 D"). It is a stylised display font tesseract
  cannot segment. The OCR gap is narrowed (clean overlay text now
  reads), not closed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

# Tesseract tuning (see module docstring for the measurements behind them).
OCR_PSM = "11"
OCR_LANGS = "eng+hin"
OCR_MIN_CONFIDENCE = 70.0
OCR_MIN_WORD_LEN = 3
OCR_FRAME_WIDTH = 1280  # max width; smaller sources are NOT upscaled
OCR_MAX_TEXT_CHARS = 500

# Value stored on a shot when OCR ran but no word passed the gate.
OCR_NO_TEXT = "no text detected"

_LETTER_RE = re.compile(r"[A-Za-z\u0900-\u097F]")
_STRIP_RE = re.compile(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$")


def tesseract_available() -> bool:
    """True when the tesseract CLI is on PATH (system package)."""
    return shutil.which("tesseract") is not None


def _clean_word(text: str) -> str:
    """Strip surrounding punctuation/quotes; the stored form is the core."""
    return _STRIP_RE.sub("", text.strip())


def _gate_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only words that are plausibly real on-screen text (see header).

    words are TSV word rows in reading order: {"text", "conf", "x", "y",
    "w", "h"} with pixel coords on the OCR frame. Returns the kept
    words, each with cleaned text and coords normalised to 0-1000.
    """
    kept: list[dict[str, Any]] = []
    prev_kept = False
    for wd in words:
        conf = float(wd.get("conf", 0.0))
        core = _clean_word(str(wd.get("text", "")))
        keep = False
        if conf >= OCR_MIN_CONFIDENCE and core:
            letters = "".join(_LETTER_RE.findall(core))
            if len(letters) >= OCR_MIN_WORD_LEN:
                keep = True
            elif core.isdigit() and prev_kept:
                keep = True
        if keep:
            kept.append({**wd, "text": core})
            prev_kept = True
        else:
            prev_kept = False
    return kept


def _ocr_png_bytes(png_bytes: bytes) -> list[dict[str, Any]]:
    """Run tesseract on PNG bytes via stdin; return raw TSV word rows."""
    cmd = [
        "tesseract", "stdin", "stdout",
        "--psm", OCR_PSM, "-l", OCR_LANGS, "tsv",
    ]
    try:
        res = subprocess.run(cmd, input=png_bytes, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if res.returncode != 0:
        return []
    words: list[dict[str, Any]] = []
    # TSV columns are level, page/block/par/line/word numbers, left,
    # top, width, height, conf, text — the TSV carries NO page size,
    # so normalise against the PNG's own dimensions.
    try:
        with Image.open(BytesIO(png_bytes)) as im:
            page_w, page_h = im.size
    except Exception:
        return []
    text = res.stdout.decode("utf-8", errors="replace")
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            level = int(parts[0])
            conf = float(parts[10])
            word_text = parts[11]
            left, top, w, h = int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])
        except ValueError:
            continue
        if level != 5 or not word_text.strip():
            continue
        words.append({
            "text": word_text,
            "conf": round(conf, 1),
            "x": round(left / max(1, page_w) * 1000),
            "y": round(top / max(1, page_h) * 1000),
            "w": round(w / max(1, page_w) * 1000),
            "h": round(h / max(1, page_h) * 1000),
        })
    return words


def extract_shot_text(
    video_path: Path,
    shots: list[dict[str, Any]],
    width: int = OCR_FRAME_WIDTH,
) -> dict[int, dict[str, Any]]:
    """Extract on-screen text from the mid frame of every shot.

    One `ffmpeg -ss <mid> -frames:v 1` per shot at `width` px, PNG
    bytes piped straight into tesseract stdin (no temp files). Seeks
    are used instead of one streaming decode because a streaming
    rawvideo pass needs the frame height up front (aspect unknown
    here) while a seek per still is exact and fast (~0.1-0.2s per
    shot, bounded by the shot count). Returns {shot_id: {"text": str,
    "words": [{text, x, y, w, h, conf}]} for every shot whose mid
    frame decoded; "text" is "" when no word passed the gate. Returns
    {} when tesseract is absent or the decode fails (callers then
    leave the cards on OCR_TEXT_STATUS).
    """
    if not shots or not tesseract_available():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for s in shots:
        sid = int(s["id"])
        mid_t = float(s["start"]) + float(s["duration"]) / 2.0
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{mid_t:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale='min({width},iw)':-2",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if res.returncode != 0 or not res.stdout:
            continue
        try:
            with Image.open(BytesIO(res.stdout)) as im:
                im.load()
        except Exception:
            continue
        words = _gate_words(_ocr_png_bytes(res.stdout))
        text = " ".join(w["text"] for w in words)[:OCR_MAX_TEXT_CHARS]
        out[sid] = {"text": text, "words": words}
    return out
