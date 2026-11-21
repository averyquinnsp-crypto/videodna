"""Deep container metadata, cadence, and AIGC/software forensics using ffprobe."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


SOFTWARE_SIGNATURES = {
    "capcut": ["capcut", "bytedance", "jianying"],
    "premiere": ["adobe", "premiere", "after effects"],
    "davinci": ["blackmagic", "davinci", "resolve"],
    "remotion_or_code": ["remotion", "chromium", "playwright", "hyperframes", "headless"],
    "final_cut": ["final cut", "apple prores", "com.apple.quicktime"],
    "ai_generator": [
        "aigc", "c2pa", "sora", "runway", "kling", "veo", "minimax",
        "hailuo", "seedance", "pika", "luma", "midjourney", "jimeng",
        "heygen", "synthesia", "d-id", "krea", "wan",
    ],
    "platform_reencode": [
        "tiktok", "douyin", "instagram", "facebook", "fbcdn", "youtube",
        "wechat", "x264", "iso5", "mp42",
    ],
}


def _parse_ratio(rate_str: str) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        try:
            d = float(den)
            return round(float(num) / d, 3) if d else 0.0
        except ValueError:
            return 0.0
    try:
        return round(float(rate_str), 3)
    except ValueError:
        return 0.0


def probe_video(video_path: Path) -> dict[str, Any]:
    """Run ffprobe and extract forensic technical metadata."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(res.stdout)

    fmt = raw.get("format", {})
    streams = raw.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not v_stream:
        raise ValueError(f"No video stream found in {video_path}")

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    duration = float(fmt.get("duration") or v_stream.get("duration") or 0.0)
    size_bytes = int(fmt.get("size") or video_path.stat().st_size)
    bitrate_kbps = round(int(fmt.get("bit_rate") or 0) / 1000, 1)

    avg_fps = _parse_ratio(v_stream.get("avg_frame_rate", "0/0"))
    r_fps = _parse_ratio(v_stream.get("r_frame_rate", "0/0"))
    is_vfr = abs(avg_fps - r_fps) > 0.5 if (avg_fps and r_fps) else False

    # Aspect ratio & form classification
    ratio = round(width / height, 3) if height else 0.0
    if ratio < 0.75:
        orientation = "vertical_9_16"
    elif ratio > 1.25:
        orientation = "horizontal_16_9"
    else:
        orientation = "square_1_1"

    # Auto mode: Short (<= 180s or vertical <= 210s) vs Long
    if duration <= 180.0 or (orientation == "vertical_9_16" and duration <= 210.0):
        recommended_mode = "short"
    else:
        recommended_mode = "long"

    # Collect all metadata tags for forensic signature scanning
    all_tags: dict[str, str] = {}
    for k, v in (fmt.get("tags") or {}).items():
        all_tags[f"format.{k}"] = str(v)
    for k, v in (v_stream.get("tags") or {}).items():
        all_tags[f"video.{k}"] = str(v)
    if a_stream:
        for k, v in (a_stream.get("tags") or {}).items():
            all_tags[f"audio.{k}"] = str(v)

    blob = " ".join(f"{k}={v}" for k, v in all_tags.items()).lower()
    detected_signals: dict[str, list[str]] = {}
    for category, keywords in SOFTWARE_SIGNATURES.items():
        hits = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}", blob) or kw in blob]
        if hits:
            detected_signals[category] = hits

    # Cadence clues (AI video generators often output 24fps or 16fps/8fps-upsampled, screen caps VFR)
    cadence_notes: list[str] = []
    if is_vfr:
        cadence_notes.append(f"Variable frame rate detected (avg {avg_fps} vs nominal {r_fps}) — common in phone screen recordings or mobile capture.")
    if avg_fps in (24.0, 23.976, 16.0, 8.0):
        cadence_notes.append(f"Cinema/AI-typical cadence ({avg_fps} fps).")
    elif avg_fps >= 58.0:
        cadence_notes.append(f"High frame rate ({avg_fps} fps) — common in gaming, modern motion graphics, or upscaled edits.")
    if "ai_generator" in detected_signals:
        cadence_notes.append(f"Explicit AI/AIGC metadata tag found: {', '.join(detected_signals['ai_generator'])}.")
    if bitrate_kbps and bitrate_kbps < 1200 and max(width, height) >= 1080:
        cadence_notes.append(f"Low bitrate ({bitrate_kbps} kbps for {width}x{height}) — likely platform-recompressed (sensor noise may be stripped).")

    chapters = []
    for idx, ch in enumerate(raw.get("chapters", []), start=1):
        chapters.append({
            "index": idx,
            "start": round(float(ch.get("start_time", 0.0)), 2),
            "end": round(float(ch.get("end_time", 0.0)), 2),
            "title": (ch.get("tags") or {}).get("title", f"Chapter {idx}"),
        })

    return {
        "file": str(video_path),
        "filename": video_path.name,
        "duration_sec": round(duration, 2),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "bitrate_kbps": bitrate_kbps,
        "width": width,
        "height": height,
        "aspect_ratio": ratio,
        "orientation": orientation,
        "recommended_mode": recommended_mode,
        "video_codec": v_stream.get("codec_name", "unknown"),
        "pix_fmt": v_stream.get("pix_fmt", "unknown"),
        "avg_fps": avg_fps,
        "nominal_fps": r_fps,
        "is_vfr": is_vfr,
        "has_audio": a_stream is not None,
        "audio_codec": a_stream.get("codec_name") if a_stream else None,
        "audio_sample_rate": int(a_stream.get("sample_rate", 0)) if a_stream else 0,
        "audio_channels": int(a_stream.get("channels", 0)) if a_stream else 0,
        "embedded_chapters": chapters,
        "forensic_signals": detected_signals,
        "cadence_notes": cadence_notes,
        "raw_tags": all_tags,
    }
