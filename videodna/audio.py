"""Audio forensics: LUFS loudness, dead-air detection, energy envelope, SFX/beat hits, and timeline chart."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LUFS_RE = re.compile(r"I:\s*(-?[0-9.]+)\s*LUFS")
LRA_RE = re.compile(r"LRA:\s*([0-9.]+)\s*LU")
PEAK_RE = re.compile(r"Peak:\s*(-?[0-9.]+)\s*dBFS")
SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


def analyze_audio(
    video_path: Path,
    duration: float,
    shots: list[dict[str, Any]],
    out_dir: Path,
    has_audio: bool = True,
) -> dict[str, Any]:
    """Measure loudness, silences, audio energy curve, transients, and cut-sync."""
    if not has_audio or duration <= 0:
        return {
            "has_audio": False,
            "integrated_lufs": None,
            "loudness_range_lu": None,
            "true_peak_dbfs": None,
            "dead_air_segments": [],
            "dead_air_total_sec": 0.0,
            "audio_transients": [],
            "cut_beat_sync_ratio": 0.0,
            "energy_curve_db": [],
            "timeline_chart": None,
        }

    # 1. Integrated Loudness (EBU R128)
    loud_cmd = [
        "ffmpeg", "-hide_banner", "-i", str(video_path),
        "-af", "ebur128=peak=true", "-vn", "-f", "null", "-",
    ]
    loud_res = subprocess.run(loud_cmd, capture_output=True, text=True)
    lufs_matches = LUFS_RE.findall(loud_res.stderr)
    lra_matches = LRA_RE.findall(loud_res.stderr)
    peak_matches = PEAK_RE.findall(loud_res.stderr)

    integrated_lufs = float(lufs_matches[-1]) if lufs_matches else None
    loudness_range = float(lra_matches[-1]) if lra_matches else None
    true_peak = float(peak_matches[-1]) if peak_matches else None

    # 2. Silence / Dead-Air Detection (>0.45s below -38dB)
    sil_cmd = [
        "ffmpeg", "-hide_banner", "-i", str(video_path),
        "-af", "silencedetect=noise=-38dB:d=0.45", "-vn", "-f", "null", "-",
    ]
    sil_res = subprocess.run(sil_cmd, capture_output=True, text=True)
    starts = [float(x) for x in SILENCE_START_RE.findall(sil_res.stderr)]
    ends = [(float(e), float(d)) for e, d in SILENCE_END_RE.findall(sil_res.stderr)]
    dead_air: list[dict[str, float]] = []
    for idx, (e, d) in enumerate(ends):
        s = starts[idx] if idx < len(starts) else max(0.0, e - d)
        dead_air.append({"start": round(s, 2), "end": round(e, 2), "duration": round(d, 2)})
    dead_air_total = round(sum(x["duration"] for x in dead_air), 2)

    # 3. Energy Envelope & Transient (SFX/Beat Hit) Detection via 8kHz mono PCM
    sr = 8000
    pcm_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    pcm_res = subprocess.run(pcm_cmd, capture_output=True)
    samples = np.frombuffer(pcm_res.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    # Window size: 0.25s for short videos, 0.5s for long videos
    win_sec = 0.25 if duration <= 210 else 0.5
    win_n = max(1, int(sr * win_sec))
    n_bins = len(samples) // win_n
    energy_db: list[dict[str, float]] = []
    transients: list[dict[str, Any]] = []

    if n_bins > 2:
        frames = samples[: n_bins * win_n].reshape(n_bins, win_n)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
        db = 20.0 * np.log10(rms)
        db = np.clip(db, -60.0, 0.0)

        for i, val in enumerate(db):
            energy_db.append({"t": round(i * win_sec, 2), "db": round(float(val), 1)})

        # Transients: sudden jump >= +5.5 dB over previous window and above -26 dB
        diffs = np.diff(db, prepend=db[0])
        for i in range(1, len(db)):
            if diffs[i] >= 5.5 and db[i] >= -26.0:
                t_hit = round(i * win_sec, 2)
                if not transients or (t_hit - transients[-1]["t"]) >= 0.4:
                    transients.append({
                        "t": t_hit,
                        "jump_db": round(float(diffs[i]), 1),
                        "peak_db": round(float(db[i]), 1),
                    })

    # 4. Audio-to-Cut Sync Score (how many visual cuts land near an audio spike)
    cut_times = [s["start"] for s in shots[1:]]
    synced_cuts = 0
    for ct in cut_times:
        if any(abs(tr["t"] - ct) <= 0.30 for tr in transients):
            synced_cuts += 1
    sync_ratio = round(synced_cuts / len(cut_times), 2) if cut_times else 0.0

    # 5. Render audio_timeline.png
    chart_path = _render_audio_chart(
        out_dir / "audio_timeline.png",
        energy_db,
        cut_times,
        dead_air,
        duration,
        integrated_lufs,
    )

    # Downsample energy curve stored in JSON for long videos (max 180 points)
    if len(energy_db) > 180:
        idx = np.linspace(0, len(energy_db) - 1, 180, dtype=int)
        compact_curve = [energy_db[i] for i in idx]
    else:
        compact_curve = energy_db

    # Loudness verdict (directly maps to voice-pro target -14 LUFS)
    loudness_note = "unknown"
    if integrated_lufs is not None:
        if integrated_lufs < -20.0:
            loudness_note = f"too_quiet ({integrated_lufs} LUFS — target -14 LUFS for Shorts/YouTube)"
        elif integrated_lufs <= -12.0:
            loudness_note = f"optimal_streaming_loudness ({integrated_lufs} LUFS)"
        else:
            loudness_note = f"very_loud_or_compressed ({integrated_lufs} LUFS)"

    return {
        "has_audio": True,
        "integrated_lufs": integrated_lufs,
        "loudness_range_lu": loudness_range,
        "true_peak_dbfs": true_peak,
        "loudness_note": loudness_note,
        "dead_air_segments": dead_air[:30],
        "dead_air_count": len(dead_air),
        "dead_air_total_sec": dead_air_total,
        "audio_transients_count": len(transients),
        "audio_transients": transients[:40],
        "cut_beat_sync_ratio": sync_ratio,
        "energy_curve_db": compact_curve,
        "timeline_chart": str(chart_path.relative_to(out_dir)) if chart_path else None,
    }


def _render_audio_chart(
    out_path: Path,
    energy_db: list[dict[str, float]],
    cut_times: list[float],
    dead_air: list[dict[str, float]],
    duration: float,
    lufs: float | None,
) -> Path | None:
    if not energy_db or duration <= 0:
        return None
    w, h = 1000, 260
    pad_l, pad_r, pad_t, pad_b = 50, 20, 38, 34
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    img = Image.new("RGB", (w, h), (15, 18, 26))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lufs_str = f"{lufs:.1f} LUFS" if lufs is not None else "N/A"
    draw.text(
        (pad_l, 10),
        f"AUDIO ENERGY & CUT TIMELINE | Integrated: {lufs_str} | Blue=Energy  Red=Dead-Air  Yellow=Cuts",
        fill=(230, 238, 255),
        font=font,
    )

    # Grid lines (-60 to 0 dB)
    for db_mark in (-50, -35, -20, -6):
        y = pad_t + int(((0.0 - db_mark) / 60.0) * plot_h)
        draw.line([(pad_l, y), (w - pad_r, y)], fill=(38, 44, 62), width=1)
        draw.text((8, y - 6), f"{db_mark}dB", fill=(130, 145, 175), font=font)

    # Dead-air highlight bars
    for seg in dead_air:
        x1 = pad_l + int((seg["start"] / duration) * plot_w)
        x2 = pad_l + int((seg["end"] / duration) * plot_w)
        draw.rectangle([x1, pad_t, max(x1 + 2, x2), pad_t + plot_h], fill=(95, 28, 38))

    # Visual cuts as vertical amber ticks
    for ct in cut_times[:120]:
        xc = pad_l + int((ct / duration) * plot_w)
        draw.line([(xc, pad_t), (xc, pad_t + plot_h)], fill=(235, 185, 55), width=1)

    # Audio energy polyline
    pts = []
    for p in energy_db:
        x = pad_l + int((p["t"] / duration) * plot_w)
        y = pad_t + int(((0.0 - p["db"]) / 60.0) * plot_h)
        pts.append((x, max(pad_t, min(pad_t + plot_h, y))))
    if len(pts) >= 2:
        draw.line(pts, fill=(75, 195, 255), width=2)

    # Time axis ticks
    for i in range(6):
        t_mark = (duration / 5.0) * i
        x = pad_l + int((t_mark / duration) * plot_w)
        draw.text((x - 10, h - 22), f"{t_mark:.0f}s", fill=(150, 165, 195), font=font)

    img.save(out_path)
    return out_path
