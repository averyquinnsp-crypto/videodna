"""Smart Audio Forensics: Silent/Speech/Music classifier, LUFS, BPM estimator, transients & timeline chart."""

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


def _estimate_bpm(samples: np.ndarray, sr: int) -> tuple[float | None, float]:
    """Estimate BPM (60-180) via onset-strength envelope autocorrelation (pure numpy)."""
    if len(samples) < sr * 3.0:
        return None, 0.0
    # Analyze up to first 60s for speed
    seg = samples[: sr * 60]
    hop = max(1, int(sr * 0.02))  # 50 Hz onset frame rate
    n_frames = len(seg) // hop
    if n_frames < 100:
        return None, 0.0
    mat = seg[: n_frames * hop].reshape(n_frames, hop)
    energy = np.sqrt(np.mean(mat ** 2, axis=1) + 1e-10)
    onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    # A beat needs real onsets: count jumps of at least 10% of the average
    # frame energy. A steady tone/drone has only micro-variation, and its
    # periodic envelope autocorrelates spuriously — v2.0 reported BPM 120
    # on a pure 440Hz sine. Fewer than 4 real onsets => no BPM.
    onset_floor = 0.10 * float(np.mean(energy))
    n_real_onsets = int(np.sum(onset > onset_floor))
    if n_real_onsets < 4:
        return None, 0.0
    onset = onset - onset.mean()
    if np.std(onset) < 1e-5:
        return None, 0.0

    # Autocorrelation via FFT
    n_fft = 1 << int(np.ceil(np.log2(len(onset) * 2)))
    spec = np.fft.rfft(onset, n=n_fft)
    ac = np.fft.irfft(spec * np.conj(spec), n=n_fft)[: len(onset)]
    ac /= (ac[0] + 1e-8)

    env_fps = sr / hop
    min_lag = max(1, int(env_fps * 60.0 / 180.0))  # 180 BPM
    max_lag = min(len(ac) - 1, int(env_fps * 60.0 / 60.0))  # 60 BPM
    if max_lag <= min_lag:
        return None, 0.0

    window = ac[min_lag : max_lag + 1]
    best_idx = int(np.argmax(window))
    peak_val = float(window[best_idx])
    if peak_val < 0.14:
        return None, round(peak_val, 2)
    lag = min_lag + best_idx

    # Octave-fold guard (v2.0.2): a periodic onset train autocorrelates
    # at every multiple of the true period, and the window's max-lag edge
    # (60 BPM) can out-score the fundamental — a 180 BPM click train
    # peaks at lag 50 (3x period, ac 0.909) above lag 17 (true period,
    # ac 0.775) and was read as 60.0 BPM. Check sub-multiples (lag/3,
    # then lag/2) and prefer the smallest lag whose autocorrelation is
    # comparably strong (>= 70% of the peak). True 60/120 BPM trains
    # are safe: their sub-multiple lags score ~0 (click60: ac[17]=-0.02,
    # ac[25]=-0.02; click120's sub-multiples fall below min_lag).
    for divisor in (3, 2):
        sub = int(round(lag / divisor))
        if sub >= min_lag and float(ac[sub]) >= 0.70 * peak_val:
            lag = sub
            peak_val = float(ac[sub])
            break

    # Parabolic interpolation around the chosen lag: the true period
    # usually falls between integer lags (180 BPM = 16.67 frames at
    # 50 fps, so lag 17 alone reads 176.5). Verified on fixtures:
    # click180 176.5 -> 178.7, click60/120 stay 60.0/120.0.
    lag_f = float(lag)
    if 0 < lag < len(ac) - 1:
        a, b, c = float(ac[lag - 1]), float(ac[lag]), float(ac[lag + 1])
        denom = a - 2.0 * b + c
        if denom != 0.0:
            delta = 0.5 * (a - c) / denom
            if abs(delta) <= 1.0:
                lag_f = lag + delta
    bpm = round(60.0 * env_fps / lag_f, 1)
    return bpm, round(min(1.0, peak_val), 2)


def _classify_audio_content(
    samples: np.ndarray,
    sr: int,
    dead_air_ratio: float,
    bpm_conf: float,
) -> tuple[str, dict[str, float]]:
    """Classify audible track into speech_only, music_only, speech_plus_bgm, or ambient_or_sfx."""
    seg = samples[: sr * 30]
    if len(seg) < sr:
        return "ambient_or_sfx", {}

    # Compute average magnitude spectrum across 512-sample windows
    win = 512
    n_w = len(seg) // win
    frames = seg[: n_w * win].reshape(n_w, win)
    mag = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) + 1e-8
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    mean_spec = mag.mean(axis=0)

    total_e = float(np.sum(mean_spec)) + 1e-8
    bass_e = float(np.sum(mean_spec[(freqs >= 40) & (freqs < 250)])) / total_e
    speech_e = float(np.sum(mean_spec[(freqs >= 300) & (freqs <= 3200)])) / total_e
    high_e = float(np.sum(mean_spec[freqs > 3200])) / total_e

    # Tonality: a near-pure tone (test signal / drone) concentrates almost
    # all energy in one spectral bin and has near-zero spectral flatness.
    # Without this check a 440Hz sine looks like "speech" (its energy sits
    # inside the 300-3200Hz speech band) and spurious BPM pushes it to
    # speech_plus_bgm (v2.0 bug) — so test for tone FIRST.
    dominant_bin_ratio = float(np.max(mean_spec)) / total_e
    log_spec = np.log(mean_spec + 1e-12)
    spectral_flatness = float(np.exp(np.mean(log_spec)) / (np.mean(mean_spec) + 1e-12))

    # Frame energy variability (speech modulates strongly between syllables; steady music is flatter)
    rms_f = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    low_energy_frac = float(np.mean(rms_f < (np.percentile(rms_f, 80) * 0.20)))

    feats = {
        "bass_band_ratio": round(bass_e, 3),
        "speech_band_ratio": round(speech_e, 3),
        "high_band_ratio": round(high_e, 3),
        "modulation_pause_ratio": round(low_energy_frac, 3),
        "dominant_bin_ratio": round(dominant_bin_ratio, 3),
        "spectral_flatness": round(spectral_flatness, 4),
    }

    if dominant_bin_ratio >= 0.45 or spectral_flatness < 0.005:
        return "test_tone", feats

    # v2.0.3: pause gate 0.14 -> 0.05, speech_e gate 0.58 -> 0.55.
    # Measured on real samples: Hindi dialogue (speech1) has
    # low_energy_frac 0.095 and speech_e 0.583 — it failed the old
    # 0.14 pause gate and sat only +0.003 above the old speech_e gate,
    # so loud continuous speech with almost no gaps (dead air 0.0,
    # bpm_conf 0.11) fell through to ambient_or_sfx. Music (U4) has
    # low_energy_frac 0.005 and speech_e 0.572, so it still fails the
    # 0.05 pause gate by a symmetric 0.045 margin (0.05 is the exact
    # midpoint of the two known samples); the pause gate, not
    # speech_e, is the separator. Re-validate 0.05 on more real
    # samples before shipping — only one real-speech and one
    # real-music sample exist so far.
    if speech_e >= 0.55 and bass_e < 0.22 and (low_energy_frac >= 0.05 or dead_air_ratio >= 0.05):
        return "speech_only", feats
    if (bass_e >= 0.22 or bpm_conf >= 0.25) and low_energy_frac < 0.12:
        if speech_e >= 0.45:
            return "speech_plus_bgm", feats
        return "music_only", feats
    if speech_e >= 0.48 and bpm_conf >= 0.15:
        # Voice-activity / pause-structure gate (v2.0.2): real speech
        # pauses between syllables and phrases, so it shows low-energy
        # frames or dead air. Continuous music has neither — U4 (music,
        # no speech) measured pause ratio 0.005 and dead air 0.0 yet
        # fired this branch (speech_e 0.572, bpm_conf 0.19) and was
        # called speech_plus_bgm. Require some pause structure before
        # calling a track speech; without it the track falls through to
        # music_only / ambient_or_sfx below.
        if low_energy_frac >= 0.05 or dead_air_ratio >= 0.02:
            return "speech_plus_bgm", feats
    if bass_e >= 0.25:
        return "music_only", feats
    return "ambient_or_sfx", feats


def analyze_audio(
    video_path: Path,
    duration: float,
    shots: list[dict[str, Any]],
    out_dir: Path,
    has_audio: bool = True,
) -> dict[str, Any]:
    """Measure loudness, detect true silent tracks, classify speech/music, estimate BPM & cut-sync."""
    if not has_audio or duration <= 0:
        return {
            "has_audio": False,
            "is_silent_track": True,
            "audio_type": "silent_no_audio",
            "integrated_lufs": None,
            "loudness_range_lu": None,
            "true_peak_dbfs": None,
            "loudness_note": "no_audio_stream (visual-only clip)",
            "estimated_bpm": None,
            "bpm_confidence": 0.0,
            "spectral_features": {},
            "dead_air_segments": [],
            "dead_air_count": 0,
            "dead_air_total_sec": 0.0,
            "audio_transients_count": 0,
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

    # 2. Decode 16kHz mono PCM for true silence check, energy curve, spectral classification & BPM
    sr = 16000
    pcm_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    pcm_res = subprocess.run(pcm_cmd, capture_output=True)
    samples = np.frombuffer(pcm_res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    max_amp = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms_all = float(np.sqrt(np.mean(samples ** 2) + 1e-12)) if len(samples) else 0.0
    overall_db = float(20.0 * np.log10(max(rms_all, 1e-6)))

    # True Silent Track check (fixes false "too quiet / breaks verbal momentum" alarm!)
    is_silent = (
        len(samples) == 0
        or max_amp < 0.0015
        or overall_db <= -56.0
        or (integrated_lufs is not None and integrated_lufs <= -58.0)
    )

    cut_times = [s["start"] for s in shots[1:]]

    if is_silent:
        chart_path = _render_audio_chart(
            out_dir / "audio_timeline.png",
            [{"t": 0.0, "db": -60.0}, {"t": duration, "db": -60.0}],
            cut_times,
            [],
            duration,
            integrated_lufs,
            is_silent=True,
        )
        return {
            "has_audio": True,
            "is_silent_track": True,
            "audio_type": "silent_no_audio",
            "integrated_lufs": integrated_lufs,
            "loudness_range_lu": loudness_range,
            "true_peak_dbfs": true_peak,
            "loudness_note": "silent_track (audio stream present but completely mute — visual template/raw clip)",
            "estimated_bpm": None,
            "bpm_confidence": 0.0,
            "spectral_features": {},
            "dead_air_segments": [],
            "dead_air_count": 0,
            "dead_air_total_sec": 0.0,
            "audio_transients_count": 0,
            "audio_transients": [],
            "cut_beat_sync_ratio": 0.0,
            "energy_curve_db": [],
            "timeline_chart": str(chart_path.relative_to(out_dir)) if chart_path else None,
        }

    # 3. Silence / Dead-Air Detection (only meaningful on audible tracks!)
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
    # Trailing silence: silencedetect emits silence_start without a matching
    # silence_end when the file ends mid-silence — count it to EOF.
    if len(starts) > len(ends):
        s = starts[-1]
        dead_air.append({
            "start": round(s, 2),
            "end": round(duration, 2),
            "duration": round(max(0.0, duration - s), 2),
        })
    dead_air_total = round(sum(x["duration"] for x in dead_air), 2)
    dead_air_ratio = dead_air_total / max(duration, 0.1)

    # 4. Energy Envelope & Transients
    win_sec = 0.25 if duration <= 210 else 0.5
    win_n = max(1, int(sr * win_sec))
    n_bins = len(samples) // win_n
    energy_db: list[dict[str, float]] = []
    transients: list[dict[str, Any]] = []

    if n_bins > 2:
        frames = samples[: n_bins * win_n].reshape(n_bins, win_n)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
        db = np.clip(20.0 * np.log10(rms), -60.0, 0.0)
        for i, val in enumerate(db):
            energy_db.append({"t": round(i * win_sec, 2), "db": round(float(val), 1)})

        diffs = np.diff(db, prepend=db[0])
        for i in range(1, len(db)):
            if diffs[i] >= 5.5 and db[i] >= -26.0:
                t_hit = round(i * win_sec, 2)
                if not transients or (t_hit - transients[-1]["t"]) >= 0.35:
                    transients.append({
                        "t": t_hit,
                        "jump_db": round(float(diffs[i]), 1),
                        "peak_db": round(float(db[i]), 1),
                    })

    # 5. BPM & Audio Content Classification
    bpm, bpm_conf = _estimate_bpm(samples, sr)
    audio_type, spec_feats = _classify_audio_content(samples, sr, dead_air_ratio, bpm_conf)
    # BPM is only meaningful for music; speech / ambient / test-tone tracks
    # get no BPM even if the onset envelope shows spurious periodicity.
    if audio_type not in ("music_only", "speech_plus_bgm"):
        bpm, bpm_conf = None, 0.0

    # 6. Cut-to-Beat Sync Score
    synced_cuts = sum(1 for ct in cut_times if any(abs(tr["t"] - ct) <= 0.30 for tr in transients))
    sync_ratio = round(synced_cuts / len(cut_times), 2) if cut_times else 0.0

    chart_path = _render_audio_chart(
        out_dir / "audio_timeline.png",
        energy_db,
        cut_times,
        dead_air,
        duration,
        integrated_lufs,
        is_silent=False,
        audio_type=audio_type,
        bpm=bpm,
    )

    if len(energy_db) > 180:
        idx = np.linspace(0, len(energy_db) - 1, 180, dtype=int)
        compact_curve = [energy_db[i] for i in idx]
    else:
        compact_curve = energy_db

    loudness_note = "unknown"
    if integrated_lufs is not None:
        if integrated_lufs < -20.0:
            loudness_note = f"too_quiet ({integrated_lufs} LUFS — target -14 LUFS for Shorts/YouTube)"
        elif integrated_lufs <= -12.0:
            loudness_note = f"optimal_streaming_loudness ({integrated_lufs} LUFS)"
        else:
            loudness_note = f"very_loud_or_compressed ({integrated_lufs} LUFS)"
        if true_peak is not None and true_peak > -0.3:
            loudness_note += f" [clipping risk: {true_peak} dBFS]"

    return {
        "has_audio": True,
        "is_silent_track": False,
        "audio_type": audio_type,
        "integrated_lufs": integrated_lufs,
        "loudness_range_lu": loudness_range,
        "true_peak_dbfs": true_peak,
        "loudness_note": loudness_note,
        "estimated_bpm": bpm,
        "bpm_confidence": bpm_conf,
        "spectral_features": spec_feats,
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
    is_silent: bool = False,
    audio_type: str = "",
    bpm: float | None = None,
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

    if is_silent:
        header = "AUDIO TIMELINE | SILENT / MUTE TRACK (No Audible Sound) | Yellow = Visual Cuts"
    else:
        lufs_str = f"{lufs:.1f} LUFS" if lufs is not None else "N/A"
        bpm_str = f" | ~{bpm} BPM" if bpm else ""
        header = f"AUDIO TIMELINE | {audio_type.upper()} | {lufs_str}{bpm_str} | Blue=Energy Red=Silence Yellow=Cuts"
    draw.text((pad_l, 10), header, fill=(230, 238, 255), font=font)

    for db_mark in (-50, -35, -20, -6):
        y = pad_t + int(((0.0 - db_mark) / 60.0) * plot_h)
        draw.line([(pad_l, y), (w - pad_r, y)], fill=(38, 44, 62), width=1)
        draw.text((8, y - 6), f"{db_mark}dB", fill=(130, 145, 175), font=font)

    if is_silent:
        draw.rectangle([pad_l, pad_t, w - pad_r, pad_t + plot_h], fill=(24, 28, 38))
        draw.text(
            (pad_l + 240, pad_t + plot_h // 2 - 8),
            "SILENT TRACK — VISUAL TEMPLATE / RAW CLIP (ADD VOICE + BGM + SFX)",
            fill=(165, 180, 210),
            font=font,
        )
    else:
        for seg in dead_air:
            x1 = pad_l + int((seg["start"] / duration) * plot_w)
            x2 = pad_l + int((seg["end"] / duration) * plot_w)
            draw.rectangle([x1, pad_t, max(x1 + 2, x2), pad_t + plot_h], fill=(95, 28, 38))

        pts = []
        for p in energy_db:
            x = pad_l + int((p["t"] / duration) * plot_w)
            y = pad_t + int(((0.0 - p["db"]) / 60.0) * plot_h)
            pts.append((x, max(pad_t, min(pad_t + plot_h, y))))
        if len(pts) >= 2:
            draw.line(pts, fill=(75, 195, 255), width=2)

    for ct in cut_times[:120]:
        xc = pad_l + int((ct / duration) * plot_w)
        draw.line([(xc, pad_t), (xc, pad_t + plot_h)], fill=(235, 185, 55), width=1)

    for i in range(6):
        t_mark = (duration / 5.0) * i
        x = pad_l + int((t_mark / duration) * plot_w)
        draw.text((x - 10, h - 22), f"{t_mark:.0f}s", fill=(150, 165, 195), font=font)

    img.save(out_path)
    return out_path
