"""Adaptive Short-Form & Long-Form shot detection, pacing telemetry, and dead-zone audit.

Cut detection runs in TWO passes:
  1. ffmpeg `scdet` — precise for hard cuts.
  2. Frame-diff pass — catches wipes / fades / dissolves / crossfades that scdet
     misses (common in code-rendered motion graphics, e.g. HyperFrames/Remotion),
     using an adaptive threshold over per-frame mean absolute pixel difference.
The two cut lists are merged (scdet wins ties inside min_gap).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


SCDET_RE = re.compile(r"lavfi\.scd\.score:\s*([0-9.]+),\s*lavfi\.scd\.time:\s*([0-9.]+)")

# v2.0.2 soft-cut guard (layout/background continuity).
# The frame-diff pass over-segments motion-graphics videos: big animated
# text/elements moving inside ONE continuous shot produce diff peaks that look
# like transitions. A real soft transition replaces the layout/background, so
# for each frame-diff candidate we compare the settled frames on either side:
#   * GUARD_OFFSET_SEC — how far either side of the candidate the comparison
#     frames sit (also the settled footage a candidate needs on both sides;
#     closer to the start/end than this, there is no before/after state to
#     compare — intro/outro animation and fade-from/to-black live there).
#   * GUARD_PIXEL_THRESHOLD — per-pixel mean |RGB| change (0-255) that counts
#     as "this pixel changed".
#   * GUARD_MIN_CHANGED_FRACTION — if fewer than this fraction of pixels
#     changed between the before/after frames, the layout/background is
#     essentially unchanged and the candidate is in-shot animation, not a cut.
# Tuned on 7 videos (v2.0.2 regression set): at offset 0.75s / threshold 25,
# in-shot animation scored 0.19-0.29 changed fraction while real transitions
# scored 0.59-1.00, so 0.45 sits mid-gap. scdet hard cuts are NEVER guarded.
GUARD_OFFSET_SEC = 0.75
GUARD_PIXEL_THRESHOLD = 25.0
GUARD_MIN_CHANGED_FRACTION = 0.45


def _changed_fraction(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fraction of pixels whose mean |RGB| change exceeds GUARD_PIXEL_THRESHOLD."""
    diff = np.mean(
        np.abs(
            frame_a.reshape(-1, 3).astype(np.float32)
            - frame_b.reshape(-1, 3).astype(np.float32)
        ),
        axis=1,
    )
    return float(np.mean(diff > GUARD_PIXEL_THRESHOLD))


def _sample_frames_at(
    video_path: Path,
    times: list[float],
    sample_fps: float,
    sample_w: int,
    sample_h: int,
) -> dict[int, np.ndarray]:
    """Decode once at the frame-diff sample rate and return {frame_index: frame}
    for the requested times (uint8 RGB arrays). Streams and stops as soon as
    the last wanted frame has passed, so only the wanted frames are ever held.
    Returns {} on any ffmpeg failure (callers then keep candidates unguarded).
    """
    wanted = {int(round(t * sample_fps)) for t in times}
    if not wanted:
        return {}
    last_idx = max(wanted)
    frame_bytes = sample_w * sample_h * 3
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"scale={sample_w}:{sample_h},fps={sample_fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return {}
    assert proc.stdout is not None
    found: dict[int, np.ndarray] = {}
    try:
        idx = 0
        while idx <= last_idx:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if idx in wanted:
                found[idx] = np.frombuffer(buf, dtype=np.uint8).copy()
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    return found


def _guard_soft_cuts(
    video_path: Path,
    duration: float,
    cuts: list[tuple[float, float]],
    sample_fps: float,
    sample_w: int,
    sample_h: int,
) -> tuple[list[tuple[float, float]], int]:
    """v2.0.2 layout/background-continuity guard for frame-diff candidates.

    Keeps a candidate only if the settled frames GUARD_OFFSET_SEC before and
    after it differ across at least GUARD_MIN_CHANGED_FRACTION of pixels.
    Candidates too close to the start/end to have settled footage on both
    sides are suppressed (no before/after state exists to compare).
    Returns (kept_cuts, suppressed_count). On any frame-grab failure the
    candidates are returned unguarded (fail open — never lose a real cut).
    """
    if not cuts:
        return cuts, 0
    keep: list[tuple[float, float]] = []
    need_times: list[float] = []
    for t, _score in cuts:
        if GUARD_OFFSET_SEC <= t <= duration - GUARD_OFFSET_SEC:
            need_times.append(t - GUARD_OFFSET_SEC)
            need_times.append(t + GUARD_OFFSET_SEC)
    frames = _sample_frames_at(video_path, need_times, sample_fps, sample_w, sample_h)
    if len(frames) < len({int(round(t * sample_fps)) for t in need_times}):
        return cuts, 0
    suppressed = 0
    for t, score in cuts:
        if not (GUARD_OFFSET_SEC <= t <= duration - GUARD_OFFSET_SEC):
            suppressed += 1
            continue
        before = frames.get(int(round((t - GUARD_OFFSET_SEC) * sample_fps)))
        after = frames.get(int(round((t + GUARD_OFFSET_SEC) * sample_fps)))
        if before is None or after is None:
            keep.append((t, score))
            continue
        if _changed_fraction(before, after) < GUARD_MIN_CHANGED_FRACTION:
            suppressed += 1
            continue
        keep.append((t, score))
    return keep, suppressed


def _frame_diff_cuts(
    video_path: Path,
    duration: float,
    width: int | None,
    height: int | None,
    min_gap: float,
) -> list[tuple[float, float]]:
    """Second-pass cut detection via streaming per-frame pixel differences.

    Decodes at low resolution + low fps through a rawvideo pipe (constant memory:
    only two frames are held at a time). Returns ([(time_sec, diff_score)],
    suppressed_count) for detected transitions, where suppressed_count is the
    number of candidates dropped by the v2.0.2 layout/background-continuity
    guard. Returns ([], 0) on any ffmpeg failure.
    """
    sample_fps = 6.0 if duration <= 210 else 3.0
    sample_w = 160
    if width and height:
        sample_h = max(2, int(round(height * sample_w / width / 2.0) * 2))
    else:
        sample_h = 90
    frame_bytes = sample_w * sample_h * 3

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"scale={sample_w}:{sample_h},fps={sample_fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return [], 0
    assert proc.stdout is not None

    diffs: list[float] = []
    prev: np.ndarray | None = None
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(frame - prev))))
            prev = frame
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()

    if len(diffs) < 3:
        return [], 0

    d = np.asarray(diffs, dtype=np.float64)
    # Threshold: twice the video's median frame-to-frame change, with a floor
    # so sensor noise / tiny motion on near-static footage never triggers.
    # (A global MAD-based threshold proved too aggressive on bimodal videos —
    # e.g. a busy first half + static title card inflates MAD and hides real
    # slide transitions.)
    threshold = max(12.0, 2.0 * float(np.median(d)))
    # ±0.75s local-max window: gradual transitions (wipes/fades/slides) take
    # ~0.5–1s to complete, so two distinct ones cannot peak closer together.
    # (Hard cuts closer than this are scdet's job — it stays frame-accurate.)
    half_win = max(1, int(round(0.75 * sample_fps)))

    # A cut is a local maximum above threshold within ±0.75s. This collapses a
    # multi-frame wipe/fade (sustained high diffs) into ONE cut at its peak
    # instead of flagging every frame of the transition.
    cuts: list[tuple[float, float]] = []
    n = len(d)
    for i in range(n):
        if d[i] < threshold:
            continue
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        if d[i] < float(np.max(d[lo:hi])):
            continue
        # diffs[k] measures the change INTO frame k+1, so its timestamp is (k+1)/fps.
        t = round((i + 1) / sample_fps, 3)
        if not (0.15 < t < duration - 0.15):
            continue
        if not cuts or (t - cuts[-1][0]) >= min_gap:
            cuts.append((t, round(float(d[i]), 2)))
        elif d[i] > cuts[-1][1]:
            cuts[-1] = (t, round(float(d[i]), 2))

    # v2.0.2 guard: drop candidates whose before/after frames show the same
    # layout/background (in-shot motion-graphics animation, intro/outro).
    # scdet hard cuts are never touched — this only filters this pass.
    cuts, suppressed = _guard_soft_cuts(
        video_path, duration, cuts, sample_fps, sample_w, sample_h
    )
    return cuts, suppressed


def detect_shots(
    video_path: Path,
    duration: float,
    mode: str = "short",
    threshold: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Detect scene cuts using ffmpeg scdet + frame-diff second pass, then compute pacing + dead-zone metrics."""
    if threshold is None:
        # Shorts use tighter cut sensitivity; long-form avoids micro-jitter over-splitting
        threshold = 10.5 if mode == "short" else 14.0

    min_gap = 0.25 if mode == "short" else 0.5

    # Pass 1: ffmpeg scdet (hard cuts).
    vf = f"scale=320:-2,scdet=threshold={threshold}:sc_pass=0"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i", str(video_path),
        "-vf", vf,
        "-an",
        "-f", "null",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stderr = res.stderr

    raw_cuts: list[tuple[float, float]] = []
    for m in SCDET_RE.finditer(stderr):
        score = float(m.group(1))
        t = float(m.group(2))
        if t <= 0.15 or (duration > 0 and t >= duration - 0.15):
            continue
        if raw_cuts and (t - raw_cuts[-1][0]) < min_gap:
            # Keep the stronger cut inside min_gap
            if score > raw_cuts[-1][1]:
                raw_cuts[-1] = (t, score)
            continue
        raw_cuts.append((t, score))
    scdet_cut_count = len(raw_cuts)
    scdet_times = {t for t, _ in raw_cuts}

    # Pass 2: frame-diff (wipes / fades / dissolves scdet misses). scdet wins
    # any tie inside min_gap because it is frame-accurate on hard cuts.
    # The v2.0.2 guard inside _frame_diff_cuts already dropped in-shot
    # animation candidates; diff_cut_count counts only the survivors.
    diff_cut_count = 0
    diff_cuts, guard_suppressed = _frame_diff_cuts(
        video_path, duration, width, height, min_gap
    )
    for t, score in diff_cuts:
        if any(abs(t - existing) < min_gap for existing, _ in raw_cuts):
            continue
        raw_cuts.append((t, score))
        diff_cut_count += 1
    raw_cuts.sort(key=lambda c: c[0])

    # Build contiguous shot list
    boundaries = [0.0] + [t for t, _ in raw_cuts] + [max(duration, 0.1)]
    scores_map = {t: s for t, s in raw_cuts}
    shots: list[dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        start = round(boundaries[i], 3)
        end = round(boundaries[i + 1], 3)
        dur = round(max(0.0, end - start), 3)
        if dur < 0.05:
            continue
        cut_src = (
            "start" if i == 0
            else ("scdet" if boundaries[i] in scdet_times else "frame_diff")
        )
        shots.append({
            "id": len(shots) + 1,
            "start": start,
            "end": end,
            "duration": dur,
            "cut_score_in": round(scores_map.get(boundaries[i], 0.0), 2),
            "cut_source_in": cut_src,
            "transition_in": (
                "opening" if cut_src == "start"
                else ("hard_cut" if cut_src == "scdet" else "soft_transition")
            ),
        })

    if not shots:
        shots = [{
            "id": 1,
            "start": 0.0,
            "end": round(duration, 3),
            "duration": round(duration, 3),
            "cut_score_in": 0.0,
            "cut_source_in": "start",
            "transition_in": "opening",
        }]

    durations = [s["duration"] for s in shots]
    total_cuts = max(0, len(shots) - 1)
    asl = round(sum(durations) / len(durations), 2)
    cuts_per_min = round((total_cuts / duration) * 60.0, 2) if duration > 0 else 0.0

    # Hook velocity (critical retention windows)
    cuts_first_3s = sum(1 for t, _ in raw_cuts if t <= 3.0)
    cuts_first_10s = sum(1 for t, _ in raw_cuts if t <= 10.0)
    cuts_first_30s = sum(1 for t, _ in raw_cuts if t <= 30.0)

    # Dead-zone detection (shots dangerously long for the format)
    dead_zone_limit = 4.5 if mode == "short" else 14.0
    dead_zones = []
    for s in shots:
        if s["duration"] >= dead_zone_limit:
            severity = "high" if s["duration"] >= dead_zone_limit * 1.8 else "medium"
            dead_zones.append({
                "shot_id": s["id"],
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "severity": severity,
                "reason": (
                    f"No visual cut for {s['duration']}s "
                    f"(threshold for {mode}-form is {dead_zone_limit}s — high swipe/drop-off risk unless strong on-screen motion/graphics exist)."
                ),
            })

    # Pacing classification
    if mode == "short":
        if asl <= 1.8:
            pacing_style = "hyper_fast_viral"
        elif asl <= 3.2:
            pacing_style = "fast_modern_short"
        elif asl <= 5.5:
            pacing_style = "moderate_story"
        else:
            pacing_style = "slow_single_take_or_talking_head"
    else:
        if asl <= 3.5:
            pacing_style = "high_retention_mrbeast_style"
        elif asl <= 7.0:
            pacing_style = "dynamic_youtube_documentary"
        elif asl <= 15.0:
            pacing_style = "standard_explainer_or_vlog"
        else:
            pacing_style = "long_take_podcast_or_lecture"

    # Build Long-form Macro Acts (or Short-form 3-phase breakdown)
    acts = _build_macro_acts(duration, shots, mode)

    return {
        "mode": mode,
        "threshold_used": threshold,
        "detection": {
            "scdet_cuts": scdet_cut_count,
            "frame_diff_cuts": diff_cut_count,
            "frame_diff_suppressed_by_guard": guard_suppressed,
            "method": "scdet+frame_diff",
        },
        "shot_count": len(shots),
        "total_cuts": total_cuts,
        "hard_cuts": scdet_cut_count,
        "soft_transitions": diff_cut_count,
        "avg_shot_length_sec": asl,
        "shortest_shot_sec": round(min(durations), 2),
        "longest_shot_sec": round(max(durations), 2),
        "cuts_per_minute": cuts_per_min,
        "hook_cuts_0_3s": cuts_first_3s,
        "hook_cuts_0_10s": cuts_first_10s,
        "hook_cuts_0_30s": cuts_first_30s,
        "pacing_style": pacing_style,
        "dead_zone_threshold_sec": dead_zone_limit,
        "dead_zones": dead_zones,
        "macro_acts": acts,
        "shots": shots,
    }


def _build_macro_acts(duration: float, shots: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Segment the timeline into structural acts with local pacing density."""
    if duration <= 0:
        return []

    if mode == "short":
        # 4 viral short zones: Hook (0-3s), Setup (3-25%), Escalation (25-80%), Payoff/Loop (80-100%)
        t_hook = min(3.0, duration * 0.2)
        t_setup = max(t_hook + 0.5, duration * 0.30)
        t_payoff = max(t_setup + 0.5, duration * 0.80)
        windows = [
            ("Hook (Stop-the-Scroll)", 0.0, t_hook),
            ("Setup & Promise", t_hook, t_setup),
            ("Escalation & Value Delivery", t_setup, t_payoff),
            ("Payoff / Twist / Loop", t_payoff, duration),
        ]
    else:
        # Long-form: Critical Intro (0-30s), Setup (30s-15%), Mid Acts, Climax & Outro
        t_intro = min(30.0, duration * 0.1)
        t_act1 = max(t_intro + 5.0, duration * 0.25)
        t_act2 = duration * 0.55
        t_act3 = duration * 0.85
        windows = [
            ("Critical Hook & Intro (0-30s)", 0.0, t_intro),
            ("Act 1: Setup & Stakes", t_intro, t_act1),
            ("Act 2: Core Development / Re-Hooks", t_act1, t_act2),
            ("Act 3: Escalation & Deep Payoff", t_act2, t_act3),
            ("Climax, Resolution & CTA", t_act3, duration),
        ]

    acts = []
    for name, w_start, w_end in windows:
        w_start = round(min(w_start, duration), 2)
        w_end = round(min(w_end, duration), 2)
        if w_end - w_start < 0.1:
            continue
        in_window = [s for s in shots if s["end"] > w_start and s["start"] < w_end]
        cuts_in = sum(1 for s in in_window if s["start"] > w_start)
        dur = max(0.1, w_end - w_start)
        acts.append({
            "name": name,
            "start": w_start,
            "end": w_end,
            "duration": round(dur, 2),
            "shot_count": len(in_window),
            "cuts_per_min": round((cuts_in / dur) * 60.0, 1),
        })
    return acts
