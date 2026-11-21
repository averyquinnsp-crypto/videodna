"""Adaptive Short-Form & Long-Form shot detection, pacing telemetry, and dead-zone audit.

Cut detection runs in TWO passes:
  1. ffmpeg `scdet` — precise for hard cuts.
  2. Frame-diff pass — catches wipes / fades / dissolves / crossfades that scdet
     misses (common in code-rendered motion graphics, e.g. HyperFrames/Remotion),
     using an adaptive threshold over per-frame mean absolute pixel difference.
The two cut lists are merged (scdet wins ties inside min_gap).
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


from videodna.forensics import _phase_shift

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


# ---------------------------------------------------------------------------
# v2.0.4 item 3: soft-transition subtyping (fade / dissolve / slide_wipe /
# soft_unknown). Runs ONLY on frame-diff survivors, AFTER the v2.0.2 guard
# has kept them — it never feeds back into cut decisions, never changes
# transition_in, and never changes a shot boundary. A subtype is a label
# on an existing cut, nothing more.
#
# Method (tuned on M3 / multi.mp4 / a true fade-through-black fixture /
# a wipeleft fixture — see CHANGELOG v2.0.4 for the measured numbers):
# re-decode a [t-1s, t+1s] window at 12fps / 160px with a seek (a 6fps
# decode gives only ~3 samples across a 0.5s dissolve — too coarse).
# A = settled frame before, B = settled frame after. Tests run in order:
#   1. FADE: mean luma dips below 0.35x min(L(A), L(B)) — or absolutely
#      below 20 (to black) / above 235 (to white) — with recovery on
#      both sides (the dip is interior, not a step to a darker scene).
#      A dissolve/slide can never trip this: a blend's luma stays
#      between L(A) and L(B), and a slide's does too.
#   2. DISSOLVE: for each mid frame M, fit scalar alpha minimising
#      ||M - (alpha*A + (1-alpha)*B)||. Label dissolve if the fit
#      residual is < 0.5x ||M - A|| with alpha in (0.2, 0.8) for >= 3
#      consecutive samples (a plateau, not a single spike). A hard cut
#      fails this (alpha ~0 or ~1); a slide fails on residual (a
#      spatial mix is not a global blend).
#   3. SLIDE / PUSH / WIPE (labelled jointly "slide_wipe"): the
#      outgoing frame TRANSLATES during a slide, so a strip test
#      against the static A frame is meaningless (measured on M3: the
#      whole frame reads "changed" within 0.1s of the slide start).
#      Instead, phase-correlate consecutive window frames: a slide is
#      a fast coherent pure-axial translation — >= 3 consecutive pairs
#      with the same-sign shift >= 4px/frame at 160px on one axis,
#      <= 1px on the other, corrected coherence >= 0.5 — AND the run
#      must displace >= half the frame width in total (M3 slide: 160px
#      in 7 pairs; a slow U2 overlay drift manages only ~50px in 7
#      pairs and must not read as a slide).
#   4. REVEAL-WIPE (same "slide_wipe" label): xfade wipeleft is NOT a
#      translating slide — the old frame stays put and the new one is
#      revealed by a moving edge. A mid frame is "wipe-clean" only when
#      its strips split into a matched group (mean diff vs B < 8 — the
#      revealed region IS B) and an unmatched group (diff > 40),
#      allowing one intermediate boundary strip, with the matched group
#      a prefix or suffix; the wipe-clean matched-counts must grow
#      monotonically through >= 4 distinct values, starting <= 2 and
#      ending >= 5. The sharpness is what separates a reveal from a
#      slow drift/pan into the new layout: a drift's strip diffs form
#      a smooth gradient (no strip is ever < 8 while its neighbour is
#      > 40). Measured: wipeleft gives wipe-clean counts 1,3,4,5,6
#      (revealed strips exactly 0.0, unrevealed 90-148); a U2 overlay
#      drift gives only 3 distinct wipe-clean counts (2,3,4).
# Gate: the settled frames at t+-0.75s must differ across >= 45% of
# pixels (the guard's own line, recomputed from the window frames) —
# on a flat/textureless pair the blend fit passes vacuously and the
# strip test is meaningless, so anything below the gate stays
# "soft_unknown". Any disagreement between tests also defaults to
# "soft_unknown". Never force a subtype.
# ---------------------------------------------------------------------------

TRANSITION_SAMPLE_FPS = 12.0
TRANSITION_WINDOW_SEC = 1.0
TRANSITION_MIN_CHANGED_FRACTION = GUARD_MIN_CHANGED_FRACTION  # 0.45
TRANSITION_FADE_RELATIVE = 0.35
TRANSITION_FADE_BLACK_LUMA = 20.0
TRANSITION_FADE_WHITE_LUMA = 235.0
TRANSITION_DISSOLVE_MIN_CONSECUTIVE = 3
TRANSITION_DISSOLVE_MAX_RESIDUAL_RATIO = 0.5
TRANSITION_SLIDE_MIN_CONSECUTIVE = 3
TRANSITION_SLIDE_MIN_SHIFT_PX = 4
TRANSITION_SLIDE_MAX_ORTHO_PX = 1
TRANSITION_SLIDE_MIN_COHERENCE = 0.5
TRANSITION_SLIDE_MIN_DISPLACEMENT_FRAC = 0.5  # run must move >= half the frame width
TRANSITION_WIPE_STRIP_MATCH_DIFF = 8.0  # a revealed strip IS the settled frame
TRANSITION_WIPE_STRIP_NOMATCH_DIFF = 40.0  # an unrevealed strip is clearly different
TRANSITION_WIPE_MIN_INTERMEDIATE_VALUES = 4


def _sample_window_frames(
    video_path: Path,
    start: float,
    end: float,
    sample_fps: float,
    sample_w: int,
    sample_h: int,
) -> dict[int, np.ndarray]:
    """Seek-decode the frames of one [start, end] window at sample_fps.

    Unlike _sample_frames_at (which streams from 0 — the guard needs
    that), this seeks straight to the window: a classifier that runs
    once per soft cut must not pay a full-file decode per cut. Returns
    {frame_index: flat uint8 RGB array} with index 0 == start (the fps
    filter re-timestamps from the seek point). Returns {} on failure
    (callers then keep the cut as soft_unknown — fail open).
    """
    if end <= start:
        return {}
    n_want = max(1, int(round((end - start) * sample_fps)) + 1)
    frame_bytes = sample_w * sample_h * 3
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}",
        "-i", str(video_path),
        "-t", f"{end - start:.3f}",
        "-vf", f"fps={sample_fps},scale={sample_w}:{sample_h}",
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
        while idx < n_want:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            found[idx] = np.frombuffer(buf, dtype=np.uint8).copy()
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    return found


def _classify_soft_transition(
    video_path: Path,
    t: float,
    duration: float,
    sample_w: int,
    sample_h: int,
) -> str:
    """Subtype one surviving soft transition at time t (see header above).

    Returns "fade" | "dissolve" | "slide_wipe" | "soft_unknown".
    """
    win_start = max(0.0, t - TRANSITION_WINDOW_SEC)
    win_end = min(duration, t + TRANSITION_WINDOW_SEC)
    raw = _sample_window_frames(
        video_path, win_start, win_end,
        TRANSITION_SAMPLE_FPS, sample_w, sample_h,
    )
    if len(raw) < 5:
        return "soft_unknown"
    idxs = sorted(raw)
    frames = {i: raw[i].reshape(sample_h, sample_w, 3) for i in idxs}
    times = {i: win_start + i / TRANSITION_SAMPLE_FPS for i in idxs}
    A = frames[idxs[0]].astype(np.float32)
    B = frames[idxs[-1]].astype(np.float32)

    # Gate: settled frames either side must differ across >= 45% of
    # pixels, else the subtype tests are meaningless (flat pairs).
    ib = min(idxs, key=lambda i: abs(times[i] - (t - GUARD_OFFSET_SEC)))
    ia = min(idxs, key=lambda i: abs(times[i] - (t + GUARD_OFFSET_SEC)))
    if _changed_fraction(raw[ib], raw[ia]) < TRANSITION_MIN_CHANGED_FRACTION:
        return "soft_unknown"

    lumas = {
        i: (
            0.299 * frames[i][..., 0].astype(np.float32)
            + 0.587 * frames[i][..., 1].astype(np.float32)
            + 0.114 * frames[i][..., 2].astype(np.float32)
        )
        for i in idxs
    }
    L = {i: float(lumas[i].mean()) for i in idxs}
    l_a, l_b = L[idxs[0]], L[idxs[-1]]
    l_min, l_max = min(L.values()), max(L.values())

    # (1) FADE — interior luma dip (through black or white).
    if (
        l_min < TRANSITION_FADE_BLACK_LUMA
        and l_a > l_min + 15.0
        and l_b > l_min + 15.0
    ):
        return "fade"
    if (
        l_max > TRANSITION_FADE_WHITE_LUMA
        and l_a < l_max - 15.0
        and l_b < l_max - 15.0
    ):
        return "fade"
    if (
        l_min < TRANSITION_FADE_RELATIVE * min(l_a, l_b)
        and l_a > l_min + 10.0
        and l_b > l_min + 10.0
    ):
        return "fade"

    mids = idxs[1:-1]
    d_ab = float(np.mean(np.abs(A - B)))

    # (2) DISSOLVE — global blend fit holds for a run of mid frames.
    if d_ab >= 10.0:
        run = 0
        best_run = 0
        for i in mids:
            M = frames[i].astype(np.float32)
            d_ma = float(np.mean(np.abs(M - A)))
            good = False
            if d_ma >= 12.0:
                num = float(np.sum((M - B) * (A - B)))
                den = float(np.sum((A - B) ** 2))
                alpha = num / den if den > 0 else 0.0
                resid = float(np.mean(np.abs(M - (alpha * A + (1.0 - alpha) * B))))
                good = (
                    0.2 < alpha < 0.8
                    and resid < TRANSITION_DISSOLVE_MAX_RESIDUAL_RATIO * d_ma
                )
            run = run + 1 if good else 0
            best_run = max(best_run, run)
        if best_run >= TRANSITION_DISSOLVE_MIN_CONSECUTIVE:
            return "dissolve"

    # (3) SLIDE / PUSH — fast coherent pure-axial translation between
    # consecutive window frames (the outgoing frame translates, so
    # strip-vs-A tests are meaningless here — see header). A slide by
    # definition moves content across a large part of the frame, so
    # the run must also displace >= half the frame width in total —
    # this is what separates a slide from a slow drift/pan that merely
    # spans the cut (measured: M3 slide displaces 160px = the full
    # 160px width in 7 pairs; a U2 overlay drift displaces only ~50px
    # in 7 pairs and must NOT read as a slide).
    best_disp = 0.0
    run = 0
    run_disp = 0.0
    run_sign: tuple[str, int] | None = None
    for k in range(len(idxs) - 1):
        dx, dy, coh, raw_diff = _phase_shift(lumas[idxs[k]], lumas[idxs[k + 1]])
        usable: tuple[str, int] | None = None
        shift = 0
        if raw_diff >= 0.8 and coh >= TRANSITION_SLIDE_MIN_COHERENCE:
            if abs(dx) >= TRANSITION_SLIDE_MIN_SHIFT_PX and abs(dy) <= TRANSITION_SLIDE_MAX_ORTHO_PX:
                usable = ("x", 1 if dx > 0 else -1)
                shift = abs(dx)
            elif abs(dy) >= TRANSITION_SLIDE_MIN_SHIFT_PX and abs(dx) <= TRANSITION_SLIDE_MAX_ORTHO_PX:
                usable = ("y", 1 if dy > 0 else -1)
                shift = abs(dy)
        if usable is not None and usable == run_sign:
            run += 1
            run_disp += shift
        elif usable is not None:
            run, run_disp, run_sign = 1, float(shift), usable
        else:
            run, run_disp, run_sign = 0, 0.0, None
        if run >= TRANSITION_SLIDE_MIN_CONSECUTIVE:
            best_disp = max(best_disp, run_disp)
    if best_disp >= TRANSITION_SLIDE_MIN_DISPLACEMENT_FRAC * sample_w:
        return "slide_wipe"

    # (4) REVEAL-WIPE — strips progressively match the settled B frame
    # (the old frame stays put; the new one is revealed by a moving
    # edge). A frame is "wipe-clean" only when its strips split into
    # a matched group (diff < 8 — the revealed region IS B, near-exact)
    # and an unmatched group (diff > 40), allowing ONE intermediate
    # strip adjacent to the matched group (the boundary strip itself),
    # with the matched group a prefix or suffix (contiguous from one
    # edge). Collect wipe-clean matched-counts in time order and
    # require >= 4 distinct values, non-decreasing, starting <= 2 and
    # ending >= 5. The sharpness is what separates a reveal from a
    # slow drift/pan into the new layout: a drift's strip diffs form a
    # smooth gradient (no strip is ever < 8 while its neighbour is
    # > 40), so it never produces a wipe-clean frame.
    for axis in (0, 1):
        clean_counts: list[int] = []
        for i in mids:
            d = np.mean(np.abs(frames[i].astype(np.float32) - B), axis=2)
            n = 8
            if axis == 0:
                fr = [float(d[:, j * sample_w // n:(j + 1) * sample_w // n].mean()) for j in range(n)]
            else:
                fr = [float(d[j * sample_h // n:(j + 1) * sample_h // n, :].mean()) for j in range(n)]
            matched = [j for j, x in enumerate(fr) if x < TRANSITION_WIPE_STRIP_MATCH_DIFF]
            if not (0 < len(matched) < n):
                continue
            if matched != list(range(len(matched))) and matched != list(range(n - len(matched), n)):
                continue
            # One intermediate strip allowed, adjacent to the matched group.
            edge = matched[-1] + 1 if matched == list(range(len(matched))) else matched[0] - 1
            ok = True
            for j, x in enumerate(fr):
                if j in matched:
                    continue
                if j == edge:
                    continue
                if x <= TRANSITION_WIPE_STRIP_NOMATCH_DIFF:
                    ok = False
                    break
            if not ok:
                continue
            clean_counts.append(len(matched))
        if len(clean_counts) < 2:
            continue
        if clean_counts[0] > 2 or clean_counts[-1] < 5:
            continue
        if not all(clean_counts[k + 1] >= clean_counts[k] for k in range(len(clean_counts) - 1)):
            continue
        if len({c for c in clean_counts if 0 < c < 8}) >= TRANSITION_WIPE_MIN_INTERMEDIATE_VALUES:
            return "slide_wipe"

    return "soft_unknown"


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


def _build_rhythm(
    shots: list[dict[str, Any]],
    duration: float,
    total_cuts: int,
) -> dict[str, Any]:
    """Rhythm/pacing detail for the v2.0.4 rhythm section (pure function of the
    full shot list — no sampling caps, no retention weights touched).

    - Shot-length stats (min/median/max) plus WHICH shots are longest/shortest
      (shot id + window), so the pacing is traceable to concrete shots.
    - cuts_per_10s: overall cut density on a 10-second grid.
    - pacing_curve: one entry per 10s window — shots starting in the window
      and cuts landing in it (shot 1 at 0.0s is a start, not a cut).
    - transitions: every cut as (time, into-shot, type). Transition duration
      is NOT measured (only the cut instant is known), so no duration field.
    """
    if not shots:
        return {
            "shot_length_min_sec": 0.0,
            "shot_length_median_sec": 0.0,
            "shot_length_max_sec": 0.0,
            "longest_shot": None,
            "shortest_shot": None,
            "cuts_per_10s": 0.0,
            "pacing_curve": [],
            "transitions": [],
        }
    durations = [float(s["duration"]) for s in shots]
    longest = max(shots, key=lambda s: float(s["duration"]))
    shortest = min(shots, key=lambda s: float(s["duration"]))

    def _window(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "shot_id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
        }

    curve: list[dict[str, Any]] = []
    if duration > 0:
        n_windows = max(1, math.ceil(duration / 10.0))
        for w in range(n_windows):
            ws = round(w * 10.0, 2)
            we = round(min((w + 1) * 10.0, duration), 2)
            started = [s for s in shots if ws <= float(s["start"]) < we]
            cuts = [s for s in started if float(s["start"]) > 0.0]
            curve.append({
                "window_start": ws,
                "window_end": we,
                "shots": len(started),
                "cuts": len(cuts),
            })

    # "subtype" is added only for soft transitions that got one — a
    # hard_cut entry stays exactly {"time", "into_shot", "type"} (the
    # v2.0.4 rhythm tests assert that shape).
    transitions: list[dict[str, Any]] = []
    for s in shots[1:]:
        entry: dict[str, Any] = {
            "time": s["start"],
            "into_shot": s["id"],
            "type": s.get("transition_in", "-"),
        }
        if s.get("transition_subtype"):
            entry["subtype"] = s["transition_subtype"]
        transitions.append(entry)
    return {
        "shot_length_min_sec": round(min(durations), 2),
        "shot_length_median_sec": round(float(np.median(durations)), 2),
        "shot_length_max_sec": round(max(durations), 2),
        "longest_shot": _window(longest),
        "shortest_shot": _window(shortest),
        "cuts_per_10s": round(total_cuts / (duration / 10.0), 2) if duration > 0 else 0.0,
        "pacing_curve": curve,
        "transitions": transitions,
    }


def export_shot_frames(
    video_path: Path,
    shots: list[dict[str, Any]],
    out_dir: Path,
    fps: float = 30.0,
    width: int = 480,
    orig_width: int | None = None,
    orig_height: int | None = None,
) -> dict[int, dict[str, str]]:
    """Export first/mid/last frame JPEGs for EVERY shot into dna/frames/.

    ONE ffmpeg streaming decode at export resolution (constant memory — each
    wanted frame is written to disk the moment it arrives; nothing is held
    except the <=6 small strip frames per shot, composed at the end). No
    per-frame ffmpeg spawn: spawn/seek would dominate on long videos.

    Frame times use a small inset so the "first" frame is inside the shot,
    not the last frame of the previous one. Timestamps map to frame indices
    via the probe fps (nearest frame; VFR sources are approximate). A strip
    (<=6 evenly spaced frames tiled into one contact image) is built from the
    same pass when the shot is >= 1s; shorter shots get no strip.

    Returns {shot_id: {"first": rel, "mid": rel, "last": rel, "strip": rel}}
    with paths relative to out_dir (e.g. "dna/frames/shot-01-first.jpg").
    Returns {} on any ffmpeg failure (fail open — the pipeline continues
    without frame paths, exactly like the v2.0.2 guard's fail-open rule).
    """
    if not shots:
        return {}
    fps = float(fps) if fps and float(fps) > 0 else 30.0
    # Cap the decode at 30 fps: stills only need ~±16 ms accuracy, and a
    # 60 fps source decodes ~30% faster with the fps filter dropping frames
    # before the scaler (measured on speech2: 14.1s -> 10.1s for the pass).
    # Index mapping below uses the effective fps.
    eff_fps = min(fps, 30.0)
    if orig_width and orig_height:
        exp_h = max(2, int(round(orig_height * width / orig_width / 2.0) * 2))
    else:
        exp_h = 270
    frames_dir = out_dir / "dna" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # wanted[idx] = list of (shot_id, label) — labels dedupe by frame index
    # so a 0.1s shot whose first/mid/last land on one frame writes one file.
    wanted: dict[int, list[tuple[int, str]]] = {}

    def _want(shot_id: int, label: str, t: float) -> None:
        wanted.setdefault(int(round(max(0.0, t) * eff_fps)), []).append((shot_id, label))

    for s in shots:
        sid = int(s["id"])
        dur = float(s["duration"])
        start = float(s["start"])
        end = float(s["end"])
        inset = min(0.05, dur * 0.1)
        _want(sid, "first", start + inset)
        _want(sid, "mid", start + dur / 2.0)
        _want(sid, "last", end - inset)
        n_strip = 6 if dur >= 2.0 else (4 if dur >= 1.0 else 0)
        if n_strip >= 2 and dur - 2 * inset > 0:
            span = dur - 2 * inset
            for i in range(n_strip):
                _want(sid, f"strip_{i}", start + inset + span * i / (n_strip - 1))

    if not wanted:
        return {}
    last_idx = max(wanted)
    frame_bytes = width * exp_h * 3
    vf = f"scale={width}:{exp_h}"
    if eff_fps < fps:
        vf = f"fps={eff_fps},scale={width}:{exp_h}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return {}
    assert proc.stdout is not None

    result: dict[int, dict[str, str]] = {}
    strip_parts: dict[int, list[Path]] = {}
    try:
        idx = 0
        while idx <= last_idx:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            labels = wanted.get(idx)
            if labels is not None:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(exp_h, width, 3)
                img = Image.fromarray(arr)
                for sid, label in labels:
                    if label.startswith("strip_"):
                        tmp = frames_dir / f".strip-{sid:02d}-{label[6:]}.jpg"
                        img.save(tmp, format="JPEG", quality=88)
                        strip_parts.setdefault(sid, []).append(tmp)
                    else:
                        name = f"shot-{sid:02d}-{label}.jpg"
                        img.save(frames_dir / name, format="JPEG", quality=88)
                        result.setdefault(sid, {})[label] = f"dna/frames/{name}"
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()

    # Compose each strip from its saved frames, then delete the parts.
    for sid, parts in strip_parts.items():
        parts = sorted(parts)
        try:
            tiles = [Image.open(p).convert("RGB") for p in parts]
            if len(tiles) < 2:
                continue
            tw, th = tiles[0].size
            strip = Image.new("RGB", (tw * len(tiles), th), (0, 0, 0))
            for i, tile in enumerate(tiles):
                strip.paste(tile, (i * tw, 0))
            name = f"shot-{sid:02d}-strip.jpg"
            strip.save(frames_dir / name, format="JPEG", quality=88)
            result.setdefault(sid, {})["strip"] = f"dna/frames/{name}"
        finally:
            for p in parts:
                try:
                    p.unlink()
                except OSError:
                    pass

    # Fallback: the container duration can run slightly past the last
    # decodable frame (speech2: container 21.32s, last video frame at
    # 21.27s), so a shot's "last" timestamp may map past the end of the
    # stream. Copy the same shot's mid (else first) frame into any missing
    # first/mid/last slot — same shot, a fraction of a second away — so
    # the per-shot contract (all three frames exist) always holds.
    for s in shots:
        sid = int(s["id"])
        have = result.get(sid, {})
        donor = have.get("mid") or have.get("first") or have.get("last")
        if not donor:
            continue
        for label in ("first", "mid", "last"):
            if label not in have:
                name = f"shot-{sid:02d}-{label}.jpg"
                # donor is stored relative to out_dir ("dna/frames/..."),
                # so take its basename inside frames_dir.
                shutil.copyfile(frames_dir / Path(donor).name, frames_dir / name)
                result.setdefault(sid, {})[label] = f"dna/frames/{name}"
    return result


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
            # v2.0.4 item 3: subtype is set below, for soft transitions
            # only. None elsewhere (opening / hard_cut have no transition
            # to subtype). Labelling only — never a cut decision.
            "transition_subtype": None,
        })

    # v2.0.4 item 3: subtype each surviving soft transition (runs AFTER
    # the guard, on final shot boundaries only). Same 160px sampling
    # geometry as the frame-diff pass.
    if width and height:
        sub_h = max(2, int(round(height * 160 / width / 2.0) * 2))
    else:
        sub_h = 90
    for s in shots:
        if s["transition_in"] == "soft_transition":
            s["transition_subtype"] = _classify_soft_transition(
                video_path, s["start"], duration, 160, sub_h,
            )

    if not shots:
        shots = [{
            "id": 1,
            "start": 0.0,
            "end": round(duration, 3),
            "duration": round(duration, 3),
            "cut_score_in": 0.0,
            "cut_source_in": "start",
            "transition_in": "opening",
            "transition_subtype": None,
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
        "rhythm": _build_rhythm(shots, duration, total_cuts),
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
