"""v2.0.2 shot-detection regression tests: motion-graphics over-segmentation.

The frame-diff pass over-counted shots when big animated text/elements moved
inside one continuous shot (U1 counted 4 shots for 1; U3 counted 3 for 2).
The layout/background-continuity guard in videodna.shots suppresses a soft
candidate when the settled frames 0.75s either side of it show the same
layout (changed fraction < 0.45), or when the candidate sits within 0.75s of
the start/end (no settled before/after state — intro/outro animation).
scdet hard cuts are never guarded.

Each test needs its real video file: paths come from $VIDEODNA_V202_<NAME>
or fall back to the known workspace paths, and SKIP gracefully when absent
(same pattern as the v1/v2 shot tests in test_videodna.py).

Run from the repo root:
    python -m pytest tests/test_v202_shots.py -q
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from videodna.shots import (
    GUARD_MIN_CHANGED_FRACTION,
    GUARD_OFFSET_SEC,
    _changed_fraction,
    detect_shots,
)

U1_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_U1",
    "/home/hatch/workspace/user/media_library/video/fd/"
    "fda867a38232d196f7799a04609ccfca811e57a38596ae0218b5949d992fd031.mp4",
))
U2_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_U2",
    "/home/hatch/workspace/user/media_library/video/5e/"
    "5ea5bf13f40626e7b1253d0d5bb517aaa79a068602dc067f20db91f80dd7b496.mp4",
))
U3_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_U3",
    "/home/hatch/workspace/user/media_library/video/47/"
    "47710088e6600a495dd384ebe2d7d71b44686980ebf7aaac98f018eb3ecca3c9.mp4",
))
U4_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_U4",
    "/home/hatch/workspace/user/media_library/video/7c/"
    "7c4d22b0e260da7b9664be6476af726f3225cc1fb8a6e4b981f0fc56e39d5152.mp4",
))
M3_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_M3", "/tmp/vidtest/textured.mp4",
))
MULTI_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_MULTI", "/home/hatch/workspace/videodna-test/multi.mp4",
))
TEST_VIDEO = Path(os.environ.get(
    "VIDEODNA_V202_TEST", "/home/hatch/workspace/videodna-test/test.mp4",
))


def _cuts(result):
    return [s["start"] for s in result["shots"][1:]]


class TestChangedFraction(unittest.TestCase):
    """Guard metric itself, on synthetic frames (no video needed)."""

    def test_identical_frames_change_nothing(self):
        frame = np.full((90, 160, 3), 128, dtype=np.uint8)
        self.assertEqual(_changed_fraction(frame, frame), 0.0)

    def test_fully_different_frames_change_everything(self):
        black = np.zeros((90, 160, 3), dtype=np.uint8)
        white = np.full((90, 160, 3), 255, dtype=np.uint8)
        self.assertEqual(_changed_fraction(black, white), 1.0)

    def test_half_frame_overlay_is_below_the_guard_threshold(self):
        # A text/element overlay covering ~half the frame must NOT read as a
        # layout change: changed fraction 0.5 is still above the 0.45 guard
        # line only when the overlay itself is large; a quarter-frame overlay
        # (the U1/U2 text scale) must sit well below it.
        base = np.full((90, 160, 3), 128, dtype=np.uint8)
        overlay = base.copy()
        overlay[:45, :80] = 255  # quarter of the frame replaced
        self.assertAlmostEqual(_changed_fraction(base, overlay), 0.25, places=2)
        self.assertLess(_changed_fraction(base, overlay), GUARD_MIN_CHANGED_FRACTION)


@unittest.skipUnless(U1_VIDEO.exists(), "U1 motion-graphics promo not present")
class TestU1MotionGraphicsOverSegmentation(unittest.TestCase):
    def test_one_continuous_shot_not_four(self):
        r = detect_shots(U1_VIDEO, 6.6, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 1)
        self.assertEqual(r["hard_cuts"], 0)
        self.assertEqual(r["soft_transitions"], 0)
        self.assertEqual(r["detection"]["frame_diff_suppressed_by_guard"], 3)


@unittest.skipUnless(U2_VIDEO.exists(), "U2 statue video not present")
class TestU2TextOverlaysStillCount(unittest.TestCase):
    def test_text_overlay_transitions_are_kept(self):
        # WARNING case: U2's soft transitions are text overlays appearing over
        # the SAME statue background. The guard must not suppress them — the
        # overlays change 0.55-0.87 of the frame, above the 0.45 guard line.
        r = detect_shots(U2_VIDEO, 10.01, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 4)
        self.assertEqual(r["hard_cuts"], 1)
        self.assertEqual(r["soft_transitions"], 2)
        self.assertEqual(r["detection"]["frame_diff_suppressed_by_guard"], 0)
        for got, want in zip(_cuts(r), [2.5, 4.5, 6.1]):
            self.assertAlmostEqual(got, want, delta=0.35)


@unittest.skipUnless(U3_VIDEO.exists(), "U3 Zuckerberg video not present")
class TestU3IntroAnimationNotAShot(unittest.TestCase):
    def test_intro_animation_suppressed_real_cut_kept(self):
        # The 0.167s frame-diff candidate is intro graphics animating in at
        # the very start (no settled before-state); the scdet hard cut at
        # 2.067s is the only real boundary.
        r = detect_shots(U3_VIDEO, 4.0, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 2)
        self.assertEqual(r["hard_cuts"], 1)
        self.assertEqual(r["soft_transitions"], 0)
        self.assertEqual(r["detection"]["frame_diff_suppressed_by_guard"], 1)
        self.assertAlmostEqual(_cuts(r)[0], 2.067, delta=0.35)


@unittest.skipUnless(U4_VIDEO.exists(), "U4 video not present")
class TestU4SingleShot(unittest.TestCase):
    def test_stays_one_shot(self):
        r = detect_shots(U4_VIDEO, 6.04, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 1)
        self.assertEqual(r["soft_transitions"], 0)


@unittest.skipUnless(M3_VIDEO.exists(), "M3 textured fade+slide video not present")
class TestM3RealSoftTransitionsKept(unittest.TestCase):
    def test_fade_and_slide_both_still_detected(self):
        r = detect_shots(M3_VIDEO, 5.0, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 3)
        self.assertEqual(r["hard_cuts"], 0)
        self.assertEqual(r["soft_transitions"], 2)
        self.assertEqual(r["detection"]["frame_diff_suppressed_by_guard"], 0)
        for got, want in zip(_cuts(r), [1.667, 3.333]):
            self.assertAlmostEqual(got, want, delta=0.35)


@unittest.skipUnless(MULTI_VIDEO.exists(), "multi.mp4 dissolve video not present")
class TestMultiDissolveKept(unittest.TestCase):
    def test_dissolve_still_detected(self):
        r = detect_shots(MULTI_VIDEO, 5.5, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 2)
        self.assertEqual(r["soft_transitions"], 1)
        self.assertEqual(r["detection"]["frame_diff_suppressed_by_guard"], 0)
        self.assertAlmostEqual(_cuts(r)[0], 2.667, delta=0.35)


@unittest.skipUnless(TEST_VIDEO.exists(), "test.mp4 not present")
class TestSingleShotControl(unittest.TestCase):
    def test_stays_one_shot(self):
        r = detect_shots(TEST_VIDEO, 12.0, mode="short", width=1280, height=720)
        self.assertEqual(r["shot_count"], 1)
        self.assertEqual(r["soft_transitions"], 0)


if __name__ == "__main__":
    unittest.main()
