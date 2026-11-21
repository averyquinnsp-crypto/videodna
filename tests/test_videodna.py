"""VideoDNA test suite (stdlib unittest — no extra test dependencies).

Run from the repo root:
    python -m unittest discover -s tests -v

The two shot-detection regression tests need real video files: they look for
them via $VIDEODNA_TEMPLATE_VIDEO / $VIDEODNA_STYLEB_VIDEO, then fall back to
the known workspace paths, and SKIP gracefully when the files are absent.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from videodna.cli import _safe_slug, _tile_dims
from videodna.diagnostics import compute_diagnostics
from videodna.ingest import find_yt_dlp, mine_comments, parse_subtitle_file
from videodna.probe import match_signatures
from videodna.visual import _write_cube_lut

import numpy as np

TEMPLATE_VIDEO = Path(
    os.environ.get(
        "VIDEODNA_TEMPLATE_VIDEO",
        "/home/hatch/workspace/user/files/1000151875_0_zejq.mp4",
    )
)
STYLEB_VIDEO = Path(
    os.environ.get(
        "VIDEODNA_STYLEB_VIDEO",
        "/home/hatch/workspace/hyperframes-test/muse_style_b.mp4",
    )
)


def _diag_inputs(**overrides):
    """Minimal valid inputs for compute_diagnostics."""
    probe = {"duration_sec": 12.0}
    shots = {
        "mode": "short",
        "hook_cuts_0_3s": 1,
        "hook_cuts_0_10s": 2,
        "dead_zones": [],
        "dead_zone_threshold_sec": 4.5,
        "avg_shot_length_sec": 3.0,
        "cuts_per_minute": 20.0,
        "total_cuts": 4,
    }
    visual = {
        "motion_summary": {"static_shot_ratio": 0.2},
        "shot_visuals": [],
    }
    audio = {
        "has_audio": True,
        "integrated_lufs": -14.0,
        "dead_air_total_sec": 0.0,
        "cut_beat_sync_ratio": 0.5,
    }
    for key, val in overrides.items():
        {"probe": probe, "shots": shots, "visual": visual, "audio": audio}[key].update(val)
    return probe, shots, visual, audio


class TestHelpers(unittest.TestCase):
    def test_safe_slug(self):
        self.assertEqual(_safe_slug("My Video! (2026)"), "My_Video_2026")
        self.assertEqual(_safe_slug("!!!"), "video")

    def test_tile_dims(self):
        self.assertEqual(_tile_dims(1), (1, 1))
        self.assertEqual(_tile_dims(12), (4, 3))
        self.assertEqual(_tile_dims(13), (4, 4))
        self.assertEqual(_tile_dims(100), (8, 8))  # capped at 64 frames

    def test_find_yt_dlp(self):
        # yt-dlp is installed in the dev venv; discovery must find it even
        # when the venv bin dir is not on PATH (the v1.0 bug).
        found = find_yt_dlp()
        self.assertIsNotNone(found)


class TestProbeSignatures(unittest.TestCase):
    def test_taiwan_is_not_wan(self):
        blob = "format.encoder=Lavf61.1.100 comment=filmed in taiwan"
        self.assertNotIn("ai_generator", match_signatures(blob))

    def test_x264_is_not_a_platform_signal(self):
        blob = "video.encoder=x264 major_brand=mp42"
        self.assertEqual(match_signatures(blob), {})

    def test_real_signals_still_match(self):
        self.assertIn("remotion_or_code", match_signatures("encoder=HyperFrames 0.9.74"))
        self.assertIn("ai_generator", match_signatures("meta=aigc veo3 render"))
        self.assertIn("capcut", match_signatures("comment=edited with CapCut"))


class TestCommentMining(unittest.TestCase):
    def _mine(self, texts):
        return mine_comments([{"text": t, "like_count": 1} for t in texts])

    def test_praise_is_not_a_question(self):
        res = self._mine(["what a video!"])
        self.assertEqual(res["questions"], [])

    def test_real_questions_detected(self):
        res = self._mine(["what software did you use?", "কিভাবে বানালে?"])
        self.assertEqual(len(res["questions"]), 2)

    def test_bare_more_is_not_a_request(self):
        res = self._mine(["more", "please", "next video when"])
        self.assertEqual(res["requests"], [])

    def test_real_requests_detected(self):
        res = self._mine(["part 2 please!", "make a tutorial on this"])
        self.assertEqual(len(res["requests"]), 2)

    def test_complaints_and_timestamps(self):
        res = self._mine(["music too loud", "the 2:34 part was confusing"])
        self.assertEqual(len(res["complaints"]), 1)
        self.assertEqual(len(res["timestamp_highlights"]), 1)


class TestSubtitleParsing(unittest.TestCase):
    def test_srt_sample(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\n<b>Second</b> line\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
            f.write(srt)
            path = Path(f.name)
        try:
            lines = parse_subtitle_file(path)
        finally:
            path.unlink()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["start"], 1.0)
        self.assertEqual(lines[0]["text"], "Hello world")
        self.assertEqual(lines[1]["text"], "Second line")


class TestDiagnostics(unittest.TestCase):
    def test_silent_video_excluded_not_punished(self):
        probe, shots, visual, audio = _diag_inputs(
            audio={"has_audio": False, "integrated_lufs": None, "dead_air_total_sec": 0.0}
        )
        d = compute_diagnostics(probe, shots, visual, audio)
        self.assertTrue(d["audio_excluded_from_score"])
        self.assertIsNone(d["scores"]["audio_dynamics"])
        self.assertEqual(d["overall_retention_score"], 8.2)  # (7.5 + 9.0) / 2
        self.assertFalse(any(w["area"] == "Audio" for w in d["weaknesses"]))

    def test_near_silent_track_counts_as_silent(self):
        probe, shots, visual, audio = _diag_inputs(
            audio={"has_audio": True, "integrated_lufs": -70.0, "dead_air_total_sec": 12.0}
        )
        d = compute_diagnostics(probe, shots, visual, audio)
        self.assertTrue(d["audio_excluded_from_score"])
        self.assertFalse(any(w["area"] == "Dead Air / Pauses" for w in d["weaknesses"]))

    def test_high_motion_long_shot_is_not_a_dead_zone(self):
        probe, shots, visual, audio = _diag_inputs(
            shots={"dead_zones": [{
                "shot_id": 1, "start": 0.0, "end": 12.0, "duration": 12.0,
                "severity": "high", "reason": "x",
            }]},
            visual={"shot_visuals": [{"shot_id": 1, "motion_class": "high_energy_whip"}]},
        )
        d = compute_diagnostics(probe, shots, visual, audio)
        self.assertEqual(d["effective_dead_zones"], [])
        self.assertFalse(any(w["area"] == "Pacing / Dead-Zones" for w in d["weaknesses"]))

    def test_static_long_shot_stays_a_dead_zone(self):
        probe, shots, visual, audio = _diag_inputs(
            shots={"dead_zones": [{
                "shot_id": 1, "start": 0.0, "end": 12.0, "duration": 12.0,
                "severity": "high", "reason": "x",
            }]},
            visual={"shot_visuals": [{"shot_id": 1, "motion_class": "static"}]},
        )
        d = compute_diagnostics(probe, shots, visual, audio)
        self.assertEqual(len(d["effective_dead_zones"]), 1)


class TestVisual(unittest.TestCase):
    def test_lut_has_17_cubed_entries(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.cube"
            _write_cube_lut(
                out,
                np.array([120.0, 100.0, 90.0]),
                np.array([60.0, 55.0, 50.0]),
                0.3,
            )
            lines = out.read_text(encoding="utf-8").splitlines()
            data = [ln for ln in lines if ln and ln[0].isdigit()]
            self.assertEqual(len(data), 17 ** 3)
            self.assertIn("LUT_3D_SIZE 17", lines)


@unittest.skipUnless(TEMPLATE_VIDEO.exists(), "template test video not present")
class TestShotDetectionTemplate(unittest.TestCase):
    def test_finds_all_four_shots(self):
        from videodna.shots import detect_shots

        r = detect_shots(TEMPLATE_VIDEO, 10.01, mode="short", width=1280, height=720)
        # Ground truth (rea-video, visually verified): cuts at 2.467, 4.467, 6.1
        self.assertEqual(r["shot_count"], 4)
        cuts = [s["start"] for s in r["shots"][1:]]
        for got, want in zip(cuts, [2.467, 4.467, 6.1]):
            self.assertAlmostEqual(got, want, delta=0.35)


@unittest.skipUnless(STYLEB_VIDEO.exists(), "style-B test video not present")
class TestShotDetectionStyleB(unittest.TestCase):
    def test_finds_all_three_scenes(self):
        from videodna.shots import detect_shots

        r = detect_shots(STYLEB_VIDEO, 12.0, mode="short", width=1280, height=720)
        # Ground truth (HyperFrames project): scenes at 0-6, 6-9.7, 9.7-12.
        # v1.0 scdet-only found 1 shot here.
        self.assertEqual(r["shot_count"], 3)
        cuts = [s["start"] for s in r["shots"][1:]]
        for got, want in zip(cuts, [6.0, 9.667]):
            self.assertAlmostEqual(got, want, delta=0.35)


if __name__ == "__main__":
    unittest.main()
