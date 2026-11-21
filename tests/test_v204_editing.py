"""v2.0.4 editing-analysis upgrade: per-shot remake frames (item 1),
mixed-production verdict (item 2), rhythm/pacing section (item 5).

- Item 1: export_shot_frames() decodes once (streaming, constant memory)
  and writes first/mid/last JPEGs + a contact strip for EVERY shot into
  dna/frames/; the paths land on each shot dict (frame_paths) and on both
  shot-card builders (remake.build_shot_cards, report._build_shot_cards).
- Item 2: the displayed production mix is the duration-weighted MEAN
  SOFTMAX (aggregate_mix), not winner-take-all — "unknown" is not a
  production method and must not swallow half a mixed video, and minority
  mass is preserved. Below 0.5 confidence the verdict still says it is
  uncertain from pixels alone, now with the full mix + that qualifier.
  Per-shot cards carry top label + runner-up + confidence.
- Item 5: _build_rhythm() (pure function of the full shot list) adds
  min/median/max shot length with longest/shortest shot identity,
  cuts/10s, a shots-per-10s pacing curve, and a transitions list.

Video-dependent tests skip gracefully when the fixture is absent (same
pattern as the v2.0.2/v2.0.3 suites).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = REPO_ROOT / ".venv" / "bin" / "videodna"
VIDEODNA_BIN = str(VENV_BIN if VENV_BIN.exists() else shutil.which("videodna"))

TEST_MP4 = Path(os.environ.get(
    "VIDEODNA_V204_TEST", "/home/hatch/workspace/videodna-test/test.mp4",
))
SPEECH2 = Path(os.environ.get(
    "VIDEODNA_V204_SPEECH2",
    "/home/hatch/workspace/user/media_library/video/1b/"
    "1bca5c0726abc4858464ccfa44feb0edba39b2e5c835280f6d4b5448bf22deea.mp4",
))

CATS = ("live", "ai_video", "3d_render", "code_motion", "screen_rec", "static_zoom")


def _probs(top: str, top_p: float = 0.6) -> dict[str, float]:
    """A plausible per-shot softmax with `top` on top."""
    rest = (1.0 - top_p) / (len(CATS) - 1)
    return {k: (round(top_p, 3) if k == top else round(rest, 3)) for k in CATS}


class TestBuildRhythm(unittest.TestCase):
    """Item 5: rhythm stats are pure functions of the full shot list."""

    def _shots(self):
        # 5 shots over 21.3s: durations 1.0, 2.5, 0.5, 4.0, 13.3 (sum 21.3)
        starts = [0.0, 1.0, 3.5, 4.0, 8.0]
        ends = [1.0, 3.5, 4.0, 8.0, 21.3]
        return [
            {"id": i + 1, "start": s, "end": e, "duration": round(e - s, 2),
             "transition_in": "opening" if i == 0 else "hard_cut"}
            for i, (s, e) in enumerate(zip(starts, ends))
        ]

    def test_median_and_longest_shortest_identity(self):
        from videodna.shots import _build_rhythm
        r = _build_rhythm(self._shots(), 21.3, 4)
        self.assertEqual(r["shot_length_min_sec"], 0.5)
        self.assertEqual(r["shot_length_median_sec"], 2.5)
        self.assertEqual(r["shot_length_max_sec"], 13.3)
        self.assertEqual(r["longest_shot"]["shot_id"], 5)
        self.assertEqual(r["longest_shot"]["start"], 8.0)
        self.assertEqual(r["longest_shot"]["end"], 21.3)
        self.assertEqual(r["shortest_shot"]["shot_id"], 3)
        self.assertEqual(r["shortest_shot"]["duration"], 0.5)

    def test_cuts_per_10s_and_pacing_curve(self):
        from videodna.shots import _build_rhythm
        r = _build_rhythm(self._shots(), 21.3, 4)
        self.assertAlmostEqual(r["cuts_per_10s"], round(4 / 2.13, 2))
        curve = r["pacing_curve"]
        self.assertEqual(len(curve), 3)  # 0-10s, 10-20s, 20-21.3s
        self.assertEqual(curve[0]["shots"], 5)  # all 5 shots start before 10s
        self.assertEqual(curve[0]["cuts"], 4)
        self.assertEqual(curve[1]["shots"], 0)
        self.assertEqual(curve[2]["shots"], 0)

    def test_transitions_list(self):
        from videodna.shots import _build_rhythm
        r = _build_rhythm(self._shots(), 21.3, 4)
        self.assertEqual(len(r["transitions"]), 4)
        self.assertEqual(r["transitions"][0], {"time": 1.0, "into_shot": 2, "type": "hard_cut"})
        self.assertEqual(r["transitions"][-1]["into_shot"], 5)
        # Transition duration is not measured — no duration field.
        self.assertNotIn("duration", r["transitions"][0])

    def test_empty_shots(self):
        from videodna.shots import _build_rhythm
        r = _build_rhythm([], 0.0, 0)
        self.assertEqual(r["pacing_curve"], [])
        self.assertEqual(r["transitions"], [])
        self.assertIsNone(r["longest_shot"])


class TestAggregateMix(unittest.TestCase):
    """Item 2: mean-softmax mix preserves minority mass and never shows
    "unknown" as a production method."""

    def test_all_six_labels_present_and_sums_to_100(self):
        from videodna.forensics import aggregate_mix
        mix = aggregate_mix([
            {"duration": 10.0, "probs": _probs("live", 0.6), "confidence": 0.7},
            {"duration": 10.0, "probs": _probs("3d_render", 0.6), "confidence": 0.7},
        ])
        pct = mix["duration_weighted_pct"]
        self.assertEqual(set(pct), set(CATS))
        self.assertAlmostEqual(sum(pct.values()), 100.0, places=0)
        self.assertEqual(list(pct)[0], "live")  # tie broken by insertion order
        self.assertEqual(mix["overall_confidence"], 0.7)

    def test_unknown_shots_do_not_swallow_the_mix(self):
        from videodna.forensics import aggregate_mix
        # speech2-like: half the duration is "unknown" per-shot calls whose
        # softmax still leans live/3D, half is confident live.
        leaning = {"live": 0.30, "3d_render": 0.28, "ai_video": 0.105,
                   "code_motion": 0.105, "screen_rec": 0.105, "static_zoom": 0.105}
        mix = aggregate_mix([
            {"duration": 10.0, "probs": leaning, "confidence": 0.30},
            {"duration": 10.0, "probs": _probs("live", 0.8), "confidence": 0.8},
        ])
        pct = mix["duration_weighted_pct"]
        self.assertNotIn("unknown", pct)
        self.assertGreater(pct["live"], 40.0)
        self.assertGreater(pct["3d_render"], 10.0)

    def test_empty_entries(self):
        from videodna.forensics import aggregate_mix
        mix = aggregate_mix([])
        self.assertEqual(mix["duration_weighted_pct"], {})
        self.assertEqual(mix["overall_confidence"], 0.0)

    def test_runner_up_helper(self):
        from videodna.forensics import _runner_up
        label, prob = _runner_up(_probs("live", 0.6), "live")
        self.assertEqual(prob, 0.08)
        self.assertIn(label, CATS)
        self.assertNotEqual(label, "live")
        self.assertEqual(_runner_up({}, "live"), (None, 0.0))


class TestMixedVerdictSummary(unittest.TestCase):
    """Item 2: the verdict string shows the full mix; below 0.5 it still
    says it is uncertain from pixels alone."""

    def test_high_confidence_shows_full_mix(self):
        from videodna.forensics import honest_verdict_summary
        breakdown = {"live": 55.0, "3d_render": 25.0, "ai_video": 20.0}
        probs = {"live": 0.55, "3d_render": 0.25, "ai_video": 0.20,
                 "code_motion": 0.0, "screen_rec": 0.0, "static_zoom": 0.0}
        verdict, uncertain, runner_up = honest_verdict_summary(breakdown, probs, 0.8)
        self.assertEqual(verdict, "live 55% + 3d_render 25% + ai_video 20%")
        self.assertFalse(uncertain)
        self.assertIsNone(runner_up)

    def test_low_confidence_mix_keeps_uncertain_qualifier(self):
        from videodna.forensics import honest_verdict_summary
        breakdown = {"live": 27.6, "3d_render": 18.5, "code_motion": 17.5,
                     "ai_video": 13.9, "screen_rec": 11.3, "static_zoom": 11.3}
        probs = {k: v / 100.0 for k, v in breakdown.items()}
        verdict, uncertain, runner_up = honest_verdict_summary(breakdown, probs, 0.44)
        self.assertTrue(uncertain)
        self.assertEqual(runner_up, "3d_render")
        self.assertIn("live 28%", verdict)
        self.assertIn("3d_render 18%", verdict)
        self.assertIn("ai_video 14%", verdict)
        self.assertIn("uncertain from pixels alone", verdict)
        self.assertNotIn("unknown", verdict)

    def test_fused_path_uses_same_mix(self):
        from videodna.forensics import fuse_review_probabilities
        fused = fuse_review_probabilities(
            {"per_shot": [
                {"shot_id": 1, "start": 0.0, "end": 10.0, "duration": 10.0,
                 "predicted_type": "live", "confidence": 0.7,
                 "probs": _probs("live", 0.7), "metrics": {}},
                {"shot_id": 2, "start": 10.0, "end": 20.0, "duration": 10.0,
                 "predicted_type": "3d_render", "confidence": 0.7,
                 "probs": _probs("3d_render", 0.7), "metrics": {}},
            ]},
            {"shot_annotations": []},
        )
        self.assertEqual(set(fused["duration_weighted_pct"]), set(CATS))
        self.assertIn("live", fused["verdict_summary"])
        self.assertIn("3d_render", fused["verdict_summary"])


class TestExportShotFrames(unittest.TestCase):
    """Item 1: one streaming pass writes first/mid/last (+strip) per shot."""

    def test_single_shot_video_exports_three_frames(self):
        if not TEST_MP4.exists():
            self.skipTest(f"test.mp4 missing: {TEST_MP4}")
        from videodna.shots import detect_shots, export_shot_frames
        from videodna.probe import probe_video
        probe = probe_video(TEST_MP4)
        shots = detect_shots(TEST_MP4, probe["duration_sec"], mode="short",
                             width=probe["width"], height=probe["height"])
        with tempfile.TemporaryDirectory(prefix="videodna_v204_frames_") as td:
            out_dir = Path(td)
            frames = export_shot_frames(
                TEST_MP4, shots["shots"], out_dir,
                fps=probe.get("avg_fps") or 30.0,
                orig_width=probe.get("width"), orig_height=probe.get("height"),
            )
            self.assertEqual(len(frames), len(shots["shots"]))
            for sid, paths in frames.items():
                for label in ("first", "mid", "last"):
                    self.assertIn(label, paths, f"shot {sid} missing {label}")
                    p = out_dir / paths[label]
                    self.assertTrue(p.exists(), f"missing file {p}")
                    self.assertGreater(p.stat().st_size, 1000)
                    # Valid JPEG, 480px wide.
                    from PIL import Image
                    with Image.open(p) as im:
                        self.assertEqual(im.format, "JPEG")
                        self.assertEqual(im.size[0], 480)

    def test_empty_shots_returns_empty(self):
        from videodna.shots import export_shot_frames
        self.assertEqual(export_shot_frames(Path("/nonexistent.mp4"), [], Path("/tmp")), {})

    def test_missing_video_fails_open(self):
        from videodna.shots import export_shot_frames
        shots = [{"id": 1, "start": 0.0, "end": 2.0, "duration": 2.0}]
        with tempfile.TemporaryDirectory(prefix="videodna_v204_frames_") as td:
            self.assertEqual(
                export_shot_frames(Path("/nonexistent.mp4"), shots, Path(td)), {}
            )


class TestShotCardFramePathsAndRunnerUp(unittest.TestCase):
    """Items 1+2: both card builders carry frame_paths and runner-up."""

    def _fake_forensics(self):
        return {"per_shot": [
            {"shot_id": 1, "start": 0.0, "end": 2.5, "duration": 2.5,
             "predicted_type": "live", "confidence": 0.68,
             "probs": _probs("live", 0.68),
             "needs_visual_review": False,
             "palette_hex": ["#080808"], "metrics": {"mean_frame_diff": 4.98}},
        ]}

    def test_remake_cards_carry_frames_and_runner_up(self):
        from videodna.remake import build_shot_cards
        frames = {1: {"first": "dna/frames/shot-01-first.jpg",
                      "mid": "dna/frames/shot-01-mid.jpg",
                      "last": "dna/frames/shot-01-last.jpg",
                      "strip": "dna/frames/shot-01-strip.jpg"}}
        cards = build_shot_cards(self._fake_forensics(), shot_frames=frames)
        c = cards[0]
        self.assertEqual(c["frame_paths"], frames[1])
        self.assertEqual(c["production_type"], "live")
        self.assertIsNotNone(c["production_runner_up"])
        self.assertNotEqual(c["production_runner_up"], "live")
        self.assertFalse(c["needs_visual_review"])

    def test_report_cards_carry_frames_and_runner_up(self):
        from videodna.report import _build_shot_cards, _render_shot_card
        bp = {
            "shots": {"shots": [
                {"id": 1, "start": 0.0, "end": 2.5, "duration": 2.5,
                 "transition_in": "opening",
                 "frame_paths": {"first": "dna/frames/shot-01-first.jpg",
                                 "mid": "dna/frames/shot-01-mid.jpg",
                                 "last": "dna/frames/shot-01-last.jpg"}},
            ]},
            "visual": {"shot_visuals": []},
            "forensics": self._fake_forensics(),
            "fused_forensics": None,
            "remake_plan": {"shot_cards": []},
        }
        cards = _build_shot_cards(bp)
        self.assertEqual(cards[0]["frame_paths"]["mid"], "dna/frames/shot-01-mid.jpg")
        self.assertIsNotNone(cards[0]["production_runner_up"])
        md = "\n".join(_render_shot_card(cards[0], "### Shot #01"))
        self.assertIn("dna/frames/shot-01-first.jpg", md)
        self.assertIn("runner-up:", md)

    def test_rhythm_renders_in_shots_md(self):
        from videodna.report import _render_shots_md
        bp = {
            "shots": {"shot_count": 2, "mode": "short",
                      "avg_shot_length_sec": 2.25, "cuts_per_minute": 13.33,
                      "hard_cuts": 1, "soft_transitions": 0,
                      "shots": [
                          {"id": 1, "start": 0.0, "end": 2.5, "duration": 2.5,
                           "transition_in": "opening"},
                          {"id": 2, "start": 2.5, "end": 4.5, "duration": 2.0,
                           "transition_in": "hard_cut"},
                      ],
                      "rhythm": {
                          "shot_length_min_sec": 2.0, "shot_length_median_sec": 2.25,
                          "shot_length_max_sec": 2.5,
                          "longest_shot": {"shot_id": 1, "start": 0.0, "end": 2.5, "duration": 2.5},
                          "shortest_shot": {"shot_id": 2, "start": 2.5, "end": 4.5, "duration": 2.0},
                          "cuts_per_10s": 13.33,
                          "pacing_curve": [{"window_start": 0.0, "window_end": 4.5,
                                            "shots": 2, "cuts": 1}],
                          "transitions": [{"time": 2.5, "into_shot": 2, "type": "hard_cut"}],
                      }},
            "visual": {"shot_visuals": []},
            "forensics": {"per_shot": []},
            "fused_forensics": None,
            "diagnostics": {"true_dead_zones": []},
            "remake_plan": {"shot_cards": []},
        }
        md = _render_shots_md(bp)
        self.assertIn("Rhythm & Pacing", md)
        self.assertIn("median `2.25s`", md)
        self.assertIn("Longest shot", md)
        self.assertIn("#01", md)
        self.assertIn("2.50s | #02 | `hard_cut`", md)


class TestEndToEndV204(unittest.TestCase):
    """Smoke: speech2 verdict becomes a decisive mixed verdict (item 2),
    frames exist for every shot (item 1), rhythm block in blueprint (item 5).
    Shot counts, audio and retention must not change vs v2.0.3."""

    def setUp(self) -> None:
        base_tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        self.tmp = Path(tempfile.mkdtemp(prefix="videodna_v204_qa_", dir=base_tmp))

    def tearDown(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _analyze(self, video: Path, out_dir: Path) -> dict:
        subprocess.run(
            [VIDEODNA_BIN, "analyze", str(video), "-o", str(out_dir)],
            check=True, capture_output=True, text=True,
        )
        return json.loads((out_dir / "blueprint.json").read_text(encoding="utf-8"))

    def test_speech2_mixed_verdict_frames_and_rhythm(self):
        if not SPEECH2.exists():
            self.skipTest(f"speech2 video missing: {SPEECH2}")
        out = self.tmp / "speech2"
        bp = self._analyze(SPEECH2, out)
        # Regression: shots, audio, retention unchanged from v2.0.3.
        self.assertEqual(bp["shots"]["shot_count"], 15)
        self.assertEqual(bp["audio"]["audio_type"], "speech_plus_bgm")
        self.assertEqual(bp["audio"]["estimated_bpm"], 115.7)
        self.assertEqual(bp["diagnostics"]["overall_retention_score"], 7.1)
        # Item 2: decisive mixed verdict naming live + AI/3D shares, honest.
        verdict = bp["forensics"]["verdict_summary"]
        self.assertIn("live", verdict)
        self.assertIn("3d_render", verdict)
        self.assertIn("ai_video", verdict)
        self.assertIn("uncertain from pixels alone", verdict)
        self.assertNotIn("unknown 48%", verdict)
        self.assertTrue(bp["forensics"]["verdict_uncertain"])
        self.assertEqual(bp["forensics"]["runner_up"], "3d_render")
        self.assertNotIn("unknown", bp["forensics"]["duration_weighted_pct"])
        # Item 1: frames for every shot.
        for s in bp["shots"]["shots"]:
            fp = s.get("frame_paths") or {}
            for label in ("first", "mid", "last"):
                self.assertIn(label, fp, f"shot {s['id']} missing {label}")
                self.assertTrue((out / fp[label]).exists())
        self.assertEqual(len(list((out / "dna" / "frames").glob("shot-*-first.jpg"))), 15)
        # Item 5: rhythm block present and consistent.
        r = bp["shots"]["rhythm"]
        self.assertEqual(r["longest_shot"]["shot_id"],
                         max(bp["shots"]["shots"], key=lambda s: s["duration"])["id"])
        self.assertEqual(r["shortest_shot"]["shot_id"],
                         min(bp["shots"]["shots"], key=lambda s: s["duration"])["id"])
        self.assertEqual(len(r["transitions"]), 14)
        self.assertEqual(len(r["pacing_curve"]), 3)  # 0-10, 10-20, 20-21.32
        dna = (out / "DNA.md").read_text(encoding="utf-8")
        self.assertIn("Rhythm:", dna)
        self.assertIn("dna/frames/", dna)
        shots_md = (out / "dna" / "shots.md").read_text(encoding="utf-8")
        self.assertIn("Rhythm & Pacing", shots_md)
        self.assertIn("dna/frames/shot-01-first.jpg", shots_md)


MULTI_MP4 = Path(os.environ.get(
    "VIDEODNA_V204_MULTI", "/home/hatch/workspace/videodna-test/multi.mp4",
))
U2_MP4 = Path(os.environ.get(
    "VIDEODNA_V204_U2",
    "/home/hatch/workspace/user/media_library/video/5e/"
    "5ea5bf13f40626e7b1253d0d5bb517aaa79a068602dc067f20db91f80dd7b496.mp4",
))
VIDTEST = Path(os.environ.get("VIDEODNA_V204_VIDTEST", "/tmp/vidtest"))


def _build_fixture(name: str, build_cmd: list[str], tmp: Path) -> Path:
    """Build one synthetic fixture with ffmpeg; raises on failure."""
    out = tmp / name
    subprocess.run(build_cmd(out), check=True, capture_output=True, text=True)
    return out


class TestPhaseShift(unittest.TestCase):
    """Item 4 helper: FFT phase correlation returns the TRUE shift.

    The FFT peak is the negated shift (verified on synthetic rolls) —
    _phase_shift negates it back, so a roll by (dy=2, dx=3) reads
    (dx=3, dy=2) with coherence ~1.0. Identical frames read (0, 0)."""

    def test_synthetic_roll_returns_true_shift(self):
        from videodna.forensics import _phase_shift
        rng = np.random.default_rng(0)
        base = rng.random((90, 160)).astype(np.float32) * 255
        moved = np.roll(np.roll(base, 2, axis=0), 3, axis=1)
        dx, dy, coh, raw = _phase_shift(base, moved)
        self.assertEqual((dx, dy), (3, 2))
        self.assertAlmostEqual(coh, 1.0, places=2)
        self.assertGreater(raw, 10.0)

    def test_identical_frames_zero_shift(self):
        from videodna.forensics import _phase_shift
        rng = np.random.default_rng(1)
        base = rng.random((90, 160)).astype(np.float32) * 255
        self.assertEqual(_phase_shift(base, base.copy()), (0, 0, 0.0, 0.0))


class TestCameraMotionFromBurst(unittest.TestCase):
    """Item 4: static/pan separate on synthetic bursts; the rest is unknown."""

    def _lumas(self, frames):
        return np.stack(frames).astype(np.float32)

    def test_static_identical_frames(self):
        from videodna.forensics import _camera_motion_from_burst
        rng = np.random.default_rng(2)
        base = rng.random((90, 160)).astype(np.float32) * 255
        label, conf = _camera_motion_from_burst(self._lumas([base] * 12))
        self.assertEqual((label, conf), ("static", 0.9))

    def test_steady_pan(self):
        from videodna.forensics import _camera_motion_from_burst
        rng = np.random.default_rng(3)
        base = rng.random((90, 160)).astype(np.float32) * 255
        frames = [np.roll(base, -2 * i, axis=1) for i in range(12)]
        label, conf = _camera_motion_from_burst(self._lumas(frames))
        self.assertEqual(label, "pan")
        self.assertGreaterEqual(conf, 0.7)

    def test_sign_flipping_jitter_is_unknown(self):
        from videodna.forensics import _camera_motion_from_burst
        rng = np.random.default_rng(4)
        base = rng.random((90, 160)).astype(np.float32) * 255
        # Alternating +-1px rolls: motion exists, no consistent direction.
        frames = [np.roll(base, (1 if i % 2 else -1), axis=1) for i in range(12)]
        label, conf = _camera_motion_from_burst(self._lumas(frames))
        self.assertEqual((label, conf), ("unknown", 0.0))

    def test_random_noise_is_unknown(self):
        from videodna.forensics import _camera_motion_from_burst
        rng = np.random.default_rng(5)
        frames = [rng.random((90, 160)).astype(np.float32) * 255 for _ in range(12)]
        label, conf = _camera_motion_from_burst(self._lumas(frames))
        self.assertEqual((label, conf), ("unknown", 0.0))

    def test_too_few_frames_is_unknown(self):
        from videodna.forensics import _camera_motion_from_burst
        rng = np.random.default_rng(6)
        base = rng.random((90, 160)).astype(np.float32) * 255
        label, conf = _camera_motion_from_burst(self._lumas([base, base]))
        self.assertEqual((label, conf), ("unknown", 0.0))


class TestClassifySoftTransition(unittest.TestCase):
    """Item 3: fade / dissolve / slide_wipe on synthetic fixtures.

    Fixtures are built with ffmpeg into a temp dir (skipped when ffmpeg
    fails). NOTE on M3's first transition: ffmpeg's xfade `fade` is a
    linear crossfade, not a fade through black (measured: luma
    126 -> 108 -> 93 across the transition, no dip; blend alpha sweeps
    0.91 -> 0) — so its pixel-truth label is "dissolve". The "fade"
    branch is validated on truefade.mp4, a real fade-out to black +
    fade-in from black (luma 126 -> 59 -> 0 -> 49 -> 93).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="videodna_v204_trans_"))
        try:
            cls.m3 = _build_fixture("textured.mp4", lambda o: [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
                "-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=2",
                "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
                "-filter_complex",
                "[0][1]xfade=transition=fade:duration=0.5:offset=1.5[x1];"
                "[x1][2]xfade=transition=slideleft:duration=0.5:offset=3.0[out]",
                "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(o),
            ], cls.tmp)
            cls.truefade = _build_fixture("truefade.mp4", lambda o: [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
                "-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=2",
                "-filter_complex",
                "[0]fade=t=out:st=1.5:d=0.5[v0];[1]fade=t=in:st=0:d=0.5[v1];"
                "[v0][v1]concat=n=2:v=1:a=0[out]",
                "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(o),
            ], cls.tmp)
            cls.wipe = _build_fixture("wipe.mp4", lambda o: [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
                "-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=2",
                "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
                "-filter_complex",
                "[0][1]xfade=transition=wipeleft:duration=0.5:offset=1.5[x1];"
                "[x1][2]xfade=transition=slideleft:duration=0.5:offset=3.0[out]",
                "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(o),
            ], cls.tmp)
        except (subprocess.CalledProcessError, OSError):
            cls.m3 = cls.truefade = cls.wipe = None

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _subtypes(self, video: Path) -> list:
        from videodna.probe import probe_video
        from videodna.shots import detect_shots
        probe = probe_video(video)
        shots = detect_shots(video, probe["duration_sec"], mode="short",
                             width=probe["width"], height=probe["height"])
        return [(s["id"], s["transition_in"], s.get("transition_subtype"))
                for s in shots["shots"]], shots

    def test_m3_crossfade_is_dissolve_and_slide_is_slide_wipe(self):
        if self.m3 is None or not self.m3.exists():
            self.skipTest("M3 fixture could not be built")
        subs, shots = self._subtypes(self.m3)
        self.assertEqual(shots["shot_count"], 3)
        self.assertEqual(subs[1], (2, "soft_transition", "dissolve"))
        self.assertEqual(subs[2], (3, "soft_transition", "slide_wipe"))
        self.assertIsNone(subs[0][2])  # opening has no subtype
        # Rhythm list carries the subtype only for soft transitions.
        tr = shots["rhythm"]["transitions"]
        self.assertEqual(tr[0]["subtype"], "dissolve")
        self.assertEqual(tr[1]["subtype"], "slide_wipe")

    def test_true_fade_through_black_is_fade(self):
        if self.truefade is None or not self.truefade.exists():
            self.skipTest("true-fade fixture could not be built")
        subs, shots = self._subtypes(self.truefade)
        self.assertEqual(shots["shot_count"], 2)
        self.assertEqual(subs[1], (2, "soft_transition", "fade"))

    def test_wipeleft_reveal_and_slideleft_are_slide_wipe(self):
        if self.wipe is None or not self.wipe.exists():
            self.skipTest("wipe fixture could not be built")
        subs, shots = self._subtypes(self.wipe)
        self.assertEqual(shots["shot_count"], 3)
        self.assertEqual(subs[1], (2, "soft_transition", "slide_wipe"))
        self.assertEqual(subs[2], (3, "soft_transition", "slide_wipe"))

    def test_multi_flat_dissolve_is_dissolve(self):
        if not MULTI_MP4.exists():
            self.skipTest(f"multi.mp4 missing: {MULTI_MP4}")
        subs, shots = self._subtypes(MULTI_MP4)
        self.assertEqual(shots["shot_count"], 2)
        self.assertEqual(subs[1], (2, "soft_transition", "dissolve"))


class TestCameraMotionFixtures(unittest.TestCase):
    """Item 4 end-to-end: synthetic static/pan/zoom/jitter videos.

    zoom and jitter must read "unknown" — the measured zoom tests do
    not separate (centre-vs-edge 1.50 vs pan 1.89; scale-search
    residual >= raw diff) and jitter is invisible at 160x90/12fps.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="videodna_v204_cam_"))
        try:
            still = cls.tmp / "still.png"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=0.1",
                "-frames:v", "1", str(still),
            ], check=True, capture_output=True)
            for name, vf in [
                ("cam-static.mp4", "scale=640:360"),
                ("cam-pan.mp4", "crop=640:360:x='200+120*t':y=180,scale=640:360"),
                ("cam-zoom.mp4", "zoompan=z='1+0.02*on':d=150:s=640x360:fps=30"),
            ]:
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-loop", "1", "-i", str(still), "-t", "5",
                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(cls.tmp / name),
                ], check=True, capture_output=True)
            # Jitter: +-6px random-walk translations of the still.
            from PIL import Image
            rng = np.random.default_rng(7)
            arr = np.asarray(Image.open(still).convert("RGB"))
            h, w, _ = arr.shape
            ox, oy = 0, 0
            proc = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
                "-i", "-", "-vf", "scale=640:360",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(cls.tmp / "cam-jitter.mp4"),
            ], stdin=subprocess.PIPE)
            assert proc.stdin is not None
            for _ in range(150):
                ox = int(np.clip(ox + rng.integers(-2, 3), -6, 6))
                oy = int(np.clip(oy + rng.integers(-2, 3), -6, 6))
                f = np.pad(arr, ((6, 6), (6, 6), (0, 0)), mode="edge")
                proc.stdin.write(f[6 + oy:6 + oy + h, 6 + ox:6 + ox + w].astype(np.uint8).tobytes())
            proc.stdin.close()
            proc.wait()
            cls.ok = proc.returncode == 0
        except (subprocess.CalledProcessError, OSError, AssertionError):
            cls.ok = False

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _motion(self, name: str) -> str:
        from videodna.forensics import classify_camera_motion_for_shots
        video = self.tmp / name
        res = classify_camera_motion_for_shots(
            video, [{"id": 1, "start": 0.0, "end": 5.0, "duration": 5.0}])
        return res[1]["camera_motion"]

    def test_static(self):
        if not getattr(self, "ok", False):
            self.skipTest("camera fixtures could not be built")
        self.assertEqual(self._motion("cam-static.mp4"), "static")

    def test_pan(self):
        if not getattr(self, "ok", False):
            self.skipTest("camera fixtures could not be built")
        self.assertEqual(self._motion("cam-pan.mp4"), "pan")

    def test_zoom_is_unknown(self):
        if not getattr(self, "ok", False):
            self.skipTest("camera fixtures could not be built")
        self.assertEqual(self._motion("cam-zoom.mp4"), "unknown")

    def test_jitter_is_unknown(self):
        if not getattr(self, "ok", False):
            self.skipTest("camera fixtures could not be built")
        self.assertEqual(self._motion("cam-jitter.mp4"), "unknown")

    def test_forensics_loop_carries_camera_motion(self):
        """The main path: run_shot_forensics labels its sampled shots."""
        if not getattr(self, "ok", False):
            self.skipTest("camera fixtures could not be built")
        from videodna.forensics import run_shot_forensics
        from videodna.probe import probe_video
        from videodna.shots import detect_shots
        video = self.tmp / "cam-pan.mp4"
        probe = probe_video(video)
        shots = detect_shots(video, probe["duration_sec"], mode="short",
                             width=probe["width"], height=probe["height"])
        fx = run_shot_forensics(video, probe, shots, mode="short")
        self.assertEqual(fx["per_shot"][0]["camera_motion"], "pan")
        self.assertGreaterEqual(fx["per_shot"][0]["camera_motion_confidence"], 0.7)


class TestOCRGate(unittest.TestCase):
    """Item 6: the quality gate keeps real words, drops confident garbage."""

    def _word(self, text, conf, x=0, y=0, w=10, h=10):
        return {"text": text, "conf": conf, "x": x, "y": y, "w": w, "h": h}

    def test_keeps_real_overlay_text_with_digits(self):
        from videodna.ocr import _gate_words
        words = [self._word("Text", 95.8), self._word("4", 88.0),
                 self._word("Text", 95.5), self._word("4", 97.0)]
        kept = _gate_words(words)
        self.assertEqual([k["text"] for k in kept], ["Text", "4", "Text", "4"])

    def test_drops_single_char_garbage_even_when_confident(self):
        from videodna.ocr import _gate_words
        words = [self._word("Y", 90.7), self._word("|", 86.9),
                 self._word("S", 85.9), self._word("20", 83.5),
                 self._word("!", 82.8)]
        self.assertEqual(_gate_words(words), [])

    def test_drops_low_confidence_three_letter_strays(self):
        from videodna.ocr import _gate_words
        # Measured strays on the test videos: hws 50.6, hens 55.4, कैट 50.7.
        words = [self._word("hws", 50.6), self._word("hens", 55.4),
                 self._word("कैट", 50.7)]
        self.assertEqual(_gate_words(words), [])

    def test_digit_kept_only_after_kept_word(self):
        from videodna.ocr import _gate_words
        self.assertEqual(
            [k["text"] for k in _gate_words([self._word("Text", 95.0), self._word("1", 93.0)])],
            ["Text", "1"],
        )
        # "20" after an unkept single char is dropped.
        self.assertEqual(
            _gate_words([self._word("S", 87.0), self._word("20", 83.5)]), [])

    def test_clean_word_strips_quotes(self):
        from videodna.ocr import _clean_word
        self.assertEqual(_clean_word("“कैट"), "कैट")
        self.assertEqual(_clean_word("Text,"), "Text")

    def test_no_tesseract_returns_empty(self):
        from unittest import mock
        import videodna.ocr as ocr_mod
        with mock.patch.object(ocr_mod, "tesseract_available", return_value=False):
            self.assertEqual(
                ocr_mod.extract_shot_text(
                    Path("/nonexistent.mp4"),
                    [{"id": 1, "start": 0.0, "end": 2.0, "duration": 2.0}]),
                {},
            )


class TestOCREndToEnd(unittest.TestCase):
    """Item 6: real extraction on U2 (clean overlay text) and speech2."""

    def test_u2_shot3_reads_text_4(self):
        from videodna.ocr import extract_shot_text, tesseract_available
        if not tesseract_available():
            self.skipTest("tesseract not installed")
        if not U2_MP4.exists():
            self.skipTest(f"U2 video missing: {U2_MP4}")
        from videodna.probe import probe_video
        from videodna.shots import detect_shots
        probe = probe_video(U2_MP4)
        shots = detect_shots(U2_MP4, probe["duration_sec"], mode="short",
                             width=probe["width"], height=probe["height"])
        texts = extract_shot_text(U2_MP4, shots["shots"])
        self.assertEqual(len(texts), 4)
        self.assertIn("Text 4", texts[3]["text"])
        for sid in (1, 2, 4):
            self.assertEqual(texts[sid]["text"], "")
        # Word boxes are normalised 0-1000 with text + confidence.
        self.assertTrue(all(0 <= w["x"] <= 1000 for w in texts[3]["words"]))
        self.assertTrue(all(w["conf"] >= 70 for w in texts[3]["words"]))

    def test_speech2_reads_real_overlay_text(self):
        from videodna.ocr import extract_shot_text, tesseract_available
        if not tesseract_available():
            self.skipTest("tesseract not installed")
        if not SPEECH2.exists():
            self.skipTest(f"speech2 video missing: {SPEECH2}")
        from videodna.probe import probe_video
        from videodna.shots import detect_shots
        probe = probe_video(SPEECH2)
        shots = detect_shots(SPEECH2, probe["duration_sec"], mode="short",
                             width=probe["width"], height=probe["height"])
        texts = extract_shot_text(SPEECH2, shots["shots"])
        self.assertEqual(len(texts), 15)
        # Shot 1 carries the video's title/comment UI text; most shots
        # have no readable text and must report "" (never garbage).
        self.assertIn("Janata Party", texts[1]["text"])
        for sid, entry in texts.items():
            for w in entry["words"]:
                self.assertGreaterEqual(len(w["text"]), 1)
                self.assertGreaterEqual(w["conf"], 70)


class TestEndToEndV204Items346(unittest.TestCase):
    """Smoke: M3 full analyze — subtypes + camera motion + OCR fields
    present in blueprint/cards, regressions unchanged vs agent A."""

    def setUp(self) -> None:
        base_tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        self.tmp = Path(tempfile.mkdtemp(prefix="videodna_v204_346_", dir=base_tmp))

    def tearDown(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_m3_subtypes_camera_ocr_in_blueprint(self):
        m3 = VIDTEST / "textured.mp4"
        if not m3.exists():
            # Rebuild M3 the same way (fade + slide) if /tmp was wiped.
            try:
                m3 = _build_fixture("textured.mp4", lambda o: [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
                    "-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=2",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
                    "-filter_complex",
                    "[0][1]xfade=transition=fade:duration=0.5:offset=1.5[x1];"
                    "[x1][2]xfade=transition=slideleft:duration=0.5:offset=3.0[out]",
                    "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(o),
                ], self.tmp)
            except (subprocess.CalledProcessError, OSError):
                self.skipTest(f"M3 missing and could not be rebuilt: {m3}")
        out = self.tmp / "m3"
        subprocess.run(
            [VIDEODNA_BIN, "analyze", str(m3), "-o", str(out)],
            check=True, capture_output=True, text=True,
        )
        bp = json.loads((out / "blueprint.json").read_text(encoding="utf-8"))
        # Regression vs agent A: shots, verdict, audio, retention identical.
        self.assertEqual(bp["shots"]["shot_count"], 3)
        self.assertEqual(bp["audio"]["audio_type"], "silent_no_audio")
        self.assertEqual(bp["diagnostics"]["overall_retention_score"], 8.2)
        self.assertIn("code_motion 70%", bp["forensics"]["verdict_summary"])
        # Item 3: subtypes on the shot dicts + rhythm list.
        subs = [s.get("transition_subtype") for s in bp["shots"]["shots"]]
        self.assertEqual(subs, [None, "dissolve", "slide_wipe"])
        tr = bp["shots"]["rhythm"]["transitions"]
        self.assertEqual([t.get("subtype") for t in tr], ["dissolve", "slide_wipe"])
        # Item 4: camera motion on every shot dict + every forensics entry.
        for s in bp["shots"]["shots"]:
            self.assertIn(s.get("camera_motion"), ("static", "pan", "unknown"))
        for fx in bp["forensics"]["per_shot"]:
            self.assertIn(fx.get("camera_motion"), ("static", "pan", "unknown"))
        # Item 6: OCR fields on every shot card (both builders).
        for card in bp["remake_plan"]["shot_cards"]:
            self.assertIn("on_screen_text", card)
            self.assertIn("on_screen_text_words", card)
            self.assertIn("transition_subtype", card)
            self.assertIn("camera_motion", card)
        shots_md = (out / "dna" / "shots.md").read_text(encoding="utf-8")
        self.assertIn("(dissolve)", shots_md)
        self.assertIn("(slide_wipe)", shots_md)
        self.assertIn("camera motion", shots_md)


if __name__ == "__main__":
    unittest.main()
