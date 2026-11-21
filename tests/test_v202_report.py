"""v2.0.2: remake-plan shot cards (measured per-shot content, no invented
text) and low-confidence verdict honesty.

- Shot cards carry duration, transition-in, dominant colors (measured from
  the shot's own pixels), motion level (measured frame differences with
  units), and on-screen text ONLY if OCR actually extracted it. No OCR
  engine is installed on this machine, so cards must state
  "text not extracted (no OCR available)" and never claim any text.
- When the duration-weighted overall production confidence is < 0.5 the
  verdict must say the call is uncertain and name the real runner-up
  (from the measured probabilities) instead of asserting one type.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = REPO_ROOT / ".venv" / "bin" / "videodna"
VIDEODNA_BIN = str(VENV_BIN if VENV_BIN.exists() else shutil.which("videodna"))

HOME = Path.home()
U2 = HOME / "workspace/user/media_library/video/5e/5ea5bf13f40626e7b1253d0d5bb517aaa79a068602dc067f20db91f80dd7b496.mp4"
U4 = HOME / "workspace/user/media_library/video/7c/7c4d22b0e260da7b9664be6476af726f3225cc1fb8a6e4b981f0fc56e39d5152.mp4"
U1 = HOME / "workspace/user/media_library/video/fd/fda867a38232d196f7799a04609ccfca811e57a38596ae0218b5949d992fd031.mp4"
TEST_MP4 = HOME / "workspace/videodna-test/test.mp4"
MULTI_MP4 = HOME / "workspace/videodna-test/multi.mp4"
M3_MP4 = Path("/tmp/vidtest/textured.mp4")

# U4's real measured overall probabilities (baseline run, conf 0.43).
U4_PROBS = {
    "live": 0.036, "ai_video": 0.036, "3d_render": 0.081,
    "code_motion": 0.469, "screen_rec": 0.139, "static_zoom": 0.239,
}
U4_BREAKDOWN = {"code_motion": 100.0}


def _analyze(video: Path, out_dir: Path) -> dict:
    subprocess.run(
        [VIDEODNA_BIN, "analyze", str(video), "-o", str(out_dir)],
        check=True, capture_output=True, text=True,
    )
    return json.loads((out_dir / "blueprint.json").read_text(encoding="utf-8"))


class TestHonestVerdict(unittest.TestCase):
    def test_low_confidence_uses_real_runner_up(self) -> None:
        from videodna.forensics import honest_verdict_summary
        verdict, uncertain, runner_up = honest_verdict_summary(U4_BREAKDOWN, U4_PROBS, 0.43)
        self.assertEqual(verdict, "code_motion / static_zoom — uncertain from pixels alone")
        self.assertTrue(uncertain)
        self.assertEqual(runner_up, "static_zoom")
        self.assertNotIn("100%", verdict)

    def test_high_confidence_verdicts_unchanged(self) -> None:
        from videodna.forensics import honest_verdict_summary
        # The must-not-change videos: U1 0.73, U2 0.78, test.mp4 0.72,
        # M3 0.71, multi.mp4 0.59 — all keep the assertive wording.
        for conf in (0.73, 0.78, 0.72, 0.71, 0.59):
            verdict, uncertain, runner_up = honest_verdict_summary(
                {"code_motion": 100.0}, {"code_motion": 0.7, "live": 0.1, "ai_video": 0.05,
                                         "3d_render": 0.05, "screen_rec": 0.05, "static_zoom": 0.05},
                conf,
            )
            self.assertEqual(verdict, "code_motion 100%")
            self.assertFalse(uncertain)
            self.assertIsNone(runner_up)

    def test_boundary_confidence(self) -> None:
        from videodna.forensics import honest_verdict_summary
        verdict, uncertain, _ = honest_verdict_summary(U4_BREAKDOWN, U4_PROBS, 0.5)
        self.assertEqual(verdict, "code_motion 100%")
        self.assertFalse(uncertain)
        verdict, uncertain, _ = honest_verdict_summary(U4_BREAKDOWN, U4_PROBS, 0.49)
        self.assertIn("uncertain from pixels alone", verdict)
        self.assertTrue(uncertain)

    def test_unknown_top_call_stays_unknown(self) -> None:
        from videodna.forensics import honest_verdict_summary
        probs = {k: round(1 / 6, 3) for k in
                 ("live", "ai_video", "3d_render", "code_motion", "screen_rec", "static_zoom")}
        verdict, uncertain, _ = honest_verdict_summary({"unknown": 100.0}, probs, 0.30)
        self.assertEqual(verdict, "unknown 100%")
        self.assertFalse(uncertain)

    def test_fused_path_applies_same_rule(self) -> None:
        from videodna.forensics import fuse_review_probabilities
        forensics = {
            "per_shot": [{
                "shot_id": 1, "start": 0.0, "end": 6.04, "duration": 6.04,
                "predicted_type": "code_motion", "confidence": 0.43,
                "probs": dict(U4_PROBS), "metrics": {},
            }],
        }
        fused = fuse_review_probabilities(forensics, {"shot_annotations": []})
        self.assertEqual(
            fused["verdict_summary"],
            "code_motion / static_zoom — uncertain from pixels alone",
        )
        self.assertTrue(fused["verdict_uncertain"])
        self.assertEqual(fused["runner_up"], "static_zoom")


class TestDominantPalette(unittest.TestCase):
    def test_solid_color_frame(self) -> None:
        from videodna.forensics import _dominant_palette
        red = np.zeros((90, 160, 3), dtype=np.float32)
        red[..., 0] = 255.0
        self.assertEqual(_dominant_palette(red), ["#F80808"])

    def test_two_color_frame_finds_both(self) -> None:
        from videodna.forensics import _dominant_palette
        half = np.zeros((90, 160, 3), dtype=np.float32)
        half[:, :80, 0] = 255.0
        half[:, 80:, 2] = 255.0
        palette = _dominant_palette(half)
        self.assertEqual(sorted(palette), ["#0808F8", "#F80808"])

    def test_empty_frame(self) -> None:
        from videodna.forensics import _dominant_palette
        self.assertEqual(_dominant_palette(np.zeros((0, 0, 3), dtype=np.float32)), [])


class TestShotCards(unittest.TestCase):
    def _fake_forensics(self) -> dict:
        return {"per_shot": [
            {"shot_id": 1, "start": 0.0, "end": 2.5, "duration": 2.5,
             "predicted_type": "code_motion", "confidence": 0.72,
             "palette_hex": ["#080808", "#780808"],
             "metrics": {"mean_frame_diff": 4.98}},
            {"shot_id": 2, "start": 2.5, "end": 4.5, "duration": 2.0,
             "predicted_type": "code_motion", "confidence": 0.72,
             "palette_hex": ["#F8F8F8"],
             "metrics": {"mean_frame_diff": 5.52}},
        ]}

    def test_cards_carry_measured_fields_and_no_invented_text(self) -> None:
        from videodna.remake import OCR_TEXT_STATUS, build_shot_cards
        cards = build_shot_cards(self._fake_forensics())
        self.assertEqual(len(cards), 2)
        c1 = cards[0]
        self.assertEqual(c1["shot_id"], 1)
        self.assertEqual(c1["duration"], 2.5)
        self.assertEqual(c1["production_type"], "code_motion")
        self.assertEqual(c1["palette_hex"], ["#080808", "#780808"])
        self.assertEqual(c1["motion_mean_frame_diff"], 4.98)
        self.assertIn("0-255", c1["motion_units"])
        # No OCR installed: the card must say so, never claim any text.
        self.assertEqual(c1["on_screen_text"], "text not extracted (no OCR available)")
        self.assertEqual(OCR_TEXT_STATUS, "text not extracted (no OCR available)")
        self.assertNotIn("BLOOD", json.dumps(cards))

    def test_remake_plan_includes_shot_cards(self) -> None:
        from videodna.remake import build_remake_plan
        plan = build_remake_plan(
            self._fake_forensics(),
            {"audio_type": "silent_no_audio"},
            {"lut_file": "reference_grade.cube", "grade_style": "test"},
        )
        self.assertEqual(len(plan["shot_cards"]), 2)
        self.assertEqual(plan["shot_cards"][1]["palette_hex"], ["#F8F8F8"])

    def _fake_blueprint(self) -> dict:
        forensics = self._fake_forensics()
        forensics["low_confidence_shots"] = []
        return {
            "shots": {"shot_count": 2, "mode": "short", "shots": [
                {"id": 1, "start": 0.0, "end": 2.5, "duration": 2.5, "transition_in": "opening"},
                {"id": 2, "start": 2.5, "end": 4.5, "duration": 2.0, "transition_in": "soft_transition"},
            ]},
            "visual": {"shot_visuals": [
                {"shot_id": 1, "motion_class": "dynamic_action", "motion_energy": 24.31},
                {"shot_id": 2, "motion_class": "high_energy_whip", "motion_energy": 42.08},
            ]},
            "forensics": forensics,
            "fused_forensics": None,
            "diagnostics": {"true_dead_zones": []},
            "remake_plan": {
                "total_batches": 1,
                "audio_pipeline": "test audio",
                "color_pipeline": "test color",
                "batches": [{"batch_id": 1, "production_type": "code_motion",
                             "start": 0.0, "end": 4.5, "total_duration": 4.5,
                             "tool": "HyperFrames", "cost_tier": "free_local",
                             "strategy": "test", "shot_ids": [1, 2]}],
            },
        }

    def test_render_remake_md_shows_cards_with_measured_values(self) -> None:
        from videodna.report import _render_remake_md
        md = _render_remake_md(self._fake_blueprint())
        self.assertIn("Shot #01", md)
        self.assertIn("`opening`", md)
        self.assertIn("`soft_transition`", md)
        self.assertIn("#080808", md)
        self.assertIn("24.31", md)
        self.assertIn("4.98", md)
        self.assertIn("text not extracted (no OCR available)", md)
        self.assertNotIn("BLOOD", md)

    def test_render_shots_md_shows_cards_with_measured_values(self) -> None:
        from videodna.report import _render_shots_md
        md = _render_shots_md(self._fake_blueprint())
        self.assertIn("Shot Cards", md)
        self.assertIn("Shot #02", md)
        self.assertIn("#F8F8F8", md)
        self.assertIn("42.08", md)
        self.assertIn("text not extracted (no OCR available)", md)
        self.assertNotIn("BLOOD", md)


class TestEndToEndV202(unittest.TestCase):
    def setUp(self) -> None:
        import os
        base_tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        self.tmp = Path(tempfile.mkdtemp(prefix="videodna_v202_qa_", dir=base_tmp))

    def tearDown(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_u4_low_confidence_verdict_is_uncertain_end_to_end(self) -> None:
        if not U4.exists():
            self.skipTest(f"U4 video missing: {U4}")
        out = self.tmp / "u4"
        bp = _analyze(U4, out)
        self.assertEqual(
            bp["forensics"]["verdict_summary"],
            "code_motion / static_zoom — uncertain from pixels alone",
        )
        self.assertLess(bp["forensics"]["overall_confidence"], 0.5)
        self.assertTrue(bp["forensics"]["verdict_uncertain"])
        self.assertEqual(bp["forensics"]["runner_up"], "static_zoom")
        self.assertNotIn("code_motion 100%", bp["forensics"]["verdict_summary"])
        # Diagnostics passthrough carries the same honest wording.
        self.assertEqual(
            bp["diagnostics"]["production_verdict_auto"],
            "code_motion / static_zoom — uncertain from pixels alone",
        )
        dna = (out / "DNA.md").read_text(encoding="utf-8")
        self.assertIn("code_motion / static_zoom — uncertain from pixels alone", dna)
        self.assertNotIn("code_motion 100%", dna)

    def test_u2_shot_cards_measured_and_no_invented_text(self) -> None:
        if not U2.exists():
            self.skipTest(f"U2 video missing: {U2}")
        out = self.tmp / "u2"
        bp = _analyze(U2, out)
        # U2 verdict must stay assertive (conf 0.78).
        self.assertEqual(bp["forensics"]["verdict_summary"], "code_motion 100%")
        self.assertEqual(bp["shots"]["shot_count"], 4)
        self.assertEqual(len(bp["remake_plan"]["shot_cards"]), 4)
        for card in bp["remake_plan"]["shot_cards"]:
            self.assertEqual(len(card["palette_hex"]), 3)
            self.assertTrue(all(h.startswith("#") and len(h) == 7 for h in card["palette_hex"]))
            self.assertIsNotNone(card["motion_mean_frame_diff"])
            self.assertEqual(card["on_screen_text"], "text not extracted (no OCR available)")
        for fname in ("dna/remake_plan.md", "dna/shots.md"):
            text = (out / fname).read_text(encoding="utf-8")
            self.assertIn("text not extracted (no OCR available)", text)
            # U2 shows a big BLOOD title on screen; with no OCR installed
            # the report must NOT claim that (or any) on-screen text.
            self.assertNotIn("BLOOD", text)
        # DNA.md carries only a pointer to the cards, not the cards themselves.
        dna_text = (out / "DNA.md").read_text(encoding="utf-8")
        self.assertIn("Per-Shot Content Cards", dna_text)
        self.assertNotIn("BLOOD", dna_text)
        remake_md = (out / "dna" / "remake_plan.md").read_text(encoding="utf-8")
        self.assertIn("Shot #01", remake_md)
        self.assertIn("Shot #04", remake_md)
        self.assertIn("`opening`", remake_md)
        self.assertIn("`hard_cut`", remake_md)

    def test_high_confidence_verdicts_unchanged_end_to_end(self) -> None:
        # U1 0.73, test.mp4 0.72, M3 0.71, multi.mp4 0.59 (baseline values):
        # the verdict string must stay exactly "code_motion 100%".
        # (U1's shot count/confidence may shift while a parallel fix to
        # shot segmentation lands — only the verdict string is asserted.)
        missing = [n for n, p in (("u1", U1), ("test", TEST_MP4), ("m3", M3_MP4),
                                   ("multi", MULTI_MP4)) if not p.exists()]
        if missing:
            self.skipTest(f"videos missing: {missing}")
        for name, path in (("u1", U1), ("test", TEST_MP4), ("m3", M3_MP4),
                           ("multi", MULTI_MP4)):
            bp = _analyze(path, self.tmp / name)
            self.assertEqual(
                bp["forensics"]["verdict_summary"], "code_motion 100%",
                f"{name} verdict changed: {bp['forensics']['verdict_summary']!r}",
            )
            self.assertGreaterEqual(bp["forensics"]["overall_confidence"], 0.5, name)
            self.assertFalse(bp["forensics"].get("verdict_uncertain"), name)


if __name__ == "__main__":
    unittest.main()
