"""v2.0.3 audio fix: real continuous speech no longer ambient_or_sfx.

v2.0.2 classified a real 51.85s Hindi reaction/edit video (loud, -7.3 LUFS,
full of continuous dialogue) as ambient_or_sfx, BPM None, because branch 1
(speech_only) required low_energy_frac >= 0.14 and speech_e >= 0.58, and
the sample measured low_energy_frac 0.095 / speech_e 0.583. The fix lowers
only the two branch-1 thresholds (pause 0.14 -> 0.05, speech_e 0.58 ->
0.55); U4 (music, no speech) still fails the pause gate (0.005) and must
stay ambient_or_sfx with BPM None.

These tests use the REAL audio files (no synthetic speech) and assert on
the measured feature vectors, so a threshold regression shows up as a
failing test rather than a silent re-classification.

Run from the repo root:
    .venv/bin/python -m pytest tests/test_v203_audio.py -q
    .venv/bin/python -m unittest tests.test_v203_audio -v

Every test that needs a real file SKIPS gracefully when it is absent, so
the suite stays green on a fresh checkout.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from videodna.audio import _classify_audio_content, _estimate_bpm, analyze_audio

SR = 16000
SPEECH1 = Path(
    "/home/hatch/workspace/user/media_library/video/5c/"
    "5cf429380dfd866ec57af18d0bdeb639aa305cf91b21fee1be895f29b0102322.mp4"
)
U4 = Path(
    "/home/hatch/workspace/user/media_library/video/7c/"
    "7c4d22b0e260da7b9664be6476af726f3225cc1fb8a6e4b981f0fc56e39d5152.mp4"
)


def _decode(path: Path, sr: int = SR) -> np.ndarray:
    """Decode a video's audio track to 16kHz mono float samples."""
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-"],
        capture_output=True,
    )
    return np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0


class TestSpeech1RealAudioIsSpeech(unittest.TestCase):
    """v2.0.3 fix: real Hindi dialogue (speech1) must be a speech class.

    Measured feature vector on this file (v2.0.2, unchanged by the fix):
    speech_e 0.583, bass_e 0.171, low_energy_frac 0.095, dead_air 0.0,
    bpm_conf 0.11. It failed the old 0.14 pause gate and fell through to
    ambient_or_sfx. With the 0.05 gate it passes branch 1 with a +0.045
    margin on the pause gate and +0.033 on speech_e.
    """

    def test_speech1_classifier_returns_speech_class(self):
        if not SPEECH1.exists():
            self.skipTest(f"speech1 file missing: {SPEECH1}")
        samples = _decode(SPEECH1)
        bpm, bpm_conf = _estimate_bpm(samples, SR)
        audio_type, feats = _classify_audio_content(samples, SR, 0.0, bpm_conf)
        # The measured feature vector is what the thresholds act on.
        self.assertAlmostEqual(feats["speech_band_ratio"], 0.583, delta=0.02)
        self.assertAlmostEqual(feats["modulation_pause_ratio"], 0.095, delta=0.02)
        self.assertAlmostEqual(feats["bass_band_ratio"], 0.171, delta=0.02)
        self.assertIn(audio_type, ("speech_only", "speech_plus_bgm"))
        self.assertNotEqual(audio_type, "ambient_or_sfx")
        self.assertNotEqual(audio_type, "music_only")

    def test_speech1_full_pipeline(self):
        if not SPEECH1.exists():
            self.skipTest(f"speech1 file missing: {SPEECH1}")
        with tempfile.TemporaryDirectory() as td:
            res = analyze_audio(SPEECH1, 51.85, [], Path(td))
        self.assertIn(res["audio_type"], ("speech_only", "speech_plus_bgm"))
        self.assertFalse(res["is_silent_track"])
        # No steady beat in the track (bpm_conf 0.11), so still no BPM.
        self.assertIsNone(res["estimated_bpm"])


class TestU4StillNotSpeech(unittest.TestCase):
    """v2.0.3 must NOT change U4: music with no speech stays non-speech.

    Measured feature vector (unchanged from v2.0.2): speech_e 0.572,
    low_energy_frac 0.005, dead_air 0.0, bpm_conf 0.19. The 0.05 pause
    gate is the separator: U4 sits 0.045 BELOW it, speech1 0.045 ABOVE.
    """

    def test_u4_classifier_still_ambient(self):
        if not U4.exists():
            self.skipTest(f"U4 file missing: {U4}")
        samples = _decode(U4)
        bpm, bpm_conf = _estimate_bpm(samples, SR)
        audio_type, feats = _classify_audio_content(samples, SR, 0.0, bpm_conf)
        self.assertAlmostEqual(feats["speech_band_ratio"], 0.572, delta=0.02)
        self.assertAlmostEqual(feats["modulation_pause_ratio"], 0.005, delta=0.02)
        self.assertEqual(audio_type, "ambient_or_sfx")

    def test_u4_full_pipeline_still_ambient_no_bpm(self):
        if not U4.exists():
            self.skipTest(f"U4 file missing: {U4}")
        with tempfile.TemporaryDirectory() as td:
            res = analyze_audio(U4, 6.0, [], Path(td))
        self.assertEqual(res["audio_type"], "ambient_or_sfx")
        self.assertIsNone(res["estimated_bpm"])


class TestClick180EstimatorUnchanged(unittest.TestCase):
    """v2.0.3 touches only branch-1 classifier thresholds; the BPM
    estimator is untouched. click180 must still read ~178.7 directly."""

    def test_click180_estimator(self):
        path = Path("/home/hatch/workspace/videodna-test/click180.mp4")
        if not path.exists():
            self.skipTest(f"fixture missing: {path}")
        est, conf = _estimate_bpm(_decode(path), SR)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est, 178.7, delta=4.0)
        self.assertGreaterEqual(conf, 0.5)


if __name__ == "__main__":
    unittest.main()
