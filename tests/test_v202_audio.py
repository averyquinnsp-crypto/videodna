"""v2.0.2 audio fixes: BPM octave fold + music-called-speech classifier gate.

Run from the repo root:
    .venv/bin/python -m pytest tests/test_v202_audio.py -q
    .venv/bin/python -m unittest tests.test_v202_audio -v

Fixture files live in ~/workspace/videodna-test/ and in the user's media
library; every test that needs a real file SKIPs gracefully when it is
absent, so the suite stays green on a fresh checkout.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from videodna.audio import _classify_audio_content, _estimate_bpm, analyze_audio

SR = 16000
FIXTURES = Path("/home/hatch/workspace/videodna-test")
U4 = Path(
    "/home/hatch/workspace/user/media_library/video/7c/"
    "7c4d22b0e260da7b9664be6476af726f3225cc1fb8a6e4b981f0fc56e39d5152.mp4"
)
U2 = Path(
    "/home/hatch/workspace/user/media_library/video/5e/"
    "5ea5bf13f40626e7b1253d0d5bb517aaa79a068602dc067f20db91f80dd7b496.mp4"
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


def _synthetic_click_train(bpm: float, dur: float = 12.0, sr: int = SR) -> np.ndarray:
    """Regular click train: 30ms 1kHz bursts on the beat, silence between."""
    n = int(sr * dur)
    sig = np.zeros(n, dtype=np.float32)
    burst_len = int(sr * 0.03)
    burst = np.sin(2.0 * np.pi * 1000.0 * np.arange(burst_len) / sr).astype(np.float32)
    burst *= np.hanning(burst_len).astype(np.float32)
    period = int(round(sr * 60.0 / bpm))
    for start in range(0, n - burst_len, period):
        sig[start:start + burst_len] += burst
    return sig


class TestBpmOctaveFold(unittest.TestCase):
    """v2.0.1 read a 180 BPM click train as 60.0 BPM (conf 0.91): the 3x
    period sits at the autocorrelation window's max-lag edge and
    out-scores the fundamental. The estimator must now prefer the
    fundamental when a sub-multiple lag is comparably strong."""

    def test_synthetic_click_trains(self):
        for bpm in (60.0, 120.0, 180.0):
            est, conf = _estimate_bpm(_synthetic_click_train(bpm), SR)
            self.assertIsNotNone(est, f"{bpm} BPM train gave no BPM")
            self.assertAlmostEqual(est, bpm, delta=4.0,
                                   msg=f"{bpm} BPM train read as {est}")
            self.assertGreaterEqual(conf, 0.5)

    def test_fixture_click_trains(self):
        for bpm, name in ((60.0, "click60"), (120.0, "click120"), (180.0, "click180")):
            path = FIXTURES / f"{name}.mp4"
            if not path.exists():
                self.skipTest(f"fixture missing: {path}")
            est, conf = _estimate_bpm(_decode(path), SR)
            self.assertIsNotNone(est, f"{name}.mp4 gave no BPM")
            self.assertAlmostEqual(est, bpm, delta=4.0,
                                   msg=f"{name}.mp4 read as {est}")
            self.assertGreaterEqual(conf, 0.5)

    def test_beat120_has_no_bpm(self):
        """Documented v2.0.1 behaviour, unchanged: beat120.mp4's onset
        confidence (0.06) is below the 0.14 floor, so no BPM is reported."""
        path = FIXTURES / "beat120.mp4"
        if not path.exists():
            self.skipTest(f"fixture missing: {path}")
        bpm, conf = _estimate_bpm(_decode(path), SR)
        self.assertIsNone(bpm)
        self.assertLess(conf, 0.14)

    def test_sine_tone_has_no_bpm(self):
        path = FIXTURES / "sine440.mp4"
        if not path.exists():
            self.skipTest(f"fixture missing: {path}")
        bpm, conf = _estimate_bpm(_decode(path), SR)
        self.assertIsNone(bpm)
        self.assertEqual(conf, 0.0)


class TestClassifierPauseGate(unittest.TestCase):
    """v2.0.1 called U4 (purple 3D character, 6s, music audio, NO speech)
    speech_plus_bgm at 115.4 BPM: it fired the third classifier branch
    (speech_e 0.572, bpm_conf 0.19) even though the track has no pause
    structure at all (pause ratio 0.005, dead air 0.0). The branch now
    requires pause structure before any track may be called speech."""

    def test_u4_not_speech(self):
        if not U4.exists():
            self.skipTest(f"U4 file missing: {U4}")
        samples = _decode(U4)
        bpm, bpm_conf = _estimate_bpm(samples, SR)
        audio_type, feats = _classify_audio_content(samples, SR, 0.0, bpm_conf)
        self.assertNotEqual(audio_type, "speech_plus_bgm")
        self.assertNotEqual(audio_type, "speech_only")
        self.assertIn(audio_type, ("music_only", "ambient_or_sfx"))
        # The gate inputs are what they were in v2.0.1: continuous audio
        # with speech-band energy — the fix is the gate, not new features.
        self.assertLess(feats["modulation_pause_ratio"], 0.05)

    def test_u4_full_pipeline(self):
        if not U4.exists():
            self.skipTest(f"U4 file missing: {U4}")
        with tempfile.TemporaryDirectory() as td:
            res = analyze_audio(U4, 6.0, [], Path(td))
        self.assertNotEqual(res["audio_type"], "speech_plus_bgm")
        self.assertIn(res["audio_type"], ("music_only", "ambient_or_sfx"))
        self.assertIsNone(res["estimated_bpm"])

    def test_u2_silence_still_silent(self):
        """U2's audio track is digital silence (max amp 0.0)."""
        if not U2.exists():
            self.skipTest(f"U2 file missing: {U2}")
        with tempfile.TemporaryDirectory() as td:
            res = analyze_audio(U2, 10.0, [], Path(td))
        self.assertTrue(res["is_silent_track"])
        self.assertEqual(res["audio_type"], "silent_no_audio")
        self.assertIsNone(res["estimated_bpm"])

    def test_test_mp4_sine_still_test_tone(self):
        path = FIXTURES / "test.mp4"
        if not path.exists():
            self.skipTest(f"fixture missing: {path}")
        with tempfile.TemporaryDirectory() as td:
            res = analyze_audio(path, 12.0, [], Path(td))
        self.assertEqual(res["audio_type"], "test_tone")
        self.assertIsNone(res["estimated_bpm"])
        self.assertEqual(res["bpm_confidence"], 0.0)

    def test_clicks_still_no_bpm_in_full_pipeline(self):
        """Click trains classify as speech_only in the pipeline, so BPM
        is suppressed even though the estimator now reads them correctly
        when called directly."""
        for name in ("click60", "click120", "click180"):
            path = FIXTURES / f"{name}.mp4"
            if not path.exists():
                self.skipTest(f"fixture missing: {path}")
            with tempfile.TemporaryDirectory() as td:
                res = analyze_audio(path, 12.0, [], Path(td))
            self.assertIsNone(res["estimated_bpm"], f"{name}.mp4 leaked a BPM")
            self.assertEqual(res["bpm_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
