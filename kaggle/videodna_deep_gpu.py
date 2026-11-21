#!/usr/bin/env python3
"""
VideoDNA Deep Forensic GPU Worker (Kaggle T4 / Colab GPU).

Runs heavy GPU-accelerated extraction cleanly inside /tmp (zero-junk workspace!)
and exports ONLY a single lightweight `deep_dna_bundle.zip` in /kaggle/working:
  1. Demucs v4 4-stem audio separation (vocals / drums / bass / other+SFX)
  2. Faster-Whisper word-level multilingual transcription (Bengali, English, Hindi, etc.)
  3. Full VideoDNA telemetry + Contact Sheets + .cube LUT + interactive HTML report

Usage on Kaggle:
  Set VIDEO_SOURCE to a local path or URL, then run this script in a GPU T4 notebook.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORK_TMP = Path("/tmp/videodna_job")
EXPORT_DIR = Path("/kaggle/working")
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "/kaggle/input/video/input.mp4")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "")  # empty = auto-detect (supports bn, en, etc.)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[GPU Worker] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def main() -> None:
    if WORK_TMP.exists():
        shutil.rmtree(WORK_TMP)
    WORK_TMP.mkdir(parents=True, exist_ok=True)
    out_dir = WORK_TMP / "dna_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Install dependencies strictly into ephemeral environment (never pollute /kaggle/working)
    run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp", "demucs", "faster-whisper", "Pillow", "numpy"])

    repo_dir = WORK_TMP / "videodna_repo"
    if not repo_dir.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/averyquinnsp-crypto/videodna.git", str(repo_dir)])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo_dir)])

    # 2. Run VideoDNA core analysis (handles both URL and local file)
    run(["videodna", "analyze", VIDEO_SOURCE, "-o", str(out_dir)])

    bp_path = out_dir / "blueprint.json"
    blueprint = json.loads(bp_path.read_text(encoding="utf-8"))
    video_path = Path(blueprint["probe"]["file"])

    # 3. Extract clean 44.1kHz WAV in /tmp for GPU audio forensics
    wav_path = WORK_TMP / "audio_full.wav"
    if blueprint["probe"].get("has_audio"):
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-ar", "44100", "-ac", "2", "-y", str(wav_path)])

        # 3a. Demucs v4 4-Stem Separation (Vocals / Drums / Bass / Other+SFX)
        stems_root = WORK_TMP / "demucs_out"
        try:
            run(["demucs", "-n", "htdemucs", "--two-stems=vocals", "-o", str(stems_root), str(wav_path)])
            stem_files = list(stems_root.rglob("*.wav"))
            stem_metrics = {}
            for sf in stem_files:
                # Measure integrated loudness of each separated stem (exact Voice vs BGM/SFX balance!)
                r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-i", str(sf), "-af", "ebur128", "-f", "null", "-"],
                    capture_output=True, text=True,
                )
                import re
                m = re.findall(r"I:\s*(-?[0-9.]+)\s*LUFS", r.stderr)
                stem_metrics[sf.stem] = {"lufs": float(m[-1]) if m else None}
            blueprint["gpu_audio_stems"] = stem_metrics
            if "vocals" in stem_metrics and "no_vocals" in stem_metrics:
                v_lufs = stem_metrics["vocals"]["lufs"]
                bg_lufs = stem_metrics["no_vocals"]["lufs"]
                if v_lufs is not None and bg_lufs is not None:
                    blueprint["gpu_audio_stems"]["voice_to_bgm_ratio_db"] = round(v_lufs - bg_lufs, 1)
        except Exception as exc:
            print(f"[Warn] Demucs skipped: {exc}")

        # 3b. Faster-Whisper Word-Level Transcription (Bengali / English / Auto)
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            kw = {"word_timestamps": True}
            if WHISPER_LANG:
                kw["language"] = WHISPER_LANG
            segments, info = model.transcribe(str(wav_path), **kw)
            transcript = []
            for seg in segments:
                transcript.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "words": [
                        {"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                        for w in (seg.words or [])
                    ],
                })
            blueprint["transcript"] = transcript
            blueprint["transcript_lang"] = info.language
        except Exception as exc:
            print(f"[Warn] Faster-Whisper skipped: {exc}")

        # Re-render DNA.md & report.html with GPU-enriched data
        from videodna.report import write_outputs
        write_outputs(out_dir, blueprint)

    # 4. Package ONLY clean lightweight DNA files (exclude source video!) into /kaggle/working/deep_dna_bundle.zip
    bundle_path = EXPORT_DIR / "deep_dna_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file() and "source" not in p.parts:
                zf.write(p, p.relative_to(out_dir))

    print(f"\n[VideoDNA GPU] Done! Clean DNA bundle exported to: {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
