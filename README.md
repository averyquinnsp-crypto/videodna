<div align="center">

# 🧬 VideoDNA

**10x Video Reverse-Engineering, Viral Retention X-Ray & Beat-The-Original Blueprint Engine**

Built for **BOTH Short-Form** (Reels / Shorts / TikTok) and **Long-Form** (YouTube) videos.
Zero heavy native dependencies — runs smoothly on **Termux (Android), Linux, macOS, Windows, and Kaggle T4 GPU**.

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Agents](https://img.shields.io/badge/agents-Antigravity%20%7C%20Claude%20Code%20%7C%20Codex-8A2BE2)

</div>

---

## Why VideoDNA v2.0?

**VideoDNA v2.0** fuses **offline mathematical shot-level production forensics** with **Viral Retention X-Ray & Beat-The-Original strategy** — zero heavy native dependencies, runs smoothly on Termux, Linux, macOS, Windows, and Kaggle GPU:

1. **Mathematical Shot-Level Production Forensics (Offline, Free):** Measures **sensor noise residual, flat-region ratio, color entropy, rectilinear UI edges, effective frame cadence, and 2D FFT planar motion coherence** per shot to classify **`live` / `ai_video` / `3d_render` / `code_motion` / `screen_rec` / `static_zoom`** with calibrated **Confidence %** BEFORE visual review!
2. **Two-Pass Shot & Transition Detector (`scdet` + `frame_diff`):** Catches BOTH frame-accurate hard cuts AND smooth wipes/fades/slides (4/4 verified ground-truth accuracy on motion-graphics templates).
3. **Smart Audio Forensics & BPM Estimator:** Accurately separates **Silent/Mute Tracks** (excluded from audio penalties — zero false alarms!), **Speech-only**, **Music-only**, and **Speech + BGM**, measures **LUFS**, estimates **BPM**, scores **Cut-to-Beat Sync**, and supports local `--transcribe` via Faster-Whisper.
4. **Bayesian Evidence Fusion (`apply-review`):** Merges deterministic tool probabilities with AI Agent visual inspection and flags tool-vs-eye conflicts automatically.
5. **Model-Aware Remake Batch Planner (`dna/remake_plan.md`):** Batches shots into minimal production units across **Google Flow (Veo), Kling, Seedance, MiniMax, HyperFrames/Remotion, and Live Shoot** + exports **SRT captions** and a **3D `.cube` Color LUT**.
6. **Dual-Brain Short & Long Mode + Audience Comment Mining + Beat-The-Original Blueprint + Style Vault (`compare` & `fuse`) + Highlight `moments` zoomer!**

---

## Quick Start (Any Account / Any AI Agent)

Tell **Antigravity, Claude Code, Codex, or Gemini CLI**:
```text
Install https://github.com/averyquinnsp-crypto/videodna, then analyze this video and tell me how to beat it: /path/to/video.mp4
```

Or install manually in 10 seconds:
```bash
git clone https://github.com/averyquinnsp-crypto/videodna.git
cd videodna
bash install.sh
```

> **Requirements:** `python3 >= 3.10`, `ffmpeg` / `ffprobe`, `numpy`, `Pillow`.
> **Optional (for direct YouTube/Shorts/TikTok URLs + Comment Mining):** `pip install yt-dlp`.
> **Optional (for local speech transcription):** `pip install 'videodna[transcribe]'` (Faster-Whisper).

---

## What's New in v1.1.0

- **Two-pass shot detection** — `scdet` (hard cuts) + a frame-diff pass (wipes / fades / slides / dissolves) merged, so code-rendered motion graphics no longer read as one long shot. Verified against rea-video's ground truth on the same video.
- **Silent-aware scoring** — silent videos (motion graphics, text-only demos) are excluded from the audio part of the retention score instead of being punished with 2.0.
- **Motion-aware dead-zones** — a long shot with hard motion (whip-pans, kinetic type) is no longer flagged as a dead-zone.
- **Local transcription** — `videodna analyze video.mp4 --transcribe` (Faster-Whisper, optional extra; graceful message when not installed).
- **Honest forensics** — metadata matching is word-boundary only (no more "wan" inside "taiwan"); `x264`/`mp42` no longer reported as platform signals.
- **Tighter comment mining** — praise ("what a video!") is no longer counted as a question; bare "more"/"please" no longer counted as requests.
- **Vault stores views/likes** — `vault compare` reports the top performer by views next to the averaged formula.
- **Real tool routings** — the skill and review template route to real tools (Cartesia, optional Kaggle upscale); no phantom tool names.
- **Tests** — 19-test stdlib suite (`python -m unittest discover -s tests -v`), including shot-detection regression tests.
- Full Apache-2.0 license text.

> **Honest limits:** the retention score is a rule-of-thumb heuristic, not measured audience retention — the telemetry (cuts, ASL, LUFS, dead-air, motion) is the truth. Forensic "AI detection" reads container metadata tags only; re-encoded videos usually have them stripped. The `.cube` LUT is a tonal approximation, not a true color transfer.

---

## CLI Commands

```bash
# 1. Analyze a local MP4 or direct YouTube/Shorts/TikTok URL (auto Short/Long mode)
videodna analyze /path/to/video.mp4 -o out/my_video --save-vault
videodna analyze "https://youtube.com/shorts/..." -o out/viral_short --save-vault

# 1b. Transcribe speech locally with Faster-Whisper (optional extra)
videodna analyze /path/to/video.mp4 --transcribe -o out/my_video

# 2. Zoom into any specific moment (e.g., 2.0s to 4.4s every 0.2s)
videodna frames /path/to/video.mp4 --start 2.0 --end 4.4 --every 0.2 -o out/my_video/zoom.jpg

# 3. Merge AI Brain visual review & Beat-The-Original upgrades into DNA.md + report.html
videodna apply-review out/my_video --save-vault

# 4. Channel Pattern Cracker & Hybrid DNA Fusion
videodna vault list
videodna vault compare
videodna vault fuse --hook video_a --pacing video_b --audio video_c

# 5. Check environment health
videodna doctor
```

---

## Outputs Generated (`out/<name>/`)

| File | Purpose |
|---|---|
| **`DNA.md`** | Compact layered brief (~1.5k–2.5k tokens) designed for AI Agents |
| **`report.html`** | Interactive offline dark-mode visual dashboard for humans |
| **`blueprint.json`** | Complete machine-readable telemetry & remake specification |
| **`hook_strip.jpg`** | Opening critical seconds frame strip (stop-the-scroll visual audit) |
| **`storyboard.jpg`** | Labeled keyframe contact sheet with shot durations & motion classes |
| **`audio_timeline.png`** | Audio energy curve + red dead-air bars + yellow visual cut ticks |
| **`reference_grade.cube`** | Auto-generated 17×17×17 3D color LUT matching the reference look |
| **`dna/shots.md`** | Full shot table & retention dead-zone alerts |
| **`dna/comments.md`** | Mined audience complaints, questions, requests & timestamp callouts |
| **`dna/transcript.md`** | Timestamped spoken script |

---

## বাংলায় দ্রুত গাইড (Quickstart in Bengali)

যেকোনো নতুন অ্যাকাউন্ট বা AI এজেন্টে (Antigravity / Claude Code / Codex) শুধু বলো:
```text
https://github.com/averyquinnsp-crypto/videodna ইনস্টল করো, তারপর এই ভিডিওটার পুরো DNA বের করে অরিজিনালের চেয়ে বেটার বানানোর প্ল্যান দাও: /path/to/video.mp4
```
এটি শর্টস এবং লং—দুই ধরনের ভিডিওতেই কাজ করে, কোনো আলাদা পেইড API ছাড়াই!

---

## License

Apache-2.0 — free to use, fork, and build upon.
