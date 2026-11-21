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

## Why VideoDNA?

Traditional video breakdown tools only give you a passive forensic report (*"this shot is AI, that shot is live"*).
**VideoDNA** pairs a **free, exact X-Ray Engine** (Python + `ffmpeg`) with your **AI Agent as the Brain** (`SKILL.md`) to uncover:

1. **Full Production Forensics ("Naari-Bhuri"):** Shot-by-shot cuts, pacing velocity, motion energy, color palette + auto-generated **3D `.cube` LUT**, audio loudness (`LUFS`), dead-air silences, beat/SFX transients, cut-to-beat sync ratio, and metadata/cadence clues.
2. **Dual-Brain Short & Long Mode:**
   - **Short-Form Mode (≤ 3 min):** Microscope audit — 0–5s Hook Strip (every 0.5s), hyper-cut detection, pattern-interrupt cadence, loop check.
   - **Long-Form Mode (> 3 min):** Hierarchical Macro Acts + Critical Intro spotlight + Dead-Zone drop-off detector (so even a 60-minute video stays crystal-clear in minimal AI tokens).
3. **Audience Comment Mining (URL Mode):** Automatically pulls Top 50–100 viewer comments (`yt-dlp`) and separates **Complaints, Unanswered Questions, Part-2 Requests, and Timestamp Callouts** — giving you real viewer proof of what the original missed!
4. **🚀 Beat-The-Original 10x Upgrade Blueprint:** Diagnoses exact drop-off weaknesses and outputs **3 stronger hook variations**, timestamped pacing surgery, visual/audio upgrades, and remake routing (Google Flow / Veo, HyperFrames / Remotion, Voice-Pro, Kaggle 2K).
5. **🧬 Style Vault, Channel Pattern Cracker & Hybrid DNA Fusion:** Save winning videos to your vault, compare multiple videos to extract shared niche formulas (`videodna vault compare`), or fuse Hook A + Pacing B + Audio C (`videodna vault fuse`).
6. **🔥 Hybrid Termux + Kaggle GPU Support:** Runs instantaneously on phone/PC in Lite Mode, and ships with `kaggle/videodna_deep_gpu.py` for GPU-accelerated **Demucs v4** (vocal/BGM separation) + **Faster-Whisper** (word-level Bengali/English/multilingual transcription).

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

---

## CLI Commands

```bash
# 1. Analyze a local MP4 or direct YouTube/Shorts/TikTok URL (auto Short/Long mode)
videodna analyze /path/to/video.mp4 -o out/my_video --save-vault
videodna analyze "https://youtube.com/shorts/..." -o out/viral_short --save-vault

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
