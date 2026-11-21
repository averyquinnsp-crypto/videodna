# VideoDNA Changelog

## v2.0.2 — 2026-11-21

Seven verified weaknesses from the 2026-11-21 head-to-head (7 videos, tools-only) fixed and regression-tested. Test suite: **61 passed, 2 skipped** (the 2 skips are the pre-existing regression-video skips — template/style-B videos absent). All fixes verified end-to-end on the 7 test videos with the repo venv binary.

### Fixed

1. **Installer PATH/upgrade bug (`install.sh`)** — the shim lands in `~/.local/bin`, which is not on PATH on many systems, and an older `videodna` elsewhere on PATH shadows it, so after a "successful" install plain `videodna --version` still reported the old version. The old PATH warning only fired when `command -v videodna` failed, which never happens for upgraders. `install.sh` now detects an existing `videodna` on PATH outside `~/.local/bin` and prints a LOUD warning naming the path and detected version, stating it will shadow the new install, with three fixes (remove it, rename it, or put `~/.local/bin` first on PATH). It never silently overwrites a binary outside its own paths. Sandbox-tested both ways (warning fires with a fake old binary; no false warning without one).
2. **BPM octave fold (`videodna/audio.py`, `_estimate_bpm`)** — 180 BPM click trains read as 60.0 BPM (conf 0.91): the 3× period sat at the window's max-lag edge and out-scored the true period. The estimator now checks sub-multiples (lag/3, lag/2) and prefers the smallest lag whose autocorrelation is ≥70% of the peak, with parabolic interpolation around the chosen lag. Click trains now read 60.0 / 120.0 / **178.7** BPM. No regressions: sine tone still test_tone with BPM None; clicks in the full pipeline still get no BPM (speech_only suppression); beat120 still no BPM (onset conf 0.06 < 0.14 floor, documented).
3. **Motion-graphics shot over-segmentation (`videodna/shots.py`)** — big animated text/elements inside one shot were counted as soft transitions (U1 AutoAE promo: 4 shots for 1 continuous shot; U3 Zuckerberg/Meta: 3 for 2). New layout/background-continuity guard on frame-diff (soft) candidates only — hard cuts untouched: (a) candidates within 0.75s of start/end are suppressed (no settled before/after state); (b) changed-fraction rule comparing settled frames 0.75s before vs after (pixel change >25, threshold 0.45): below = same layout, suppress; above = keep. U1 is now **1 shot**, U3 **2 shots**. No regressions: M3 textured fade+slide stays 3 (both soft transitions kept), multi.mp4 stays 2, test.mp4 stays 1, U2 stays 4, U4 stays 1. Guard reports `detection.frame_diff_suppressed_by_guard` and fails open if frame-grabbing fails.
4. **Audio classifier called music "speech" (`videodna/audio.py`, `_classify_audio_content`)** — U4 (music audio, no speech) was classified speech_plus_bgm 115.4 BPM. The firing branch now requires pause structure (`low_energy_frac ≥ 0.05` or `dead_air_ratio ≥ 0.02`); U4 has pause ratio 0.005 and dead air 0.0, so it now reads **ambient_or_sfx, BPM None**. No regressions: U2's digital-silence track still silent_no_audio; test.mp4 sine still test_tone, BPM None; clicks still no BPM in pipeline.
5. **Doctor cosmetic bug (`videodna/cli.py`)** — `doctor` reported `requests` MISSING, but nothing in the code imports requests (it is only an optional `[ai]` extra). The check is dropped; doctor now passes clean (ffmpeg/ffprobe/git/yt-dlp/numpy/PIL all OK).
6. **Remake plan had no per-shot content (`videodna/remake.py`, `videodna/report.py`, `videodna/forensics.py`)** — the plan was one generic batch with no per-shot description. Every shot now gets a measured **shot card** in `dna/remake_plan.md`, `dna/shots.md`, and `blueprint.json`: window/duration, transition-in, production call + confidence, dominant colors (measured from the mid-shot frame), motion level (mean frame diff, 0–255 scale, units labeled), and on-screen text **only if actually extracted**. Nothing is invented. OCR is NOT available on this machine (no tesseract binary; no rapidocr/easyocr/pytesseract in either Python), so no OCR dependency was added — cards honestly state `text not extracted (no OCR available)`.
7. **Verdict honesty at low confidence (`videodna/forensics.py`)** — U4 was labelled `code_motion 100%` at confidence 0.43 though it is a 3D render. When the top production confidence is < 0.5, the verdict now says so using the real runner-up: U4 reads **`code_motion / static_zoom — uncertain from pixels alone`** (conf 0.43). At conf ≥ 0.5, or when the top call is already `unknown`, wording is byte-identical to before. No 3D classifier was invented.

### Before / after (7 test videos, v2.0.1 → v2.0.2)

| Video | Shots | Verdict | Audio | Retention |
|---|---|---|---|---|
| U1 AutoAE promo (truth 1) | 4 → **1** | code_motion 100% (conf 0.73 → 0.88) | silent | 8.2 → 6.8 |
| U2 statue/BLOOD (truth 4) | 4 → 4 | code_motion 100% 0.78 | silent | 8.2 → 8.2 |
| U3 Zuckerberg/Meta (truth 2) | 3 → **2** | unknown 0.30 | silent | 9.0 → 8.2 |
| U4 3D "?" (truth 1, 3D render) | 1 → 1 | code_motion 100% 0.43 → **code_motion / static_zoom — uncertain from pixels alone** | speech_plus_bgm 115.4 → **ambient_or_sfx, BPM None** | 6.7 → 6.7 |
| test.mp4 (truth 1) | 1 → 1 | code_motion 100% 0.72 | test_tone | 6.0 → 6.0 |
| multi.mp4 (truth 2, dissolve) | 2 → 2 | code_motion 100% 0.59 | silent | 8.2 → 8.2 |
| M3 textured fade+slide (truth 3) | 3 → 3 | code_motion 100% 0.71 | silent | 8.2 → 8.2 |

Notes: U1/U3 retention changed only because the shot counts are now correct (retention is a heuristic derived from pacing). U1's verdict confidence moved 0.73 → 0.88 for the same reason (forensics re-scored on 1 shot instead of 4).

### Known remaining work (not fixed in v2.0.2)

- **OCR**: no tesseract/rapidocr/easyocr on this machine. Shot cards report the gap honestly; adding OCR means installing one of those and filling `on_screen_text` only from real extraction output.
- **Speech gate validation**: no real speech sample exists on this VM, so the pause gate is validated only in the negative direction (continuous music no longer called speech). A real speech / speech+bgm sample could still be affected if its pause ratio falls under 0.05 with dead air under 0.02. Also, the second classifier branch can still return `speech_plus_bgm` for continuous audio when `bpm_conf ≥ 0.25` or `bass ≥ 0.22` with `speech_e ≥ 0.45` — only the branch that fired for U4 was gated.
- **Shot-guard thin margin**: U2's 2.5s overlay scores 0.591 changed-fraction vs the 0.45 line. The guard is tuned on these 7 videos; a future motion-graphics video with a larger in-shot overlay (or a subtler real transition) could land near the line. The new regression tests pin all 7 counts so drift fails loudly.
- **PATH shadow still needs a human**: the installer now warns loudly, but the stale `/usr/local/bin/videodna` (v2.0.0) is still on this VM and still shadows the venv install for a plain `videodna` call. Removing/renaming it is a user decision. Note the existing test suite shells out to bare `videodna` in one test, so run the suite with `.venv/bin` first on PATH (`PATH="$PWD/.venv/bin:$PATH"`), otherwise that one test fails against the old binary — a live demonstration of the bug itself.
- **BPM precision**: click180 reads 178.7, not exactly 180 (1.9% off — the 50 fps onset grid plus interpolation).

## v2.0.1 — 2026-11-21

Audio classifier tuning (test_tone detection via dominant-bin + spectral flatness, onset floor for BPM, BPM only for music, trailing silence to EOF), forensics tie-break fix (no rule fired / near-tie → "unknown" 0.30 instead of silent "live"), phantom tools removed ("voice-pro" → "Cartesia voice (or your TTS of choice)"; review template now routes Flow/Veo/Kling, HyperFrames/Remotion, Cartesia, Kaggle + LUT), install.sh rewritten properly (venv + apt deps + ~/.local/bin shim — correct PEP 668 fix). 26 tests: 25 pass + 2 skipped. Known open bugs carried into v2.0.2's fix list: upgrade/PATH shadowing, BPM octave fold (180 → 60).

## v2.0.0 — 2026-11-21

Mathematical shot forensics (Laplacian sensor noise, flat ratio, color entropy, rectilinear edges, FFT planar coherence, duplicate frames, softmax capped 88%), Bayesian review fusion (35% tool + 65% agent), smart audio classifier + BPM, model-aware remake batching, new `moments` command, `dna/remake_plan.md`.

## v1.1.0 — 2026-11-21

Two-pass shot detection (scdet + frame-diff for wipes/fades/slides), honest scoring (silent excluded from audio score, dead-zones motion-aware, score labeled heuristic), probe word-boundary fix, --transcribe via Faster-Whisper, yt-dlp venv discovery, tighter comment patterns, vault views/likes, real routings (Cartesia, no phantom voice-pro/Stickman), 19-test stdlib suite, full Apache LICENSE.
