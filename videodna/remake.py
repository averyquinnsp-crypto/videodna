"""Model-aware remake batch planner (groups shots into minimal generation/production units)."""

from __future__ import annotations

from typing import Any

# v2.0.4: tesseract-ocr IS installed on this machine (system package,
# 5.3.4 + eng/hin data, installed 2026-11-21 — see videodna/ocr.py).
# Shot cards therefore carry real extracted text when the quality gate
# passes; OCR_TEXT_STATUS below is only the fallback for shots where
# no OCR ran (binary absent) or nothing was extracted. OCR stays
# optional — never a hard dependency.
OCR_TEXT_STATUS = "text not extracted (no OCR available)"

# Units for the measured motion value carried on each shot card.
MOTION_UNITS = "mean abs luma diff per pixel, 0-255 scale, across the burst frames inside the shot"


def build_shot_cards(
    forensics: dict[str, Any],
    fused: dict[str, Any] | None = None,
    shot_frames: dict[int, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build one measured content card per shot for the remake plan.

    Every value comes from the pipeline's own measurements — nothing is
    described or invented:
    - duration / window: shot detector
    - production_type / confidence: per-shot forensics (fused when a visual
      review exists); production_runner_up: second-highest class in the
      same per-shot distribution, so the mix is traceable to its shots
      (v2.0.4 — like rea-video's per-shot top % + runner-up lines)
    - palette_hex: top-3 dominant colors of the shot's mid burst frame
      (4-bit histogram over the real pixels — see forensics._dominant_palette)
    - motion_mean_frame_diff: mean frame-to-frame luma difference inside
      the shot (forensics metrics)
    - frame_paths: first/mid/last/strip JPEGs exported by
      shots.export_shot_frames into dna/frames/ (v2.0.4; {} when the
      export failed or the shot was not exported)
    - on_screen_text: only ever extracted text; tesseract OCR (see
      videodna/ocr.py) fills it when the quality gate passes, "no text
      detected" when OCR ran but found nothing, and OCR_TEXT_STATUS
      when no OCR ran at all.
    """
    from videodna.forensics import _runner_up

    shot_list = (fused or forensics).get("per_shot", [])
    cards: list[dict[str, Any]] = []
    for sf in shot_list:
        m = sf.get("metrics") or {}
        probs = sf.get("fused_probs") or sf.get("probs") or {}
        top = sf.get("fused_type") or sf.get("predicted_type") or "unknown"
        runner_up, runner_prob = _runner_up(probs, top)
        cards.append({
            "shot_id": sf["shot_id"],
            "start": sf["start"],
            "end": sf["end"],
            "duration": sf["duration"],
            "production_type": top,
            "production_confidence": sf.get("fused_confidence") or sf.get("confidence"),
            "production_runner_up": runner_up,
            "production_runner_up_confidence": round(runner_prob, 3) if runner_up else None,
            "needs_visual_review": bool(sf.get("needs_visual_review", False)),
            "transition_subtype": sf.get("transition_subtype"),
            "camera_motion": sf.get("camera_motion", "unknown"),
            "camera_motion_confidence": sf.get("camera_motion_confidence", 0.0),
            "palette_hex": list(sf.get("palette_hex") or []),
            "motion_mean_frame_diff": m.get("mean_frame_diff"),
            "motion_units": MOTION_UNITS,
            "frame_paths": dict((shot_frames or {}).get(sf["shot_id"], {})),
            "on_screen_text": sf.get("on_screen_text") or OCR_TEXT_STATUS,
            "on_screen_text_words": list(sf.get("on_screen_text_words") or []),
        })
    return cards


MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "code_motion": {
        "recommended_tool": "HyperFrames (HTML/CSS/GSAP) or Remotion",
        "max_batch_sec": 60.0,
        "cost_tier": "free_local",
        "strategy": "Combine consecutive motion-graphics / text / template shots into a single unified timeline composition.",
    },
    "ai_video": {
        "recommended_tool": "Google Flow (Veo 3) / Kling 2.1 / Seedance / MiniMax",
        "max_batch_sec": 8.0,
        "cost_tier": "ai_credits",
        "strategy": "Batch consecutive AI shots sharing the same scene/palette up to 8s per generation clip using 1 consistent reference image.",
    },
    "3d_render": {
        "recommended_tool": "Blender / Three.js (HyperFrames WebGL) OR Veo/Kling I2V",
        "max_batch_sec": 10.0,
        "cost_tier": "free_or_ai",
        "strategy": "Render clean 3D explainers/products in Blender/Three.js for 100% consistency, or batch stylized 3D shots into 8–10s I2V clips.",
    },
    "live": {
        "recommended_tool": "Live Camera / Phone Rig Shoot",
        "max_batch_sec": 120.0,
        "cost_tier": "real_shoot",
        "strategy": "Film consecutive live shots in 1 continuous setup & lighting rig, then punch-cut in editing.",
    },
    "screen_rec": {
        "recommended_tool": "Fresh Screen Recording (OBS / Phone Capture)",
        "max_batch_sec": 120.0,
        "cost_tier": "free_local",
        "strategy": "Record clean original UI/screen footage directly; overlay zoom & highlights in post.",
    },
    "static_zoom": {
        "recommended_tool": "ImageGen + HyperFrames / FFmpeg Ken Burns",
        "max_batch_sec": 30.0,
        "cost_tier": "free_local",
        "strategy": "Generate high-res still images once, then animate smooth scale/pan/parallax curves in code.",
    },
    "unknown": {
        "recommended_tool": "Visual review first — classify the shot, then route it",
        "max_batch_sec": 8.0,
        "cost_tier": "review_first",
        "strategy": "No forensic signal fired for these shots — inspect them visually, set the real production type in the agent review, then re-run apply-review to re-batch.",
    },
}


def build_remake_plan(
    forensics: dict[str, Any],
    audio: dict[str, Any],
    color_dna: dict[str, Any],
    fused: dict[str, Any] | None = None,
    shot_frames: dict[int, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Group consecutive same-type shots into minimal production batches."""
    shot_list = (fused or forensics).get("per_shot", [])
    batches: list[dict[str, Any]] = []

    for sf in shot_list:
        stype = sf.get("fused_type") or sf.get("predicted_type") or "code_motion"
        cap = MODEL_CAPABILITIES.get(stype, MODEL_CAPABILITIES["code_motion"])
        max_dur = float(cap["max_batch_sec"])
        dur = float(sf["duration"])

        if (
            batches
            and batches[-1]["production_type"] == stype
            and (batches[-1]["total_duration"] + dur) <= max_dur
        ):
            b = batches[-1]
            b["shot_ids"].append(sf["shot_id"])
            b["end"] = sf["end"]
            b["total_duration"] = round(b["total_duration"] + dur, 2)
        else:
            batches.append({
                "batch_id": len(batches) + 1,
                "production_type": stype,
                "tool": cap["recommended_tool"],
                "cost_tier": cap["cost_tier"],
                "shot_ids": [sf["shot_id"]],
                "start": sf["start"],
                "end": sf["end"],
                "total_duration": round(dur, 2),
                "strategy": cap["strategy"],
            })

    # Audio & Finish routing (real tools only — no phantom names)
    audio_type = audio.get("audio_type", "unknown")
    if audio_type == "silent_no_audio":
        audio_plan = "Original track is SILENT — add fresh studio voiceover + dark/matching BGM + beat-synced SFX mastered to -14 LUFS with Cartesia voice (or your TTS of choice)."
    elif audio_type == "test_tone":
        audio_plan = "Original audio is a pure test tone / drone (no real content) — add fresh voiceover + BGM mastered to -14 LUFS with Cartesia voice (or your TTS of choice)."
    elif audio_type == "music_only":
        audio_plan = f"Music-driven edit (est. {audio.get('estimated_bpm') or 'N/A'} BPM) — source/generate copyright-safe track matching BPM and align cuts to beats."
    else:
        audio_plan = "Voice + BGM pipeline — record/clone narration with Cartesia (or your TTS of choice) and master to -14 LUFS (tight compression, -12dB BGM ducking)."

    return {
        "total_batches": len(batches),
        "batches": batches,
        "shot_cards": build_shot_cards(forensics, fused=fused, shot_frames=shot_frames),
        "audio_pipeline": audio_plan,
        "color_pipeline": f"Apply auto-generated LUT `{color_dna.get('lut_file')}` ({color_dna.get('grade_style')}) + optional GPU upscale on Kaggle (only when the user asks).",
    }
