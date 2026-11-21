"""Deterministic Retention Scorecard & Beat-The-Original Diagnostic Engine."""

from __future__ import annotations

from typing import Any


def compute_diagnostics(
    probe: dict[str, Any],
    shots: dict[str, Any],
    visual: dict[str, Any],
    audio: dict[str, Any],
    comments_intel: dict[str, Any] | None = None,
    forensics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score retention signals and flag concrete opportunities to beat the reference video.

    Scores are RULE-OF-THUMB heuristics (fixed ladders + weights), not measured
    retention — the returned telemetry is the truth, the score is a summary.
    """
    mode = shots["mode"]
    duration = max(0.1, probe["duration_sec"])
    weaknesses: list[dict[str, str]] = []
    strengths: list[str] = []
    upgrade_actions: list[str] = []
    notes: list[str] = []

    # 0. Effective dead-zones: a long shot is only a dead-zone if it is ALSO
    # visually quiet. A long shot with hard motion (whip-pans, kinetic type,
    # slides, zooms) holds attention — flagging it as dead is a false positive.
    motion_by_shot = {
        v["shot_id"]: v.get("motion_class", "unknown")
        for v in visual.get("shot_visuals", [])
    }
    effective_dead_zones: list[dict[str, Any]] = []
    for dz in shots["dead_zones"]:
        mc = motion_by_shot.get(dz["shot_id"], "unknown")
        if mc in ("dynamic_action", "high_energy_whip"):
            continue  # moving hard — not dead
        if mc == "gentle_motion" and dz["severity"] == "high":
            dz = {**dz, "severity": "medium"}
        effective_dead_zones.append(dz)

    # 1. Hook Velocity Score (0-10)
    hook_cuts_3s = shots["hook_cuts_0_3s"]
    hook_cuts_10s = shots["hook_cuts_0_10s"]
    if mode == "short":
        if hook_cuts_3s >= 2:
            hook_score = 9.0
            strengths.append(f"High-velocity opening ({hook_cuts_3s} cuts inside first 3s — strong visual pattern interrupt).")
        elif hook_cuts_3s == 1:
            hook_score = 7.5
            strengths.append("Clean single cut inside the first 3 seconds.")
        else:
            hook_score = 4.5
            weaknesses.append({
                "area": "Hook (0-3s)",
                "severity": "high",
                "finding": "Zero visual cuts in the first 3 seconds — high swipe-away risk on Shorts/Reels.",
            })
            upgrade_actions.append(
                "BEAT HOOK: Open with a visual shock/contrast cut at 1.2s–1.8s + bold 3–5 word on-screen hook + subtle Whoosh/Riser SFX."
            )
    else:
        if hook_cuts_10s >= 3:
            hook_score = 9.0
            strengths.append(f"Dynamic YouTube intro ({hook_cuts_10s} cuts in first 10s).")
        elif hook_cuts_10s >= 1:
            hook_score = 7.0
        else:
            hook_score = 4.5
            weaknesses.append({
                "area": "Intro Hook (0-10s)",
                "severity": "high",
                "finding": "Static opening for 10+ seconds — long-form viewers decide in the first 15s.",
            })
            upgrade_actions.append(
                "BEAT INTRO: Add 2–3 quick visual proof B-rolls or kinetic title callouts inside the first 10 seconds to visually confirm the title/thumbnail promise."
            )

    # 2. Pacing & Dead-Zone Score (0-10)
    dead_zones = effective_dead_zones
    asl = shots["avg_shot_length_sec"]
    static_ratio = visual["motion_summary"]["static_shot_ratio"]

    pacing_score = 9.0
    if dead_zones:
        pacing_score -= min(4.5, len(dead_zones) * 1.2)
        first_dz = dead_zones[0]
        weaknesses.append({
            "area": "Pacing / Dead-Zones",
            "severity": "high" if any(d["severity"] == "high" for d in dead_zones) else "medium",
            "finding": f"{len(dead_zones)} long uncut segment(s) detected (e.g., {first_dz['start']}s–{first_dz['end']}s = {first_dz['duration']}s).",
        })
        upgrade_actions.append(
            f"PACING SURGERY: Break every uncut stretch >{shots['dead_zone_threshold_sec']}s with a 1.12x punch-in zoom, B-roll metaphor, or HyperFrames kinetic callout."
        )
    else:
        strengths.append(f"Tight pacing with zero dead-zones (avg shot length: {asl}s, {shots['cuts_per_minute']} cuts/min).")

    if static_ratio >= 0.55:
        pacing_score = max(3.0, pacing_score - 1.5)
        weaknesses.append({
            "area": "Visual Motion",
            "severity": "medium",
            "finding": f"{int(static_ratio * 100)}% of sampled shots are visually static.",
        })
        upgrade_actions.append(
            "MOTION UPGRADE: Add subtle 3–5% slow push-in, floating graphic cards, or parallax motion to static frames."
        )

    # 3. Audio Dynamics & Loudness Score (0-10)
    # Silence is often INTENTIONAL (motion graphics, text-only demos, title
    # cards). Scoring a silent video 2.0 punishes the format, so silent audio
    # is excluded from the overall score instead of treated as a failure.
    lufs = audio.get("integrated_lufs")
    silent_track = (not audio.get("has_audio")) or (lufs is not None and lufs < -50.0)
    audio_excluded = False
    audio_score: float | None = 8.0
    if silent_track:
        audio_score = None
        audio_excluded = True
        notes.append(
            "Silent video (no audible audio track) — audio excluded from the "
            "retention score; silence may be intentional (e.g. motion-graphics demo)."
        )
    else:
        if lufs is not None and lufs < -20.0:
            audio_score -= 2.5
            weaknesses.append({
                "area": "Audio Loudness",
                "severity": "medium",
                "finding": f"Under-leveled audio ({lufs} LUFS) — sounds weak compared to -14 LUFS viral standard.",
            })
            upgrade_actions.append(
                "AUDIO UPGRADE: Master voice & BGM to -14 LUFS with tight compression and BGM ducking under the voice."
            )
        elif lufs is not None and -16.5 <= lufs <= -12.0:
            strengths.append(f"Broadcast-ready loudness ({lufs} LUFS).")

        dead_air_ratio = audio.get("dead_air_total_sec", 0.0) / duration
        if dead_air_ratio > 0.12:
            audio_score -= 2.0
            weaknesses.append({
                "area": "Dead Air / Pauses",
                "severity": "medium",
                "finding": f"{audio['dead_air_total_sec']}s of silence ({int(dead_air_ratio * 100)}% of runtime) — breaks verbal momentum.",
            })
            upgrade_actions.append(
                "TIGHTEN SPEECH: Trim pauses >0.35s with J-cuts/L-cuts so speech feels effortlessly punchy."
            )

        sync = audio.get("cut_beat_sync_ratio", 0.0)
        if sync >= 0.45:
            audio_score = min(10.0, audio_score + 1.0)
            strengths.append(f"Strong audio-visual rhythm ({int(sync * 100)}% of cuts align with audio transients/beats).")
        elif shots["total_cuts"] >= 5 and sync < 0.2:
            upgrade_actions.append(
                "BEAT-SYNC EDIT: Align major visual cuts and text reveals within ±100ms of beat hits or Whoosh/Pop SFX."
            )

    # 4. Audience Comment Exploitation
    if comments_intel and comments_intel.get("total_mined", 0) > 0:
        if comments_intel.get("complaints"):
            upgrade_actions.append(
                f"FIX VIEWER COMPLAINTS: {len(comments_intel['complaints'])} viewer complaint(s) mined — directly solve those pain points in your version."
            )
        if comments_intel.get("questions"):
            upgrade_actions.append(
                f"ANSWER UNANSWERED QUESTIONS: {len(comments_intel['questions'])} top viewer question(s) found — weave those answers right into your script for higher saves & shares."
            )
        if comments_intel.get("requests"):
            upgrade_actions.append(
                f"CAPTURE DEMAND: {len(comments_intel['requests'])} viewer request(s)/Part-2 asks detected — turn the #1 request into your hook or bonus payoff."
            )

    if audio_excluded:
        overall = round((hook_score * 0.5 + pacing_score * 0.5), 1)
    else:
        overall = round((hook_score * 0.35 + pacing_score * 0.35 + (audio_score or 0.0) * 0.30), 1)

    return {
        "overall_retention_score": overall,
        "score_method": "heuristic",
        "score_note": (
            "Retention scores are rule-of-thumb heuristics (fixed ladders + weights), "
            "not measured audience retention. Treat the underlying telemetry "
            "(cuts, ASL, LUFS, dead-air, motion) as the truth."
        ),
        "audio_excluded_from_score": audio_excluded,
        "effective_dead_zones": effective_dead_zones,
        "true_dead_zones": effective_dead_zones,
        "production_verdict_auto": (forensics or {}).get("verdict_summary", "unknown"),
        "production_confidence_auto": (forensics or {}).get("overall_confidence", 0.0),
        "notes": notes,
        "scores": {
            "hook_velocity": round(hook_score, 1),
            "pacing_health": round(pacing_score, 1),
            "audio_dynamics": round(audio_score, 1) if audio_score is not None else None,
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "beat_the_original_actions": upgrade_actions,
    }
