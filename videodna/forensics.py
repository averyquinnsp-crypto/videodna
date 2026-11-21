"""Mathematical per-shot production forensics (pure numpy + ffmpeg, zero heavy native deps).

Measures 6 physical signals per shot to classify production method BEFORE visual review:
  1. High-frequency sensor noise residual (organic camera grain vs clean digital render)
  2. Flat-region ratio (solid fills / vector graphics / UI backgrounds)
  3. Color entropy & bimodal contrast density (synthetic templates vs continuous real scenes)
  4. Axis-aligned sharp synthetic edge ratio (text, UI panels, screen capture)
  5. Effective frame cadence & duplicate-frame ratio (8/12/16/24fps AI/anim inside 30/60fps)
  6. Global 2D planar motion coherence via FFT cross-correlation (2D pan/slide vs 3D/live)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

CATEGORIES = ("live", "ai_video", "3d_render", "code_motion", "screen_rec", "static_zoom")

# v2.0.2: a production call is only asserted when the duration-weighted
# overall confidence reaches this threshold. Below it the verdict must say
# the call is uncertain (with the real runner-up) instead of asserting one
# type — e.g. U4 (purple 3D character) read "code_motion 100%" at 0.43.
LOW_CONFIDENCE_VERDICT_THRESHOLD = 0.5


def _dominant_palette(frame_rgb: np.ndarray, k: int = 3) -> list[str]:
    """Top-k dominant colors of a frame via a 4-bit-per-channel histogram.

    Measured from the actual pixels (no model, no guessing): each channel is
    quantized to 16 levels, the 4096-bin histogram is counted, and the k
    most-populated bins are returned as hex colors at their bin centers.
    Ties break by bin index so the result is deterministic.
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return []
    q = (np.clip(frame_rgb, 0, 255).astype(np.int32) >> 4)  # 0..15 per channel
    bins = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    counts = np.bincount(bins.ravel(), minlength=4096)
    order = np.lexsort((np.arange(4096), -counts))  # count desc, bin asc
    out: list[str] = []
    for b in order[:k]:
        if counts[b] <= 0:
            break
        r = ((int(b) >> 8) & 0xF) * 16 + 8
        g = ((int(b) >> 4) & 0xF) * 16 + 8
        bl = (int(b) & 0xF) * 16 + 8
        out.append(f"#{r:02X}{g:02X}{bl:02X}")
    return out


def honest_verdict_summary(
    breakdown_pct: dict[str, float],
    overall_probs: dict[str, float],
    overall_conf: float,
) -> tuple[str, bool, str | None]:
    """Build the verdict label, refusing to assert a type at low confidence.

    Returns (verdict_summary, verdict_uncertain, runner_up_label).
    - confidence >= 0.5 (or an already-honest "unknown" top call): the usual
      "code_motion 100%" / "live 60% + ai_video 40%" wording, unchanged.
    - confidence < 0.5: "code_motion / static_zoom — uncertain from pixels
      alone", where the runner-up is the real second-highest class from the
      duration-weighted overall probabilities — never invented.
    """
    if not breakdown_pct:
        return "indeterminate", False, None
    top_label = next(iter(breakdown_pct))
    summary = " + ".join(f"{k} {v:.0f}%" for k, v in list(breakdown_pct.items())[:3])
    if top_label == "unknown" or overall_conf >= LOW_CONFIDENCE_VERDICT_THRESHOLD:
        return summary, False, None
    ranked = sorted(overall_probs.items(), key=lambda kv: -kv[1])
    runner_up = next((k for k, _ in ranked if k != top_label), None)
    if runner_up is None:
        return summary, False, None
    return f"{top_label} / {runner_up} — uncertain from pixels alone", True, runner_up


def _decode_burst_rgb(
    video_path: Path,
    start: float,
    duration: float,
    width: int = 160,
    height: int = 90,
    max_frames: int = 12,
    sample_fps: float = 12.0,
) -> np.ndarray:
    """Decode a small burst of consecutive RGB frames from inside a shot via ffmpeg."""
    burst_dur = min(max(0.35, duration * 0.8), max_frames / sample_fps)
    seek_t = max(0.0, start + max(0.0, (duration - burst_dur) * 0.5))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{seek_t:.3f}",
        "-i", str(video_path),
        "-t", f"{burst_dur:.3f}",
        "-vf", f"fps={sample_fps},scale={width}:{height}:flags=bilinear",
        "-frames:v", str(max_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    raw = res.stdout
    frame_bytes = width * height * 3
    n = len(raw) // frame_bytes
    if n == 0:
        return np.empty((0, height, width, 3), dtype=np.float32)
    return np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, height, width, 3).astype(np.float32)


def _laplacian_sensor_noise(luma: np.ndarray) -> float:
    """Estimate organic high-frequency sensor noise on smooth (non-edge) regions."""
    # Discrete 3x3 Laplacian
    center = luma[1:-1, 1:-1]
    lap = (
        -luma[:-2, 1:-1] - luma[2:, 1:-1]
        - luma[1:-1, :-2] - luma[1:-1, 2:]
        + 4.0 * center
    )
    # Local gradient magnitude to mask out strong object/text edges
    gx = np.abs(luma[1:-1, 2:] - luma[1:-1, :-2])
    gy = np.abs(luma[2:, 1:-1] - luma[:-2, 1:-1])
    grad = gx + gy
    smooth_mask = grad < 14.0
    if np.sum(smooth_mask) < 50:
        return float(np.median(np.abs(lap)))
    # MAD-based noise estimator on smooth pixels
    return float(np.median(np.abs(lap[smooth_mask])) * 1.4826)


def _color_entropy(frame_rgb: np.ndarray) -> float:
    """Compute normalized Shannon entropy over 4-bit quantized RGB (4096 bins)."""
    q = (frame_rgb.astype(np.int32) >> 4)
    bins = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    counts = np.bincount(bins.ravel(), minlength=4096)
    probs = counts[counts > 0].astype(np.float64)
    probs /= probs.sum()
    ent = -np.sum(probs * np.log2(probs))
    return float(ent / 12.0)  # normalized 0..1


def _planar_motion_coherence(f1: np.ndarray, f2: np.ndarray) -> tuple[float, float]:
    """Check if motion between f1 and f2 is mostly rigid 2D translation (pan/slide/wipe)."""
    raw_diff = float(np.mean(np.abs(f2 - f1)))
    if raw_diff < 0.8:
        return 0.0, 0.0
    # Phase correlation via 2D FFT
    a = f1 - f1.mean()
    b = f2 - f2.mean()
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    denom = np.abs(cross) + 1e-6
    corr = np.fft.irfft2(cross / denom, s=a.shape)
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = a.shape
    dy = int(py if py <= h // 2 else py - h)
    dx = int(px if px <= w // 2 else px - w)

    if abs(dx) > w // 3 or abs(dy) > h // 3:
        return raw_diff, 0.0

    # Measure residual after shifting f1 by (dy, dx)
    shifted = np.roll(np.roll(f1, dy, axis=0), dx, axis=1)
    # Crop border affected by roll
    m = max(abs(dy), abs(dx), 2)
    if h - 2 * m < 10 or w - 2 * m < 10:
        return raw_diff, 0.0
    aligned_diff = float(np.mean(np.abs(f2[m:-m, m:-m] - shifted[m:-m, m:-m])))
    coherence = float(np.clip(1.0 - (aligned_diff / max(raw_diff, 1e-3)), 0.0, 1.0))
    return raw_diff, coherence


def _score_shot_metrics(
    metrics: dict[str, Any],
    probe: dict[str, Any],
) -> tuple[dict[str, float], str, float, list[str]]:
    """Map physical forensic metrics to soft probabilities across 6 production classes."""
    noise = metrics["sensor_noise"]
    flat = metrics["flat_region_ratio"]
    ent = metrics["color_entropy"]
    rect_edge = metrics["rectilinear_edge_ratio"]
    dup_ratio = metrics["duplicate_frame_ratio"]
    mean_diff = metrics["mean_frame_diff"]
    coherence = metrics["planar_coherence"]
    high_contrast_bw = metrics["extreme_bw_ratio"]

    # Base logits (start neutral)
    scores: dict[str, float] = {k: 0.15 for k in CATEGORIES}
    evidence: list[str] = []

    # 1. Strong AIGC container metadata signal
    if "ai_generator" in (probe.get("forensic_signals") or {}):
        scores["ai_video"] += 2.2
        evidence.append("Container contains explicit AI generator metadata tag.")

    if "remotion_or_code" in (probe.get("forensic_signals") or {}):
        scores["code_motion"] += 2.0
        evidence.append("Container metadata references code-rendering engine.")

    # 2. Flat regions + low entropy + clean digital floor -> Code/Motion Graphics or Screen Rec
    if flat >= 0.45 and noise < 1.6:
        scores["code_motion"] += 1.6
        evidence.append(f"High flat-fill area ({int(flat*100)}%) with clean noise floor ({noise:.2f}) — characteristic of motion graphics / vector / code templates.")
    elif flat >= 0.30 and noise < 2.0:
        scores["code_motion"] += 0.9
        scores["3d_render"] += 0.6

    if high_contrast_bw >= 0.35 and ent < 0.48:
        scores["code_motion"] += 1.1
        evidence.append(f"Stark pure black/white backdrop ({int(high_contrast_bw*100)}%) & low color entropy ({ent:.2f}) — strong graphic template signature.")

    if rect_edge >= 0.18 and flat >= 0.40:
        scores["screen_rec"] += 1.0
        scores["code_motion"] += 0.5
        evidence.append(f"Strong rectilinear edges ({int(rect_edge*100)}%) — UI / typography / screen capture clue.")

    if probe.get("is_vfr") and flat >= 0.30:
        scores["screen_rec"] += 1.2
        evidence.append("Variable frame rate + clean flat surfaces — strong screen recording indicator.")

    # 3. Sensor noise & rich continuous color entropy -> Live Footage vs 3D/AI
    if noise >= 2.8 and flat < 0.22 and ent >= 0.58:
        scores["live"] += 1.8
        evidence.append(f"Organic sensor grain ({noise:.2f}) + rich color entropy ({ent:.2f}) + low flat area ({int(flat*100)}%) — consistent with real camera capture.")
    elif noise < 1.4 and flat < 0.35 and ent >= 0.42:
        scores["3d_render"] += 1.2
        scores["ai_video"] += 0.8
        evidence.append(f"Ultra-clean noise floor ({noise:.2f}) with smooth shaded gradients — typical of 3D CGI render or AI video generation.")

    # 4. Static / Ken Burns / 2D Planar Slide
    if mean_diff < 1.0:
        scores["static_zoom"] += 1.4
        scores["code_motion"] += 0.5
        evidence.append(f"Near-static frame progression (diff={mean_diff:.2f}).")
    elif coherence >= 0.55 and flat >= 0.25:
        scores["code_motion"] += 1.0
        scores["static_zoom"] += 0.7
        evidence.append(f"High 2D planar motion coherence ({int(coherence*100)}%) — flat layer slide/zoom rather than 3D camera parallax.")
    elif coherence >= 0.65 and flat < 0.25:
        scores["static_zoom"] += 1.3
        evidence.append(f"Rigid 2D image pan/zoom detected (coherence {int(coherence*100)}%).")

    # 5. Frame cadence (duplicate frames inside 30/60fps container)
    if dup_ratio >= 0.35 and mean_diff >= 1.0:
        scores["ai_video"] += 0.8
        scores["code_motion"] += 0.5
        evidence.append(f"Lower effective cadence ({int(dup_ratio*100)}% repeated frames in {probe.get('avg_fps')}fps container).")

    # Softmax -> calibrated probabilities
    vals = np.array([scores[k] for k in CATEGORIES], dtype=np.float64)
    vals = np.exp((vals - vals.max()) * 1.35)
    probs_arr = vals / vals.sum()
    probs = {k: round(float(p), 3) for k, p in zip(CATEGORIES, probs_arr)}

    sorted_p = sorted(probs.values(), reverse=True)
    margin = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
    # Honest offline calibration: never claim >88% before visual agent review.
    # If NO rule fired, all classes tie at 1/6 and max() would silently pick
    # the first label ("live") — a pure dict-order artifact (v2.0 bug: it
    # reported "live 41%" on a video with no live footage). A tie or a
    # near-tie is "unknown", never a confident-sounding class.
    if not evidence or margin < 0.005:
        top_label = "unknown"
        confidence = 0.30
        if not evidence:
            evidence.append("No forensic rule fired — flat/uniform profile, classify visually (unknown).")
        else:
            evidence.append("Top classes nearly tied — classify visually (unknown).")
    else:
        top_label = max(probs, key=probs.get)
        top_prob = probs[top_label]
        confidence = round(float(np.clip(top_prob * 0.78 + margin * 0.28, 0.30, 0.88)), 2)

    return probs, top_label, confidence, evidence


def run_shot_forensics(
    video_path: Path,
    probe: dict[str, Any],
    shots_data: dict[str, Any],
    mode: str = "short",
) -> dict[str, Any]:
    """Run mathematical production forensics across shots and compute duration-weighted breakdown."""
    shots = shots_data["shots"]
    max_s = 32 if mode == "short" else 24
    if len(shots) <= max_s:
        target_shots = shots
    else:
        idx = np.linspace(0, len(shots) - 1, max_s, dtype=int)
        target_shots = [shots[i] for i in sorted(set(idx))]

    per_shot: list[dict[str, Any]] = []
    weighted_totals: dict[str, float] = {k: 0.0 for k in CATEGORIES}
    total_dur = 0.0
    conf_sum = 0.0

    for s in target_shots:
        frames = _decode_burst_rgb(video_path, s["start"], s["duration"])
        if len(frames) == 0:
            continue

        lumas = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
        mid_idx = len(frames) // 2
        mid_rgb = frames[mid_idx]
        mid_luma = lumas[mid_idx]

        noise = _laplacian_sensor_noise(mid_luma)

        gx = np.abs(mid_luma[1:-1, 2:] - mid_luma[1:-1, :-2])
        gy = np.abs(mid_luma[2:, 1:-1] - mid_luma[:-2, 1:-1])
        grad = gx + gy
        flat_ratio = float(np.mean(grad < 2.2))
        strong_edges = grad > 35.0
        if np.sum(strong_edges) > 20:
            axis_aligned = np.logical_and(
                strong_edges,
                np.logical_or(gx > gy * 3.0, gy > gx * 3.0),
            )
            rect_ratio = float(np.sum(axis_aligned) / np.sum(strong_edges))
        else:
            rect_ratio = 0.0

        extreme_bw = float(np.mean((mid_luma < 10.0) | (mid_luma > 245.0)))
        ent = _color_entropy(mid_rgb)

        diffs = []
        coherences = []
        dup_count = 0
        for i in range(len(lumas) - 1):
            d_raw, coh = _planar_motion_coherence(lumas[i], lumas[i + 1])
            diffs.append(d_raw)
            if d_raw < 0.8:
                dup_count += 1
            elif coh > 0:
                coherences.append(coh)

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        dup_ratio = float(dup_count / max(1, len(diffs)))
        planar_coh = float(np.mean(coherences)) if coherences else 0.0

        metrics = {
            "sensor_noise": round(noise, 2),
            "flat_region_ratio": round(flat_ratio, 3),
            "rectilinear_edge_ratio": round(rect_ratio, 3),
            "extreme_bw_ratio": round(extreme_bw, 3),
            "color_entropy": round(ent, 3),
            "mean_frame_diff": round(mean_diff, 2),
            "duplicate_frame_ratio": round(dup_ratio, 3),
            "planar_coherence": round(planar_coh, 3),
        }

        probs, top_label, conf, evidence = _score_shot_metrics(metrics, probe)
        needs_review = conf < 0.65

        dur = max(0.1, float(s["duration"]))
        for k, p_val in probs.items():
            weighted_totals[k] += p_val * dur
        total_dur += dur
        conf_sum += conf * dur

        per_shot.append({
            "shot_id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
            "predicted_type": top_label,
            "confidence": conf,
            "needs_visual_review": needs_review,
            "probs": probs,
            "palette_hex": _dominant_palette(mid_rgb, k=3),
            "metrics": metrics,
            "evidence": evidence,
        })

    # Duration-weighted top-class production breakdown (sums cleanly to 100%)
    if total_dur > 0:
        winner_dur: dict[str, float] = {}
        for ps in per_shot:
            lbl = ps["predicted_type"]
            winner_dur[lbl] = winner_dur.get(lbl, 0.0) + max(0.1, float(ps["duration"]))
        breakdown_pct = {
            k: round((v / total_dur) * 100.0, 1)
            for k, v in sorted(winner_dur.items(), key=lambda kv: -kv[1])
        }
        overall_conf = round(conf_sum / total_dur, 2)
        # Duration-weighted overall class probabilities (for the runner-up in
        # the low-confidence verdict — measured, not guessed).
        overall_probs = {
            k: round(float(v / total_dur), 3) for k, v in weighted_totals.items()
        }
    else:
        breakdown_pct = {}
        overall_conf = 0.0
        overall_probs = {k: 0.0 for k in CATEGORIES}

    primary_type = next(iter(breakdown_pct.keys()), "unknown")
    verdict_line, verdict_uncertain, runner_up = honest_verdict_summary(
        breakdown_pct, overall_probs, overall_conf
    )

    return {
        "primary_type": primary_type,
        "verdict_summary": verdict_line,
        "verdict_uncertain": verdict_uncertain,
        "runner_up": runner_up,
        "overall_probs": overall_probs,
        "overall_confidence": overall_conf,
        "duration_weighted_pct": breakdown_pct,
        "low_confidence_shots": [x["shot_id"] for x in per_shot if x["needs_visual_review"]],
        "per_shot": per_shot,
    }


def fuse_review_probabilities(
    forensics: dict[str, Any],
    agent_review: dict[str, Any],
) -> dict[str, Any]:
    """Fuse deterministic tool probabilities with AI Agent visual review & flag conflicts."""
    annotations = {
        int(a["shot_id"]): a
        for a in (agent_review.get("shot_annotations") or [])
        if "shot_id" in a
    }
    label_alias = {
        "live": "live",
        "ai_video": "ai_video",
        "avatar": "ai_video",
        "3d": "3d_render",
        "3d_render": "3d_render",
        "2d_anim": "code_motion",
        "code_ui": "code_motion",
        "code_motion": "code_motion",
        "screen_rec": "screen_rec",
        "static_zoom": "static_zoom",
    }

    fused_shots = []
    conflicts = []
    weighted: dict[str, float] = {k: 0.0 for k in CATEGORIES}
    total_dur = 0.0
    conf_sum = 0.0

    for sf in forensics.get("per_shot", []):
        sid = sf["shot_id"]
        dur = max(0.1, float(sf["duration"]))
        tool_probs = dict(sf["probs"])
        ann = annotations.get(sid)

        if ann and ann.get("source_type"):
            raw_lbl = str(ann["source_type"]).strip().lower()
            mapped = label_alias.get(raw_lbl)
            agent_conf = float(ann.get("confidence", 0.82))
            if mapped:
                if mapped != sf["predicted_type"] and sf["confidence"] >= 0.65:
                    conflicts.append({
                        "shot_id": sid,
                        "tool_call": f"{sf['predicted_type']} ({int(sf['confidence']*100)}%)",
                        "agent_call": f"{mapped} ({int(agent_conf*100)}%)",
                        "note": "Visual review overrode high-confidence tool estimate — verify if ambiguous.",
                    })
                # Weighted fusion: 35% deterministic tool + 65% visual agent inspection
                agent_dist = {k: (agent_conf if k == mapped else (1.0 - agent_conf) / (len(CATEGORIES) - 1)) for k in CATEGORIES}
                fused = {k: round(0.35 * tool_probs.get(k, 0.0) + 0.65 * agent_dist[k], 3) for k in CATEGORIES}
                norm = sum(fused.values()) or 1.0
                fused = {k: round(v / norm, 3) for k, v in fused.items()}
                final_lbl = max(fused, key=fused.get)
                final_conf = round(float(np.clip(fused[final_lbl] * 0.9 + 0.12, 0.4, 0.98)), 2)
            else:
                fused = tool_probs
                final_lbl = sf["predicted_type"]
                final_conf = sf["confidence"]
        else:
            fused = tool_probs
            final_lbl = sf["predicted_type"]
            final_conf = sf["confidence"]

        for k, pv in fused.items():
            weighted[k] += pv * dur
        total_dur += dur
        conf_sum += final_conf * dur

        fused_shots.append({
            **sf,
            "fused_type": final_lbl,
            "fused_confidence": final_conf,
            "fused_probs": fused,
        })

    winner_dur: dict[str, float] = {}
    for fs in fused_shots:
        lbl = fs["fused_type"]
        winner_dur[lbl] = winner_dur.get(lbl, 0.0) + max(0.1, float(fs["duration"]))
    breakdown_pct = {
        k: round((v / total_dur) * 100.0, 1)
        for k, v in sorted(winner_dur.items(), key=lambda kv: -kv[1])
    } if total_dur > 0 else {}
    overall_conf = round(conf_sum / total_dur, 2) if total_dur > 0 else 0.0
    # Duration-weighted overall class probabilities (runner-up source).
    overall_probs = (
        {k: round(float(v / total_dur), 3) for k, v in weighted.items()}
        if total_dur > 0 else {k: 0.0 for k in CATEGORIES}
    )
    summary, verdict_uncertain, runner_up = honest_verdict_summary(
        breakdown_pct, overall_probs, overall_conf
    )

    return {
        "verdict_summary": summary,
        "verdict_uncertain": verdict_uncertain,
        "runner_up": runner_up,
        "overall_probs": overall_probs,
        "overall_confidence": overall_conf,
        "duration_weighted_pct": breakdown_pct,
        "conflicts": conflicts,
        "per_shot": fused_shots,
    }
