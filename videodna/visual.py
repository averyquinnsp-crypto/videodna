"""Visual forensics: keyframes, motion energy, color DNA, auto .cube LUT, and AI contact sheets."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _extract_frame(video_path: Path, timestamp: float, out_path: Path, max_w: int = 480) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(0.0, timestamp):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_w},iw)':-2",
        "-q:v", "3",
        "-y", str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0 and out_path.exists()


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _extract_palette(arr_list: list[np.ndarray], k: int = 6) -> list[str]:
    """Fast numpy k-means color palette extraction without sklearn."""
    if not arr_list:
        return []
    pixels = np.concatenate([a.reshape(-1, 3) for a in arr_list], axis=0).astype(np.float32)
    # Subsample for speed
    if len(pixels) > 8000:
        idx = np.linspace(0, len(pixels) - 1, 8000, dtype=int)
        pixels = pixels[idx]

    # Deterministic initial centroids via quantiles
    rng = np.random.default_rng(42)
    centers = pixels[rng.choice(len(pixels), size=min(k, len(pixels)), replace=False)]

    for _ in range(8):
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = []
        for ci in range(len(centers)):
            pts = pixels[labels == ci]
            new_centers.append(pts.mean(axis=0) if len(pts) > 0 else centers[ci])
        centers = np.array(new_centers)

    # Sort by population descending
    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(-counts)
    return [_rgb_to_hex(tuple(np.clip(centers[i], 0, 255).astype(int))) for i in order]


def _classify_color_grade(mean_rgb: np.ndarray, contrast: float, saturation: float) -> str:
    r, g, b = mean_rgb / 255.0
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if saturation < 0.08:
        return "monochrome_or_noir"
    if contrast >= 0.30 and saturation < 0.28:
        return "high_contrast_noir_graphic"
    if luma < 0.28 and contrast > 0.22:
        return "dark_moody_cinematic"
    if luma > 0.70 and saturation < 0.28:
        return "bright_airy_high_key"
    if (r - b) > 0.10 and b < 0.42:
        return "warm_golden_or_film"
    if (b - r) > 0.08:
        return "cool_tech_or_cyber"
    if saturation > 0.42 and contrast > 0.24:
        return "vibrant_punchy_viral"
    return "natural_balanced"


def _write_cube_lut(out_path: Path, mean_rgb: np.ndarray, std_rgb: np.ndarray, sat_target: float) -> None:
    """Generate a 17x17x17 3D .cube LUT approximating the reference tonal/color signature."""
    size = 17
    target_mean = np.clip(mean_rgb / 255.0, 0.20, 0.80)
    # Gentle shift so LUT stylizes footage without clipping
    tint_shift = (target_mean - target_mean.mean()) * 0.45
    contrast_gain = float(np.clip((std_rgb.mean() / 64.0), 0.85, 1.25))
    sat_gain = float(np.clip(0.85 + sat_target * 0.6, 0.80, 1.35))

    lines = [
        'TITLE "VideoDNA_Reference_Grade"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    grid = np.linspace(0.0, 1.0, size)
    for b in grid:
        for g in grid:
            for r in grid:
                vec = np.array([r, g, b], dtype=np.float32)
                # Contrast around mid-grey
                vec = (vec - 0.5) * contrast_gain + 0.5
                # Saturation
                luma = 0.299 * vec[0] + 0.587 * vec[1] + 0.114 * vec[2]
                vec = luma + (vec - luma) * sat_gain
                # Midtone-weighted color tint
                mid_weight = 1.0 - abs(luma - 0.5) * 1.5
                vec = vec + tint_shift * max(0.0, mid_weight)
                vec = np.clip(vec, 0.0, 1.0)
                lines.append(f"{vec[0]:.5f} {vec[1]:.5f} {vec[2]:.5f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_contact_sheet(
    items: list[tuple[Path, str]],
    out_path: Path,
    cols: int = 4,
    tile_w: int = 320,
    title: str = "",
) -> Path | None:
    """Create a clean labeled contact sheet optimized for AI vision & human review."""
    valid = [(p, lbl) for p, lbl in items if p.exists()]
    if not valid:
        return None

    with Image.open(valid[0][0]) as sample:
        sw, sh = sample.size
    tile_h = max(120, int(tile_w * (sh / sw))) if sw else 180
    label_h = 26
    header_h = 38 if title else 8
    pad = 8

    rows = math.ceil(len(valid) / cols)
    sheet_w = cols * tile_w + (cols + 1) * pad
    sheet_h = header_h + rows * (tile_h + label_h + pad) + pad

    canvas = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    if title:
        draw.text((pad, 10), title, fill=(235, 240, 255), font=font)

    for idx, (img_path, label) in enumerate(valid):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile_w + pad)
        y = header_h + r * (tile_h + label_h + pad)
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(im_rgb, (x, y))
        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + label_h], fill=(28, 32, 44))
        draw.text((x + 6, y + tile_h + 6), label[:48], fill=(220, 230, 250), font=font)

    canvas.save(out_path, format="JPEG", quality=88)
    return out_path


def analyze_visuals(
    video_path: Path,
    shots_data: dict[str, Any],
    out_dir: Path,
    mode: str = "short",
) -> dict[str, Any]:
    """Extract keyframes, motion scores, color DNA, .cube LUT, and contact sheets."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    duration = shots_data["shots"][-1]["end"] if shots_data["shots"] else 1.0
    shots = shots_data["shots"]

    # 1. Hook Strip (0.0s to min(5.0s, duration) every 0.5s — crucial for stop-the-scroll audit!)
    hook_end = min(5.0 if mode == "short" else 12.0, duration)
    hook_step = 0.5 if mode == "short" else 1.5
    hook_items: list[tuple[Path, str]] = []
    t = 0.0
    h_idx = 0
    while t <= hook_end + 0.01 and h_idx < 12:
        hp = frames_dir / f"hook_{h_idx:02d}.jpg"
        if _extract_frame(video_path, min(t, max(0.0, duration - 0.05)), hp, max_w=360):
            hook_items.append((hp, f"HOOK t={t:.1f}s"))
        t += hook_step
        h_idx += 1

    hook_sheet = _build_contact_sheet(
        hook_items,
        out_dir / "hook_strip.jpg",
        cols=4,
        tile_w=300,
        title=f"HOOK ANATOMY (0.0s - {hook_end:.1f}s) | First-Impressions & Pattern Interrupt Audit",
    )

    # 2. Shot sampling strategy:
    #    Short-form: inspect up to 24 shots directly (first + mid + last for motion score)
    #    Long-form: hierarchical sampling — first 8 shots (critical intro) + evenly spaced macro shots (up to 24 total)
    if len(shots) <= 24:
        selected_shots = shots
    elif mode == "short":
        idx = np.linspace(0, len(shots) - 1, 24, dtype=int)
        selected_shots = [shots[i] for i in sorted(set(idx))]
    else:
        first_part = shots[:8]
        rest = shots[8:]
        idx = np.linspace(0, len(rest) - 1, 16, dtype=int)
        selected_shots = first_part + [rest[i] for i in sorted(set(idx))]

    storyboard_items: list[tuple[Path, str]] = []
    sampled_arrays: list[np.ndarray] = []
    motion_scores: list[float] = []
    shot_visuals: list[dict[str, Any]] = []

    for s in selected_shots:
        sid = s["id"]
        t_start = s["start"] + min(0.15, s["duration"] * 0.15)
        t_mid = s["start"] + s["duration"] * 0.5
        t_end = max(t_start, s["end"] - min(0.15, s["duration"] * 0.15))

        p_start = frames_dir / f"shot_{sid:03d}_a.jpg"
        p_mid = frames_dir / f"shot_{sid:03d}_mid.jpg"
        p_end = frames_dir / f"shot_{sid:03d}_b.jpg"

        ok_a = _extract_frame(video_path, t_start, p_start, max_w=360)
        ok_m = _extract_frame(video_path, t_mid, p_mid, max_w=360)
        ok_b = _extract_frame(video_path, t_end, p_end, max_w=360)

        # Estimate motion energy via normalized pixel diff across start->mid->end
        motion_val = 0.0
        luma_val = 0.0
        arrs = []
        for p, ok in [(p_start, ok_a), (p_mid, ok_m), (p_end, ok_b)]:
            if ok:
                with Image.open(p) as im:
                    small = np.asarray(im.convert("RGB").resize((96, 96)), dtype=np.float32)
                    arrs.append(small)
        if arrs:
            sampled_arrays.append(arrs[len(arrs) // 2])
            luma_val = float(np.mean(arrs[len(arrs) // 2]) / 255.0)
        if len(arrs) >= 2:
            diffs = [float(np.mean(np.abs(arrs[i + 1] - arrs[i]))) for i in range(len(arrs) - 1)]
            motion_val = round(sum(diffs) / len(diffs), 2)

        if motion_val < 4.5:
            motion_label = "static"
        elif motion_val < 14.0:
            motion_label = "gentle_motion"
        elif motion_val < 28.0:
            motion_label = "dynamic_action"
        else:
            motion_label = "high_energy_whip"

        motion_scores.append(motion_val)
        shot_visuals.append({
            "shot_id": sid,
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
            "motion_energy": motion_val,
            "motion_class": motion_label,
            "brightness": round(luma_val, 3),
            "keyframe": str(p_mid.relative_to(out_dir)) if ok_m else None,
        })

        # Keep storyboard clean: clean up _a and _b after motion measurement to save space
        for temp_p in (p_start, p_end):
            if temp_p.exists():
                temp_p.unlink()

        if ok_m:
            storyboard_items.append(
                (p_mid, f"#{sid:02d} {s['start']:.1f}-{s['end']:.1f}s ({s['duration']:.1f}s) | {motion_label}")
            )

    storyboard_sheet = _build_contact_sheet(
        storyboard_items,
        out_dir / "storyboard.jpg",
        cols=4,
        tile_w=320,
        title=f"STORYBOARD ({len(storyboard_items)} key shots of {len(shots)} total) | Mode: {mode.upper()}",
    )

    # 3. Color DNA & Auto .cube LUT
    if sampled_arrays:
        stacked = np.stack(sampled_arrays, axis=0)  # (N, 96, 96, 3)
        mean_rgb = stacked.mean(axis=(0, 1, 2))
        std_rgb = stacked.std(axis=(0, 1, 2))
        # HSV-like saturation approximation
        mx = stacked.max(axis=-1)
        mn = stacked.min(axis=-1)
        sat = float(np.mean((mx - mn) / np.maximum(mx, 1.0)))
        contrast = float(std_rgb.mean() / 255.0)
        palette = _extract_palette(sampled_arrays, k=6)
        grade_style = _classify_color_grade(mean_rgb, contrast, sat)
        lut_path = out_dir / "reference_grade.cube"
        _write_cube_lut(lut_path, mean_rgb, std_rgb, sat)
    else:
        mean_rgb = np.array([128.0, 128.0, 128.0])
        contrast = 0.0
        sat = 0.0
        palette = []
        grade_style = "unknown"
        lut_path = None

    avg_motion = round(sum(motion_scores) / len(motion_scores), 2) if motion_scores else 0.0
    static_ratio = round(
        sum(1 for v in shot_visuals if v["motion_class"] == "static") / max(1, len(shot_visuals)),
        2,
    )

    return {
        "hook_strip": str(hook_sheet.relative_to(out_dir)) if hook_sheet else None,
        "storyboard": str(storyboard_sheet.relative_to(out_dir)) if storyboard_sheet else None,
        "color_dna": {
            "palette_hex": palette,
            "avg_brightness": round(float(mean_rgb.mean() / 255.0), 3),
            "contrast_score": round(contrast, 3),
            "saturation_score": round(sat, 3),
            "grade_style": grade_style,
            "lut_file": str(lut_path.relative_to(out_dir)) if lut_path else None,
        },
        "motion_summary": {
            "avg_motion_energy": avg_motion,
            "static_shot_ratio": static_ratio,
        },
        "shot_visuals": shot_visuals,
    }
