"""VideoDNA v2.0.4 CLI — 10x Video Forensics, Viral X-Ray & Beat-The-Original Engine."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from videodna import __version__
from videodna.audio import analyze_audio
from videodna.diagnostics import compute_diagnostics
from videodna.forensics import (
    classify_camera_motion_for_shots,
    fuse_review_probabilities,
    run_shot_forensics,
)
from videodna.ingest import find_yt_dlp, ingest_url, is_url, parse_subtitle_file, transcribe_local
from videodna.ocr import OCR_NO_TEXT, extract_shot_text, tesseract_available
from videodna.probe import probe_video
from videodna.remake import build_remake_plan
from videodna.report import write_outputs
from videodna.shots import detect_shots, export_shot_frames
from videodna.vault import crack_channel_patterns, fuse_dna, list_vault, save_to_vault
from videodna.visual import analyze_visuals


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")[:48] or "video"


def _attach_shot_labels(
    shots: dict[str, Any],
    forensics: dict[str, Any],
    video_path: Path,
) -> None:
    """Attach v2.0.4 per-shot labels to the shot dicts + forensics entries.

    - camera_motion (item 4): run_shot_forensics() labels the sampled
      shots in its own loop; the unsampled remainder (shots beyond the
      32/24 cap) is classified here with the same burst + classifier,
      so every shot dict carries a value. Forensics entries already
      carry theirs; they are not touched.
    - transition_subtype (item 3): set on the shot dicts by
      detect_shots(); copied onto the forensics entries so the remake
      cards (built from forensics) can carry it.
    Defaults are "unknown" (camera) / None (subtype). Labelling only.
    """
    shot_list = shots.get("shots", [])
    fx_map = {x["shot_id"]: x for x in forensics.get("per_shot", [])}
    missing = [s for s in shot_list if s["id"] not in fx_map]
    extra = classify_camera_motion_for_shots(video_path, missing) if missing else {}
    for s in shot_list:
        sid = int(s["id"])
        fx = fx_map.get(sid)
        if fx is not None:
            s["camera_motion"] = fx.get("camera_motion", "unknown")
            s["camera_motion_confidence"] = fx.get("camera_motion_confidence", 0.0)
        else:
            cm = extra.get(sid, {})
            s["camera_motion"] = cm.get("camera_motion", "unknown")
            s["camera_motion_confidence"] = cm.get("camera_motion_confidence", 0.0)
        if fx is not None and s.get("transition_subtype"):
            fx["transition_subtype"] = s["transition_subtype"]


def cmd_analyze(args: argparse.Namespace) -> int:
    source = args.source
    platform_meta: dict[str, Any] = {}
    transcript: list[dict[str, Any]] = []
    comments_intel: dict[str, Any] = {}
    thumbnail: str | None = None

    if is_url(source):
        out_dir = Path(args.output) if args.output else Path("out") / _safe_slug(source.split("/")[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        print("[VideoDNA] Downloading URL + metadata + comments via yt-dlp...")
        ing = ingest_url(source, out_dir, max_comments=args.max_comments, sub_langs=args.sub_langs)
        video_path = ing["video_path"]
        platform_meta = ing["platform_meta"]
        transcript = ing["transcript"]
        comments_intel = ing["comments_intel"]
        thumbnail = ing["thumbnail"]
    else:
        video_path = Path(source).expanduser().resolve()
        if not video_path.exists():
            print(f"[Error] Video file not found: {video_path}", file=sys.stderr)
            return 1
        out_dir = Path(args.output) if args.output else Path("out") / _safe_slug(video_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.transcript:
            t_path = Path(args.transcript).expanduser().resolve()
            if t_path.exists():
                transcript = parse_subtitle_file(t_path)

    print("[VideoDNA] 1/5 Probing container metadata & cadence forensics...")
    probe = probe_video(video_path)
    mode = probe["recommended_mode"] if args.mode == "auto" else args.mode

    print(f"[VideoDNA] 2/5 Running two-pass shot detector (mode={mode}, duration={probe['duration_sec']}s)...")
    shots = detect_shots(
        video_path, probe["duration_sec"], mode=mode, threshold=args.threshold,
        width=probe["width"], height=probe["height"],
    )

    print("[VideoDNA] 3/5 Running mathematical per-shot production forensics (noise, flatness, cadence, 2D coherence)...")
    forensics = run_shot_forensics(video_path, probe, shots, mode=mode)

    # v2.0.4 item 4: camera motion is computed inside run_shot_forensics
    # for the sampled shots; cover the unsampled remainder here so EVERY
    # shot dict carries a camera_motion value (labelling only — no
    # scoring input, no verdict change). Also copy the transition
    # subtype (item 3, set on the shot dicts by detect_shots) onto the
    # forensics entries so the remake cards can carry it.
    _attach_shot_labels(shots, forensics, video_path)

    # v2.0.4: export first/mid/last (+strip) frames for EVERY shot in one
    # streaming pass; paths land on each shot dict so blueprint.json,
    # dna/shots.md and dna/remake_plan.md all reference the same files.
    print("[VideoDNA] 3b/5 Exporting per-shot first/mid/last frames (dna/frames/)...")
    shot_frames = export_shot_frames(
        video_path, shots["shots"], out_dir,
        fps=probe.get("avg_fps") or 30.0,
        orig_width=probe.get("width"), orig_height=probe.get("height"),
    )
    for s in shots["shots"]:
        s["frame_paths"] = dict(shot_frames.get(s["id"], {}))

    # v2.0.4 item 6: on-screen text via the tesseract CLI (system
    # package, no pip dependency). One mid frame per shot, quality-gated
    # (see videodna/ocr.py) — garbage never reaches the cards. When
    # tesseract is absent this is a no-op and the cards keep stating
    # that no OCR is available.
    if tesseract_available():
        import time as _time
        _ocr_t0 = _time.monotonic()
        print("[VideoDNA] 3c/5 Extracting on-screen text per shot (tesseract OCR)...")
        ocr_texts = extract_shot_text(video_path, shots["shots"])
        if ocr_texts:
            for s in shots["shots"]:
                entry = ocr_texts.get(s["id"])
                if entry is None:
                    continue
                s["on_screen_text"] = entry["text"] or OCR_NO_TEXT
                s["on_screen_text_words"] = list(entry["words"])
            for sf in forensics.get("per_shot", []):
                entry = ocr_texts.get(sf["shot_id"])
                if entry is None:
                    continue
                sf["on_screen_text"] = entry["text"] or OCR_NO_TEXT
                sf["on_screen_text_words"] = list(entry["words"])
            print(f"[VideoDNA] OCR done in {_time.monotonic() - _ocr_t0:.1f}s "
                  f"({len(ocr_texts)} shots, "
                  f"{sum(1 for e in ocr_texts.values() if e['text'])} with text).")

    print("[VideoDNA] 4/5 Extracting keyframes, motion energy, color DNA & 3D .cube LUT...")
    visual = analyze_visuals(video_path, shots, out_dir, mode=mode)

    print("[VideoDNA] 5/5 Running smart audio forensics (silence/speech/music classifier, LUFS, BPM, cut-sync)...")
    audio = analyze_audio(video_path, probe["duration_sec"], shots["shots"], out_dir, has_audio=probe["has_audio"])

    if args.transcribe and probe["has_audio"] and not audio.get("is_silent_track") and not transcript:
        transcript = transcribe_local(video_path, model=args.transcribe_model)

    diagnostics = compute_diagnostics(
        probe, shots, visual, audio, comments_intel=comments_intel, forensics=forensics
    )
    shots["dead_zones"] = diagnostics["effective_dead_zones"]
    remake_plan = build_remake_plan(
        forensics, audio, visual["color_dna"], shot_frames=shot_frames
    )

    blueprint: dict[str, Any] = {
        "schema_version": "videodna/2.0",
        "probe": probe,
        "platform_meta": platform_meta,
        "thumbnail": thumbnail,
        "shots": shots,
        "forensics": forensics,
        "fused_forensics": None,
        "visual": visual,
        "audio": audio,
        "transcript": transcript,
        "comments_intel": comments_intel,
        "diagnostics": diagnostics,
        "remake_plan": remake_plan,
        "agent_review": None,
    }

    paths = write_outputs(out_dir, blueprint)

    if args.save_vault:
        v_path = save_to_vault(blueprint, name=args.vault_name)
        print(f"[VideoDNA] Saved DNA snapshot to Style Vault: {v_path}")

    conf_pct = int(round(forensics["overall_confidence"] * 100))
    audio_summary = (
        "SILENT TRACK (no audible sound)"
        if audio.get("is_silent_track")
        else f"{audio.get('audio_type')} | {audio.get('integrated_lufs')} LUFS"
        + (f" | ~{audio['estimated_bpm']} BPM" if audio.get("estimated_bpm") else "")
    )

    print("\n=== VideoDNA v2.0.4 Analysis Complete ===")
    print(f"How It Was Made: {forensics['verdict_summary']} (Confidence: {conf_pct}%)")
    print(f"Mode / Size:     {mode.upper()} ({probe['duration_sec']}s, {probe['width']}x{probe['height']})")
    print(f"Shots Detected:  {shots['shot_count']} ({shots.get('hard_cuts', 0)} hard cuts + {shots.get('soft_transitions', 0)} soft transitions) | ASL {shots['avg_shot_length_sec']}s")
    print(f"Retention Score: {diagnostics['overall_retention_score']}/10 (heuristic) | Dead-Zones: {len(shots['dead_zones'])}")
    print(f"Audio Status:    {audio_summary}")
    print(f"Remake Batches:  {remake_plan['total_batches']} batch(es)")
    print(f"DNA Brief (AI):  {paths['dna_md']}")
    print(f"HTML Dashboard:  {paths['report_html']}")
    print(f"Review Template: {paths['agent_review_template']}")
    return 0


def cmd_apply_review(args: argparse.Namespace) -> int:
    out_dir = Path(args.dir).expanduser().resolve()
    bp_path = out_dir / "blueprint.json"
    rev_path = Path(args.review_file).expanduser().resolve() if args.review_file else (out_dir / "agent_review.json")
    if not bp_path.exists() or not rev_path.exists():
        print(f"[Error] Missing blueprint.json or agent_review.json in {out_dir}", file=sys.stderr)
        return 1

    blueprint = json.loads(bp_path.read_text(encoding="utf-8"))
    review = json.loads(rev_path.read_text(encoding="utf-8"))
    blueprint["agent_review"] = review

    if blueprint.get("forensics"):
        fused = fuse_review_probabilities(blueprint["forensics"], review)
        blueprint["fused_forensics"] = fused
        # v2.0.4: keep the frame paths recorded on the shots in the
        # blueprint (apply-review re-renders the cards from them).
        shot_frames = {
            int(s["id"]): dict(s.get("frame_paths") or {})
            for s in (blueprint.get("shots") or {}).get("shots", [])
        }
        blueprint["remake_plan"] = build_remake_plan(
            blueprint["forensics"], blueprint["audio"], blueprint["visual"]["color_dna"],
            fused=fused, shot_frames=shot_frames,
        )
        print(
            f"[VideoDNA] Fused Verdict: {fused['verdict_summary']} "
            f"(Confidence: {int(round(fused['overall_confidence']*100))}% | Conflicts: {len(fused['conflicts'])})"
        )

    paths = write_outputs(out_dir, blueprint)
    if args.save_vault:
        save_to_vault(blueprint)

    print("[VideoDNA] Merged AI Brain review into:")
    print(f"  - {paths['dna_md']}")
    print(f"  - {paths['report_html']}")
    return 0


def cmd_moments(args: argparse.Namespace) -> int:
    """Extract 6fps zoomed frame strips around top transitions/highlights/tricks."""
    out_dir = Path(args.dir).expanduser().resolve()
    bp_path = out_dir / "blueprint.json"
    if not bp_path.exists():
        print(f"[Error] blueprint.json not found in {out_dir}", file=sys.stderr)
        return 1
    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    video_path = Path(bp["probe"]["file"])
    duration = bp["probe"]["duration_sec"]

    peaks: list[tuple[float, str]] = []
    for s in bp["shots"]["shots"][1:]:
        peaks.append((s["start"], f"{s.get('transition_in', 'cut')}_shot_{s['id']}"))
    for tr in (bp["audio"].get("audio_transients") or [])[:5]:
        if all(abs(tr["t"] - pt) > 1.2 for pt, _ in peaks):
            peaks.append((tr["t"], f"audio_peak_{tr['jump_db']}dB"))

    peaks = sorted(peaks, key=lambda x: x[0])[: args.limit]
    m_dir = out_dir / "moments"
    m_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for idx, (t, label) in enumerate(peaks, start=1):
        s_t = max(0.0, t - 0.8)
        e_t = min(duration, t + 1.0)
        out_jpg = m_dir / f"moment_{idx:02d}_{label}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{s_t:.2f}", "-to", f"{e_t:.2f}",
            "-i", str(video_path),
            "-vf", "scale=320:-2,fps=6,tile=4x3",
            "-frames:v", "1", "-q:v", "2", "-y", str(out_jpg),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and out_jpg.exists():
            generated.append({"t": t, "label": label, "sheet": str(out_jpg)})

    print(json.dumps({"moments_extracted": len(generated), "moments": generated}, indent=2))
    return 0


def _tile_dims(n_frames: int) -> tuple[int, int]:
    """Smallest near-square tile grid holding n_frames (caps at 8x8 = 64)."""
    n = max(1, min(n_frames, 64))
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def cmd_frames(args: argparse.Namespace) -> int:
    """Extract zoomed micro-frames between start and end seconds for forensic inspection."""
    video_path = Path(args.video).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps_expr = f"fps=1/{max(0.05, args.every)}"
    n_frames = max(1, int((args.end - args.start) / max(0.05, args.every)) + 1)
    cols, rows = _tile_dims(n_frames)
    vf = f"scale=320:-2,{fps_expr},tile={cols}x{rows}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(args.start), "-to", str(args.end),
        "-i", str(video_path),
        "-vf", vf, "-frames:v", "1", "-q:v", "2", "-y", str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Error] ffmpeg failed: {res.stderr}", file=sys.stderr)
        return 1
    print(f"[VideoDNA] Zoomed frame sheet saved: {out_path}")
    return 0


def cmd_vault(args: argparse.Namespace) -> int:
    if args.vault_cmd == "list":
        items = list_vault()
        if not items:
            print("Vault is empty. Run `videodna analyze <video> --save-vault`.")
            return 0
        for x in items:
            print(
                f"- {x['name']:28s} | {x['mode']:5s} | {x['duration_sec']:6.1f}s | "
                f"ASL {x['avg_shot_length_sec']:4.1f}s | Score {x['retention_score']}/10 | {x['color_grade']}"
            )
        return 0
    if args.vault_cmd == "compare":
        print(json.dumps(crack_channel_patterns(), indent=2, ensure_ascii=False))
        return 0
    if args.vault_cmd == "fuse":
        res = fuse_dna(args.hook, args.pacing, args.audio)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    print(f"VideoDNA v{__version__} Environment Check:")
    for tool in ("ffmpeg", "ffprobe", "git"):
        path = shutil.which(tool)
        status = f"OK ({path})" if path else "MISSING"
        print(f"  - {tool:8s}: {status}")
    yt_dlp_cmd = find_yt_dlp()
    print(f"  - yt-dlp  : {'OK (' + ' '.join(yt_dlp_cmd) + ')' if yt_dlp_cmd else 'MISSING (optional for local MP4)'}")
    # Only check modules the CLI actually imports. `requests` is NOT checked:
    # nothing in the code imports it (it is only an optional [ai] extra), so
    # a fresh base install must not report it MISSING.
    for mod in ("numpy", "PIL"):
        try:
            __import__(mod)
            print(f"  - {mod:8s}: OK")
        except ImportError:
            print(f"  - {mod:8s}: MISSING")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videodna",
        description="VideoDNA v2.0.4 — 10x Video Forensics, Viral X-Ray & Beat-The-Original Engine.",
    )
    p.add_argument("--version", action="version", version=f"videodna {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="Analyze a local video file or YouTube/Shorts/TikTok URL.")
    pa.add_argument("source", help="Path to local video (.mp4/.mkv/.mov) OR video URL.")
    pa.add_argument("-o", "--output", help="Output directory (default: out/<name>).")
    pa.add_argument("--mode", choices=["auto", "short", "long"], default="auto", help="Analysis mode.")
    pa.add_argument("--threshold", type=float, default=None, help="Override ffmpeg scdet cut threshold.")
    pa.add_argument("--transcript", help="Optional local .srt/.vtt subtitle file.")
    pa.add_argument("--transcribe", action="store_true", help="Transcribe speech locally with Faster-Whisper.")
    pa.add_argument("--transcribe-model", default="base", help="Faster-Whisper model (default: base).")
    pa.add_argument("--max-comments", type=int, default=80, help="Max audience comments to mine on URL input.")
    pa.add_argument("--sub-langs", default="en,bn,hi", help="Subtitle languages for yt-dlp.")
    pa.add_argument("--save-vault", action="store_true", help="Save DNA snapshot into Style Vault.")
    pa.add_argument("--vault-name", help="Custom name for Style Vault entry.")
    pa.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("apply-review", help="Fuse AI Agent visual review & Beat-The-Original plan into DNA.md & report.html.")
    pr.add_argument("dir", help="Output folder generated by `videodna analyze`.")
    pr.add_argument("--review-file", help="Custom JSON review path (default: <dir>/agent_review.json).")
    pr.add_argument("--save-vault", action="store_true", help="Update Style Vault with reviewed DNA.")
    pr.set_defaults(func=cmd_apply_review)

    pm = sub.add_parser("moments", help="Extract 6fps zoomed frame strips around top transitions/highlights/tricks.")
    pm.add_argument("dir", help="Output folder generated by `videodna analyze`.")
    pm.add_argument("--limit", type=int, default=5, help="Max peak moments to extract.")
    pm.set_defaults(func=cmd_moments)

    pf = sub.add_parser("frames", help="Extract dense zoomed frame grid between two timestamps.")
    pf.add_argument("video", help="Path to video file.")
    pf.add_argument("--start", type=float, required=True)
    pf.add_argument("--end", type=float, required=True)
    pf.add_argument("--every", type=float, default=0.2, help="Seconds between frames.")
    pf.add_argument("-o", "--output", required=True, help="Output JPG path.")
    pf.set_defaults(func=cmd_frames)

    pv = sub.add_parser("vault", help="Style Vault: list, compare (Channel Pattern Cracker), or fuse DNA.")
    vsub = pv.add_subparsers(dest="vault_cmd", required=True)
    vsub.add_parser("list", help="List saved video DNA snapshots.")
    vsub.add_parser("compare", help="Extract common viral patterns across all saved videos.")
    pvf = vsub.add_parser("fuse", help="Fuse Hook + Pacing + Audio DNA from 3 saved videos.")
    pvf.add_argument("--hook", required=True, help="Vault video name for Hook DNA.")
    pvf.add_argument("--pacing", required=True, help="Vault video name for Pacing/Color DNA.")
    pvf.add_argument("--audio", required=True, help="Vault video name for Audio/Rhythm DNA.")
    pv.set_defaults(func=cmd_vault)

    pd = sub.add_parser("doctor", help="Check system environment (ffmpeg, python, yt-dlp).")
    pd.set_defaults(func=cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
