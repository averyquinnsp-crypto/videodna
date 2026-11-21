"""Layered AI-Native DNA.md, blueprint.json, sub-docs, and interactive offline report.html builder."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_outputs(out_dir: Path, blueprint: dict[str, Any]) -> dict[str, str]:
    """Generate DNA.md, dna/*.md, blueprint.json, and interactive report.html."""
    dna_sub = out_dir / "dna"
    dna_sub.mkdir(parents=True, exist_ok=True)

    # 1. Machine-readable blueprint.json
    bp_path = out_dir / "blueprint.json"
    bp_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Sub-files for token-efficient layered AI reading
    shots_md = _render_shots_md(blueprint)
    (dna_sub / "shots.md").write_text(shots_md, encoding="utf-8")

    comments_md = _render_comments_md(blueprint)
    (dna_sub / "comments.md").write_text(comments_md, encoding="utf-8")

    transcript_md = _render_transcript_md(blueprint)
    (dna_sub / "transcript.md").write_text(transcript_md, encoding="utf-8")

    # 3. Review template for AI Agent Brain (only create if not already present)
    review_tpl_path = out_dir / "agent_review.json"
    if not review_tpl_path.exists():
        review_tpl_path.write_text(
            json.dumps(_build_review_template(blueprint), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # 4. Top-level DNA.md (compact ~1.5k-2.5k tokens for instant AI comprehension)
    dna_md = _render_dna_md(blueprint)
    dna_path = out_dir / "DNA.md"
    dna_path.write_text(dna_md, encoding="utf-8")

    # 5. Interactive visual HTML dashboard
    html_str = _render_html_report(blueprint)
    html_path = out_dir / "report.html"
    html_path.write_text(html_str, encoding="utf-8")

    return {
        "dna_md": str(dna_path),
        "blueprint_json": str(bp_path),
        "report_html": str(html_path),
        "agent_review_template": str(review_tpl_path),
    }


def _build_review_template(bp: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewer": "AI Agent Brain",
        "production_verdict": "Fill: e.g., Mixed (60% AI video + 25% Motion Graphics + 15% Screen Cap) or 100% Talking-Head Live",
        "visual_style_bible": "Fill: Lighting, color grade, camera feel, typography & character consistency rules",
        "story_formula": {
            "hook_0_3s": "Fill: How the original stopped the scroll (visual + text + audio)",
            "core_mechanic": "Fill: Curiosity gap / storytelling arc / emotional trigger",
            "ending_or_loop": "Fill: Payoff / CTA / seamless loop mechanic",
            "reusable_template": "Fill: Fill-in-the-blanks script structure for any new topic",
        },
        "beat_the_original": {
            "biggest_weaknesses": ["Fill weakness #1 backed by timestamps/comments"],
            "stronger_hooks": [
                "Hook Variation 1 (Curiosity): ...",
                "Hook Variation 2 (Contrarian/Shock): ...",
                "Hook Variation 3 (Visual Proof): ...",
            ],
            "pacing_and_visual_upgrades": [
                "Fill exact timestamp fixes and higher-impact visual replacements"
            ],
            "audio_and_sfx_upgrades": [
                "Fill exact LUFS/ducking/SFX timing upgrades"
            ],
            "remake_tool_routing": {
                "cinematic_ai_shots": "Google Flow / Veo / Kling",
                "motion_graphics_and_captions": "HyperFrames / Remotion",
                "voice_and_mix": "Cartesia (or your TTS of choice) — master to -14 LUFS, tight compression, BGM ducking under the voice",
                "upscale_and_grade": "Optional GPU upscale on Kaggle (only when the user asks) + reference_grade.cube LUT",
            },
        },
        "shot_annotations": [
            {
                "shot_id": s["shot_id"],
                "source_type": "live | ai_video | avatar | 3d | 2d_anim | code_ui | screen_rec | static_zoom",
                "what_happens": "",
                "remake_prompt_or_plan": "",
            }
            for s in bp["visual"]["shot_visuals"][:12]
        ],
    }


def _render_dna_md(bp: dict[str, Any]) -> str:
    p = bp["probe"]
    sh = bp["shots"]
    v = bp["visual"]
    a = bp["audio"]
    d = bp["diagnostics"]
    meta = bp.get("platform_meta") or {}
    c_intel = bp.get("comments_intel") or {}
    review = bp.get("agent_review")

    audio_score = d["scores"]["audio_dynamics"]
    audio_score_txt = f"`{audio_score}`" if audio_score is not None else "`n/a (silent, excluded)`"
    det = sh.get("detection") or {}
    det_txt = (
        f" · Detection: `{det.get('scdet_cuts', '?')} scdet + {det.get('frame_diff_cuts', '?')} frame-diff cuts`"
        if det else ""
    )

    # Sections are numbered dynamically so the doc never jumps 4 -> 7 when
    # the optional comments/review sections are absent.
    sections: list[tuple[str, list[str]]] = []

    telemetry = [
        f"- **Mode:** `{sh['mode'].upper()}` | **Duration:** `{p['duration_sec']}s` | **Resolution:** `{p['width']}x{p['height']}` (`{p['orientation']}`) | **FPS:** `{p['avg_fps']}` ({'VFR' if p['is_vfr'] else 'CFR'})",
        f"- **Retention Health Score:** **{d['overall_retention_score']}/10** (Hook: `{d['scores']['hook_velocity']}` · Pacing: `{d['scores']['pacing_health']}` · Audio: {audio_score_txt})",
        f"- *{d.get('score_note', 'Retention scores are rule-of-thumb heuristics, not measured audience retention.')}*",
        f"- **Pacing:** `{sh['shot_count']} shots` (`{sh['total_cuts']} cuts`) · Avg Shot Length: `{sh['avg_shot_length_sec']}s` · `{sh['cuts_per_minute']} cuts/min` · Style: `{sh['pacing_style']}`{det_txt}",
        f"- **Hook Velocity:** `{sh['hook_cuts_0_3s']}` cuts in 0–3s · `{sh['hook_cuts_0_10s']}` cuts in 0–10s",
        f"- **Audio:** `{a.get('loudness_note', 'N/A')}` · Dead-Air: `{a.get('dead_air_total_sec', 0)}s` · Cut-to-Beat Sync: `{int(a.get('cut_beat_sync_ratio', 0)*100)}%`",
        f"- **Color DNA:** `{v['color_dna']['grade_style']}` · Palette: `{', '.join(v['color_dna']['palette_hex'])}` · LUT: `{v['color_dna']['lut_file']}`",
    ]
    if meta.get("title"):
        telemetry.append(
            f"- **Platform Meta:** `{meta.get('platform')}` · **{meta.get('title')}** by `{meta.get('channel')}` "
            f"(Views: `{meta.get('view_count')}` · Likes: `{meta.get('like_count')}`)"
        )
    if d.get("notes"):
        for n in d["notes"]:
            telemetry.append(f"- ℹ️ {n}")
    sections.append(("Executive Telemetry Card", telemetry))

    forensic: list[str] = []
    if p["forensic_signals"]:
        for k, vals in p["forensic_signals"].items():
            forensic.append(f"- `{k}`: {', '.join(vals)}")
    else:
        forensic.append("- No explicit editor/AIGC container tags (platform-stripped or clean export — verify visually via `hook_strip.jpg` & `storyboard.jpg`).")
    for note in p["cadence_notes"]:
        forensic.append(f"- {note}")
    sections.append(("Forensic & Production Signals", forensic))

    acts = [
        "| Act / Phase | Window | Duration | Shots | Cuts/Min |",
        "|---|---|---:|---:|---:|",
    ]
    for act in sh["macro_acts"]:
        acts.append(
            f"| {act['name']} | {act['start']}s – {act['end']}s | {act['duration']}s | {act['shot_count']} | {act['cuts_per_min']} |"
        )
    sections.append(("Structural Acts & Pacing Map", acts))

    audit: list[str] = []
    if d["strengths"]:
        audit.append("### What Works Well")
        for s in d["strengths"]:
            audit.append(f"- ✅ {s}")
    if d["weaknesses"]:
        audit.append("### Weaknesses / Drop-Off Risks")
        for w in d["weaknesses"]:
            audit.append(f"- ⚠️ **[{w['severity'].upper()}] {w['area']}:** {w['finding']}")
    if d["beat_the_original_actions"]:
        audit.append("### Auto-Detected Ways to Beat This Video")
        for act_txt in d["beat_the_original_actions"]:
            audit.append(f"- 🚀 {act_txt}")
    sections.append(("Retention Audit: Strengths, Weaknesses & Beat-The-Original Signals", audit))

    if c_intel.get("total_mined", 0) > 0:
        sections.append((
            f"Audience Comment Intelligence ({c_intel['total_mined']} comments mined)",
            [
                f"- Questions: `{len(c_intel.get('questions', []))}` · Complaints: `{len(c_intel.get('complaints', []))}` · Requests: `{len(c_intel.get('requests', []))}` · Timestamp Callouts: `{len(c_intel.get('timestamp_highlights', []))}`",
                "- Full comment breakdown: `dna/comments.md`",
            ],
        ))

    if review:
        rev = [
            f"- **Production Verdict:** {review.get('production_verdict', '')}",
            f"- **Style Bible:** {review.get('visual_style_bible', '')}",
        ]
        sf = review.get("story_formula") or {}
        if sf:
            rev += [
                f"- **Hook (0–3s):** {sf.get('hook_0_3s', '')}",
                f"- **Core Mechanic:** {sf.get('core_mechanic', '')}",
                f"- **Reusable Template:** {sf.get('reusable_template', '')}",
            ]
        bto = review.get("beat_the_original") or {}
        if bto.get("stronger_hooks"):
            rev.append("### Upgraded Hooks (Stronger Than Original)")
            for h in bto["stronger_hooks"]:
                rev.append(f"- 🔥 {h}")
        if bto.get("pacing_and_visual_upgrades"):
            rev.append("### Pacing & Visual Surgery")
            for u in bto["pacing_and_visual_upgrades"]:
                rev.append(f"- 🎬 {u}")
        sections.append(("AI Brain Visual Review & 10x Beat-The-Original Blueprint", rev))

    sections.append((
        "Layered File Map (Read On-Demand to Save Tokens)",
        [
            "| File | Purpose |",
            "|---|---|",
            "| `hook_strip.jpg` | Visual strip of the critical opening seconds (inspect FIRST!) |",
            "| `storyboard.jpg` | Labeled contact sheet of key shots with timestamps & motion classes |",
            "| `audio_timeline.png` | Audio energy curve + red dead-air zones + yellow visual cut ticks |",
            "| `reference_grade.cube` | Auto-generated 3D color LUT matching the reference tonal grade |",
            "| `dna/shots.md` | Full shot-by-shot table + dead-zone list |",
            "| `dna/transcript.md` | Timestamped spoken script (if available) |",
            "| `dna/comments.md` | Audience questions, complaints, requests & timestamp highlights |",
            "| `report.html` | Interactive visual dashboard for humans |",
            "| `blueprint.json` | Complete machine-readable JSON for automated remake pipelines |",
        ],
    ))

    lines = [f"# VideoDNA Report — `{p['filename']}`", ""]
    for idx, (title, body) in enumerate(sections, start=1):
        lines.append(f"## {idx}. {title}")
        lines.extend(body)
        lines.append("")
    return "\n".join(lines)


def _render_shots_md(bp: dict[str, Any]) -> str:
    sh = bp["shots"]
    vis_map = {v["shot_id"]: v for v in bp["visual"]["shot_visuals"]}
    lines = [
        f"# Shot-by-Shot Timeline ({sh['shot_count']} shots | Mode: {sh['mode']})",
        "",
        "| Shot | Start | End | Dur | Motion | Brightness | Keyframe |",
        "|---:|---:|---:|---:|---|---:|---|",
    ]
    for s in sh["shots"]:
        v = vis_map.get(s["id"], {})
        lines.append(
            f"| #{s['id']:02d} | {s['start']:.2f}s | {s['end']:.2f}s | {s['duration']:.2f}s "
            f"| {v.get('motion_class', 'unsampled')} ({v.get('motion_energy', '-')}) "
            f"| {v.get('brightness', '-')} | `{v.get('keyframe') or '-'}` |"
        )
    if sh["dead_zones"]:
        lines += ["", "## Dead-Zone Alerts (Retention Drop-Off Risks)"]
        for dz in sh["dead_zones"]:
            lines.append(
                f"- **Shot #{dz['shot_id']} ({dz['start']}s–{dz['end']}s, {dz['duration']}s) [{dz['severity'].upper()}]:** {dz['reason']}"
            )
    return "\n".join(lines) + "\n"


def _render_comments_md(bp: dict[str, Any]) -> str:
    c = bp.get("comments_intel") or {}
    if not c.get("total_mined"):
        return "# Audience Comments\n\nNo platform comments mined (local file or comments unavailable).\n"
    lines = [f"# Audience Comment Intelligence ({c['total_mined']} mined)", ""]
    for title, key in [
        ("Viewer Complaints (Fix These to Beat the Original!)", "complaints"),
        ("Unanswered Viewer Questions (Answer These in Your Video!)", "questions"),
        ("Viewer Requests / Part-2 Demand", "requests"),
        ("Top Comments Overall", "top_comments"),
    ]:
        items = c.get(key) or []
        lines.append(f"## {title}")
        if not items:
            lines.append("- None detected.")
        else:
            for it in items:
                lines.append(f"- `👍 {it['likes']}` {it['text']}")
        lines.append("")
    ts_items = c.get("timestamp_highlights") or []
    if ts_items:
        lines.append("## Timestamp Callouts (Exact Moments Viewers Replayed/Talked About)")
        for it in ts_items:
            lines.append(f"- `{', '.join(it['timestamps'])}` (`👍 {it['likes']}`) {it['text']}")
    return "\n".join(lines) + "\n"


def _render_transcript_md(bp: dict[str, Any]) -> str:
    tr = bp.get("transcript") or []
    if not tr:
        return "# Transcript\n\nNo subtitle/transcript track available. Use URL ingest (yt-dlp subtitles), `--transcribe` (local Faster-Whisper, optional extra), or Kaggle Deep GPU mode (Faster-Whisper) for transcription.\n"
    lines = ["# Timestamped Transcript", ""]
    for row in tr:
        lines.append(f"- **[{row['start']:.1f}s – {row['end']:.1f}s]** {row['text']}")
    return "\n".join(lines) + "\n"


def _render_html_report(bp: dict[str, Any]) -> str:
    p = bp["probe"]
    sh = bp["shots"]
    v = bp["visual"]
    a = bp["audio"]
    d = bp["diagnostics"]
    meta = bp.get("platform_meta") or {}
    review = bp.get("agent_review") or {}

    title = html.escape(meta.get("title") or p["filename"])
    swatches = "".join(
        f'<div class="swatch"><span style="background:{hex_c}"></span><code>{hex_c}</code></div>'
        for hex_c in v["color_dna"]["palette_hex"]
    )
    weak_html = "".join(
        f'<li class="sev-{html.escape(w["severity"])}"><b>{html.escape(w["area"])}:</b> {html.escape(w["finding"])}</li>'
        for w in d["weaknesses"]
    ) or "<li>No major structural weaknesses detected automatically.</li>"
    str_html = "".join(f"<li>✅ {html.escape(s)}</li>" for s in d["strengths"]) or "<li>Review visually to confirm strengths.</li>"
    up_html = "".join(f"<li>🚀 {html.escape(u)}</li>" for u in d["beat_the_original_actions"]) or "<li>Run AI Brain review for custom 10x upgrade hooks.</li>"

    acts_rows = "".join(
        f"<tr><td>{html.escape(act['name'])}</td><td>{act['start']}s–{act['end']}s</td>"
        f"<td>{act['duration']}s</td><td>{act['shot_count']}</td><td>{act['cuts_per_min']}</td></tr>"
        for act in sh["macro_acts"]
    )

    review_block = ""
    if review:
        hooks_li = "".join(
            f"<li>🔥 {html.escape(str(h))}</li>"
            for h in (review.get("beat_the_original") or {}).get("stronger_hooks", [])
        )
        review_block = f"""
        <section class="card highlight">
          <h2>🧠 AI Brain Audit & 10x Beat-The-Original Blueprint</h2>
          <p><b>Production Verdict:</b> {html.escape(str(review.get('production_verdict', '')))}</p>
          <p><b>Style Bible:</b> {html.escape(str(review.get('visual_style_bible', '')))}</p>
          <h3>Upgraded Hooks</h3>
          <ul>{hooks_li}</ul>
        </section>
        """

    hook_img = f'<img src="{html.escape(v["hook_strip"])}" alt="Hook Strip">' if v.get("hook_strip") else ""
    board_img = f'<img src="{html.escape(v["storyboard"])}" alt="Storyboard">' if v.get("storyboard") else ""
    audio_img = f'<img src="{html.escape(a["timeline_chart"])}" alt="Audio Timeline">' if a.get("timeline_chart") else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VideoDNA — {title}</title>
<style>
  :root {{ --bg:#0b0f19; --card:#131a2a; --border:#23304d; --text:#e6edf8; --muted:#8fa3c7; --accent:#58a6ff; --good:#3fb950; --warn:#d29922; --bad:#f85149; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.55; padding:18px; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  header {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:12px; border-bottom:1px solid var(--border); padding-bottom:14px; margin-bottom:20px; }}
  h1 {{ font-size:1.35rem; margin:0; }}
  .badge {{ display:inline-block; background:#1c2944; color:var(--accent); padding:4px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; }}
  .kpi .lbl {{ color:var(--muted); font-size:0.78rem; text-transform:uppercase; }}
  .kpi .val {{ font-size:1.5rem; font-weight:700; margin-top:4px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px; margin-bottom:18px; }}
  .card.highlight {{ border-color:var(--accent); background:linear-gradient(145deg,#131d33,#101726); }}
  h2 {{ font-size:1.08rem; margin:0 0 12px; color:#fff; }}
  h3 {{ font-size:0.95rem; margin:14px 0 8px; color:var(--accent); }}
  img {{ width:100%; border-radius:10px; border:1px solid var(--border); margin-top:6px; }}
  .swatches {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .swatch {{ display:flex; align-items:center; gap:6px; background:#0d1322; padding:6px 10px; border-radius:8px; border:1px solid var(--border); }}
  .swatch span {{ width:18px; height:18px; border-radius:4px; display:inline-block; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); }}
  ul {{ margin:0; padding-left:18px; }}
  li {{ margin-bottom:6px; }}
  .sev-high {{ color:#ff8b85; }}
  .sev-medium {{ color:#f0c46c; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <span class="badge">VideoDNA X-Ray · {sh['mode'].upper()} MODE</span>
      <h1>{title}</h1>
    </div>
    <div class="badge">Retention Score: {d['overall_retention_score']}/10</div>
  </header>

  <div class="grid">
    <div class="kpi"><div class="lbl">Duration</div><div class="val">{p['duration_sec']}s</div></div>
    <div class="kpi"><div class="lbl">Shots / Cuts</div><div class="val">{sh['shot_count']} / {sh['total_cuts']}</div></div>
    <div class="kpi"><div class="lbl">Avg Shot Length</div><div class="val">{sh['avg_shot_length_sec']}s</div></div>
    <div class="kpi"><div class="lbl">Hook Cuts (0–3s)</div><div class="val">{sh['hook_cuts_0_3s']}</div></div>
    <div class="kpi"><div class="lbl">Audio Loudness</div><div class="val">{a.get('integrated_lufs') or 'N/A'} LUFS</div></div>
    <div class="kpi"><div class="lbl">Dead-Zones</div><div class="val">{len(sh['dead_zones'])}</div></div>
  </div>

  {review_block}

  <section class="card">
    <h2>🚀 Beat-The-Original Diagnostic Scorecard</h2>
    <p style="color:var(--muted);font-size:0.85rem;">{html.escape(d.get('score_note', ''))}</p>
    <h3>What Works Well</h3><ul>{str_html}</ul>
    <h3>Weaknesses & Drop-Off Leaks</h3><ul>{weak_html}</ul>
    <h3>Concrete Upgrade Actions</h3><ul>{up_html}</ul>
  </section>

  <section class="card">
    <h2>🎯 Hook Strip (Critical Opening Frames)</h2>
    {hook_img}
  </section>

  <section class="card">
    <h2>🔊 Audio Energy, Dead-Air & Cut Alignment</h2>
    {audio_img}
  </section>

  <section class="card">
    <h2>🎨 Color DNA ({html.escape(v['color_dna']['grade_style'])})</h2>
    <div class="swatches">{swatches}</div>
    <p style="color:var(--muted);font-size:0.85rem;">Auto-exported 3D LUT: <code>{html.escape(str(v['color_dna']['lut_file']))}</code></p>
  </section>

  <section class="card">
    <h2>📐 Structural Acts & Pacing Map</h2>
    <table>
      <thead><tr><th>Act / Zone</th><th>Window</th><th>Duration</th><th>Shots</th><th>Cuts/Min</th></tr></thead>
      <tbody>{acts_rows}</tbody>
    </table>
  </section>

  <section class="card">
    <h2>🎞️ Storyboard (Key Shots & Motion Energy)</h2>
    {board_img}
  </section>
</div>
</body>
</html>
"""
