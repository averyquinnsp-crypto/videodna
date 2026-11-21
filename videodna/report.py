"""Layered AI-Native DNA.md, blueprint.json, sub-docs (shots, remake_plan, SRT, comments), and report.html (v2.0.1)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _sec_to_srt_time(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    hrs, rem = divmod(ms, 3600000)
    mins, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def write_outputs(out_dir: Path, blueprint: dict[str, Any]) -> dict[str, str]:
    """Generate DNA.md, dna/* layered files, blueprint.json, and interactive report.html."""
    dna_sub = out_dir / "dna"
    dna_sub.mkdir(parents=True, exist_ok=True)

    # 1. Machine-readable blueprint.json
    bp_path = out_dir / "blueprint.json"
    bp_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Sub-files for token-efficient layered AI reading
    (dna_sub / "shots.md").write_text(_render_shots_md(blueprint), encoding="utf-8")
    (dna_sub / "remake_plan.md").write_text(_render_remake_md(blueprint), encoding="utf-8")
    (dna_sub / "comments.md").write_text(_render_comments_md(blueprint), encoding="utf-8")
    (dna_sub / "transcript.md").write_text(_render_transcript_md(blueprint), encoding="utf-8")

    # Export SRT captions when transcript exists
    if blueprint.get("transcript"):
        srt_lines = []
        for idx, row in enumerate(blueprint["transcript"], start=1):
            srt_lines.append(f"{idx}\n{_sec_to_srt_time(row['start'])} --> {_sec_to_srt_time(row['end'])}\n{row['text']}\n")
        (dna_sub / "captions.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    # 3. Review template for AI Agent Brain (create if missing)
    review_tpl_path = out_dir / "agent_review.json"
    if not review_tpl_path.exists():
        review_tpl_path.write_text(
            json.dumps(_build_review_template(blueprint), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # 4. Top-level DNA.md
    dna_path = out_dir / "DNA.md"
    dna_path.write_text(_render_dna_md(blueprint), encoding="utf-8")

    # 5. Interactive HTML dashboard
    html_path = out_dir / "report.html"
    html_path.write_text(_render_html_report(blueprint), encoding="utf-8")

    return {
        "dna_md": str(dna_path),
        "blueprint_json": str(bp_path),
        "report_html": str(html_path),
        "remake_plan_md": str(dna_sub / "remake_plan.md"),
        "agent_review_template": str(review_tpl_path),
    }


def _build_review_template(bp: dict[str, Any]) -> dict[str, Any]:
    forensics = bp.get("forensics") or {}
    per_shot = forensics.get("per_shot") or []
    return {
        "reviewer": "AI Agent Brain",
        "production_verdict": forensics.get("verdict_summary", "Verify visually"),
        "visual_style_bible": "Fill: Lighting, color grade, camera feel, typography & character consistency rules",
        "story_formula": {
            "hook_0_3s": "Fill: How the original stopped the scroll (visual + text + audio)",
            "core_mechanic": "Fill: Curiosity gap / storytelling arc / contrast trick",
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
            "pacing_and_visual_upgrades": ["Fill exact timestamp fixes and visual upgrades"],
            "audio_and_sfx_upgrades": ["Fill exact LUFS/ducking/SFX upgrades"],
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
                "source_type": s["predicted_type"],
                "confidence": s["confidence"],
                "what_happens": "",
                "remake_prompt_or_plan": "",
            }
            for s in per_shot[:16]
        ],
    }


def _render_dna_md(bp: dict[str, Any]) -> str:
    p = bp["probe"]
    sh = bp["shots"]
    v = bp["visual"]
    a = bp["audio"]
    f = bp.get("forensics") or {}
    fused = bp.get("fused_forensics")
    rm = bp.get("remake_plan") or {}
    d = bp["diagnostics"]
    meta = bp.get("platform_meta") or {}
    c_intel = bp.get("comments_intel") or {}
    review = bp.get("agent_review")

    verdict_src = fused or f
    verdict_str = verdict_src.get("verdict_summary", "indeterminate")
    verdict_conf = int(round(float(verdict_src.get("overall_confidence", 0.0)) * 100))
    verdict_tag = "Fused Tool + Visual Review" if fused else "Offline Pixel/Motion Forensics"

    audio_line = (
        f"`{a.get('audio_type', 'unknown')}` · {a.get('loudness_note', 'N/A')}"
        if a.get("is_silent_track")
        else (
            f"`{a.get('audio_type', 'unknown')}` · `{a.get('loudness_note', 'N/A')}`"
            + (f" · Est. BPM: `{a['estimated_bpm']}`" if a.get("estimated_bpm") else "")
            + f" · Silence: `{a.get('dead_air_total_sec', 0)}s` · Cut-Sync: `{int(a.get('cut_beat_sync_ratio', 0)*100)}%`"
        )
    )

    audio_score_disp = "N/A (silent)" if d["scores"]["audio_dynamics"] is None else f"`{d['scores']['audio_dynamics']}`"
    lines = [
        f"# VideoDNA v2.0.1 Report — `{p['filename']}`",
        "",
        "## 1. Executive Telemetry & Production Verdict",
        f"- **How It Was Made ({verdict_tag}):** **{verdict_str}** (Confidence: **{verdict_conf}%**)",
        f"- **Mode:** `{sh['mode'].upper()}` | **Duration:** `{p['duration_sec']}s` | **Resolution:** `{p['width']}x{p['height']}` (`{p['orientation']}`) | **FPS:** `{p['avg_fps']}` ({'VFR' if p['is_vfr'] else 'CFR'})",
        f"- **Retention Health Score (heuristic):** **{d['overall_retention_score']}/10** (Hook: `{d['scores']['hook_velocity']}` · Pacing: `{d['scores']['pacing_health']}` · Audio: {audio_score_disp})",
        f"- *{d.get('score_note', '')}*" if d.get("score_note") else "",
        f"- **Shots & Transitions:** `{sh['shot_count']} shots` (`{sh.get('hard_cuts', 0)} hard cuts` + `{sh.get('soft_transitions', 0)} soft wipes/reveals`) · Avg Shot: `{sh['avg_shot_length_sec']}s` · `{sh['cuts_per_minute']} transitions/min` (`{sh['pacing_style']}`)",
        f"- **Hook Velocity:** `{sh['hook_cuts_0_3s']}` transitions in 0–3s · `{sh['hook_cuts_0_10s']}` in 0–10s",
        f"- **Audio Forensics:** {audio_line}",
        f"- **Color DNA:** `{v['color_dna']['grade_style']}` · Palette: `{', '.join(v['color_dna']['palette_hex'])}` · 3D LUT: `{v['color_dna']['lut_file']}`",
    ]

    if meta.get("title"):
        lines.append(
            f"- **Platform Meta:** `{meta.get('platform')}` · **{meta.get('title')}** by `{meta.get('channel')}` "
            f"(Views: `{meta.get('view_count')}` · Likes: `{meta.get('like_count')}`)"
        )

    sec = 2
    lines += ["", f"## {sec}. Shot-Level Production Forensics (Tool Evidence)"]
    sec += 1
    if f.get("low_confidence_shots"):
        lines.append(f"- ⚠️ **Low-Confidence Shots (Inspect Visually):** `{', '.join(f'#{sid}' for sid in f['low_confidence_shots'])}`")
    if fused and fused.get("conflicts"):
        lines.append("- ⚡ **Tool vs. Visual Conflicts Flagged:**")
        for cf in fused["conflicts"]:
            lines.append(f"  - Shot #{cf['shot_id']}: Tool `{cf['tool_call']}` vs Agent `{cf['agent_call']}`")

    lines.append("| Shot | Window | Transition In | Production Call | Conf | Noise | Flat% | 2D Coh | Evidence |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
    for sf in (verdict_src.get("per_shot") or [])[:20]:
        sh_match = next((x for x in sh["shots"] if x["id"] == sf["shot_id"]), {})
        m = sf.get("metrics") or {}
        lbl = sf.get("fused_type") or sf.get("predicted_type")
        conf_p = int(round(float(sf.get("fused_confidence") or sf.get("confidence") or 0) * 100))
        ev = (sf.get("evidence") or [""])[0][:75]
        coh_str = "static" if float(m.get("mean_frame_diff", 0)) < 1.0 else f"{int(float(m.get('planar_coherence', 0))*100)}%"
        lines.append(
            f"| #{sf['shot_id']:02d} | {sf['start']:.1f}–{sf['end']:.1f}s ({sf['duration']:.1f}s) "
            f"| `{sh_match.get('transition_in', '-')}` | **`{lbl}`** | {conf_p}% "
            f"| {m.get('sensor_noise', '-')} | {int(float(m.get('flat_region_ratio', 0))*100)}% "
            f"| {coh_str} | {ev} |"
        )

    lines += ["", f"## {sec}. Container & Cadence Signals"]
    sec += 1
    if p["forensic_signals"]:
        for k, vals in p["forensic_signals"].items():
            lines.append(f"- `{k}`: {', '.join(vals)}")
    for note in p["cadence_notes"]:
        lines.append(f"- {note}")

    lines += ["", f"## {sec}. Model-Aware Remake Batch Plan"]
    sec += 1
    if rm.get("batches"):
        lines.append("| Batch | Shots | Window | Production Type | Recommended Tool / Pipeline |")
        lines.append("|---:|---|---|---|---|")
        for b in rm["batches"]:
            s_ids = ", ".join(f"#{i}" for i in b["shot_ids"])
            lines.append(f"| B{b['batch_id']} | {s_ids} | {b['start']:.1f}–{b['end']:.1f}s ({b['total_duration']}s) | `{b['production_type']}` | {b['tool']} |")
        lines.append(f"- **Audio Route:** {rm.get('audio_pipeline', '')}")
        lines.append(f"- **Color Route:** {rm.get('color_pipeline', '')}")

    lines += ["", f"## {sec}. Retention Audit & Beat-The-Original Signals"]
    sec += 1
    if d.get("notes"):
        for n in d["notes"]:
            lines.append(f"- ℹ️ {n}")
    if d["strengths"]:
        lines.append("### What Works Well")
        for s in d["strengths"]:
            lines.append(f"- ✅ {s}")
    if d["weaknesses"]:
        lines.append("### Weaknesses / Drop-Off Risks")
        for w in d["weaknesses"]:
            lines.append(f"- ⚠️ **[{w['severity'].upper()}] {w['area']}:** {w['finding']}")
    if d["beat_the_original_actions"]:
        lines.append("### Auto-Detected Ways to Beat This Video")
        for act_txt in d["beat_the_original_actions"]:
            lines.append(f"- 🚀 {act_txt}")

    if c_intel.get("total_mined", 0) > 0:
        lines += [
            "",
            f"## {sec}. Audience Comment Intelligence ({c_intel['total_mined']} comments mined)",
            f"- Questions: `{len(c_intel.get('questions', []))}` · Complaints: `{len(c_intel.get('complaints', []))}` · Requests: `{len(c_intel.get('requests', []))}` · Timestamp Callouts: `{len(c_intel.get('timestamp_highlights', []))}`",
            "- Read full breakdown in `dna/comments.md`",
        ]
        sec += 1

    if review:
        lines += [
            "",
            f"## {sec}. AI Brain Visual Review & 10x Beat-The-Original Blueprint",
            f"- **Production Verdict:** {review.get('production_verdict', '')}",
            f"- **Style Bible:** {review.get('visual_style_bible', '')}",
        ]
        sec += 1
        sf_rev = review.get("story_formula") or {}
        if sf_rev:
            lines += [
                f"- **Hook (0–3s):** {sf_rev.get('hook_0_3s', '')}",
                f"- **Core Mechanic:** {sf_rev.get('core_mechanic', '')}",
                f"- **Reusable Template:** {sf_rev.get('reusable_template', '')}",
            ]
        bto = review.get("beat_the_original") or {}
        if bto.get("stronger_hooks"):
            lines.append("### Upgraded Hooks (Stronger Than Original)")
            for h in bto["stronger_hooks"]:
                lines.append(f"- 🔥 {h}")
        if bto.get("pacing_and_visual_upgrades"):
            lines.append("### Pacing & Visual Surgery")
            for u in bto["pacing_and_visual_upgrades"]:
                lines.append(f"- 🎬 {u}")

    lines += [
        "",
        f"## {sec}. Layered File Map (Token-Smart Reading)",
        "| File | Purpose |",
        "|---|---|",
        "| `hook_strip.jpg` | Opening critical seconds frame strip (inspect FIRST!) |",
        "| `storyboard.jpg` | Labeled keyframe contact sheet across all shots |",
        "| `audio_timeline.png` | Audio energy curve / silent indicator + yellow cut markers |",
        "| `reference_grade.cube` | Auto-generated 17×17×17 3D color LUT |",
        "| `dna/shots.md` | Complete shot table + physical forensic metrics + dead-zones |",
        "| `dna/remake_plan.md` | Model-aware batched remake strategy & tool routing |",
        "| `dna/transcript.md` | Timestamped transcript (+ `dna/captions.srt` when present) |",
        "| `dna/comments.md` | Mined viewer complaints, questions, requests & timestamps |",
        "| `report.html` | Interactive offline dark-mode visual dashboard |",
    ]
    return "\n".join(lines) + "\n"


def _render_shots_md(bp: dict[str, Any]) -> str:
    sh = bp["shots"]
    vis_map = {v["shot_id"]: v for v in bp["visual"]["shot_visuals"]}
    f_src = bp.get("fused_forensics") or bp.get("forensics") or {}
    f_map = {x["shot_id"]: x for x in f_src.get("per_shot", [])}

    lines = [
        f"# Shot-by-Shot Forensic Timeline ({sh['shot_count']} shots | Mode: {sh['mode']})",
        "",
        "| Shot | Window | Dur | Transition | Predicted Type | Conf | Motion | Noise | Flat% | Keyframe |",
        "|---:|---|---:|---|---|---:|---|---:|---:|---|",
    ]
    for s in sh["shots"]:
        v = vis_map.get(s["id"], {})
        fx = f_map.get(s["id"], {})
        m = fx.get("metrics") or {}
        ptype = fx.get("fused_type") or fx.get("predicted_type") or "-"
        conf = f"{int(round(float(fx.get('fused_confidence') or fx.get('confidence') or 0)*100))}%" if fx else "-"
        lines.append(
            f"| #{s['id']:02d} | {s['start']:.2f}–{s['end']:.2f}s | {s['duration']:.2f}s | `{s.get('transition_in', '-')}` "
            f"| `{ptype}` | {conf} | {v.get('motion_class', '-')} ({v.get('motion_energy', '-')}) "
            f"| {m.get('sensor_noise', '-')} | {int(float(m.get('flat_region_ratio', 0))*100)}% | `{v.get('keyframe') or '-'}` |"
        )
    true_dz = bp["diagnostics"].get("true_dead_zones") or []
    if true_dz:
        lines += ["", "## True Static Dead-Zones (Retention Drop-Off Risks)"]
        for dz in true_dz:
            lines.append(
                f"- **Shot #{dz['shot_id']} ({dz['start']}s–{dz['end']}s, {dz['duration']}s) [{dz['severity'].upper()}]:** {dz['reason']}"
            )
    return "\n".join(lines) + "\n"


def _render_remake_md(bp: dict[str, Any]) -> str:
    rm = bp.get("remake_plan") or {}
    lines = [
        "# Model-Aware Batched Remake Plan",
        "",
        f"- **Total Production Batches:** `{rm.get('total_batches', 0)}`",
        f"- **Audio Pipeline:** {rm.get('audio_pipeline', '')}",
        f"- **Color & Finish Pipeline:** {rm.get('color_pipeline', '')}",
        "",
        "## Shot Batches",
    ]
    for b in rm.get("batches", []):
        s_ids = ", ".join(f"#{i}" for i in b["shot_ids"])
        lines += [
            f"### Batch {b['batch_id']} — `{b['production_type']}` ({b['start']:.1f}s – {b['end']:.1f}s | {b['total_duration']}s)",
            f"- **Shots Included:** {s_ids}",
            f"- **Recommended Tool:** {b['tool']} (`{b['cost_tier']}`)",
            f"- **Strategy:** {b['strategy']}",
            "",
        ]
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
        return "# Transcript\n\nNo subtitle/transcript track available in Lite mode. Use URL ingest or Kaggle Deep GPU mode (Faster-Whisper) for word-level transcription.\n"
    lines = ["# Timestamped Transcript", ""]
    for row in tr:
        lines.append(f"- **[{row['start']:.1f}s – {row['end']:.1f}s]** {row['text']}")
    return "\n".join(lines) + "\n"


def _render_html_report(bp: dict[str, Any]) -> str:
    p = bp["probe"]
    sh = bp["shots"]
    v = bp["visual"]
    a = bp["audio"]
    f_src = bp.get("fused_forensics") or bp.get("forensics") or {}
    rm = bp.get("remake_plan") or {}
    d = bp["diagnostics"]
    meta = bp.get("platform_meta") or {}
    review = bp.get("agent_review") or {}

    title = html.escape(meta.get("title") or p["filename"])
    verdict_str = html.escape(str(f_src.get("verdict_summary", "indeterminate")))
    verdict_conf = int(round(float(f_src.get("overall_confidence", 0.0)) * 100))

    audio_kpi = "SILENT" if a.get("is_silent_track") else f"{a.get('integrated_lufs') or 'N/A'} LUFS"
    swatches = "".join(
        f'<div class="swatch"><span style="background:{hex_c}"></span><code>{hex_c}</code></div>'
        for hex_c in v["color_dna"]["palette_hex"]
    )
    weak_html = "".join(
        f'<li class="sev-{html.escape(w["severity"])}"><b>{html.escape(w["area"])}:</b> {html.escape(w["finding"])}</li>'
        for w in d["weaknesses"]
    ) or "<li>No major structural weaknesses detected automatically.</li>"
    str_html = "".join(f"<li>✅ {html.escape(s)}</li>" for s in d["strengths"]) or "<li>Review visually to confirm strengths.</li>"
    up_html = "".join(f"<li>🚀 {html.escape(u)}</li>" for u in d["beat_the_original_actions"]) or "<li>Run AI Brain review for custom 10x hooks.</li>"

    shot_rows = ""
    for sf in (f_src.get("per_shot") or [])[:24]:
        sh_m = next((x for x in sh["shots"] if x["id"] == sf["shot_id"]), {})
        m = sf.get("metrics") or {}
        lbl = html.escape(str(sf.get("fused_type") or sf.get("predicted_type")))
        cp = int(round(float(sf.get("fused_confidence") or sf.get("confidence") or 0) * 100))
        shot_rows += (
            f"<tr><td>#{sf['shot_id']:02d}</td><td>{sf['start']:.1f}–{sf['end']:.1f}s ({sf['duration']:.1f}s)</td>"
            f"<td><code>{html.escape(str(sh_m.get('transition_in', '-')))}</code></td>"
            f"<td><b>{lbl}</b> ({cp}%)</td>"
            f"<td>{m.get('sensor_noise', '-')}</td><td>{int(float(m.get('flat_region_ratio', 0))*100)}%</td></tr>"
        )

    batch_rows = "".join(
        f"<tr><td>B{b['batch_id']}</td><td>{', '.join(f'#{i}' for i in b['shot_ids'])}</td>"
        f"<td>{b['start']:.1f}–{b['end']:.1f}s</td><td><code>{html.escape(b['production_type'])}</code></td>"
        f"<td>{html.escape(b['tool'])}</td></tr>"
        for b in rm.get("batches", [])
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
          <h3>Upgraded Hooks</h3><ul>{hooks_li}</ul>
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
<title>VideoDNA v2.0.1 — {title}</title>
<style>
  :root {{ --bg:#0b0f19; --card:#131a2a; --border:#23304d; --text:#e6edf8; --muted:#8fa3c7; --accent:#58a6ff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.55; padding:18px; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  header {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:12px; border-bottom:1px solid var(--border); padding-bottom:14px; margin-bottom:20px; }}
  h1 {{ font-size:1.3rem; margin:4px 0 0; }}
  .badge {{ display:inline-block; background:#1c2944; color:var(--accent); padding:4px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; }}
  .kpi .lbl {{ color:var(--muted); font-size:0.75rem; text-transform:uppercase; }}
  .kpi .val {{ font-size:1.35rem; font-weight:700; margin-top:4px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px; margin-bottom:18px; }}
  .card.highlight {{ border-color:var(--accent); background:linear-gradient(145deg,#131d33,#101726); }}
  h2 {{ font-size:1.08rem; margin:0 0 12px; color:#fff; }}
  h3 {{ font-size:0.95rem; margin:14px 0 8px; color:var(--accent); }}
  img {{ width:100%; border-radius:10px; border:1px solid var(--border); margin-top:6px; }}
  .swatches {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .swatch {{ display:flex; align-items:center; gap:6px; background:#0d1322; padding:6px 10px; border-radius:8px; border:1px solid var(--border); }}
  .swatch span {{ width:18px; height:18px; border-radius:4px; display:inline-block; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.86rem; }}
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
      <span class="badge">VideoDNA v2.0.1 · {sh['mode'].upper()} MODE</span>
      <h1>{title}</h1>
    </div>
    <div>
      <span class="badge">Made By: {verdict_str} ({verdict_conf}%)</span>
      <span class="badge">Retention: {d['overall_retention_score']}/10</span>
    </div>
  </header>

  <div class="grid">
    <div class="kpi"><div class="lbl">Duration</div><div class="val">{p['duration_sec']}s</div></div>
    <div class="kpi"><div class="lbl">Shots (Hard+Soft)</div><div class="val">{sh['shot_count']} ({sh.get('hard_cuts',0)}+{sh.get('soft_transitions',0)})</div></div>
    <div class="kpi"><div class="lbl">Avg Shot Length</div><div class="val">{sh['avg_shot_length_sec']}s</div></div>
    <div class="kpi"><div class="lbl">Audio Status</div><div class="val">{audio_kpi}</div></div>
    <div class="kpi"><div class="lbl">True Dead-Zones</div><div class="val">{len(d.get('true_dead_zones', []))}</div></div>
    <div class="kpi"><div class="lbl">Remake Batches</div><div class="val">{rm.get('total_batches', 0)}</div></div>
  </div>

  {review_block}

  <section class="card">
    <h2>🔬 Mathematical Shot-Level Forensics ({verdict_str} · {verdict_conf}% conf)</h2>
    <table>
      <thead><tr><th>Shot</th><th>Window</th><th>Transition</th><th>Production Call</th><th>Noise</th><th>Flat%</th></tr></thead>
      <tbody>{shot_rows}</tbody>
    </table>
  </section>

  <section class="card">
    <h2>🛠️ Model-Aware Batched Remake Plan</h2>
    <table>
      <thead><tr><th>Batch</th><th>Shots</th><th>Window</th><th>Type</th><th>Recommended Tool</th></tr></thead>
      <tbody>{batch_rows}</tbody>
    </table>
    <p style="color:var(--muted);font-size:0.86rem;"><b>Audio:</b> {html.escape(str(rm.get('audio_pipeline', '')))}<br><b>Color:</b> {html.escape(str(rm.get('color_pipeline', '')))}</p>
  </section>

  <section class="card">
    <h2>🚀 Beat-The-Original Diagnostic Scorecard</h2>
    <h3>What Works Well</h3><ul>{str_html}</ul>
    <h3>Weaknesses & Drop-Off Leaks</h3><ul>{weak_html}</ul>
    <h3>Concrete Upgrade Actions</h3><ul>{up_html}</ul>
  </section>

  <section class="card"><h2>🎯 Hook Strip</h2>{hook_img}</section>
  <section class="card"><h2>🔊 Audio Forensics & Cut Alignment</h2>{audio_img}</section>
  <section class="card"><h2>🎨 Color DNA ({html.escape(v['color_dna']['grade_style'])})</h2><div class="swatches">{swatches}</div></section>
  <section class="card"><h2>🎞️ Storyboard</h2>{board_img}</section>
</div>
</body>
</html>
"""
