"""Style Vault, Channel Pattern Cracker (multi-video comparison), and Hybrid DNA Fusion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_VAULT = Path.home() / ".videodna" / "vault"


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:60] or "video"


def save_to_vault(blueprint: dict[str, Any], name: str | None = None, vault_dir: Path = DEFAULT_VAULT) -> Path:
    vault_dir.mkdir(parents=True, exist_ok=True)
    raw_name = name or (blueprint.get("platform_meta") or {}).get("title") or blueprint["probe"]["filename"]
    slug = _slug(raw_name)
    target = vault_dir / f"{slug}.json"
    # Store compact DNA snapshot
    compact = {
        "name": slug,
        "title": (blueprint.get("platform_meta") or {}).get("title") or blueprint["probe"]["filename"],
        "mode": blueprint["shots"]["mode"],
        "duration_sec": blueprint["probe"]["duration_sec"],
        "orientation": blueprint["probe"]["orientation"],
        "avg_shot_length_sec": blueprint["shots"]["avg_shot_length_sec"],
        "cuts_per_minute": blueprint["shots"]["cuts_per_minute"],
        "hook_cuts_0_3s": blueprint["shots"]["hook_cuts_0_3s"],
        "pacing_style": blueprint["shots"]["pacing_style"],
        "integrated_lufs": blueprint["audio"].get("integrated_lufs"),
        "cut_beat_sync_ratio": blueprint["audio"].get("cut_beat_sync_ratio"),
        "color_grade": blueprint["visual"]["color_dna"]["grade_style"],
        "palette_hex": blueprint["visual"]["color_dna"]["palette_hex"],
        "retention_score": blueprint["diagnostics"]["overall_retention_score"],
        "strengths": blueprint["diagnostics"]["strengths"],
        "weaknesses": blueprint["diagnostics"]["weaknesses"],
        "agent_review": blueprint.get("agent_review"),
    }
    target.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def list_vault(vault_dir: Path = DEFAULT_VAULT) -> list[dict[str, Any]]:
    if not vault_dir.exists():
        return []
    items = []
    for p in sorted(vault_dir.glob("*.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def crack_channel_patterns(vault_dir: Path = DEFAULT_VAULT) -> dict[str, Any]:
    """Compare all saved videos in the vault to uncover winning channel/niche formulas."""
    items = list_vault(vault_dir)
    if not items:
        return {"count": 0, "message": "Vault is empty. Analyze videos with `--save-vault` first."}

    def _avg(key: str) -> float | None:
        vals = [float(x[key]) for x in items if x.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    grades: dict[str, int] = {}
    pacings: dict[str, int] = {}
    for x in items:
        g = x.get("color_grade") or "unknown"
        grades[g] = grades.get(g, 0) + 1
        ps = x.get("pacing_style") or "unknown"
        pacings[ps] = pacings.get(ps, 0) + 1

    top_grade = max(grades, key=grades.get)
    top_pacing = max(pacings, key=pacings.get)

    return {
        "count": len(items),
        "videos": [x["name"] for x in items],
        "niche_formula": {
            "avg_duration_sec": _avg("duration_sec"),
            "avg_shot_length_sec": _avg("avg_shot_length_sec"),
            "avg_cuts_per_min": _avg("cuts_per_minute"),
            "avg_hook_cuts_0_3s": _avg("hook_cuts_0_3s"),
            "avg_loudness_lufs": _avg("integrated_lufs"),
            "avg_cut_beat_sync": _avg("cut_beat_sync_ratio"),
            "dominant_pacing_style": top_pacing,
            "dominant_color_grade": top_grade,
        },
    }


def fuse_dna(hook_name: str, pacing_name: str, audio_name: str, vault_dir: Path = DEFAULT_VAULT) -> dict[str, Any]:
    """Create a Hybrid DNA Fusion spec combining best traits from 3 vault entries."""
    items = {x["name"]: x for x in list_vault(vault_dir)}
    missing = [n for n in (hook_name, pacing_name, audio_name) if n not in items]
    if missing:
        raise ValueError(f"Vault entry not found: {', '.join(missing)}. Available: {', '.join(items.keys())}")

    h, p, a = items[hook_name], items[pacing_name], items[audio_name]
    return {
        "fusion_recipe": {
            "hook_dna_from": h["name"],
            "target_hook_cuts_0_3s": h["hook_cuts_0_3s"],
            "pacing_dna_from": p["name"],
            "target_avg_shot_length_sec": p["avg_shot_length_sec"],
            "target_cuts_per_min": p["cuts_per_minute"],
            "target_color_grade": p["color_grade"],
            "target_palette_hex": p["palette_hex"],
            "audio_dna_from": a["name"],
            "target_loudness_lufs": a["integrated_lufs"] or -14.0,
            "target_cut_beat_sync": a["cut_beat_sync_ratio"],
        }
    }
