"""URL ingest (YouTube/Shorts/TikTok/Reels via yt-dlp) + Transcript & Audience Comment Miner."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


QUESTION_PATTERNS = re.compile(
    r"(\?|\bhow\b|\bwhy\b|\bwhat\b|\bwhich\b|\bwhere\b|\bname\b|কিভাবে|কেন|কোন|নাম কি|বুঝি\s*নাই|বুঝলাম\s*না)",
    re.IGNORECASE,
)
COMPLAINT_PATTERNS = re.compile(
    r"(\btoo fast\b|\btoo slow\b|\bboring\b|\bloud\b|\bclickbait\b|\bfake\b|\bwaste\b|\bannoying\b|\bcan't hear\b|\bmusic\b.*\bloud\b|বিরক্ত|মিউজিক.*বেশি|ফেক|সময় নষ্ট|বুঝা যায় না)",
    re.IGNORECASE,
)
REQUEST_PATTERNS = re.compile(
    r"(\bpart\s*2\b|\bplease\b|\bmake a video\b|\btutorial\b|\bnext\b|\bmore\b|\blink\b|পার্ট\s*২|ভিডিও বানান|টিউটোরিয়াল|আরও চাই|লিংক)",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def ingest_url(
    url: str,
    out_dir: Path,
    max_comments: int = 80,
    sub_langs: str = "en,bn,hi",
) -> dict[str, Any]:
    """Download video, metadata, subtitles, and top audience comments using yt-dlp."""
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        raise RuntimeError(
            "URL input requires 'yt-dlp', which is not installed. "
            "Install it with `pip install yt-dlp` or pass a local .mp4 file instead."
        )

    ingest_dir = out_dir / "source"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(ingest_dir / "video.%(ext)s")

    cmd = [
        yt_dlp,
        "--no-playlist",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-thumbnail",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--sub-format", "vtt/srt/best",
        "--get-comments",
        "--extractor-args", f"youtube:max_comments={max_comments},all,0;comment_sort=top",
        "-o", out_tpl,
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback without comments if platform doesn't support comment extraction
        cmd_fallback = [
            yt_dlp, "--no-playlist",
            "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "--merge-output-format", "mp4",
            "--write-info-json", "--write-thumbnail",
            "--write-subs", "--write-auto-subs", "--sub-langs", sub_langs,
            "-o", out_tpl, url,
        ]
        res2 = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if res2.returncode != 0:
            raise RuntimeError(f"yt-dlp failed:\n{res2.stderr[-800:]}")

    # Find downloaded video, info.json, thumbnail, and subtitle files
    video_files = sorted(ingest_dir.glob("video*.mp4")) or sorted(
        p for p in ingest_dir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    )
    if not video_files:
        raise RuntimeError("yt-dlp completed but no video file was found.")
    video_path = video_files[0]

    info_files = sorted(ingest_dir.glob("*.info.json"))
    meta: dict[str, Any] = {}
    comments_intel: dict[str, Any] = {}
    if info_files:
        raw_info = json.loads(info_files[0].read_text(encoding="utf-8", errors="ignore"))
        meta = {
            "url": url,
            "platform": raw_info.get("extractor_key") or raw_info.get("extractor"),
            "title": raw_info.get("title"),
            "channel": raw_info.get("channel") or raw_info.get("uploader"),
            "view_count": raw_info.get("view_count"),
            "like_count": raw_info.get("like_count"),
            "comment_count": raw_info.get("comment_count"),
            "upload_date": raw_info.get("upload_date"),
            "tags": (raw_info.get("tags") or [])[:20],
            "chapters": raw_info.get("chapters") or [],
            "description_snippet": (raw_info.get("description") or "")[:600],
        }
        comments_intel = mine_comments(raw_info.get("comments") or [])

    # Parse first available subtitle file into clean timestamped transcript
    sub_files = sorted(ingest_dir.glob("*.vtt")) + sorted(ingest_dir.glob("*.srt"))
    transcript = parse_subtitle_file(sub_files[0]) if sub_files else []

    thumbs = [p for p in ingest_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    thumb_rel = str(thumbs[0].relative_to(out_dir)) if thumbs else None

    return {
        "video_path": video_path,
        "platform_meta": meta,
        "thumbnail": thumb_rel,
        "transcript": transcript,
        "comments_intel": comments_intel,
    }


def mine_comments(raw_comments: list[dict[str, Any]]) -> dict[str, Any]:
    """Bucket viewer comments into actionable Beat-The-Original intelligence."""
    if not raw_comments:
        return {"total_mined": 0, "questions": [], "complaints": [], "requests": [], "timestamp_highlights": [], "top_comments": []}

    # Sort by like_count descending
    sorted_c = sorted(raw_comments, key=lambda c: int(c.get("like_count") or 0), reverse=True)
    questions, complaints, requests_list, ts_highlights, top_comments = [], [], [], [], []

    for c in sorted_c[:100]:
        text = " ".join(str(c.get("text") or "").split())
        if not text:
            continue
        likes = int(c.get("like_count") or 0)
        entry = {"likes": likes, "text": text[:280]}
        if len(top_comments) < 15:
            top_comments.append(entry)
        if QUESTION_PATTERNS.search(text) and len(questions) < 12:
            questions.append(entry)
        if COMPLAINT_PATTERNS.search(text) and len(complaints) < 12:
            complaints.append(entry)
        if REQUEST_PATTERNS.search(text) and len(requests_list) < 12:
            requests_list.append(entry)
        ts_found = TIMESTAMP_RE.findall(text)
        if ts_found and len(ts_highlights) < 12:
            ts_highlights.append({"timestamps": ts_found, "likes": likes, "text": text[:240]})

    return {
        "total_mined": len(sorted_c),
        "top_comments": top_comments,
        "questions": questions,
        "complaints": complaints,
        "requests": requests_list,
        "timestamp_highlights": ts_highlights,
    }


def parse_subtitle_file(sub_path: Path) -> list[dict[str, Any]]:
    """Parse VTT or SRT subtitle file into compact timestamped lines."""
    text = sub_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    time_re = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})"
    )
    lines: list[dict[str, Any]] = []
    seen_last = ""

    for blk in blocks:
        m = time_re.search(blk)
        if not m:
            continue
        start_s = _ts_to_sec(m.group(1))
        end_s = _ts_to_sec(m.group(2))
        body_lines = []
        for ln in blk.splitlines():
            if "-->" in ln or ln.strip().isdigit() or ln.strip().startswith("WEBVTT"):
                continue
            clean = re.sub(r"<[^>]+>", "", ln).strip()
            if clean:
                body_lines.append(clean)
        joined = " ".join(body_lines).strip()
        if not joined or joined == seen_last:
            continue
        seen_last = joined
        lines.append({"start": round(start_s, 2), "end": round(end_s, 2), "text": joined})

    return lines


def _ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return int(parts[0]) * 60 + float(parts[1])
