#!/usr/bin/env python3
"""Validate the local podcast archive without modifying episode contents."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import shutil
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import archive_xiaoyuzhou_channels as archive


ROOT = archive.ROOT
REPORT_JSON = ROOT / "_metadata" / "archive_validation_report.json"
REPORT_MD = ROOT / "_metadata" / "archive_validation_report.md"
MIN_VALID_DURATION_SECONDS = 5.0


@dataclass(frozen=True)
class Problem:
    section: str
    severity: str
    path: str
    problem: str
    detail: str = ""

    def to_json(self) -> dict[str, str]:
        item = {
            "section": self.section,
            "severity": self.severity,
            "path": self.path,
            "problem": self.problem,
        }
        if self.detail:
            item["detail"] = self.detail
        return item


@dataclass(frozen=True)
class AudioCheck:
    path: str
    ext: str
    size: int
    duration_seconds: float | None
    parser: str
    error: str = ""


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def expected_markdown(description_html: str) -> str:
    markdown = archive.html_to_markdown(description_html or "")
    return markdown + ("\n" if markdown else "")


def expected_text(description_html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", description_html or "")
    text = re.sub(r"\s+", " ", archive.html.unescape(text)).strip()
    return text + ("\n" if text else "")


def load_archive() -> tuple[list[str], dict[str, list[archive.Episode]]]:
    channels = archive.read_channels()
    episodes_by_channel = {
        channel: archive.load_saved_episodes(channel)
        for channel in channels
    }
    return channels, episodes_by_channel


def mp4_duration(path: Path) -> float:
    size = path.stat().st_size
    with path.open("rb") as fh:
        return find_mp4_duration(fh, 0, size)


def find_mp4_duration(fh: Any, start: int, end: int, depth: int = 0) -> float:
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        header = fh.read(8)
        if len(header) != 8:
            break
        raw_size, raw_type = struct.unpack(">I4s", header)
        box_type = raw_type.decode("latin-1", "replace")
        header_size = 8
        if raw_size == 1:
            large_size = fh.read(8)
            if len(large_size) != 8:
                break
            raw_size = struct.unpack(">Q", large_size)[0]
            header_size = 16
        elif raw_size == 0:
            raw_size = end - pos
        if raw_size < header_size:
            break
        content_start = pos + header_size
        content_end = min(pos + raw_size, end)
        if box_type == "mvhd":
            return parse_mvhd(fh, content_start, content_end)
        if box_type in {"moov", "trak", "mdia", "minf", "stbl", "edts", "udta", "meta"} and depth < 8:
            nested_start = content_start + (4 if box_type == "meta" else 0)
            try:
                return find_mp4_duration(fh, nested_start, content_end, depth + 1)
            except ValueError:
                pass
        pos += raw_size
    raise ValueError("mvhd atom not found")


def parse_mvhd(fh: Any, start: int, end: int) -> float:
    fh.seek(start)
    version_flags = fh.read(4)
    if len(version_flags) != 4:
        raise ValueError("short mvhd header")
    version = version_flags[0]
    if version == 1:
        payload = fh.read(28)
        if len(payload) != 28:
            raise ValueError("short mvhd v1")
        timescale = struct.unpack(">I", payload[16:20])[0]
        duration = struct.unpack(">Q", payload[20:28])[0]
    elif version == 0:
        payload = fh.read(16)
        if len(payload) != 16:
            raise ValueError("short mvhd v0")
        timescale = struct.unpack(">I", payload[8:12])[0]
        duration = struct.unpack(">I", payload[12:16])[0]
    else:
        raise ValueError(f"unsupported mvhd version {version}")
    if timescale <= 0:
        raise ValueError("mvhd timescale is zero")
    seconds = duration / timescale
    if not math.isfinite(seconds):
        raise ValueError("non-finite mvhd duration")
    return seconds


BITRATE_TABLE: dict[tuple[str, int], list[int | None]] = {
    ("1", 3): [None, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, None],
    ("1", 2): [None, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, None],
    ("1", 1): [None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None],
    ("2", 3): [None, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, None],
    ("2", 2): [None, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None],
    ("2", 1): [None, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None],
}
SAMPLE_RATE_TABLE = {
    "1": [44100, 48000, 32000, None],
    "2": [22050, 24000, 16000, None],
    "2.5": [11025, 12000, 8000, None],
}


def skip_id3v2(data: bytes) -> int:
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size_bytes = data[6:10]
    if any(byte & 0x80 for byte in size_bytes):
        return 0
    tag_size = (
        (size_bytes[0] << 21)
        | (size_bytes[1] << 14)
        | (size_bytes[2] << 7)
        | size_bytes[3]
    )
    footer = 10 if data[5] & 0x10 else 0
    return min(len(data), 10 + tag_size + footer)


def id3v2_size(header: bytes) -> int:
    if len(header) < 10 or header[:3] != b"ID3":
        return 0
    size_bytes = header[6:10]
    if any(byte & 0x80 for byte in size_bytes):
        return 0
    tag_size = (
        (size_bytes[0] << 21)
        | (size_bytes[1] << 14)
        | (size_bytes[2] << 7)
        | size_bytes[3]
    )
    footer = 10 if header[5] & 0x10 else 0
    return 10 + tag_size + footer


def parse_mp3_frame_header(header: bytes) -> tuple[int, int, int, int] | None:
    if len(header) != 4:
        return None
    value = int.from_bytes(header, "big")
    if (value >> 21) & 0x7FF != 0x7FF:
        return None
    version_bits = (value >> 19) & 0x3
    layer_bits = (value >> 17) & 0x3
    bitrate_index = (value >> 12) & 0xF
    sample_rate_index = (value >> 10) & 0x3
    padding = (value >> 9) & 0x1
    if version_bits == 1 or layer_bits == 0:
        return None
    version = {3: "1", 2: "2", 0: "2.5"}[version_bits]
    layer = {3: 3, 2: 2, 1: 1}[layer_bits]
    table_version = "1" if version == "1" else "2"
    bitrate = BITRATE_TABLE[(table_version, layer)][bitrate_index]
    sample_rate = SAMPLE_RATE_TABLE[version][sample_rate_index]
    if not bitrate or not sample_rate:
        return None
    if layer == 3:
        frame_length = ((12 * bitrate * 1000) // sample_rate + padding) * 4
        samples = 384
    elif layer == 2:
        frame_length = (144 * bitrate * 1000) // sample_rate + padding
        samples = 1152
    else:
        coefficient = 144 if version == "1" else 72
        frame_length = (coefficient * bitrate * 1000) // sample_rate + padding
        samples = 1152 if version == "1" else 576
    if frame_length <= 4:
        return None
    return frame_length, samples, sample_rate, bitrate


def mp3_duration(path: Path) -> float:
    # Fast validity and duration check: confirm several consecutive frames near
    # the start, then estimate duration from MPEG bitrate and payload size.
    with path.open("rb") as fh:
        header = fh.read(10)
        start = id3v2_size(header)
        fh.seek(start)
        data = fh.read(1024 * 1024)
    pos = 0
    frames = 0
    misses = 0
    first_bitrate: int | None = None
    first_frame_pos: int | None = None
    while pos + 4 <= len(data):
        parsed = parse_mp3_frame_header(data[pos:pos + 4])
        if parsed is None:
            pos += 1
            misses += 1
            if frames == 0 and misses > 512 * 1024:
                break
            continue
        frame_length, _samples, _sample_rate, bitrate = parsed
        if pos + frame_length > len(data) + 2:
            break
        if first_bitrate is None:
            first_bitrate = bitrate
            first_frame_pos = start + pos
        frames += 1
        if frames >= 8:
            break
        pos += frame_length
    if frames == 0:
        raise ValueError("no valid mp3 frames found")
    if first_bitrate is None or first_frame_pos is None:
        raise ValueError("no mp3 bitrate found")
    payload_size = max(0, path.stat().st_size - first_frame_pos)
    return payload_size * 8 / (first_bitrate * 1000)


def audio_duration(path: Path) -> tuple[float | None, str, str]:
    ext = path.suffix.lower().lstrip(".")
    try:
        if ext in {"m4a", "mp4", "aac"}:
            return mp4_duration(path), "mp4_mvhd", ""
        if ext == "mp3":
            return mp3_duration(path), "mp3_frame_scan", ""
        return None, "unsupported", f"unsupported audio extension .{ext}"
    except Exception as exc:
        return None, ext or "unknown", str(exc)


def check_one_audio(path: Path) -> AudioCheck:
    size = path.stat().st_size if path.exists() else 0
    duration, parser, error = audio_duration(path)
    return AudioCheck(
        path=rel(path),
        ext=path.suffix.lower().lstrip("."),
        size=size,
        duration_seconds=duration,
        parser=parser,
        error=error,
    )


def check_audio(
    episodes_by_channel: dict[str, list[archive.Episode]],
    *,
    workers: int,
) -> tuple[dict[str, Any], list[Problem]]:
    problems: list[Problem] = []
    audio_paths: list[Path] = []
    expected_audio_paths: set[Path] = set()
    missing_audio = 0
    empty_audio = 0

    for channel, episodes in episodes_by_channel.items():
        for episode in episodes:
            path = archive.audio_path_for(channel, episode)
            expected_audio_paths.add(path)
            if not path.exists():
                missing_audio += 1
                problems.append(Problem("audio", "error", rel(path), "missing audio"))
                continue
            size = path.stat().st_size
            if size <= 0:
                empty_audio += 1
                problems.append(Problem("audio", "error", rel(path), "empty audio"))
                continue
            audio_paths.append(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        checks = list(executor.map(check_one_audio, audio_paths))

    by_extension = Counter(check.ext for check in checks)
    by_parser = Counter(check.parser for check in checks)
    durations = [
        check.duration_seconds
        for check in checks
        if check.duration_seconds is not None and check.duration_seconds > 0
    ]
    invalid_duration = 0
    short_duration = 0
    for check in checks:
        duration = check.duration_seconds
        if duration is None or duration <= 0:
            invalid_duration += 1
            problems.append(
                Problem(
                    "audio",
                    "error",
                    check.path,
                    "invalid audio duration",
                    check.error or "duration is missing or <= 0",
                )
            )
        elif duration < MIN_VALID_DURATION_SECONDS:
            short_duration += 1
            problems.append(
                Problem(
                    "audio",
                    "error",
                    check.path,
                    "invalid audio duration",
                    f"{duration:.3f}s < {MIN_VALID_DURATION_SECONDS:.1f}s",
                )
            )
        elif duration < 30:
            short_duration += 1
            problems.append(
                Problem(
                    "audio",
                    "warning",
                    check.path,
                    "short audio duration; manual review suggested",
                    f"{duration:.3f}s < 30.0s",
                )
            )
        elif duration > 8 * 60 * 60:
            problems.append(
                Problem(
                    "audio",
                    "warning",
                    check.path,
                    "long audio duration; manual review suggested",
                    f"{duration:.3f}s > 8h",
                )
            )

    extra_audio = []
    for path in ROOT.glob("*/*/audio.*"):
        if path not in expected_audio_paths:
            extra_audio.append(rel(path))
            problems.append(Problem("audio", "warning", rel(path), "extra audio file"))

    part_files = sorted(rel(path) for path in ROOT.rglob("*.part"))
    for part_file in part_files:
        problems.append(Problem("audio", "error", part_file, "partial file remains"))

    summary = {
        "ffprobe_available": shutil.which("ffprobe") is not None,
        "checked_audio_files": len(checks),
        "expected_audio_files": sum(len(items) for items in episodes_by_channel.values()),
        "missing_audio": missing_audio,
        "empty_audio": empty_audio,
        "invalid_duration": invalid_duration,
        "short_duration_warnings": short_duration,
        "extra_audio_files": len(extra_audio),
        "part_files": len(part_files),
        "by_extension": dict(sorted(by_extension.items())),
        "by_parser": dict(sorted(by_parser.items())),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "total": sum(durations),
        },
    }
    return summary, problems


def check_naming(channels: list[str], episodes_by_channel: dict[str, list[archive.Episode]]) -> tuple[dict[str, Any], list[Problem]]:
    problems: list[Problem] = []
    expected_episode_dirs: set[Path] = set()
    expected_channel_dirs = {ROOT / channel for channel in channels}
    numbered_dirs_total = 0
    metadata_files = 0

    for channel in channels:
        channel_dir = ROOT / channel
        episodes = episodes_by_channel.get(channel, [])
        if not channel_dir.exists():
            problems.append(Problem("naming", "error", rel(channel_dir), "missing channel directory"))
            continue
        width = max(3, len(str(len(episodes))))
        expected_numbers = {f"{index:0{width}d}" for index in range(1, len(episodes) + 1)}
        actual_numbered_dirs = {
            path.name.split("-", 1)[0]: path
            for path in channel_dir.iterdir()
            if path.is_dir() and re.match(r"^\d+-", path.name)
        }
        numbered_dirs_total += len(actual_numbered_dirs)
        if set(actual_numbered_dirs) != expected_numbers:
            missing = sorted(expected_numbers - set(actual_numbered_dirs))
            extra = sorted(set(actual_numbered_dirs) - expected_numbers)
            if missing:
                problems.append(Problem("naming", "error", rel(channel_dir), "missing episode numbers", ", ".join(missing[:20])))
            if extra:
                problems.append(Problem("naming", "warning", rel(channel_dir), "extra episode numbers", ", ".join(extra[:20])))
        for episode in episodes:
            expected_dir = archive.episode_dir(channel, episode)
            expected_episode_dirs.add(expected_dir)
            if not expected_dir.exists():
                problems.append(Problem("naming", "error", rel(expected_dir), "missing expected episode directory"))
                continue
            expected_name = f"{episode.number}-{archive.safe_component(episode.title)}"
            if expected_dir.name != expected_name:
                problems.append(Problem("naming", "error", rel(expected_dir), "episode directory name mismatch", f"expected {expected_name}"))
            metadata_path = expected_dir / "metadata.json"
            if metadata_path.exists():
                metadata_files += 1
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    problems.append(Problem("naming", "error", rel(metadata_path), "metadata is not valid json", str(exc)))
                    metadata = {}
                checks = {
                    "index": episode.index,
                    "number": episode.number,
                    "title": episode.title,
                    "audio_url": episode.audio_url,
                }
                for key, expected in checks.items():
                    if metadata.get(key) != expected:
                        problems.append(
                            Problem(
                                "naming",
                                "error",
                                rel(metadata_path),
                                f"metadata {key} mismatch",
                                f"expected {expected!r}, got {metadata.get(key)!r}",
                            )
                        )
            else:
                problems.append(Problem("naming", "error", rel(metadata_path), "missing metadata.json"))
            expected_audio = archive.audio_path_for(channel, episode)
            audio_files = sorted(expected_dir.glob("audio.*"))
            if expected_audio.exists():
                unexpected = [path for path in audio_files if path != expected_audio]
                for path in unexpected:
                    problems.append(Problem("naming", "warning", rel(path), "unexpected audio filename"))
            elif audio_files:
                for path in audio_files:
                    problems.append(
                        Problem(
                            "naming",
                            "error",
                            rel(path),
                            "audio filename does not match expected extension",
                            f"expected {expected_audio.name}",
                        )
                    )

    for channel_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        if channel_dir.name.startswith(".") or channel_dir.name == "_metadata":
            continue
        if channel_dir not in expected_channel_dirs:
            continue
        for episode_path in channel_dir.iterdir():
            if episode_path.is_dir() and re.match(r"^\d+-", episode_path.name) and episode_path not in expected_episode_dirs:
                problems.append(Problem("naming", "warning", rel(episode_path), "extra numbered episode directory"))

    summary = {
        "channels": len(channels),
        "expected_episode_dirs": sum(len(items) for items in episodes_by_channel.values()),
        "numbered_episode_dirs": numbered_dirs_total,
        "metadata_files": metadata_files,
    }
    return summary, problems


def check_show_notes(episodes_by_channel: dict[str, list[archive.Episode]]) -> tuple[dict[str, Any], list[Problem]]:
    problems: list[Problem] = []
    show_notes_md = 0
    show_notes_html = 0
    show_notes_txt = 0
    nonempty_descriptions = 0
    empty_descriptions = 0

    for channel, episodes in episodes_by_channel.items():
        for episode in episodes:
            episode_path = archive.episode_dir(channel, episode)
            description_html = episode.description_html or ""
            if description_html:
                nonempty_descriptions += 1
            else:
                empty_descriptions += 1
            md_path = episode_path / "show_notes.md"
            html_path = episode_path / "show_notes.html"
            txt_path = episode_path / "show_notes.txt"
            if md_path.exists():
                show_notes_md += 1
                actual_md = md_path.read_text(encoding="utf-8")
                expected_md = expected_markdown(description_html)
                if actual_md != expected_md:
                    problems.append(Problem("show_notes", "error", rel(md_path), "show_notes.md content mismatch"))
            else:
                problems.append(Problem("show_notes", "error", rel(md_path), "missing show_notes.md"))
            if description_html:
                if html_path.exists():
                    show_notes_html += 1
                    if html_path.read_text(encoding="utf-8") != description_html:
                        problems.append(Problem("show_notes", "error", rel(html_path), "show_notes.html content mismatch"))
                else:
                    problems.append(Problem("show_notes", "error", rel(html_path), "missing show_notes.html for nonempty description"))
                if txt_path.exists():
                    show_notes_txt += 1
                    if txt_path.read_text(encoding="utf-8") != expected_text(description_html):
                        problems.append(Problem("show_notes", "error", rel(txt_path), "show_notes.txt content mismatch"))
                else:
                    problems.append(Problem("show_notes", "error", rel(txt_path), "missing show_notes.txt for nonempty description"))
            else:
                if html_path.exists():
                    show_notes_html += 1
                    problems.append(Problem("show_notes", "warning", rel(html_path), "show_notes.html exists for empty description"))
                if txt_path.exists():
                    show_notes_txt += 1
                    problems.append(Problem("show_notes", "warning", rel(txt_path), "show_notes.txt exists for empty description"))

    summary = {
        "episodes": sum(len(items) for items in episodes_by_channel.values()),
        "show_notes_md": show_notes_md,
        "show_notes_html": show_notes_html,
        "show_notes_txt": show_notes_txt,
        "nonempty_descriptions": nonempty_descriptions,
        "empty_descriptions": empty_descriptions,
    }
    return summary, problems


def group_problem_counts(problems: Iterable[Problem]) -> dict[str, Any]:
    by_section = Counter(problem.section for problem in problems)
    by_severity = Counter(problem.severity for problem in problems)
    by_problem = Counter(problem.problem for problem in problems)
    return {
        "by_section": dict(sorted(by_section.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "by_problem": dict(sorted(by_problem.items())),
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    problems = report["problems"]
    lines = [
        "# Podcast Archive Validation Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Root: `{report['root']}`",
        "",
        "## Totals",
        "",
    ]
    for key, value in report["totals"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Section Summaries", ""])
    for section, summary in report["sections"].items():
        lines.append(f"### {section}")
        for key, value in summary.items():
            if isinstance(value, dict):
                lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False)}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(["## Problems", ""])
    if not problems:
        lines.append("No problems found.")
    else:
        for item in problems:
            detail = f" - {item['detail']}" if item.get("detail") else ""
            lines.append(
                f"- `{item['severity']}` `{item['section']}` `{item['problem']}`: `{item['path']}`{detail}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8, help="parallel audio header parsers")
    parser.add_argument("--json", type=Path, default=REPORT_JSON, help="JSON report path")
    parser.add_argument("--markdown", type=Path, default=REPORT_MD, help="Markdown report path")
    args = parser.parse_args(argv)

    channels, episodes_by_channel = load_archive()
    sections: dict[str, Any] = {}
    all_problems: list[Problem] = []

    naming_summary, naming_problems = check_naming(channels, episodes_by_channel)
    sections["naming"] = naming_summary
    all_problems.extend(naming_problems)

    show_summary, show_problems = check_show_notes(episodes_by_channel)
    sections["show_notes"] = show_summary
    all_problems.extend(show_problems)

    audio_summary, audio_problems = check_audio(episodes_by_channel, workers=args.workers)
    sections["audio"] = audio_summary
    all_problems.extend(audio_problems)

    all_problem_json = [problem.to_json() for problem in all_problems]
    totals = {
        "channels": len(channels),
        "episodes": sum(len(items) for items in episodes_by_channel.values()),
        "problems": len(all_problem_json),
        "errors": sum(1 for problem in all_problems if problem.severity == "error"),
        "warnings": sum(1 for problem in all_problems if problem.severity == "warning"),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "totals": totals,
        "problem_counts": group_problem_counts(all_problems),
        "sections": sections,
        "problems": all_problem_json,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(report, args.markdown)
    print(
        "[validate] "
        + " ".join(f"{key}={value}" for key, value in totals.items())
        + f" report={args.json.relative_to(ROOT)} markdown={args.markdown.relative_to(ROOT)}"
    )
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
