#!/usr/bin/env python3
"""Archive Xiaoyuzhou podcast channels from RSS feeds.

The command is deliberately dependency-free so it can be resumed on a bare
Python install. It keeps RSS metadata as the source of truth, numbers episodes
oldest-to-newest, and downloads each audio enclosure into:

    $PODTRANS_DATA_DIR/<channel>/<sequence>-<episode title>/audio.<ext>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(
    os.environ.get("PODTRANS_DATA_DIR", Path.cwd() / "data")
).expanduser().resolve()
CHANNELS_FILE = ROOT / "channels.txt"
FEEDS_FILE = ROOT / "feeds.json"
REPORT_FILE = ROOT / "_metadata" / "verification_report.json"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
TIMEOUT = 45
MAX_TITLE_CHARS = 140

ITUNES_SEARCH = "https://itunes.apple.com/search"
XML_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
}


@dataclass(frozen=True)
class Episode:
    index: int
    total: int
    title: str
    pub_date: str
    published_ts: float
    guid: str
    audio_url: str
    audio_type: str
    expected_bytes: int | None
    link: str
    description_html: str

    @property
    def number(self) -> str:
        width = max(3, len(str(self.total)))
        return f"{self.index:0{width}d}"

    @property
    def audio_ext(self) -> str:
        parsed_path = urllib.parse.urlparse(self.audio_url).path
        ext = Path(parsed_path).suffix.lower().strip(".")
        if ext in {"mp3", "m4a", "mp4", "aac", "wav", "ogg", "flac"}:
            return ext
        if "mpeg" in self.audio_type or "mp3" in self.audio_type:
            return "mp3"
        if "mp4" in self.audio_type or "m4a" in self.audio_type:
            return "m4a"
        return "m4a"


def log(message: str) -> None:
    print(message, flush=True)


def read_channels(path: Path = CHANNELS_FILE) -> list[str]:
    channels = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            channels.append(line)
    return channels


def load_feeds(path: Path = FEEDS_FILE) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_feeds(feeds: dict[str, dict[str, Any]], path: Path = FEEDS_FILE) -> None:
    path.write_text(
        json.dumps(feeds, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def request_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = TIMEOUT,
) -> urllib.response.addinfourl:
    merged = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, headers=merged)
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_bytes(url: str, *, timeout: int = TIMEOUT) -> bytes:
    with request_url(url, timeout=timeout) as response:
        return response.read()


def normalize_name(value: str) -> str:
    value = html.unescape(value).casefold()
    value = value.replace("｜", "|").replace("·", "")
    return re.sub(r"[\s\-_【】\[\]()（）:：|]+", "", value)


def score_candidate(channel: str, collection_name: str) -> int:
    wanted = normalize_name(channel)
    got = normalize_name(collection_name)
    if got == wanted:
        return 100
    if wanted in got:
        return 80
    if got in wanted:
        return 70
    channel_parts = [normalize_name(p) for p in re.split(r"[|｜]", channel) if p.strip()]
    if channel_parts and all(part in got for part in channel_parts):
        return 65
    if channel_parts and any(part and part in got for part in channel_parts):
        return 45
    return 0


def itunes_search_terms(channel: str) -> list[str]:
    terms = [channel]
    if "｜" in channel:
        terms.append(channel.replace("｜", " "))
        terms.extend([part.strip() for part in channel.split("｜") if part.strip()])
    if "|" in channel:
        terms.append(channel.replace("|", " "))
        terms.extend([part.strip() for part in channel.split("|") if part.strip()])
    if "播客" in channel:
        terms.append(channel.replace("播客", "").strip())
    seen = set()
    unique = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique.append(term)
    return unique


def discover_itunes_candidates(channel: str, *, limit: int = 20) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for term in itunes_search_terms(channel):
        params = {
            "term": term,
            "media": "podcast",
            "entity": "podcast",
            "country": "CN",
            "limit": str(limit),
        }
        url = ITUNES_SEARCH + "?" + urllib.parse.urlencode(params)
        try:
            payload = json.loads(fetch_bytes(url).decode("utf-8"))
        except Exception as exc:
            log(f"[resolve] {channel}: iTunes search failed for {term!r}: {exc}")
            continue
        for result in payload.get("results", []):
            feed_url = result.get("feedUrl")
            name = result.get("collectionName") or ""
            if not feed_url or not name:
                continue
            score = score_candidate(channel, name)
            item = dict(result)
            item["score"] = score
            item["searchTerm"] = term
            current = candidates.get(feed_url)
            if current is None or score > current.get("score", 0):
                candidates[feed_url] = item
    return sorted(
        candidates.values(),
        key=lambda item: (item.get("score", 0), item.get("trackCount", 0)),
        reverse=True,
    )


def resolve_feeds(channels: list[str], *, min_score: int = 80) -> dict[str, dict[str, Any]]:
    feeds = load_feeds()
    meta_dir = ROOT / "_metadata"
    meta_dir.mkdir(exist_ok=True)
    all_candidates: dict[str, list[dict[str, Any]]] = {}
    changed = False

    for channel in channels:
        existing = feeds.get(channel, {}).get("feed_url")
        if existing:
            log(f"[resolve] {channel}: already mapped to {existing}")
            continue
        candidates = discover_itunes_candidates(channel)
        all_candidates[channel] = candidates
        if not candidates:
            log(f"[resolve] {channel}: no iTunes candidates")
            continue
        best = candidates[0]
        name = best.get("collectionName") or ""
        score = int(best.get("score", 0))
        feed_url = best.get("feedUrl")
        log(f"[resolve] {channel}: best score={score} name={name!r} feed={feed_url}")
        if feed_url and score >= min_score:
            feeds[channel] = {
                "feed_url": feed_url,
                "source": "itunes_search",
                "collection_name": name,
                "artist_name": best.get("artistName"),
                "track_count": best.get("trackCount"),
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            changed = True

    candidates_path = meta_dir / "feed_candidates.json"
    candidates_path.write_text(
        json.dumps(all_candidates, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if changed:
        save_feeds(feeds)
    return feeds


def text_of(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip()


def child_text(parent: ET.Element, names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = text_of(parent.find(name), "")
        if value:
            return value
    return default


def parse_pub_date(value: str) -> float:
    if not value:
        return 0.0
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def strip_namespaces(root: ET.Element) -> None:
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.rsplit("}", 1)[1]


def parse_feed(feed_xml: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parser = ET.XMLParser(encoding="utf-8")
    root = ET.fromstring(feed_xml, parser=parser)
    strip_namespaces(root)
    channel_node = root.find("channel")
    if channel_node is None:
        raise ValueError("RSS feed has no channel node")

    feed_meta = {
        "title": child_text(channel_node, ["title"]),
        "link": child_text(channel_node, ["link"]),
        "description": child_text(channel_node, ["description"]),
        "last_build_date": child_text(channel_node, ["lastBuildDate"]),
        "generator": child_text(channel_node, ["generator"]),
    }

    items = []
    for item in channel_node.findall("item"):
        enclosure = item.find("enclosure")
        audio_url = ""
        audio_type = ""
        expected_bytes = None
        if enclosure is not None:
            audio_url = enclosure.attrib.get("url", "").strip()
            audio_type = enclosure.attrib.get("type", "").strip()
            length = enclosure.attrib.get("length")
            if length and length.isdigit():
                parsed_length = int(length)
                if parsed_length > 0:
                    expected_bytes = parsed_length
        if not audio_url:
            continue
        title = child_text(item, ["title"], "untitled")
        pub_date = child_text(item, ["pubDate"])
        description_html = child_text(item, ["encoded", "description"])
        guid = child_text(item, ["guid"], audio_url)
        items.append(
            {
                "title": title,
                "pub_date": pub_date,
                "published_ts": parse_pub_date(pub_date),
                "guid": guid,
                "audio_url": audio_url,
                "audio_type": audio_type,
                "expected_bytes": expected_bytes,
                "link": child_text(item, ["link"]),
                "description_html": description_html,
            }
        )

    items.sort(key=lambda item: (item["published_ts"], item["guid"]))
    return feed_meta, items


def fetch_channel_metadata(channel: str, feed_url: str) -> tuple[dict[str, Any], list[Episode]]:
    log(f"[metadata] {channel}: fetching {feed_url}")
    feed_xml = fetch_bytes(feed_url)
    feed_meta, raw_items = parse_feed(feed_xml)

    channel_dir = ROOT / channel
    meta_dir = channel_dir / "_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "feed.xml").write_bytes(feed_xml)

    total = len(raw_items)
    episodes = [
        Episode(
            index=index,
            total=total,
            title=item["title"],
            pub_date=item["pub_date"],
            published_ts=item["published_ts"],
            guid=item["guid"],
            audio_url=item["audio_url"],
            audio_type=item["audio_type"],
            expected_bytes=item["expected_bytes"],
            link=item["link"],
            description_html=item["description_html"],
        )
        for index, item in enumerate(raw_items, start=1)
    ]
    meta_payload = {
        "channel": channel,
        "feed_url": feed_url,
        "feed_meta": feed_meta,
        "episode_count": total,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "episodes": [episode_to_json(ep) for ep in episodes],
    }
    (meta_dir / "episodes.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"[metadata] {channel}: {total} episodes; feed title={feed_meta.get('title')!r}")
    return feed_meta, episodes


def episode_to_json(episode: Episode) -> dict[str, Any]:
    return {
        "index": episode.index,
        "number": episode.number,
        "total": episode.total,
        "title": episode.title,
        "pub_date": episode.pub_date,
        "published_ts": episode.published_ts,
        "guid": episode.guid,
        "audio_url": episode.audio_url,
        "audio_type": episode.audio_type,
        "expected_bytes": episode.expected_bytes,
        "link": episode.link,
        "description_html": episode.description_html,
    }


def safe_component(value: str, *, max_chars: int = MAX_TITLE_CHARS) -> str:
    value = html.unescape(value)
    value = re.sub(r"[\x00-\x1f]+", " ", value)
    value = re.sub(r'[<>:"/\\|?*]+', "-", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        value = "untitled"
    if len(value) > max_chars:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        value = value[: max_chars - 9].rstrip(" .") + "-" + digest
    return value


def episode_dir(channel: str, episode: Episode) -> Path:
    return ROOT / channel / f"{episode.number}-{safe_component(episode.title)}"


def audio_path_for(channel: str, episode: Episode) -> Path:
    return episode_dir(channel, episode) / f"audio.{episode.audio_ext}"


class SimpleMarkdownConverter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.href_stack: list[str | None] = []
        self.list_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value for key, value in attrs}
        if tag in {"p", "div", "section", "article"}:
            self._block()
        elif tag == "br":
            self.parts.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._block()
            level = int(tag[1])
            self.parts.append("#" * level + " ")
        elif tag in {"strong", "b"}:
            self.parts.append("**")
        elif tag in {"em", "i"}:
            self.parts.append("*")
        elif tag in {"ul", "ol"}:
            self.list_depth += 1
            self._block()
        elif tag == "li":
            self._block()
            self.parts.append("  " * max(0, self.list_depth - 1) + "- ")
        elif tag == "blockquote":
            self._block()
            self.parts.append("> ")
        elif tag == "a":
            self.href_stack.append(attrs_dict.get("href"))
            self.parts.append("[")
        elif tag == "img":
            alt = attrs_dict.get("alt") or ""
            src = attrs_dict.get("src") or ""
            if src:
                self.parts.append(f"![{alt}]({src})")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "section", "article", "blockquote"}:
            self._block()
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._block()
        elif tag in {"strong", "b"}:
            self.parts.append("**")
        elif tag in {"em", "i"}:
            self.parts.append("*")
        elif tag in {"ul", "ol"}:
            self.list_depth = max(0, self.list_depth - 1)
            self._block()
        elif tag == "li":
            self.parts.append("\n")
        elif tag == "a":
            href = self.href_stack.pop() if self.href_stack else None
            self.parts.append("]")
            if href:
                self.parts.append(f"({href})")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def _block(self) -> None:
        current = "".join(self.parts)
        if not current:
            return
        if current.endswith("\n\n"):
            return
        if current.endswith("\n"):
            self.parts.append("\n")
        else:
            self.parts.append("\n\n")

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\[\s*\]\(([^)]+)\)", r"\1", text)
        return text.strip()


def html_to_markdown(value: str) -> str:
    if not value:
        return ""
    converter = SimpleMarkdownConverter()
    converter.feed(value)
    converter.close()
    markdown = converter.markdown()
    if markdown:
        return markdown
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def write_episode_sidecars(channel: str, episode: Episode) -> None:
    out_dir = episode_dir(channel, episode)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(episode_to_json(episode), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    description_html = episode.description_html or ""
    markdown = html_to_markdown(description_html)
    (out_dir / "show_notes.md").write_text(markdown + ("\n" if markdown else ""), encoding="utf-8")
    if description_html:
        (out_dir / "show_notes.html").write_text(description_html, encoding="utf-8")
        text = re.sub(r"<[^>]+>", " ", description_html)
        text = re.sub(r"\s+", " ", html.unescape(text)).strip()
        (out_dir / "show_notes.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")


def get_remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit():
                return int(length)
    except Exception:
        return None
    return None


def is_audio_complete(path: Path, expected_bytes: int | None) -> bool:
    if expected_bytes is not None and expected_bytes <= 0:
        expected_bytes = None
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if expected_bytes is None:
        return True
    return path.stat().st_size == expected_bytes


def is_audio_complete_for_url(path: Path, expected_bytes: int | None, url: str) -> bool:
    if is_audio_complete(path, expected_bytes):
        return True
    if not path.exists() or path.stat().st_size <= 0:
        return False
    size = path.stat().st_size
    remote_size = get_remote_size(url)
    if remote_size is not None:
        return size == remote_size
    return is_audio_complete(path, expected_bytes)


def content_range_total(headers: Any) -> int | None:
    if not headers:
        return None
    value = headers.get("Content-Range")
    if not value:
        return None
    match = re.search(r"/(\d+)\s*$", value)
    if match:
        return int(match.group(1))
    return None


def download_audio(channel: str, episode: Episode, *, retries: int = 120) -> dict[str, Any]:
    out_dir = episode_dir(channel, episode)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_path_for(channel, episode)
    part_path = audio_path.with_suffix(audio_path.suffix + ".part")
    write_episode_sidecars(channel, episode)

    expected = episode.expected_bytes if episode.expected_bytes and episode.expected_bytes > 0 else None
    if is_audio_complete_for_url(audio_path, expected, episode.audio_url):
        return {
            "channel": channel,
            "number": episode.number,
            "status": "skipped",
            "path": str(audio_path.relative_to(ROOT)),
            "bytes": audio_path.stat().st_size,
        }

    if part_path.exists():
        remote_size = get_remote_size(episode.audio_url)
        if remote_size is not None:
            expected = remote_size
        if expected is not None and part_path.stat().st_size == expected:
            part_path.replace(audio_path)
            return {
                "channel": channel,
                "number": episode.number,
                "status": "recovered",
                "path": str(audio_path.relative_to(ROOT)),
                "bytes": audio_path.stat().st_size,
            }

    if audio_path.exists() and not is_audio_complete_for_url(audio_path, expected, episode.audio_url):
        shutil.move(str(audio_path), str(part_path))

    for attempt in range(1, retries + 1):
        existing = part_path.stat().st_size if part_path.exists() else 0
        headers = {}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        try:
            with request_url(episode.audio_url, headers=headers, timeout=TIMEOUT) as response:
                status = getattr(response, "status", None)
                if existing > 0 and status != 206:
                    mode = "wb"
                    existing = 0
                with part_path.open(mode + "") as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            actual = part_path.stat().st_size
            remote_size = get_remote_size(episode.audio_url)
            if remote_size is not None:
                expected = remote_size
            if expected is not None and actual != expected:
                if actual > existing and attempt < retries:
                    time.sleep(0.2)
                    continue
                raise IOError(f"downloaded {actual} bytes; expected {expected}")
            part_path.replace(audio_path)
            return {
                "channel": channel,
                "number": episode.number,
                "status": "downloaded",
                "path": str(audio_path.relative_to(ROOT)),
                "bytes": audio_path.stat().st_size,
            }
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                return {
                    "channel": channel,
                    "number": episode.number,
                    "status": "failed",
                    "path": str(audio_path.relative_to(ROOT)),
                    "error": str(exc),
                }
            if exc.code == 416 and part_path.exists():
                part_size = part_path.stat().st_size
                remote_size = content_range_total(exc.headers) or get_remote_size(episode.audio_url)
                if remote_size is not None and part_size == remote_size:
                    part_path.replace(audio_path)
                    return {
                        "channel": channel,
                        "number": episode.number,
                        "status": "recovered",
                        "path": str(audio_path.relative_to(ROOT)),
                        "bytes": audio_path.stat().st_size,
                    }
                if remote_size is not None and part_size > remote_size:
                    part_path.unlink()
            if attempt == retries:
                return {
                    "channel": channel,
                    "number": episode.number,
                    "status": "failed",
                    "path": str(audio_path.relative_to(ROOT)),
                    "error": str(exc),
                }
            time.sleep(min(5, attempt))
        except Exception as exc:
            if attempt == retries:
                return {
                    "channel": channel,
                    "number": episode.number,
                    "status": "failed",
                    "path": str(audio_path.relative_to(ROOT)),
                    "error": str(exc),
                }
            time.sleep(min(5, attempt))

    raise AssertionError("unreachable")


def load_saved_episodes(channel: str) -> list[Episode]:
    path = ROOT / channel / "_metadata" / "episodes.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = []
    for item in payload.get("episodes", []):
        episodes.append(
            Episode(
                index=int(item["index"]),
                total=int(item["total"]),
                title=item["title"],
                pub_date=item.get("pub_date", ""),
                published_ts=float(item.get("published_ts") or 0),
                guid=item.get("guid", item.get("audio_url", "")),
                audio_url=item["audio_url"],
                audio_type=item.get("audio_type", ""),
                expected_bytes=(
                    item.get("expected_bytes")
                    if item.get("expected_bytes") and item.get("expected_bytes") > 0
                    else None
                ),
                link=item.get("link", ""),
                description_html=item.get("description_html", ""),
            )
        )
    return episodes


def verify_channel(channel: str, episodes: list[Episode]) -> dict[str, Any]:
    channel_dir = ROOT / channel
    part_files = list(channel_dir.rglob("*.part")) if channel_dir.exists() else []
    numbered_dirs = [
        path
        for path in channel_dir.iterdir()
        if path.is_dir() and re.match(r"^\d+-", path.name)
    ] if channel_dir.exists() else []
    problems = []
    rss_size_mismatches = []
    audio_files = 0
    show_note_files = 0
    total_expected_bytes = 0
    total_actual_bytes = 0

    for episode in episodes:
        show_notes_path = episode_dir(channel, episode) / "show_notes.md"
        if show_notes_path.exists():
            show_note_files += 1
        else:
            problems.append({"number": episode.number, "title": episode.title, "problem": "missing show_notes.md"})
        path = audio_path_for(channel, episode)
        if episode.expected_bytes:
            total_expected_bytes += episode.expected_bytes
        if not path.exists():
            problems.append({"number": episode.number, "title": episode.title, "problem": "missing audio"})
            continue
        size = path.stat().st_size
        if size <= 0:
            problems.append({"number": episode.number, "title": episode.title, "problem": "empty audio"})
            continue
        audio_files += 1
        total_actual_bytes += size
        if episode.expected_bytes is not None and size != episode.expected_bytes:
            rss_size_mismatches.append(
                {
                    "number": episode.number,
                    "title": episode.title,
                    "problem": "rss size mismatch",
                    "actual": size,
                    "expected": episode.expected_bytes,
                }
            )

    for part_file in part_files:
        problems.append({"path": str(part_file.relative_to(ROOT)), "problem": "partial file remains"})

    return {
        "channel": channel,
        "episodes": len(episodes),
        "episode_dirs": len(numbered_dirs),
        "audio_files": audio_files,
        "show_note_files": show_note_files,
        "part_files": len(part_files),
        "expected_bytes": total_expected_bytes,
        "actual_bytes": total_actual_bytes,
        "rss_size_mismatches": len(rss_size_mismatches),
        "problems": problems,
    }


def run_metadata(channels: list[str], feeds: dict[str, dict[str, Any]]) -> dict[str, list[Episode]]:
    all_episodes = {}
    for channel in channels:
        feed_url = feeds.get(channel, {}).get("feed_url")
        if not feed_url:
            log(f"[metadata] {channel}: unresolved feed")
            all_episodes[channel] = []
            continue
        try:
            _feed_meta, episodes = fetch_channel_metadata(channel, feed_url)
        except Exception as exc:
            log(f"[metadata] {channel}: failed: {exc}")
            episodes = load_saved_episodes(channel)
        all_episodes[channel] = episodes
    return all_episodes


def run_download(
    channels: list[str],
    all_episodes: dict[str, list[Episode]],
    *,
    workers: int,
    limit_per_channel: int | None,
    retry_problem_numbers: dict[str, set[str]] | None = None,
) -> None:
    tasks: list[tuple[str, Episode]] = []
    for channel in channels:
        episodes = all_episodes.get(channel) or load_saved_episodes(channel)
        if limit_per_channel is not None:
            episodes = episodes[:limit_per_channel]
        for episode in episodes:
            if retry_problem_numbers is not None and episode.number not in retry_problem_numbers.get(channel, set()):
                continue
            tasks.append((channel, episode))

    if not tasks:
        log("[download] no episodes to download")
        return

    log(f"[download] queued {len(tasks)} episodes with workers={workers}")
    done = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(download_audio, channel, episode): (channel, episode)
            for channel, episode in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            channel, episode = future_map[future]
            done += 1
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                log(f"[download] {done}/{len(tasks)} {channel} {episode.number}: failed {exc}")
                continue
            status = result.get("status")
            if status == "failed":
                failed += 1
                log(
                    f"[download] {done}/{len(tasks)} {channel} {episode.number}: "
                    f"failed {result.get('error')}"
                )
            elif done % 10 == 0 or status == "downloaded":
                log(f"[download] {done}/{len(tasks)} {channel} {episode.number}: {status}")
    if failed:
        log(f"[download] completed with {failed} failures")
    else:
        log("[download] completed without failures")


def run_sidecars(channels: list[str], all_episodes: dict[str, list[Episode]]) -> None:
    count = 0
    for channel in channels:
        episodes = all_episodes.get(channel) or load_saved_episodes(channel)
        for episode in episodes:
            write_episode_sidecars(channel, episode)
            count += 1
    log(f"[sidecars] wrote metadata/show note files for {count} episodes")


def run_verify(channels: list[str], all_episodes: dict[str, list[Episode]]) -> dict[str, Any]:
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "channels": []}
    for channel in channels:
        episodes = all_episodes.get(channel) or load_saved_episodes(channel)
        report["channels"].append(verify_channel(channel, episodes))
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    totals = {
        "channels": len(report["channels"]),
        "episodes": sum(item["episodes"] for item in report["channels"]),
        "episode_dirs": sum(item["episode_dirs"] for item in report["channels"]),
        "audio_files": sum(item["audio_files"] for item in report["channels"]),
        "show_note_files": sum(item["show_note_files"] for item in report["channels"]),
        "part_files": sum(item["part_files"] for item in report["channels"]),
        "problems": sum(len(item["problems"]) for item in report["channels"]),
    }
    report["totals"] = totals
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(
        "[verify] "
        + " ".join(f"{key}={value}" for key, value in totals.items())
        + f" report={REPORT_FILE.relative_to(ROOT)}"
    )
    return report


def load_problem_numbers(path: Path = REPORT_FILE) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    problem_numbers: dict[str, set[str]] = {}
    for channel_report in report.get("channels", []):
        channel = channel_report.get("channel")
        if not channel:
            continue
        for problem in channel_report.get("problems", []):
            number = problem.get("number")
            if not number:
                problem_path = problem.get("path", "")
                parts = Path(problem_path).parts
                if len(parts) >= 2:
                    match = re.match(r"^(\d+)-", parts[1])
                    if match:
                        number = match.group(1)
            if number:
                problem_numbers.setdefault(channel, set()).add(str(number))
    return problem_numbers


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podtrans archive", description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="discover missing feed URLs via iTunes search")
    parser.add_argument("--metadata-only", action="store_true", help="fetch RSS and write metadata without audio")
    parser.add_argument("--download", action="store_true", help="download audio files")
    parser.add_argument("--sidecars-only", action="store_true", help="rewrite per-episode metadata and show note sidecars from saved metadata")
    parser.add_argument("--verify", action="store_true", help="verify downloaded files")
    parser.add_argument("--retry-problems", action="store_true", help="only download episodes from the last verification report's problems")
    parser.add_argument("--workers", type=int, default=4, help="parallel audio downloads")
    parser.add_argument("--limit-per-channel", type=int, help="debug limit for downloads")
    parser.add_argument("--channel", action="append", help="only process this channel; can be repeated")
    args = parser.parse_args(argv)

    channels = read_channels()
    if args.channel:
        wanted = set(args.channel)
        channels = [channel for channel in channels if channel in wanted]
        missing = wanted.difference(channels)
        if missing:
            log(f"[warn] requested channels not in channels.txt: {', '.join(sorted(missing))}")

    if args.resolve:
        feeds = resolve_feeds(channels)
    else:
        feeds = load_feeds()

    all_episodes: dict[str, list[Episode]] = {}
    if args.metadata_only or args.download:
        all_episodes = run_metadata(channels, feeds)

    if args.sidecars_only:
        all_episodes = {channel: load_saved_episodes(channel) for channel in channels}
        run_sidecars(channels, all_episodes)

    if args.download:
        retry_problem_numbers = load_problem_numbers() if args.retry_problems else None
        if args.retry_problems:
            count = sum(len(numbers) for numbers in retry_problem_numbers.values())
            log(f"[download] retry-problems selected {count} episodes")
        run_download(
            channels,
            all_episodes,
            workers=args.workers,
            limit_per_channel=args.limit_per_channel,
            retry_problem_numbers=retry_problem_numbers,
        )

    if args.verify:
        if not all_episodes:
            all_episodes = {channel: load_saved_episodes(channel) for channel in channels}
        report = run_verify(channels, all_episodes)
        return 1 if report["totals"]["problems"] else 0

    if not any([args.resolve, args.metadata_only, args.download, args.sidecars_only, args.verify]):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
