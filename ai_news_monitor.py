#!/usr/bin/env python3
"""AI Model News Monitor - Discord notification bot for AI model releases.

Monitors RSS/Atom blog feeds, Hugging Face model uploads, and GitHub releases
from companies competing for #1 on the Chatbot Arena text leaderboard.
Sends Discord webhook notifications when relevant updates are detected.

Designed for use with Kalshi prediction markets -- the goal is to surface
new-model signals as fast as possible so bets can be placed before the
market adjusts.

Architecture follows the same poll-diff-notify pattern used in
https://github.com/acarter881/kalshi_top_model_events
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

import requests

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("ai_news_monitor")

# ============================================================
# Constants
# ============================================================
HF_API_BASE = "https://huggingface.co/api"
GH_API_BASE = "https://api.github.com"
HN_API_BASE = "https://hn.algolia.com/api/v1"
ARENA_URL = "https://arena.ai/leaderboard/text/overall-no-style-control"
DEFAULT_STATE_FILE = ".github/state/ai_news_state.json"
REQUEST_TIMEOUT = 30
USER_AGENT = "AINewsMonitor/1.0 (+https://github.com/acarter881/ai_model_monitor)"
MAX_STORED_ENTRIES_PER_SOURCE = 200
MAX_HN_STORIES_STORED = 500

# ============================================================
# Source Configuration
#
# Each source represents a company or lab that has historically
# been competitive on the Chatbot Arena text leaderboard.
#
# rss_urls    : candidate RSS/Atom feed URLs (tried in order)
# hf_orgs     : Hugging Face organization slugs to monitor
# gh_repos    : GitHub repos (owner/name) to watch for releases
# model_names : keywords specific to this source's models
# ============================================================
SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic",
        "color": 0xD4A574,
        "rss_urls": [
            "https://www.anthropic.com/rss.xml",
            "https://www.anthropic.com/feed.xml",
        ],
        "hf_orgs": [],
        "gh_repos": [],
        "model_names": ["claude", "opus", "sonnet", "haiku"],
    },
    "openai": {
        "label": "OpenAI",
        "color": 0x10A37F,
        "rss_urls": [
            "https://openai.com/index/rss.xml",
            "https://openai.com/blog/rss/",
            "https://openai.com/blog/rss.xml",
        ],
        "hf_orgs": [],
        "gh_repos": [],
        "model_names": ["gpt", "o1", "o3", "o4", "chatgpt"],
    },
    "google": {
        "label": "Google DeepMind",
        "color": 0x4285F4,
        "rss_urls": [
            "https://blog.google/technology/ai/rss/",
            "https://deepmind.google/blog/rss.xml",
        ],
        "hf_orgs": ["google"],
        "gh_repos": [],
        "model_names": ["gemini", "bard", "palm", "deepmind"],
    },
    "meta": {
        "label": "Meta AI",
        "color": 0x0668E1,
        "rss_urls": [
            "https://ai.meta.com/blog/rss/",
        ],
        "hf_orgs": ["meta-llama"],
        "gh_repos": ["meta-llama/llama-models"],
        "model_names": ["llama"],
    },
    "xai": {
        "label": "xAI",
        "color": 0x1DA1F2,
        "rss_urls": [
            "https://x.ai/blog/rss.xml",
            "https://x.ai/blog/rss",
        ],
        "hf_orgs": ["xai-org"],
        "gh_repos": [],
        "model_names": ["grok"],
    },
    "mistral": {
        "label": "Mistral AI",
        "color": 0xFF7000,
        "rss_urls": [
            "https://mistral.ai/news/rss.xml",
            "https://mistral.ai/feed.xml",
        ],
        "hf_orgs": ["mistralai"],
        "gh_repos": [],
        "model_names": ["mistral", "mixtral", "pixtral", "codestral"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "color": 0x4D6BFE,
        "rss_urls": [],
        "hf_orgs": ["deepseek-ai"],
        "gh_repos": ["deepseek-ai/DeepSeek-V3"],
        "model_names": ["deepseek"],
    },
    "qwen": {
        "label": "Qwen (Alibaba)",
        "color": 0xFF6A00,
        "rss_urls": [
            "https://qwenlm.github.io/feed.xml",
        ],
        "hf_orgs": ["Qwen"],
        "gh_repos": [],
        "model_names": ["qwen"],
    },
    "amazon": {
        "label": "Amazon",
        "color": 0xFF9900,
        "rss_urls": [],
        "hf_orgs": ["amazon"],
        "gh_repos": [],
        "model_names": ["nova"],
    },
}

# Broad keywords indicating a model release or benchmark result.
# Used to flag high-priority entries in Discord notifications.
RELEASE_KEYWORDS: list[str] = [
    "release",
    "released",
    "launch",
    "launching",
    "introducing",
    "announcing",
    "announced",
    "new model",
    "available now",
    "generally available",
    "preview",
    "state-of-the-art",
    "sota",
    "benchmark",
    "leaderboard",
    "arena",
    "evaluation",
    "evals",
    "frontier",
    "#1",
    "number one",
    "top spot",
]

# Hacker News search queries for catching announcements quickly.
HN_SEARCH_QUERIES: list[str] = [
    "chatbot arena leaderboard",
    "lmsys arena",
    "claude opus",
    "claude sonnet",
    "gemini pro",
    "gemini ultra",
    "gpt-5",
    "o3 openai",
    "o4 openai",
    "llama 4",
    "grok 3",
    "grok 4",
    "mistral large",
    "deepseek v4",
    "deepseek v5",
    "qwen 3",
    "frontier model release",
]


# ============================================================
# HTTP Helpers
# ============================================================
def fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = REQUEST_TIMEOUT,
    retries: int = 3,
    backoff: float = 2.0,
    token: str | None = None,
) -> requests.Response:
    """GET *url* with exponential-backoff retries and jitter."""
    hdrs = {"User-Agent": USER_AGENT}
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=hdrs, timeout=timeout)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff * attempt))
                log.warning("Rate-limited on %s, sleeping %.1fs", url, retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                wait = backoff * attempt + random.uniform(0, 1)
                log.warning(
                    "Attempt %d/%d failed for %s: %s – retrying in %.1fs",
                    attempt,
                    retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ============================================================
# RSS / Atom Feed Helpers
# ============================================================
def fetch_blog_entries(source_key: str, config: dict[str, Any]) -> dict[str, dict]:
    """Return {entry_id: entry_dict} for the first working RSS URL.

    Tries each URL in *config["rss_urls"]* until one succeeds.
    Handles both RSS 2.0 ``<item>`` and Atom 1.0 ``<entry>`` feeds
    using the standard-library XML parser.
    """
    entries: dict[str, dict] = {}
    for url in config.get("rss_urls", []):
        try:
            resp = fetch_url(url, retries=2)
            parsed = _parse_feed_xml(resp.text)
            if not parsed:
                log.warning("No entries parsed from %s", url)
                continue
            for entry in parsed:
                eid = entry.get("id") or entry.get("url", "")
                if not eid:
                    continue
                entries[eid] = entry
            log.info(
                "Fetched %d blog entries for %s from %s",
                len(entries),
                source_key,
                url,
            )
            break  # success – skip remaining URLs
        except Exception as exc:
            log.warning("Failed to fetch RSS for %s from %s: %s", source_key, url, exc)
    return entries


# Atom namespace
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_feed_xml(xml_text: str) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom 1.0 XML into a list of normalised entry dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("XML parse error: %s", exc)
        return []

    # Detect format from root tag (strip namespace if present)
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag == "rss":
        return _parse_rss_items(root)
    if tag == "feed":
        return _parse_atom_entries(root)
    # Some feeds use <rdf:RDF> (RSS 1.0) – try both parsers
    items = _parse_rss_items(root)
    if not items:
        items = _parse_atom_entries(root)
    return items


def _parse_rss_items(root: ET.Element) -> list[dict[str, str]]:
    """Extract items from an RSS 2.0 feed."""
    entries: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = _xml_text(item, "title")
        link = _xml_text(item, "link")
        guid = _xml_text(item, "guid") or link
        pub_date = _xml_text(item, "pubDate")
        description = _xml_text(item, "description")
        if not guid:
            continue
        entries.append(
            {
                "id": guid,
                "title": title.strip(),
                "url": link.strip(),
                "published": pub_date,
                "summary": _clean_html(description)[:500],
            }
        )
    return entries


def _parse_atom_entries(root: ET.Element) -> list[dict[str, str]]:
    """Extract entries from an Atom 1.0 feed."""
    entries: list[dict[str, str]] = []
    # Try with and without namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    for entry in root.iter(f"{ns}entry"):
        title = _xml_text(entry, f"{ns}title")
        # Atom <link> uses href attribute
        link_el = entry.find(f"{ns}link[@rel='alternate']")
        if link_el is None:
            link_el = entry.find(f"{ns}link")
        link = (link_el.get("href", "") if link_el is not None else "")
        entry_id = _xml_text(entry, f"{ns}id") or link
        published = _xml_text(entry, f"{ns}published") or _xml_text(entry, f"{ns}updated")
        summary = _xml_text(entry, f"{ns}summary") or _xml_text(entry, f"{ns}content")
        if not entry_id:
            continue
        entries.append(
            {
                "id": entry_id,
                "title": title.strip(),
                "url": link.strip(),
                "published": published,
                "summary": _clean_html(summary)[:500],
            }
        )
    return entries


def _xml_text(parent: ET.Element, tag: str) -> str:
    """Return text content of the first child matching *tag*, or ""."""
    el = parent.find(tag)
    return (el.text or "") if el is not None else ""


def _clean_html(text: str) -> str:
    """Rough strip of HTML tags for embed descriptions."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# Hugging Face API
# ============================================================
def fetch_hf_models(org: str) -> dict[str, dict]:
    """Return {model_id: info} for recent models from *org*."""
    url = f"{HF_API_BASE}/models?author={org}&sort=createdAt&direction=-1&limit=20"
    models: dict[str, dict] = {}
    try:
        resp = fetch_url(url, retries=2)
        for m in resp.json():
            mid = m.get("id", "")
            if not mid:
                continue
            models[mid] = {
                "model_id": mid,
                "created_at": m.get("createdAt", ""),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "tags": m.get("tags", []),
                "likes": m.get("likes", 0),
                "downloads": m.get("downloads", 0),
            }
        log.info("Fetched %d HF models for org %s", len(models), org)
    except Exception as exc:
        log.warning("Failed to fetch HF models for %s: %s", org, exc)
    return models


# ============================================================
# GitHub Releases API
# ============================================================
def fetch_gh_releases(repo: str, token: str | None = None) -> dict[str, dict]:
    """Return {tag: info} for the latest releases in *repo*."""
    url = f"{GH_API_BASE}/repos/{repo}/releases?per_page=10"
    releases: dict[str, dict] = {}
    try:
        resp = fetch_url(url, retries=2, token=token)
        for r in resp.json():
            tag = r.get("tag_name", "")
            if not tag:
                continue
            releases[tag] = {
                "tag": tag,
                "name": r.get("name", tag),
                "url": r.get("html_url", ""),
                "published_at": r.get("published_at", ""),
                "body": (r.get("body") or "")[:500],
            }
        log.info("Fetched %d GH releases for %s", len(releases), repo)
    except Exception as exc:
        log.warning("Failed to fetch GH releases for %s: %s", repo, exc)
    return releases


# ============================================================
# Hacker News (Algolia) API
# ============================================================
def fetch_hn_stories(
    queries: list[str],
    since_ts: int = 0,
) -> dict[str, dict]:
    """Search HN for recent stories matching *queries*.

    Only returns stories created after *since_ts* (Unix timestamp).
    De-duplicates across queries by story objectID.
    """
    stories: dict[str, dict] = {}
    for q in queries:
        url = (
            f"{HN_API_BASE}/search_by_date"
            f"?query={requests.utils.quote(q)}"
            f"&tags=story"
            f"&hitsPerPage=5"
        )
        if since_ts:
            url += f"&numericFilters=created_at_i>{since_ts}"
        try:
            resp = fetch_url(url, retries=1, timeout=15)
            data = resp.json()
            for hit in data.get("hits", []):
                sid = hit.get("objectID", "")
                if not sid or sid in stories:
                    continue
                stories[sid] = {
                    "title": hit.get("title", ""),
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={sid}",
                    "hn_url": f"https://news.ycombinator.com/item?id={sid}",
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "created_at_i": hit.get("created_at_i", 0),
                    "author": hit.get("author", ""),
                }
        except Exception as exc:
            log.debug("HN search failed for query %r: %s", q, exc)
    log.info("Fetched %d unique HN stories across %d queries", len(stories), len(queries))
    return stories


# ============================================================
# Snapshot Building
# ============================================================
def build_snapshot(
    gh_token: str | None = None,
    hn_since_ts: int = 0,
) -> dict[str, Any]:
    """Aggregate data from all sources into a single snapshot dict.

    Structure::

        {
            "timestamp_utc": "...",
            "blogs": {"anthropic": {"id": {...}, ...}, ...},
            "hf_models": {"meta-llama": {"model_id": {...}, ...}, ...},
            "gh_releases": {"owner/repo": {"tag": {...}, ...}, ...},
            "hn_stories": {"objectID": {...}, ...},
        }
    """
    snapshot: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "blogs": {},
        "hf_models": {},
        "gh_releases": {},
        "hn_stories": {},
    }

    for source_key, cfg in SOURCE_CONFIG.items():
        # Blog feeds
        if cfg.get("rss_urls"):
            snapshot["blogs"][source_key] = fetch_blog_entries(source_key, cfg)

        # Hugging Face
        for org in cfg.get("hf_orgs", []):
            snapshot["hf_models"][org] = fetch_hf_models(org)

        # GitHub releases
        for repo in cfg.get("gh_repos", []):
            snapshot["gh_releases"][repo] = fetch_gh_releases(repo, token=gh_token)

    # Hacker News (not tied to a single source)
    snapshot["hn_stories"] = fetch_hn_stories(HN_SEARCH_QUERIES, since_ts=hn_since_ts)

    return snapshot


# ============================================================
# Diffing
# ============================================================
def diff_snapshots(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare *old* and *new* snapshots, returning a list of change dicts.

    Each change dict has at least:
        type        : "blog" | "hf_model" | "gh_release" | "hn_story"
        source_key  : key in SOURCE_CONFIG (or "hn" for Hacker News)
        source_label: human-readable label
        source_color: Discord embed colour int
        title       : short description of what changed
        url         : link
    """
    changes: list[dict[str, Any]] = []

    # --- Blog entries ---
    for source_key, new_entries in new.get("blogs", {}).items():
        old_entries = old.get("blogs", {}).get(source_key, {})
        if not old_entries:
            # No prior baseline for this source – absorb silently (seed)
            log.info("Seeding blog baseline for %s (%d entries)", source_key, len(new_entries))
            continue
        cfg = SOURCE_CONFIG.get(source_key, {})
        for eid, entry in new_entries.items():
            if eid not in old_entries:
                changes.append(
                    {
                        "type": "blog",
                        "source_key": source_key,
                        "source_label": cfg.get("label", source_key),
                        "source_color": cfg.get("color", 0x99AAB5),
                        "title": entry.get("title", "Untitled"),
                        "url": entry.get("url", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", ""),
                    }
                )

    # --- Hugging Face models ---
    for org, new_models in new.get("hf_models", {}).items():
        old_models = old.get("hf_models", {}).get(org, {})
        if not old_models:
            log.info("Seeding HF baseline for %s (%d models)", org, len(new_models))
            continue
        # Find the source_key for this org
        src_key, src_cfg = _source_for_hf_org(org)
        for mid, model in new_models.items():
            if mid not in old_models:
                changes.append(
                    {
                        "type": "hf_model",
                        "source_key": src_key,
                        "source_label": src_cfg.get("label", org),
                        "source_color": src_cfg.get("color", 0x99AAB5),
                        "title": mid,
                        "url": f"https://huggingface.co/{mid}",
                        "created_at": model.get("created_at", ""),
                        "pipeline_tag": model.get("pipeline_tag", ""),
                        "likes": model.get("likes", 0),
                        "downloads": model.get("downloads", 0),
                    }
                )

    # --- GitHub releases ---
    for repo, new_releases in new.get("gh_releases", {}).items():
        old_releases = old.get("gh_releases", {}).get(repo, {})
        if not old_releases:
            log.info("Seeding GH baseline for %s (%d releases)", repo, len(new_releases))
            continue
        src_key, src_cfg = _source_for_gh_repo(repo)
        for tag, rel in new_releases.items():
            if tag not in old_releases:
                changes.append(
                    {
                        "type": "gh_release",
                        "source_key": src_key,
                        "source_label": src_cfg.get("label", repo),
                        "source_color": src_cfg.get("color", 0x99AAB5),
                        "title": f"{repo} — {rel.get('name', tag)}",
                        "url": rel.get("url", ""),
                        "published_at": rel.get("published_at", ""),
                        "body": rel.get("body", ""),
                    }
                )

    # --- Hacker News stories ---
    old_hn = old.get("hn_stories", {})
    if not old_hn:
        log.info("Seeding HN baseline (%d stories)", len(new.get("hn_stories", {})))
    else:
        for sid, story in new.get("hn_stories", {}).items():
            if sid not in old_hn:
                changes.append(
                    {
                        "type": "hn_story",
                        "source_key": "hn",
                        "source_label": "Hacker News",
                        "source_color": 0xFF6600,
                        "title": story.get("title", ""),
                        "url": story.get("url", ""),
                        "hn_url": story.get("hn_url", ""),
                        "points": story.get("points", 0),
                        "num_comments": story.get("num_comments", 0),
                    }
                )

    return changes


def _source_for_hf_org(org: str) -> tuple[str, dict]:
    """Look up the SOURCE_CONFIG entry that owns a given HF org."""
    for key, cfg in SOURCE_CONFIG.items():
        if org in cfg.get("hf_orgs", []):
            return key, cfg
    return org, {"label": org, "color": 0x99AAB5}


def _source_for_gh_repo(repo: str) -> tuple[str, dict]:
    """Look up the SOURCE_CONFIG entry that owns a given GH repo."""
    for key, cfg in SOURCE_CONFIG.items():
        if repo in cfg.get("gh_repos", []):
            return key, cfg
    return repo, {"label": repo, "color": 0x99AAB5}


# ============================================================
# Priority / keyword helpers
# ============================================================
def is_high_priority(text: str) -> bool:
    """Return *True* if *text* contains model-release-related keywords."""
    lower = text.lower()
    for kw in RELEASE_KEYWORDS:
        if kw in lower:
            return True
    # Check model names
    for src_cfg in SOURCE_CONFIG.values():
        for name in src_cfg.get("model_names", []):
            if name.lower() in lower:
                return True
    return False


# ============================================================
# Discord Embed Construction
# ============================================================
PRIORITY_EMOJI = "\U0001f6a8"  # 🚨
TYPE_LABELS: dict[str, str] = {
    "blog": "Blog Post",
    "hf_model": "Hugging Face Model",
    "gh_release": "GitHub Release",
    "hn_story": "Hacker News",
}


def build_embeds(
    changes: list[dict[str, Any]],
    force_summary: bool = False,
) -> list[dict[str, Any]]:
    """Build Discord embed objects from a list of changes.

    Returns a flat list of embed dicts (max 10 per Discord message,
    handled during delivery).
    """
    embeds: list[dict[str, Any]] = []

    for ch in changes:
        change_type = ch["type"]
        label = ch["source_label"]
        color = ch["source_color"]
        title_text = ch.get("title", "")
        url = ch.get("url", "")
        type_label = TYPE_LABELS.get(change_type, change_type)

        high = is_high_priority(title_text + " " + ch.get("summary", "") + " " + ch.get("body", ""))
        prefix = f"{PRIORITY_EMOJI} " if high else ""

        embed: dict[str, Any] = {
            "color": color,
            "title": f"{prefix}[{label}] {type_label}",
            "url": url if url.startswith("http") else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Description
        desc_parts: list[str] = []
        if title_text:
            if url:
                desc_parts.append(f"**[{title_text}]({url})**")
            else:
                desc_parts.append(f"**{title_text}**")

        if change_type == "blog":
            if ch.get("published"):
                desc_parts.append(f"Published: {ch['published']}")
            if ch.get("summary"):
                desc_parts.append(f"\n{ch['summary']}")

        elif change_type == "hf_model":
            details: list[str] = []
            if ch.get("pipeline_tag"):
                details.append(f"Pipeline: `{ch['pipeline_tag']}`")
            if ch.get("likes"):
                details.append(f"Likes: {ch['likes']}")
            if ch.get("downloads"):
                details.append(f"Downloads: {ch['downloads']:,}")
            if details:
                desc_parts.append(" | ".join(details))

        elif change_type == "gh_release":
            if ch.get("published_at"):
                desc_parts.append(f"Published: {ch['published_at']}")
            if ch.get("body"):
                desc_parts.append(f"```\n{ch['body'][:400]}\n```")

        elif change_type == "hn_story":
            hn_meta: list[str] = []
            if ch.get("points"):
                hn_meta.append(f"{ch['points']} points")
            if ch.get("num_comments"):
                hn_meta.append(f"{ch['num_comments']} comments")
            if hn_meta:
                desc_parts.append(" | ".join(hn_meta))
            if ch.get("hn_url") and ch.get("hn_url") != ch.get("url"):
                desc_parts.append(f"[HN Discussion]({ch['hn_url']})")

        embed["description"] = "\n".join(desc_parts) if desc_parts else "No details"

        # Footer
        footer_parts = [type_label, ch.get("source_key", "")]
        embed["footer"] = {
            "text": " | ".join(p for p in footer_parts if p)
            + f" | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        }

        # Remove None url to avoid Discord API errors
        if embed.get("url") is None:
            embed.pop("url", None)

        embeds.append(embed)

    # Force-send summary embed when there are no changes
    if not embeds and force_summary:
        embeds.append(
            {
                "color": 0x5865F2,
                "title": "AI News Monitor - Status Check",
                "description": (
                    "No new updates detected across all monitored sources.\n\n"
                    f"Sources checked: {len(SOURCE_CONFIG)} companies, "
                    f"HN queries: {len(HN_SEARCH_QUERIES)}"
                ),
                "footer": {
                    "text": f"Force-send | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                },
            }
        )

    return embeds


# ============================================================
# Discord Webhook Delivery
# ============================================================
def validate_webhook_url(url: str) -> bool:
    """Check that *url* looks like a Discord webhook."""
    if not url:
        return False
    return url.startswith("https://discord.com/api/webhooks/") or url.startswith(
        "https://discordapp.com/api/webhooks/"
    )


def send_discord(
    webhook_url: str,
    embeds: list[dict[str, Any]],
    *,
    retries: int = 3,
    backoff: float = 2.0,
    dry_run: bool = False,
) -> bool:
    """Post *embeds* to a Discord webhook, batching 10 per message.

    Returns *True* if all batches were delivered successfully.
    """
    if not embeds:
        log.info("No embeds to send")
        return True

    if dry_run:
        for i, e in enumerate(embeds, 1):
            log.info("[DRY-RUN] Embed %d: %s", i, e.get("title", "?"))
        return True

    if not validate_webhook_url(webhook_url):
        log.error("Invalid Discord webhook URL")
        return False

    ok = True
    for batch_start in range(0, len(embeds), 10):
        batch = embeds[batch_start : batch_start + 10]
        payload = {"embeds": batch}
        if not _post_discord_payload(webhook_url, payload, retries=retries, backoff=backoff):
            ok = False
        if batch_start + 10 < len(embeds):
            time.sleep(1.5)  # rate-limit cushion between batches
    return ok


def _post_discord_payload(
    webhook_url: str,
    payload: dict,
    *,
    retries: int = 3,
    backoff: float = 2.0,
) -> bool:
    """POST JSON *payload* to *webhook_url* with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                webhook_url,
                json=payload,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 204:
                return True
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", backoff * attempt)
                log.warning("Discord rate-limited, sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                continue
            if 400 <= resp.status_code < 500:
                log.error(
                    "Discord client error %d: %s",
                    resp.status_code,
                    resp.text[:300],
                )
                return False
            log.warning("Discord server error %d, retrying…", resp.status_code)
        except requests.RequestException as exc:
            log.warning("Discord request failed: %s", exc)
        if attempt < retries:
            wait = backoff * attempt + random.uniform(0, 1)
            time.sleep(wait)
    log.error("Failed to deliver Discord payload after %d attempts", retries)
    return False


# ============================================================
# State Persistence
# ============================================================
def load_state(path: str) -> dict[str, Any]:
    """Load JSON state from *path*, returning empty dict if missing."""
    p = Path(path)
    if not p.exists():
        log.info("No state file at %s – this will be a seed run", path)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        log.info("Loaded state from %s (last check: %s)", path, data.get("timestamp_utc", "?"))
        return data
    except Exception as exc:
        log.warning("Failed to load state from %s: %s – starting fresh", path, exc)
        return {}


def save_state(path: str, snapshot: dict[str, Any]) -> None:
    """Persist *snapshot* to *path* as JSON, trimming old entries."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Trim stored entries to avoid unbounded growth
    trimmed = _trim_snapshot(snapshot)
    p.write_text(json.dumps(trimmed, indent=2, default=str), encoding="utf-8")
    log.info("Saved state to %s", path)


def _trim_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Limit stored entries per source to prevent state file bloat."""
    out = dict(snapshot)

    # Trim blog entries
    blogs = out.get("blogs", {})
    for source_key in blogs:
        entries = blogs[source_key]
        if len(entries) > MAX_STORED_ENTRIES_PER_SOURCE:
            # Keep most recent by insertion order (Python 3.7+ dicts are ordered)
            keys = list(entries.keys())[-MAX_STORED_ENTRIES_PER_SOURCE:]
            blogs[source_key] = {k: entries[k] for k in keys}

    # Trim HF models
    hf = out.get("hf_models", {})
    for org in hf:
        models = hf[org]
        if len(models) > MAX_STORED_ENTRIES_PER_SOURCE:
            keys = list(models.keys())[-MAX_STORED_ENTRIES_PER_SOURCE:]
            hf[org] = {k: models[k] for k in keys}

    # Trim GH releases
    gh = out.get("gh_releases", {})
    for repo in gh:
        releases = gh[repo]
        if len(releases) > MAX_STORED_ENTRIES_PER_SOURCE:
            keys = list(releases.keys())[-MAX_STORED_ENTRIES_PER_SOURCE:]
            gh[repo] = {k: releases[k] for k in keys}

    # Trim HN stories
    hn = out.get("hn_stories", {})
    if len(hn) > MAX_HN_STORIES_STORED:
        keys = list(hn.keys())[-MAX_HN_STORIES_STORED:]
        out["hn_stories"] = {k: hn[k] for k in keys}

    return out


# ============================================================
# Merge helper: carry forward old state for sources that failed
# ============================================================
def merge_with_fallback(
    old: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    """For each source category, if new fetch returned nothing but old had
    data, carry the old data forward.  This prevents false-positive
    'new entry' alerts when a fetch temporarily fails.
    """
    merged = dict(new)

    for category in ("blogs", "hf_models", "gh_releases"):
        old_cat = old.get(category, {})
        new_cat = merged.get(category, {})
        for key, old_data in old_cat.items():
            if key not in new_cat or not new_cat[key]:
                if old_data:
                    log.info(
                        "Carrying forward old %s data for %s (new fetch was empty)",
                        category,
                        key,
                    )
                    new_cat[key] = old_data
        merged[category] = new_cat

    return merged


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor AI model news and send Discord notifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  DISCORD_WEBHOOK_URL   Discord webhook (fallback for --webhook-url)\n"
            "  GITHUB_TOKEN          GitHub token for higher API rate limits\n"
        ),
    )

    p.add_argument(
        "--webhook-url",
        default=os.environ.get("DISCORD_WEBHOOK_URL", ""),
        help="Discord webhook URL (default: $DISCORD_WEBHOOK_URL)",
    )
    p.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        help=f"Path to JSON state file (default: {DEFAULT_STATE_FILE})",
    )
    p.add_argument(
        "--force-send",
        action="store_true",
        default=False,
        help="Send a status embed even when there are no changes",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log what would be sent without posting to Discord",
    )
    p.add_argument(
        "--seed",
        action="store_true",
        default=False,
        help="Force a seed run: build state without sending notifications",
    )

    # Loop mode
    p.add_argument("--loop", action="store_true", help="Run multiple checks in a loop")
    p.add_argument("--max-checks", type=int, default=4, help="Max iterations in loop mode")
    p.add_argument(
        "--min-interval-seconds",
        type=int,
        default=60,
        help="Min sleep between loop iterations",
    )
    p.add_argument(
        "--max-interval-seconds",
        type=int,
        default=120,
        help="Max sleep between loop iterations",
    )

    # Tuning
    p.add_argument("--timeout-seconds", type=int, default=REQUEST_TIMEOUT)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-backoff-seconds", type=float, default=2.0)

    return p.parse_args()


# ============================================================
# Single-check orchestration
# ============================================================
def run_single_check(args: argparse.Namespace) -> None:
    """Execute one poll-diff-notify cycle."""
    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = args.timeout_seconds

    gh_token = os.environ.get("GITHUB_TOKEN")

    # 1. Load previous state
    old_state = load_state(args.state_file)
    is_seed = args.seed or (not old_state)

    # Determine HN "since" timestamp from old state
    hn_since_ts = 0
    if old_state:
        # Use the oldest created_at_i from stored HN stories, or 0
        old_hn = old_state.get("hn_stories", {})
        if old_hn:
            # Only fetch stories newer than our last check
            timestamps = [s.get("created_at_i", 0) for s in old_hn.values() if s.get("created_at_i")]
            if timestamps:
                hn_since_ts = max(timestamps) - 3600  # 1-hour overlap for safety

    # 2. Build new snapshot
    log.info("Building snapshot from all sources…")
    new_snapshot = build_snapshot(gh_token=gh_token, hn_since_ts=hn_since_ts)

    # 3. Merge with fallback for failed fetches
    if old_state:
        new_snapshot = merge_with_fallback(old_state, new_snapshot)

    # 4. Seed run — save state, skip notifications
    if is_seed:
        log.info("Seed run – saving initial state without sending notifications")
        save_state(args.state_file, new_snapshot)
        return

    # 5. Diff
    changes = diff_snapshots(old_state, new_snapshot)
    log.info("Detected %d change(s)", len(changes))

    # 6. Notify
    if changes or args.force_send:
        embeds = build_embeds(changes, force_summary=args.force_send)
        log.info("Sending %d embed(s) to Discord", len(embeds))
        send_discord(
            args.webhook_url,
            embeds,
            retries=args.retries,
            backoff=args.retry_backoff_seconds,
            dry_run=args.dry_run,
        )
    else:
        log.info("No changes detected – nothing to send")

    # 7. Persist
    save_state(args.state_file, new_snapshot)


# ============================================================
# Entrypoint
# ============================================================
def main() -> None:
    args = parse_args()

    if not args.dry_run and not args.seed and not validate_webhook_url(args.webhook_url):
        log.error(
            "No valid Discord webhook URL provided. "
            "Set DISCORD_WEBHOOK_URL or pass --webhook-url."
        )
        raise SystemExit(1)

    if args.loop:
        log.info(
            "Loop mode: up to %d checks, interval %d–%ds",
            args.max_checks,
            args.min_interval_seconds,
            args.max_interval_seconds,
        )
        for i in range(args.max_checks):
            log.info("=== Check %d/%d ===", i + 1, args.max_checks)
            run_single_check(args)
            if i < args.max_checks - 1:
                interval = random.randint(args.min_interval_seconds, args.max_interval_seconds)
                log.info("Sleeping %ds before next check…", interval)
                time.sleep(interval)
    else:
        run_single_check(args)

    log.info("Done.")


if __name__ == "__main__":
    main()
