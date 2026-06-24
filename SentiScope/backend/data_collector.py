"""
backend/data_collector.py
Real-time collectors from FREE public APIs (no API keys needed):
1. RedditRSSCollector — Reddit public JSON endpoint
2. HackerNewsCollector — Algolia HN search API
3. NewsRSSCollector — BBC / NPR / Al Jazeera / TechCrunch / The Verge / Ars Technica RSS
4. MultiSourceCollector — runs all three concurrently
"""

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Callable, Optional, Set
import threading
import httpx
from loguru import logger


# ── Helpers ───────────────────────────────────────────────────────────────────
def _short_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

HEADERS = {
    "User-Agent": "SentimentAnalysisBot/1.0 (college project; python-httpx)"
}


# ── 1. Reddit public JSON (no auth) ──────────────────────────────────────────
REDDIT_SUBREDDITS = [
    "worldnews", "technology", "science", "business",
    "sports", "entertainment", "politics", "gaming",
    "environment", "health",
]

class RedditRSSCollector:
    def __init__(self, callback: Callable, interval: float = 12.0):
        self.callback = callback
        self.interval = interval
        self._seen: Set[str] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("RedditRSSCollector started")

    def stop(self):
        self._stop.set()

    def _run(self):
        idx = 0
        while not self._stop.is_set():
            sub = REDDIT_SUBREDDITS[idx % len(REDDIT_SUBREDDITS)]
            idx += 1
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit=10"
                with httpx.Client(headers=HEADERS, timeout=10) as client:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        time.sleep(5)
                        continue

                    for child in resp.json().get("data", {}).get("children", []):
                        p = child.get("data", {})
                        pid = p.get("id", "")
                        if not pid or pid in self._seen:
                            continue
                        self._seen.add(pid)

                        title = p.get("title", "")
                        body = p.get("selftext", "")[:200]
                        text = f"{title}. {body}".strip(". ") if body else title
                        if not text:
                            continue

                        self.callback({
                            "id": f"reddit_{pid}",
                            "source": "Reddit",
                            "text": text,
                            "author": p.get("author", "anonymous"),
                            "topic": f"r/{sub}",
                            "url": f"https://reddit.com{p.get('permalink', '')}",
                            "likes": p.get("score", 0),
                            "retweets": p.get("num_comments", 0),
                            "timestamp": datetime.fromtimestamp(
                                p.get("created_utc", time.time()), tz=timezone.utc
                            ).isoformat(),
                        })
                    logger.debug(f"Reddit r/{sub}: OK")

            except httpx.RequestError as exc:
                logger.warning(f"Reddit network error: {exc}")
            except Exception as exc:
                logger.warning(f"Reddit error: {exc}")

            time.sleep(self.interval)


# ── 2. HackerNews via Algolia API (free, no auth) ────────────────────────────
HN_QUERIES = [
    "AI", "technology", "climate", "science", "startup",
    "programming", "machine learning", "cybersecurity", "space",
]

class HackerNewsCollector:
    def __init__(self, callback: Callable, interval: float = 18.0):
        self.callback = callback
        self.interval = interval
        self._seen: Set[str] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._idx = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("HackerNewsCollector started")

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            query = HN_QUERIES[self._idx % len(HN_QUERIES)]
            self._idx += 1
            try:
                url = (
                    f"https://hn.algolia.com/api/v1/search_by_date"
                    f"?query={query}&tags=story&hitsPerPage=8"
                )
                with httpx.Client(headers=HEADERS, timeout=10) as client:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        time.sleep(5)
                        continue

                    for hit in resp.json().get("hits", []):
                        oid = hit.get("objectID", "")
                        if not oid or oid in self._seen:
                            continue
                        self._seen.add(oid)
                        title = hit.get("title") or hit.get("story_title") or ""
                        if not title:
                            continue
                        self.callback({
                            "id": f"hn_{oid}",
                            "source": "HackerNews",
                            "text": title,
                            "author": hit.get("author", "anonymous"),
                            "topic": f"#{query}",
                            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                            "likes": hit.get("points", 0) or 0,
                            "retweets": hit.get("num_comments", 0) or 0,
                            "timestamp": hit.get("created_at", _now_iso()),
                        })
                    logger.debug(f"HN '{query}': OK")

            except httpx.RequestError as exc:
                logger.warning(f"HN network error: {exc}")
            except Exception as exc:
                logger.warning(f"HN error: {exc}")

            time.sleep(self.interval)


# ── 3. Public RSS news feeds ──────────────────────────────────────────────────
RSS_FEEDS = [
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Tech", "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Science Daily", "https://www.sciencedaily.com/rss/top.xml"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
]

class NewsRSSCollector:
    def __init__(self, callback: Callable, interval: float = 25.0):
        self.callback = callback
        self.interval = interval
        self._seen: Set[str] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._idx = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("NewsRSSCollector started")

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            name, url = RSS_FEEDS[self._idx % len(RSS_FEEDS)]
            self._idx += 1
            try:
                with httpx.Client(headers=HEADERS, timeout=10, follow_redirects=True) as client:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        time.sleep(5)
                        continue

                    root = ET.fromstring(resp.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

                    for item in items[:8]:
                        title_el = item.find("title") or item.find("atom:title", ns)
                        title = (title_el.text or "").strip() if title_el is not None else ""
                        if not title:
                            continue

                        desc_el = item.find("description") or item.find("atom:summary", ns)
                        desc = ""
                        if desc_el is not None and desc_el.text:
                            desc = re.sub(r"<[^>]+>", "", desc_el.text).strip()[:150]

                        text = f"{title}. {desc}".strip(". ") if desc else title

                        link_el = item.find("link") or item.find("atom:link", ns)
                        link = ""
                        if link_el is not None:
                            link = (link_el.get("href") or link_el.text or "").strip()

                        uid = _short_id(title + url)
                        if uid in self._seen:
                            continue
                        self._seen.add(uid)

                        self.callback({
                            "id": f"rss_{uid}",
                            "source": name,
                            "text": text,
                            "author": name,
                            "topic": f"#{name.split()[0]}",
                            "url": link,
                            "likes": 0,
                            "retweets": 0,
                            "timestamp": _now_iso(),
                        })
                    logger.debug(f"RSS '{name}': OK")

            except ET.ParseError as exc:
                logger.warning(f"RSS parse error ({name}): {exc}")
            except httpx.RequestError as exc:
                logger.warning(f"RSS network error ({name}): {exc}")
            except Exception as exc:
                logger.warning(f"RSS error ({name}): {exc}")

            time.sleep(self.interval)


# ── 4. Multi-source (all three in parallel) ───────────────────────────────────
class MultiSourceCollector:
    def __init__(self, callback: Callable):
        self._collectors = [
            RedditRSSCollector(callback, interval=12),
            HackerNewsCollector(callback, interval=18),
            NewsRSSCollector(callback, interval=25),
        ]

    def start(self):
        for c in self._collectors:
            c.start()
        logger.info("MultiSourceCollector started (Reddit + HN + RSS News)")

    def stop(self):
        for c in self._collectors:
            c.stop()


# ── Factory ───────────────────────────────────────────────────────────────────
def create_collector(source: str, callback: Callable, **kwargs):
    """source: 'multi' | 'reddit' | 'hackernews' | 'news'"""
    s = source.lower()
    if s == "reddit":
        return RedditRSSCollector(callback, kwargs.get("interval", 12))
    elif s == "hackernews":
        return HackerNewsCollector(callback, kwargs.get("interval", 18))
    elif s == "news":
        return NewsRSSCollector(callback, kwargs.get("interval", 25))
    else:
        return MultiSourceCollector(callback)
