"""Polite HTTP fetching: robots.txt checks and a GET that retries.

Small file, but I wanted the crawl-politeness rules in one obvious place
rather than scattered through the adapters.
"""

from __future__ import annotations

import asyncio
import json
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import aiohttp

from .base import UA

#: One parsed robots.txt per host, so we fetch it once per run.
_robots: dict = {}


async def robots_ok(session: aiohttp.ClientSession, url: str) -> bool:
    """Return True if robots.txt allows our user agent to fetch `url`.

    Checked live on every run instead of being written down once in the source
    YAML, because it does change: sandesh.com answered DISALLOW on an early
    probe and ALLOW when I checked again a day later. If robots.txt cannot be
    reached at all we treat that as permission, which is what the standard
    says to do.
    """
    host = urlparse(url).netloc
    if host not in _robots:
        parser = robotparser.RobotFileParser()
        try:
            async with session.get(f"{urlparse(url).scheme}://{host}/robots.txt",
                                   timeout=aiohttp.ClientTimeout(total=20)) as r:
                parser.parse((await r.text(errors="ignore")).splitlines()
                             if r.status == 200 else [""])
        except Exception:
            parser.parse([""])
        _robots[host] = parser
    try:
        return _robots[host].can_fetch(UA, url)
    except Exception:
        return True


async def fetch(session: aiohttp.ClientSession, url: str, as_json: bool = False,
                retries: int = 2, timeout: float = 45.0):
    """GET a URL and return text, parsed JSON, or None on failure.

    Only 429 and 5xx and network errors are retried; a 404 is final, since
    asking again will not change it. Backoff is 1s then 2s.

    Pass ``retries=0`` for URLs that may simply not exist, such as a guessed
    /sitemap.xml. Some hosts hang instead of answering 404, and at a 45 second
    timeout three attempts burn over two minutes to learn nothing.
    """
    for attempt in range(retries + 1):
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status == 200:
                    body = await r.text(errors="replace")
                    if not as_json:
                        return body
                    try:
                        return json.loads(body)
                    except ValueError:
                        return None
                if r.status not in (408, 429, 500, 502, 503, 504):
                    return None
        except Exception:
            pass
        if attempt < retries:
            await asyncio.sleep(2 ** attempt)
    return None
