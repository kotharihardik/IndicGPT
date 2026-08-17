"""HTTP access layer for the collection adapters.

Contains the two network primitives every adapter depends on: a robots.txt
permission check and a GET with bounded retries. Both are kept here rather
than in the individual adapters so that crawl politeness is defined in exactly
one place and cannot diverge between collection methods.
"""

from __future__ import annotations

import asyncio
import json
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import aiohttp

from .base import UA

#: Parsed robots.txt per host. Populated on first use so each host's
#: robots.txt is retrieved at most once per run.
_robots: dict = {}


async def robots_ok(session: aiohttp.ClientSession, url: str) -> bool:
    """Check whether robots.txt permits this crawler to fetch a URL.

    The check is performed on every run rather than recorded once in the
    source registry, because the directive is not stable: one source returned
    DISALLOW on an initial probe and ALLOW when re-checked a day later. An
    unreachable or unparseable robots.txt is treated as permission, which is
    the behaviour specified by RFC 9309.

    Args:
        session: Shared aiohttp session for the run.
        url: Absolute URL whose permission is being tested.

    Returns:
        True if the configured user agent is permitted to fetch `url`.
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
    """Retrieve a URL, retrying only failures that a retry can resolve.

    Retries are limited to HTTP 408, 429 and 5xx responses and to network
    errors. A 404 is treated as final, since repeating the request cannot
    change the outcome. Backoff is exponential: 1s, then 2s.

    `retries=0` is intended for speculative URLs such as a guessed
    /sitemap.xml. Some hosts hold the connection open instead of returning
    404, in which case three attempts at the default timeout consume over two
    minutes without producing information.

    Args:
        session: Shared aiohttp session for the run.
        url: Absolute URL to retrieve.
        as_json: Parse the response body as JSON before returning it.
        retries: Additional attempts after the first. 0 disables retrying.
        timeout: Total per-attempt timeout in seconds.

    Returns:
        The response body as `str`, or the parsed object when `as_json` is
        set, or None if the request failed or the body was not valid JSON.
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
