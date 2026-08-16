"""Method `sitemap`: fetch article pages one at a time and extract the body.

For sites with no WordPress API. Read sitemap.xml, fetch every article page,
and pull the body out with trafilatura. That is one request per document
instead of one per hundred, so it is roughly 70x more expensive per character
than `wp_rest` and I only use it where the register is worth the cost:
literature (aksharnaad, layastaro), opinion writing, and a general portal.

If a site publishes no usable sitemap there is a breadth-first link crawl as a
fallback. I needed it because layastaro.com found exactly zero URLs and
finished "successfully" in two seconds without collecting anything.

Boilerplate removal is trafilatura's job here, not ours. It reads the page
structure and drops navigation, sidebars, footers, adverts and comment
sections; `favor_precision=True` means it throws a block away when unsure,
which is the right trade for a training corpus.
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import unquote, urldefrag, urljoin, urlparse

from .base import SKIP_EXT, clean_text, make_doc, reject_reason
from .http import fetch, robots_ok
from .progress import Progress, log


def wanted_url(src: dict, url: str) -> bool:
    """True if a URL is on-host, is not an asset, and is not in `deny:`."""
    if not url.startswith("http"):
        return False
    hosts = [h.removeprefix("www.") for h in
             (src.get("hosts") or [urlparse(src["url"]).netloc])]
    host = urlparse(url).netloc.removeprefix("www.")
    if not any(host == h or host.endswith("." + h) for h in hosts):
        return False
    if url.lower().split("?")[0].endswith(SKIP_EXT):
        return False
    # Test the decoded form too. Deny patterns written with Gujarati characters
    # would never match the percent-encoded URLs a crawler actually sees.
    decoded = unquote(url)
    return not any(re.search(p, url) or re.search(p, decoded)
                   for p in src.get("deny", []))


async def sitemap_urls(session, src: dict, prog: Progress) -> list:
    """Walk the sitemap tree, following sitemap indexes, and return article URLs."""
    from bs4 import BeautifulSoup

    declared = list(src.get("sitemaps") or [])
    if not declared:
        robots = await fetch(session, f"https://{urlparse(src['url']).netloc}"
                                      f"/robots.txt", retries=1, timeout=20)
        declared = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots or "")

    # The guessed /sitemap.xml gets one short attempt only: some hosts hang on
    # a missing sitemap rather than returning 404.
    queue = [(u, False) for u in declared] + [
        (f"https://{urlparse(src['url']).netloc}/sitemap.xml", True)]

    max_pages = int(src.get("max_pages", 20000))
    seen_maps, out = set(), []
    while queue and len(out) < max_pages and len(seen_maps) < 400:
        url, guessed = queue.pop(0)
        if url in seen_maps:
            continue
        seen_maps.add(url)

        body = await fetch(session, url, retries=0 if guessed else 1,
                           timeout=10 if guessed else 30)
        prog.requests += 1
        if not body:
            continue
        try:
            soup = BeautifulSoup(body, "xml")
        except Exception:
            continue

        locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        if soup.find("sitemapindex"):
            queue += [(u, False) for u in locs if u not in seen_maps]
        else:
            out += [u for u in locs if wanted_url(src, u)]
    return list(dict.fromkeys(out))[:max_pages]


async def crawl_links(session, src: dict, prog: Progress, max_pages: int) -> list:
    """Breadth-first link discovery, for sites that publish no sitemap.

    Returns URLs only. The caller fetches them again to extract the text, which
    costs one extra request per page but keeps this function to one job and
    means every document goes through the same quality gate.
    """
    from bs4 import BeautifulSoup

    seed = src["url"]
    max_depth = int(src.get("max_depth", 3))
    frontier = [(seed, 0)]
    seen = {seed}
    found: list = []

    # The `len(seen)` cap stops a crawl that keeps finding pages but no
    # articles from running forever.
    while frontier and len(found) < max_pages and len(seen) < max_pages * 3:
        batch, frontier = frontier[:8], frontier[8:]
        pages = await asyncio.gather(
            *(fetch(session, u, retries=0, timeout=20) for u, _ in batch))
        prog.requests += len(batch)

        for (url, depth), body in zip(batch, pages):
            if not body:
                continue
            if wanted_url(src, url) and url != seed:
                found.append(url)
            if depth >= max_depth:
                continue
            try:
                soup = BeautifulSoup(body, "lxml")
            except Exception:
                continue
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(url, a["href"]))[0]
                if nxt not in seen and wanted_url(src, nxt):
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))

        if len(found) % 200 < 8:
            log(f"    {src['name']}: link-crawl found {len(found):,} URLs "
                f"(frontier {len(frontier):,})")
    return found


async def collect_sitemap(session, src: dict, seen: set, prog: Progress):
    """Yield Documents by fetching each article page and extracting its body."""
    import trafilatura

    if not await robots_ok(session, src["url"]):
        log(f"  {src['name']}: robots.txt DISALLOWS {src['url']} - skipping")
        return

    urls = [u for u in await sitemap_urls(session, src, prog) if u not in seen]
    log(f"  {src['name']}: {len(urls):,} new article URLs from sitemap")

    # No sitemap, or one with almost nothing in it: follow links instead.
    if len(urls) < 100:
        log(f"  {src['name']}: sitemap thin - falling back to link crawl")
        discovered = await crawl_links(session, src, prog,
                                       int(src.get("max_pages", 40_000)))
        urls = [u for u in dict.fromkeys(urls + discovered) if u not in seen]
        log(f"  {src['name']}: {len(urls):,} new article URLs after link crawl")

    if not urls:
        return

    # Producer/consumer rather than one big gather: the queue bounds how many
    # extracted pages sit in memory at once, and the consumer can stop early
    # when the source hits its budget.
    concurrency = int(src.get("concurrency", 8))
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    sem = asyncio.Semaphore(concurrency)
    DONE = object()

    async def one(url: str):
        async with sem:
            html = await fetch(session, url)
            prog.requests += 1
            if not html:
                await queue.put(("", url))
                return
            raw = trafilatura.extract(html, url=url, include_comments=False,
                                      include_tables=False, favor_precision=True)
            await queue.put((clean_text(raw or ""), url))

    async def producer():
        await asyncio.gather(*(one(u) for u in urls))
        await queue.put(DONE)

    task = asyncio.create_task(producer())
    try:
        while True:
            item = await queue.get()
            if item is DONE:
                break
            text, url = item
            seen.add(url)
            if not text:
                prog.reject("no-text-extracted")
                continue
            reason = reject_reason(text)
            if reason:
                prog.reject(reason)
                continue
            yield make_doc(src, text, url)
    finally:
        # The consumer can break out early, so the producer has to be cancelled
        # explicitly or its pending requests keep the event loop alive.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
