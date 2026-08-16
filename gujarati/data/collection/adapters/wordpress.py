"""Method `wp_rest`: read articles from a site's own WordPress REST API.

Most of the Gujarati manual corpus comes from here, and the reason is
throughput. One request to /wp-json/wp/v2/posts?per_page=100 returns 100
complete article bodies as JSON, with no navigation, adverts or scripts to
strip afterwards.

What I measured on these sites:

    wp_rest                         ~570 K chars/s
    sitemap (one request per page)    ~7 K chars/s
    my first serial crawler           ~0.9 K chars/s

My first version was that serial crawler, with a 2 second sleep and two
requests per page (one for the text, one for its links). It managed 10.3M
chars in 3h16m and then stalled, because you cannot reach a news archive by
following links from the homepage. The API is faster for a structural reason,
not because it hammers the server: it amortises one request over 100 documents
and sends no markup. The per-host request rate here is lower than a browser
loading the same pages.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from .base import clean_text, make_doc, reject_reason
from .http import fetch, robots_ok
from .progress import Progress, log

#: Only ask for the fields we use. The default payload is several times bigger.
WP_FIELDS = "id,date,link,title,content"


async def collect_wp(session, src: dict, seen: set, prog: Progress):
    """Yield Documents from a WordPress site, newest posts first.

    Pages are fetched in small concurrent batches. Plain ?page=N paging is
    enough: the biggest budget here needs about 600 pages and these hosts serve
    page=1600. Some cap lower, which is why a failed or empty page just ends
    the source instead of raising.
    """
    api = f"https://{urlparse(src['url']).netloc}/wp-json/wp/v2/posts"
    if not await robots_ok(session, api):
        log(f"  {src['name']}: robots.txt DISALLOWS {api} - skipping")
        return

    per_page = int(src.get("per_page", 100))
    batch = int(src.get("batch", 4))

    # We always read newest-first and take every post, so whatever is already
    # stored is a prefix of the archive. On resume, jump most of the way past
    # it rather than re-requesting hundreds of pages only to throw them away.
    # The 5-page margin covers articles published since the last run, and the
    # URL check below still catches anything already stored.
    page = max(1, len(seen) // per_page - 5)
    if page > 1:
        log(f"  {src['name']}: {len(seen):,} posts already stored, "
            f"resuming from page {page}")

    while True:
        urls = [f"{api}?per_page={per_page}&page={p}&orderby=date&order=desc"
                f"&_fields={WP_FIELDS}" for p in range(page, page + batch)]
        results = await asyncio.gather(*(fetch(session, u, as_json=True)
                                        for u in urls))
        prog.requests += len(urls)

        got_full_page = False
        for payload in results:
            if not isinstance(payload, list) or not payload:
                continue
            if len(payload) >= per_page:
                got_full_page = True

            for post in payload:
                link = post.get("link") or ""
                if link in seen:
                    # Resuming re-walks pages we already have, because the
                    # resume state is the stored data and not a saved page
                    # number. Tick anyway so a long skip looks like progress
                    # instead of a hang.
                    prog.reject("already-collected")
                    prog.tick()
                    continue
                seen.add(link)

                text = clean_text((post.get("content") or {}).get("rendered", ""),
                                  is_html=True)
                reason = reject_reason(text)
                if reason:
                    prog.reject(reason)
                    continue
                title = clean_text((post.get("title") or {}).get("rendered", ""),
                                   is_html=True)
                yield make_doc(src, text, link, title,
                               {"wp_id": post.get("id"),
                                "published": post.get("date")})

        # A short page means we hit the end of the archive, or the host's page
        # ceiling. Either way there is nothing more to ask for.
        if not got_full_page:
            return
        page += batch
