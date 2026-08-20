"""Collection method `wp_rest`: read articles from a site's WordPress REST API.

The most efficient collection route available, where a source supports it. A
single request to /wp-json/wp/v2/posts?per_page=100 returns 100 complete
article bodies as JSON, containing no navigation, advertising or scripts to
remove afterwards. Measured against the sitemap method on comparable pages it
yields roughly 70x more text per second, because one request is amortised over
100 documents and no markup is transferred. The resulting per-host request rate
is lower than that of a browser loading the same pages.

Only 4 of the 23 Nepali sources expose a usable REST API, so this method
supplies 163.9M characters, approximately 28.5% of the manual corpus. The
remaining sources run custom content management systems and are collected via
adapters/sitemap.py instead. The Gujarati pipeline shows the opposite
distribution, where most publishers run WordPress and this method dominates.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from .base import clean_text, make_doc, reject_reason
from .http import fetch, robots_ok
from .progress import Progress, log

#: Restrict the response to the fields actually used. The default payload is
#: several times larger.
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

    # Posts are always read newest-first and taken in full, so the stored
    # data is a prefix of the archive. On resume, skip most of that prefix
    # rather than re-requesting hundreds of pages and discarding them. The
    # 5-page margin covers articles published since the previous run; the URL
    # check below still catches anything already stored.
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
                    # Resume re-walks already-collected pages, because the
                    # resume state is the stored data rather than a saved page
                    # number. The progress tick keeps a long skip visible as
                    # activity rather than appearing to hang.
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

        # A short page indicates the end of the archive or the host's page
        # ceiling. In either case no further pages are available.
        if not got_full_page:
            return
        page += batch
