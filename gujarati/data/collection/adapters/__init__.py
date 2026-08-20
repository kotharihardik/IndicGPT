"""Collection adapters, one per source type.

Each adapter is an async generator with the same signature:

    async def collect_x(session, src, seen, prog) -> yields Document

    session  shared aiohttp session (ignored by the pdf adapter)
    src      one entry from data/sources/manual_sources.yaml
    seen     URLs/ids already collected, so a rerun skips them
    prog     Progress counters for the status line

Keeping the signature identical is what lets ADAPTERS below be a plain dict.
Adding a new source type means writing one file and adding one line here,
instead of extending an if/elif chain inside the collector.

The `method:` value in the YAML is the key, so a typo there fails with a clear
"unknown method" instead of silently collecting nothing.
"""

from .archive_org import collect_ia_fulltext
from .pdf import collect_pdf
from .sitemap import collect_sitemap
from .wordpress import collect_wp

#: method name in manual_sources.yaml -> the adapter that handles it
ADAPTERS = {
    "wp_rest": collect_wp,
    "sitemap": collect_sitemap,
    "pdf": collect_pdf,
    "ia_fulltext": collect_ia_fulltext,
}

__all__ = ["ADAPTERS"]
