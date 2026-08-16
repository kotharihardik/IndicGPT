#!/usr/bin/env python3
"""Collect the Gujarati MANUAL corpus (Phase 1, Model H).

Phase 1 requires at least 20% of the final training tokens to be manually
collected. The corpus target is ~500M tokens, so the floor is ~100M manual
tokens; the other ~80% comes from a public dataset via `collect_public.py`.

This file only drives the run. The actual collection logic lives in
`adapters/`, one module per source type, picked by the `method:` field in
`data/sources/manual_sources.yaml`:

    wp_rest      adapters/wordpress.py    the site's own WordPress REST API
    sitemap      adapters/sitemap.py      sitemap.xml, or a link crawl
    pdf          adapters/pdf.py          text layer of local PDF books
    ia_fulltext  adapters/archive_org.py  archive.org's server-side OCR text

Each adapter module explains why that method exists and what it measured.

Usage:
    python data/collection/collect_manual.py --check     # robots + reachability
    python data/collection/collect_manual.py             # collect to target
    python data/collection/collect_manual.py --status    # progress vs the 20% floor
    python data/collection/collect_manual.py --only aksharnaad --limit-chars 1000000

Safe to Ctrl-C and rerun. The resume state is the collected data itself: at
startup each source's shards are scanned for the URLs already stored and those
are skipped, so there is no separate state file that can drift out of sync
with the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import JsonlWriter, human                       # noqa: E402
from adapters import ADAPTERS                                  # noqa: E402
from adapters.base import (CFG, CHARS_PER_TOKEN, MANUAL_FLOOR, ROOT,  # noqa: E402
                           TARGET_TOKENS, UA, existing)
from adapters.http import fetch, robots_ok                     # noqa: E402
from adapters.progress import Progress, fmt_dur, log           # noqa: E402


# Running one source


async def run_source(session, src: dict, limit_chars: int,
                     deadline: float) -> None:
    """Collect one source into its own shard file, up to its character budget.

    Every run writes a new shard rather than appending to one file. Shard names
    carry a timestamp and pid, so two runs can never corrupt each other's
    output and an interrupted run costs at most the shard it was writing.
    """
    name, method = src["name"], src["method"]
    budget = int(src.get("max_chars", 0))
    prev_docs, prev_chars, seen = existing(name)

    log(f"\n--- {name}  ({method})  budget "
        f"{human(budget) if budget else 'none'} chars ---")
    if prev_chars:
        log(f"  resuming: {prev_docs:,} docs / {human(prev_chars)} chars "
            f"already collected, {len(seen):,} URLs known")
    if budget and prev_chars >= budget:
        log("  budget already met - skipping")
        return

    adapter = ADAPTERS.get(method)
    if adapter is None:
        log(f"  ! unknown method {method!r} - expected one of "
            f"{', '.join(sorted(ADAPTERS))}")
        return

    prog = Progress(name, already=prev_chars, budget=budget)
    out_dir = CFG["paths"]["raw_manual"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"{name}.part-{int(time.time())}-{os.getpid()}.jsonl.gz"
    writer = JsonlWriter(shard)
    why = ""

    try:
        async for doc in adapter(session, src, seen, prog):
            writer.write(doc)
            prog.keep(doc.n_chars)
            prog.tick()
            if budget and prog.source_chars >= budget:
                why = "source budget reached"
                break
            if limit_chars and Progress.total_chars >= limit_chars:
                why = "--limit-chars reached"
                break
            if deadline and time.time() > deadline:
                why = "--max-hours reached"
                break
    except asyncio.CancelledError:
        why = "interrupted"
        raise
    finally:
        # Always close the writer, so a Ctrl-C still leaves a readable shard.
        writer.close()
        prog.done(why)
        log(f"  shard: {shard.name} ({writer.n:,} docs / {human(writer.chars)} chars)")


# Source registry


def load_sources(only: list, include_disabled: bool) -> list:
    """Load manual_sources.yaml, merging the shared `defaults:` block into each.

    Every source therefore gets the common crawl settings without repeating
    them, but can still override any single key.
    """
    path = ROOT / "data" / "sources" / "manual_sources.yaml"
    blob = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = blob.get("defaults") or {}
    out = [{**defaults, **s} for s in (blob.get("sources") or [])]
    if not include_disabled:
        out = [s for s in out if s.get("enabled", True)]
    if only:
        out = [s for s in out if s["name"] in only]
    return out


# Reports (--status and --check). Neither writes any corpus data.


def report_status(sources: list) -> None:
    """Print per-source totals and progress toward the 20% manual-token floor."""
    print("\nGUJARATI MANUAL COLLECTION STATUS")
    print(f"  token estimate at {CHARS_PER_TOKEN} chars/token "
          f"(exact counts: data/splits/corpus_stats.py --tokenizer)\n")
    print(f"  {'source':<24}{'method':<12}{'docs':>9}{'chars':>11}"
          f"{'~tokens':>11}{'budget':>10}{'':>7}")
    print("  " + "-" * 84)

    total_docs = total_chars = 0
    by_method: dict = {}
    for src in sources:
        docs, chars, _ = existing(src["name"])
        total_docs += docs
        total_chars += chars
        by_method[src["method"]] = by_method.get(src["method"], 0) + chars
        budget = int(src.get("max_chars", 0))
        pct = f"{100 * chars / budget:.0f}%" if budget else "-"
        print(f"  {src['name']:<24}{src['method']:<12}{docs:>9,}"
              f"{human(chars):>11}{human(chars / CHARS_PER_TOKEN):>11}"
              f"{(human(budget) if budget else '-'):>10}{pct:>7}")

    print("  " + "-" * 84)
    tokens = total_chars / CHARS_PER_TOKEN
    print(f"  {'TOTAL':<24}{'':<12}{total_docs:>9,}{human(total_chars):>11}"
          f"{human(tokens):>11}")

    print("\n  by method:")
    for method, chars in sorted(by_method.items(), key=lambda kv: -kv[1]):
        share = 100 * chars / total_chars if total_chars else 0
        print(f"    {method:<12}{human(chars):>10} chars  "
              f"{human(chars / CHARS_PER_TOKEN):>9} tok  ({share:4.1f}%)")

    # Two bars: the requirement itself, and the over-collection target that
    # leaves room for what preprocessing and dedup remove later.
    for label, goal in (("hard floor (20% of 500M)", MANUAL_FLOOR),
                        ("collection target", TARGET_TOKENS)):
        frac = min(1.0, tokens / goal)
        bar = "#" * int(frac * 34) + "." * (34 - int(frac * 34))
        print(f"\n  {label}: {human(goal)} tokens")
        print(f"  [{bar}] {100 * tokens / goal:5.1f}%")

    if tokens < TARGET_TOKENS:
        need = TARGET_TOKENS * CHARS_PER_TOKEN - total_chars
        print(f"\n  still need ~{human(need)} chars "
              f"(~{human(need / CHARS_PER_TOKEN)} tokens)")
    else:
        print("\n  target met.")


async def report_check(sources: list) -> None:
    """Pre-flight: check robots.txt and reachability for every source.

    Worth running before a long collection. It is how I found that two sites
    had no working REST API before spending an hour discovering it the slow way.
    """
    print("\nPRE-FLIGHT CHECK (nothing is written)\n")
    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:
        for src in sources:
            name, method = src["name"], src["method"]
            if method == "pdf":
                pdf_dir = ROOT / src["local_dir"]
                n = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.exists() else 0
                print(f"  {name:<24}{method:<12}{n} PDFs on disk")
                continue

            allowed = await robots_ok(session, src["url"])
            note = ""
            if method == "wp_rest" and allowed:
                api = (f"https://{urlparse(src['url']).netloc}"
                       f"/wp-json/wp/v2/posts?per_page=1&_fields=id")
                data = await fetch(session, api, as_json=True, retries=1)
                note = "REST API ok" if isinstance(data, list) and data \
                    else "REST API UNREACHABLE"
            print(f"  {name:<24}{method:<12}"
                  f"{'ALLOW' if allowed else 'DISALLOW':<10}{note}")

    budget = sum(int(s.get("max_chars", 0)) for s in sources)
    print(f"\n  sum of budgets: {human(budget)} chars "
          f"(~{human(budget / CHARS_PER_TOKEN)} tokens)")
    print(f"  collection target: {human(TARGET_TOKENS)} tokens")


# Entry point


async def collect_all(sources: list, args) -> None:
    """Run every selected source in registry order, sharing one HTTP session."""
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0

    # Seed the run total with what is already on disk, so the ETA and the
    # percentage-of-target describe real progress and not just this run.
    Progress.total_chars = sum(existing(s["name"])[1] for s in sources)

    log("=" * 88)
    log(f"GUJARATI MANUAL COLLECTION   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  sources: {', '.join(s['name'] for s in sources)}")
    log(f"  target : {human(TARGET_TOKENS)} manual tokens "
        f"(~{human(TARGET_TOKENS * CHARS_PER_TOKEN)} chars)")
    log(f"  already: {human(Progress.total_chars)} chars collected")
    log("=" * 88)

    started = time.time()
    # One session for the whole run: connection reuse matters when a source
    # makes thousands of requests to the same host.
    connector = aiohttp.TCPConnector(limit=40, ttl_dns_cache=600)
    async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}) as s:
        for src in sources:
            if deadline and time.time() > deadline:
                log("\n  time box reached - stopping")
                break
            if args.limit_chars and Progress.total_chars >= args.limit_chars:
                log("\n  --limit-chars reached - stopping")
                break
            if Progress.total_chars >= TARGET_TOKENS * CHARS_PER_TOKEN:
                log("\n  collection target reached - stopping")
                break
            await run_source(s, src, args.limit_chars, deadline)

    tokens = Progress.total_chars / CHARS_PER_TOKEN
    log("\n" + "=" * 88)
    log(f"TOTAL {human(Progress.total_chars)} chars  ~{human(tokens)} tokens "
        f"({100 * tokens / TARGET_TOKENS:.1f}% of target) "
        f"in {fmt_dur(time.time() - started)}")
    log("=" * 88)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect the Gujarati manual corpus (Phase 1).")
    ap.add_argument("--only", nargs="*", default=[], metavar="NAME",
                    help="restrict the run to these source names")
    ap.add_argument("--include-disabled", action="store_true",
                    help="also run sources marked 'enabled: false'")
    ap.add_argument("--limit-chars", type=int, default=0,
                    help="stop after this many characters in total")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop after this many hours (resumable)")
    ap.add_argument("--check", action="store_true",
                    help="check robots.txt and reachability, write nothing")
    ap.add_argument("--status", action="store_true",
                    help="report progress toward the 20%% floor and exit")
    args = ap.parse_args()

    sources = load_sources(args.only, args.include_disabled)
    if not sources:
        print("No sources selected. Check --only spelling and 'enabled:' in "
              "data/sources/manual_sources.yaml.")
        sys.exit(1)

    if args.status:
        report_status(sources)
        return
    if args.check:
        asyncio.run(report_check(sources))
        return

    # Logs are filed by stage so phases 2 and 3 can add their own without the
    # directory turning into an unsearchable pile.
    log_dir = CFG["paths"]["logs"] / "collection"
    log_dir.mkdir(parents=True, exist_ok=True)
    Progress.log_file = open(log_dir / "collect_manual.log", "a", encoding="utf-8")
    try:
        asyncio.run(collect_all(sources, args))
    except KeyboardInterrupt:
        log("\ninterrupted - shards are flushed, rerun to resume")
    finally:
        Progress.log_file.close()
    print("\nprogress: python3 data/collection/collect_manual.py --status")


if __name__ == "__main__":
    main()
