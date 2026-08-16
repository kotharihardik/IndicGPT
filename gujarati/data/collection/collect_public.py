"""Collect PUBLIC datasets listed in `data/sources/public_sources.yaml`.

Everything written here is labelled `public_hf` and therefore does NOT count
towards the >=20% manual-collection requirement.

Only the configured subset is pulled (e.g. `data_dir: verified/guj` for
Sangraha) and streaming is used, so the whole multi-language dataset is never
downloaded. Each source keeps its own resume state, so an interrupted run
continues where it stopped.

    python data/collection/collect_public.py
    python data/collection/collect_public.py --only ai4bharat_sangraha
    python data/collection/collect_public.py --dry-run     # inspect fields first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import (Document, JsonlWriter, PUBLIC_HF, human,  # noqa: E402
                       lang_root, load_config, load_sources)

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
LANG = CFG["language"]


def open_stream(src: dict):
    """Open one configured source as a streaming Hugging Face dataset."""
    from datasets import load_dataset

    kw = {"streaming": True, "split": src.get("split", "train")}
    config_name = src.get("config_name") or src.get("builder_name")

    # Hugging Face builder configs for ai4bharat/sangraha are like: verified,
    # unverified, synthetic. Do not pass the project-local registry key as the
    # config name. Also, the verified subset uses the `guj` split, not `train`.
    known_configs = {"verified", "unverified", "synthetic"}
    if config_name in known_configs:
        kw["name"] = config_name

    for key in ("data_dir", "data_files"):
        if src.get(key):
            kw[key] = src[key]
    if src.get("trust_remote_code"):
        kw["trust_remote_code"] = True
    return load_dataset(src["hf_path"], **kw)


def collect(src: dict, dry_run: bool = False) -> dict:
    """Stream one public source into `data/raw/public/<name>.jsonl.gz`."""
    name = src["name"]
    field = src.get("text_field", "text")
    budget = int(float(src.get("max_chars", 0) or 0))     # 0 = uncapped
    min_chars = int(src.get("min_chars", 200))

    out = CFG["paths"]["raw_public"] / f"{name}.jsonl.gz"
    state_path = CFG["paths"]["state"] / f"public_{name}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    consumed, chars, docs = state.get("consumed", 0), state.get("chars", 0), state.get("docs", 0)

    print(f"\n=== {name}: {src['hf_path']} {src.get('data_dir') or src.get('name') or ''} ===")
    if budget and chars >= budget:
        print(f"  budget already reached ({human(chars)} chars) - skipping")
        return {"name": name, "docs": docs, "chars": chars}

    try:
        ds = open_stream(src)
    except Exception as exc:
        print(f"  ! cannot open: {type(exc).__name__}: {exc}")
        print("    gated dataset? run `huggingface-cli login` and accept the terms")
        return {"name": name, "docs": docs, "chars": chars, "error": str(exc)}

    if dry_run:
        for i, ex in enumerate(ds):
            print(f"  fields: {list(ex.keys())}")
            print(f"  {field}[:200]: {str(ex.get(field, ''))[:200]}")
            if i >= 1:
                break
        return {"name": name, "dry_run": True}

    if consumed:
        print(f"  resuming: skipping {consumed:,} examples already consumed")
        ds = ds.skip(consumed)

    writer = JsonlWriter(out)
    i = 0
    try:
        for i, ex in enumerate(ds, start=1):
            text = ex.get(field) or ""
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            writer.write(Document(
                text=text, language=LANG, source_name=name,
                source_type="public", collection_method="public_download",
                manual_or_public=PUBLIC_HF,
                source_url=ex.get("url") or src.get("url"),
                title=ex.get("title"), license=src.get("license", ""),
                extra={"hf_path": src["hf_path"], "subset": src.get("data_dir") or src.get("name")},
            ))
            docs += 1
            chars += len(text)
            if i % 20000 == 0:
                state_path.write_text(json.dumps({"consumed": consumed + i, "chars": chars, "docs": docs}))
                print(f"  {docs:,} docs | {human(chars)} chars", flush=True)
            if budget and chars >= budget:
                print(f"  budget reached: {human(chars)} chars")
                break
    except KeyboardInterrupt:
        print("  interrupted - state saved, rerun to resume")
    finally:
        writer.close()
        state_path.write_text(json.dumps({"consumed": consumed + i, "chars": chars, "docs": docs}))

    print(f"  done: {docs:,} docs, {human(chars)} chars -> {out}")
    return {"name": name, "docs": docs, "chars": chars}


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect public (downloaded) datasets.")
    ap.add_argument("--only", nargs="*", help="Restrict to these source names")
    ap.add_argument("--dry-run", action="store_true", help="Print field names and two samples")
    args = ap.parse_args()

    sources = load_sources(ROOT, "public_sources.yaml")
    if args.only:
        sources = [s for s in sources if s["name"] in args.only]

    results = [collect(s, args.dry_run) for s in sources]
    total = sum(r.get("chars", 0) for r in results)
    print(f"\n==== public total: {sum(r.get('docs', 0) for r in results):,} docs / "
          f"{human(total)} chars ====")
    print("These are PUBLIC documents; they do not count towards the 20% manual quota.")


if __name__ == "__main__":
    main()
