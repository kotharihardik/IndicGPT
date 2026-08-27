"""Generation quality: continue held-out prefixes and score the continuations.

The protocol is the same for every decoding setting, so the settings are the
only thing that varies:

  1. take a held-out document, encode it, and cut it into a prompt of
     `prefix_tokens` and a reference continuation of `continuation_tokens`;
  2. let the model continue the prompt for exactly as many tokens as the
     reference has;
  3. decode both sides back to text and score them.

The reference is the *decoded reference token ids*, not the raw document text.
That is the fair target: it is what the model would have produced had it
predicted perfectly, so the score measures the model rather than the tokenizer's
round trip.

Four settings are reported, as the brief asks: greedy, and sampling at
temperature 0.5, 1.0 and 1.5. Greedy is deterministic; the sampled ones use a
fixed seed so the numbers reproduce.

    python3 eval/generate.py
    python3 eval/generate.py --prompts 32 --continuation-tokens 64   # quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import (LANGUAGE, PIPELINE, add_common_args, eval_config,  # noqa: E402
                    load_model, load_tokenizer, pick_device, report_dir,
                    resolve_checkpoint, save_json)
from corpus_io import read_docs, split_file  # noqa: E402
from dataset import load_meta  # noqa: E402
import text_metrics  # noqa: E402


def collect_prompts(cfg: Dict, split: str, sp, prefix_tokens: int,
                    continuation_tokens: int, wanted: int) -> List[Dict]:
    """Cut held-out documents into (prompt, reference) pairs.

    Documents are taken at a stride through the split rather than from the
    front, because the split is written manual sources first: reading the first
    N documents would evaluate on one register only.
    """
    path = split_file(PIPELINE["paths"]["splits"], split)
    needed = prefix_tokens + continuation_tokens
    total_docs = load_meta(Path(cfg["data_dir"]))["splits"][split]["docs"]
    stride = max(1, total_docs // (wanted * 4))

    prompts: List[Dict] = []
    for index, doc in enumerate(read_docs(path)):
        if index % stride:
            continue
        ids = sp.encode(doc.text)
        if len(ids) < needed:
            continue
        prompts.append({
            "source": doc.source_name,
            "prompt_ids": [sp.bos_id()] + ids[:prefix_tokens],
            "reference_ids": ids[prefix_tokens:needed],
        })
        if len(prompts) >= wanted:
            break
    return prompts


@torch.no_grad()
def run_setting(model, prompts: Sequence[Dict], sp, setting: Dict,
                device: torch.device, batch_size: int, seed: int) -> Dict:
    """Generate a continuation for every prompt under one decoding setting."""
    torch.manual_seed(seed)
    started = time.time()
    outputs: List[str] = []
    stopped_early = 0

    for first in range(0, len(prompts), batch_size):
        chunk = prompts[first:first + batch_size]
        idx = torch.tensor([p["prompt_ids"] for p in chunk],
                           dtype=torch.long, device=device)
        n_new = max(len(p["reference_ids"]) for p in chunk)
        out = model.generate(idx, max_new_tokens=n_new,
                             temperature=setting["temperature"],
                             greedy=setting["greedy"],
                             eos_id=sp.eos_id())
        for row, prompt in zip(out.tolist(), chunk):
            new_ids = row[len(prompt["prompt_ids"]):]
            new_ids = new_ids[:len(prompt["reference_ids"])]
            if sp.eos_id() in new_ids:
                stopped_early += 1
                new_ids = new_ids[:new_ids.index(sp.eos_id())]
            outputs.append(sp.decode(new_ids))

    references = [sp.decode(p["reference_ids"]) for p in prompts]
    metrics = text_metrics.all_metrics(outputs, references)
    metrics["setting"] = setting["name"]
    metrics["stopped_at_eos"] = stopped_early
    metrics["seconds"] = round(time.time() - started, 1)
    return {"metrics": metrics, "outputs": outputs, "references": references}


def print_table(results: Dict[str, Dict]) -> None:
    """One row per decoding setting."""
    print(f"\n  {LANGUAGE}: generation against held-out continuations")
    print(f"  {'setting':<12}{'BLEU-4':>9}{'chrF++':>9}{'ROUGE-L':>9}"
          f"{'dist-1':>9}{'dist-2':>9}{'rep-4':>8}")
    print("  " + "-" * 65)
    for name, blob in results.items():
        m = blob["metrics"]
        d = m["diversity"]
        print(f"  {name:<12}{m['bleu']['bleu']:>9.2f}{m['chrf']['chrf']:>9.2f}"
              f"{m['rouge_l']['rouge_l']:>9.2f}{d['distinct_1']:>9.3f}"
              f"{d['distinct_2']:>9.3f}{d['repetition_4']:>8.3f}")
    ref = next(iter(results.values()))["metrics"]["reference_diversity"]
    print(f"  {'(reference)':<12}{'-':>9}{'-':>9}{'-':>9}"
          f"{ref['distinct_1']:>9.3f}{ref['distinct_2']:>9.3f}"
          f"{ref['repetition_4']:>8.3f}")
    print()


def quote(text: str) -> str:
    """Render generated text as a markdown blockquote.

    Continuations often contain paragraph breaks, which would otherwise end the
    block and scramble the layout of the samples file.
    """
    body = text.strip() or "(empty)"
    return "\n".join("> " + line if line else ">" for line in body.split("\n"))


def samples_markdown(prompts: Sequence[Dict], results: Dict[str, Dict], sp,
                     count: int) -> str:
    """Readable side-by-side samples for the report."""
    first = next(iter(results))
    lines = [f"# {LANGUAGE} — generated continuations", "",
             "Each block shows one held-out prompt, the true continuation, and "
             "what the model produced under each decoding setting. Prompts come "
             "from the test split, which the model never saw.", ""]
    for i in range(min(count, len(prompts))):
        lines += [f"## Sample {i + 1} — source `{prompts[i]['source']}`", "",
                  "**Prompt**", "", quote(sp.decode(prompts[i]["prompt_ids"][1:])), "",
                  "**Reference continuation**", "",
                  quote(results[first]["references"][i]), ""]
        for name, blob in results.items():
            lines += [f"**{name}**", "", quote(blob["outputs"][i]), ""]
        lines += ["---", ""]
    return "\n".join(lines)


def main() -> None:
    """Generate under every decoding setting and write metrics and samples."""
    ap = argparse.ArgumentParser(description="Generation quality evaluation.")
    add_common_args(ap)
    ap.add_argument("--split", default=None)
    ap.add_argument("--prompts", type=int, default=None)
    ap.add_argument("--prefix-tokens", type=int, default=None)
    ap.add_argument("--continuation-tokens", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="generation")
    args = ap.parse_args()

    cfg = eval_config(args)
    gen = cfg.get("generation", {})
    split = args.split or gen.get("split", "test")
    n_prompts = args.prompts or int(gen.get("prompts", 200))
    prefix_tokens = args.prefix_tokens or int(gen.get("prefix_tokens", 64))
    cont_tokens = args.continuation_tokens or int(gen.get("continuation_tokens", 96))
    seed = int(gen.get("seed", 1337))

    device = pick_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg)
    model, model_cfg, meta = load_model(checkpoint, device)
    sp = load_tokenizer()

    print(f"\n  checkpoint {checkpoint.name} at step {meta['state']['step']:,}  "
          f"device {device}")
    prompts = collect_prompts(cfg, split, sp, prefix_tokens, cont_tokens, n_prompts)
    print(f"  {len(prompts)} prompts from the {split} split, "
          f"{prefix_tokens} prompt tokens -> {cont_tokens} generated tokens")

    settings = [{"name": "greedy", "greedy": True, "temperature": 1.0}]
    settings += [{"name": f"temp {t}", "greedy": False, "temperature": t}
                 for t in gen.get("temperatures", [0.5, 1.0, 1.5])]

    results = {}
    for setting in settings:
        results[setting["name"]] = run_setting(model, prompts, sp, setting,
                                               device, args.batch_size, seed)
        print(f"    {setting['name']:<10} done in "
              f"{results[setting['name']]['metrics']['seconds']}s")
    print_table(results)

    save_json(cfg, args.out, {
        "language": LANGUAGE,
        "checkpoint": str(checkpoint),
        "step": meta["state"]["step"],
        "split": split,
        "prompts": len(prompts),
        "prefix_tokens": prefix_tokens,
        "continuation_tokens": cont_tokens,
        "seed": seed,
        "settings": {name: blob["metrics"] for name, blob in results.items()},
        "sources": sorted({p["source"] for p in prompts}),
    })

    md = samples_markdown(prompts, results, sp, int(gen.get("samples_shown", 8)))
    path = Path(cfg["reports_dir"]) / "samples.md"
    path.write_text(md, encoding="utf-8")
    print(f"  wrote {path}")

    raw = report_dir(cfg, "json") / f"{args.out}_outputs.json"
    raw.write_text(json.dumps(
        {name: {"outputs": blob["outputs"], "references": blob["references"]}
         for name, blob in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"  wrote {raw}")


if __name__ == "__main__":
    main()
