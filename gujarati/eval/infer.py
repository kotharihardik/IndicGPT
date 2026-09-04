"""Generate text from a trained checkpoint, one prompt at a time.

`eval/generate.py` scores 200 held-out prompts and writes metrics. This script
is for reading the output by hand.

    python3 eval/infer.py --prompt "ગુજરાત સરકારે"
    python3 eval/infer.py --prompt "..." --temperature 0.8 --samples 3
    python3 eval/infer.py --input eval/input.txt      # a file of prompts
    python3 eval/infer.py                             # interactive; Ctrl-D quits

Output is appended to `manual_samples.md` in the run's report directory (set by
`reports_dir` in `configs/eval.yaml`) unless `--no-save` is given. Read it from
the file rather than the terminal: terminals use one fixed-width cell per
character and do not shape complex scripts, so Gujarati matras and Devanagari
conjuncts appear misplaced on screen. The text itself is correct.

With no prompt the model continues from `<s>`, the document-start token, so it
writes an opening rather than continuing a fragment.

Decoding settings: `--greedy` is deterministic and repeats itself, temperature
0.5 is fluent but still repetitive, 1.0 reads best on these models, and 1.5
drifts. `--top-k` trims the long tail that produces nonsense at high
temperature.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LANGUAGE, add_common_args, eval_config, load_model,  # noqa: E402
                    load_tokenizer, pick_device, resolve_checkpoint)


def generate(model, sp, prompt: str, device: torch.device, args) -> list:
    """Continue `prompt` `args.samples` times and return the continuations."""
    ids = [sp.bos_id()] + (sp.encode(prompt) if prompt else [])
    ids = ids[-model.cfg.context_length:]
    idx = torch.tensor([ids] * args.samples, dtype=torch.long, device=device)

    started = time.time()
    out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, greedy=args.greedy,
                         top_k=args.top_k, eos_id=sp.eos_id())
    elapsed = time.time() - started

    texts = []
    for row in out.tolist():
        new = row[len(ids):]
        if sp.eos_id() in new:                      # stop at the document end
            new = new[:new.index(sp.eos_id())]
        texts.append(sp.decode(new))
    return texts, elapsed


def read_prompts(path: Path) -> list:
    """Read prompts from a text file, one per line.

    Blank lines and lines starting with `#` are skipped, so the file can carry
    notes. Everything else is taken literally, trailing punctuation included.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found. Put one prompt per line in it.")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [p for p in prompts if p and not p.startswith("#")]
    if not prompts:
        raise SystemExit(f"{path} has no prompts (blank lines and #comments are skipped)")
    return prompts


def describe_setting(args) -> str:
    """The decoding settings as one readable phrase."""
    if args.greedy:
        return "greedy"
    return (f"temperature {args.temperature}"
            + (f", top-k {args.top_k}" if args.top_k else "")
            + (f", seed {args.seed}" if args.seed is not None else ""))


def show(prompt: str, texts: list, elapsed: float, args) -> None:
    """Print the prompt and each continuation."""
    print(f"\n  [{describe_setting(args)}, {args.max_new_tokens} new tokens, "
          f"{args.max_new_tokens * len(texts) / max(elapsed, 1e-9):.0f} tok/s]")
    for n, text in enumerate(texts, 1):
        label = f"  sample {n}" if len(texts) > 1 else "  continuation"
        print(f"\n{label}:")
        print(f"    {prompt}\033[1m{text}\033[0m" if prompt else f"    {text}")
    print()


def quote(text: str) -> str:
    """Render text as a markdown blockquote, keeping its paragraph breaks."""
    body = text.strip() or "(empty)"
    return "\n".join("> " + line if line else ">" for line in body.split("\n"))


def save(path: Path, prompt: str, texts: list, args, meta: dict) -> None:
    """Append this prompt and its continuations to the session log.

    Appended rather than overwritten, so one session reads as a series of
    prompts. Markdown, because these are the qualitative examples the brief
    asks for and they have to be readable alongside the metrics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            f"# {LANGUAGE} — manual prompts\n\n"
            f"Generated with `eval/infer.py`. Each block is one prompt typed by "
            f"hand and what the model wrote after it. Appended to, so the file "
            f"is the whole session.\n", encoding="utf-8")

    lines = ["", f"## {time.strftime('%Y-%m-%d %H:%M')} — {describe_setting(args)}",
             "",
             f"`{meta['checkpoint']}` at step {meta['step']:,}, "
             f"{args.max_new_tokens} new tokens", "",
             "**Prompt**", "", quote(prompt) if prompt else "> _(none — the model "
             "started a document itself)_", ""]
    for n, text in enumerate(texts, 1):
        lines += [f"**Continuation {n}**" if len(texts) > 1 else "**Continuation**",
                  "", quote(text), ""]
    lines.append("---")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    """Generate from one prompt, or loop reading prompts from the terminal."""
    ap = argparse.ArgumentParser(
        description=f"Generate {LANGUAGE} text from a trained checkpoint.")
    add_common_args(ap)
    ap.add_argument("--prompt", default=None,
                    help="text to continue; omitted means start a new document")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true",
                    help="take the most likely token every step (deterministic)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1,
                    help="how many continuations to draw from the same prompt")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the sampling seed to make a run reproducible")
    ap.add_argument("--input", default=None,
                    help="a text file of prompts, one per line")
    ap.add_argument("--out", default=None,
                    help="file to append prompts and continuations to "
                         "(default: <reports_dir>/manual_samples.md)")
    ap.add_argument("--no-save", action="store_true",
                    help="print only, write nothing")
    args = ap.parse_args()

    cfg = eval_config(args)
    device = pick_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg)
    model, model_cfg, meta = load_model(checkpoint, device)
    sp = load_tokenizer()

    print(f"\n  {LANGUAGE}  |  {checkpoint.name} at step {meta['state']['step']:,}"
          f"  |  {model.num_parameters():,} parameters  |  {device}")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Output always goes to the report directory, never beside the prompts:
    # generated text is a report artifact and does not belong in eval/.
    source = Path(args.input) if args.input else None
    default_out = Path(cfg["reports_dir"]) / "manual_samples.md"
    out = None if args.no_save else Path(args.out or default_out)
    record = {"checkpoint": checkpoint.name, "step": meta["state"]["step"]}
    if out:
        print(f"  writing to {out}")

    def run(prompt: str) -> None:
        """Generate, print, and append to the session log."""
        texts, elapsed = generate(model, sp, prompt, device, args)
        show(prompt, texts, elapsed, args)
        if out:
            save(out, prompt, texts, args, record)

    if source is not None:
        prompts = read_prompts(source)
        print(f"  {len(prompts)} prompt(s) from {source}\n")
        for n, prompt in enumerate(prompts, 1):
            print(f"  ── {n}/{len(prompts)} ──")
            run(prompt)
        if out:
            print(f"  all {len(prompts)} written to {out}\n")
        return

    if args.prompt is not None:
        run(args.prompt)
        return

    print("  Enter a prompt and press return. A blank line or Ctrl-D quits.")
    print("  Type  /new  on its own to have the model start a document itself.\n")

    def ask(question: str) -> str:
        """Read one line, treating end-of-input as a request to stop."""
        try:
            return input(question).strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n")

    while True:
        prompt = ask("  > ")
        if not prompt:
            return
        if prompt == "/new":            # start from <s> with nothing to continue
            prompt = ""
        run(prompt)


if __name__ == "__main__":
    main()
