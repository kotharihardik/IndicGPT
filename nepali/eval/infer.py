"""Generate text from a trained checkpoint, for reading by hand.

    python3 eval/infer.py --input eval/input.txt     # every prompt in the file
    python3 eval/infer.py --prompt "नेपाल सरकारले"
    python3 eval/infer.py                            # interactive; Ctrl-D quits

With `--input`, every prompt is answered once per temperature (0.5, 1.0, 1.5)
into `model_output.md`, overwritten each run; the other modes append to
`manual_samples.md`. With no prompt the model continues from `<s>`.

Read the output from the file, not the terminal: terminals do not shape
complex scripts, so matras and conjuncts appear misplaced.
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


def generate(model, sp, prompt: str, device: torch.device, max_new_tokens: int,
             temperature: float = 1.0, greedy: bool = False,
             top_k: int = None, samples: int = 1) -> tuple:
    """Continue `prompt` `samples` times and return (continuations, seconds)."""
    ids = [sp.bos_id()] + (sp.encode(prompt) if prompt else [])
    ids = ids[-model.cfg.context_length:]
    idx = torch.tensor([ids] * samples, dtype=torch.long, device=device)

    started = time.time()
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, greedy=greedy,
                         top_k=top_k, eos_id=sp.eos_id())
    elapsed = time.time() - started

    texts = []
    for row in out.tolist():
        new = row[len(ids):]
        if sp.eos_id() in new:                      # stop at the document end
            new = new[:new.index(sp.eos_id())]
        texts.append(sp.decode(new))
    return texts, elapsed


def read_prompts(path: Path) -> list:
    """Read prompts, one per line; blank lines and `#` comments are skipped."""
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
    """Append this prompt and its continuations to the session log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            f"# {LANGUAGE} - manual prompts\n\n"
            f"Generated with `eval/infer.py`. Each block is one prompt typed by "
            f"hand and what the model wrote after it. Appended to, so the file "
            f"is the whole session.\n", encoding="utf-8")

    lines = ["", f"## {time.strftime('%Y-%m-%d %H:%M')} - {describe_setting(args)}",
             "",
             f"`{meta['checkpoint']}` at step {meta['step']:,}, "
             f"{args.max_new_tokens} new tokens", "",
             "**Prompt**", "", quote(prompt) if prompt else "> _(none - the model "
             "started a document itself)_", ""]
    for n, text in enumerate(texts, 1):
        lines += [f"**Continuation {n}**" if len(texts) > 1 else "**Continuation**",
                  "", quote(text), ""]
    lines.append("---")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_answers(path: Path, rows: list, meta: dict, max_new_tokens: int) -> None:
    """Write the prompts and their continuations, one block per prompt.

    Overwritten each run, so it always matches the current input.txt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {LANGUAGE} - model output", "",
             f"`{meta['checkpoint']}` at step {meta['step']:,}, "
             f"{max_new_tokens} new tokens per continuation.", ""]
    for n, (prompt, answers) in enumerate(rows, 1):
        lines += ["---", "", f"### Prompt {n}", "", quote(prompt), ""]
        for temperature, text in answers:
            lines += [f"**temperature {temperature}**", "", quote(text), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


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
                    help="a text file of prompts, one per line; each is answered "
                         "once per --temperatures into a single output file")
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=[0.5, 1.0, 1.5],
                    help="temperatures to answer each --input prompt at "
                         "(default: the three the evaluation reports)")
    ap.add_argument("--out", default=None,
                    help="where to write (default: <reports_dir>/model_output.md "
                         "with --input, else <reports_dir>/manual_samples.md)")
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

    # Output goes to the report directory, not beside the prompts.
    source = Path(args.input) if args.input else None
    default_out = (Path(cfg["reports_dir"]) /
                   ("model_output.md" if source else "manual_samples.md"))
    out = None if args.no_save else Path(args.out or default_out)
    record = {"checkpoint": checkpoint.name, "step": meta["state"]["step"]}
    if out:
        print(f"  writing to {out}")

    def run(prompt: str) -> None:
        """Generate, print, and append to the session log."""
        texts, elapsed = generate(model, sp, prompt, device, args.max_new_tokens,
                                  temperature=args.temperature, greedy=args.greedy,
                                  top_k=args.top_k, samples=args.samples)
        show(prompt, texts, elapsed, args)
        if out:
            save(out, prompt, texts, args, record)

    if source is not None:
        prompts = read_prompts(source)
        temperatures = args.temperatures
        print(f"  {len(prompts)} prompt(s) from {source}, "
              f"temperatures {', '.join(str(t) for t in temperatures)}\n")
        rows = []
        for n, prompt in enumerate(prompts, 1):
            answers = []
            for temperature in temperatures:
                # Re-seed per continuation so the file reproduces on a re-run.
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                texts, _ = generate(model, sp, prompt, device,
                                    args.max_new_tokens, temperature=temperature,
                                    top_k=args.top_k)
                answers.append((temperature, texts[0]))
            rows.append((prompt, answers))
            print(f"  {n}/{len(prompts)} done")
        if out:
            write_answers(out, rows, record, args.max_new_tokens)
            print(f"\n  wrote {out}\n")
        else:
            for prompt, answers in rows:
                print(f"\n  {prompt}")
                for temperature, text in answers:
                    print(f"    [{temperature}] {text}")
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
