"""
Generate solutions to the smell-inducing prompts with Claude, via the Batch API.

The 476 prompts are independent and nothing here is latency-sensitive, which is
what the Batch API is for: one submission, one poll, and half the per-token cost.

Two settings exist to keep this comparable with the open models we run alongside
it, not because they are the best settings for Claude:

  * thinking is DISABLED. Sonnet 5 thinks by default; the open models cannot, so
    leaving it on would compare a reasoning model against two non-reasoning ones
    and confound the model with the mode.
  * max_tokens matches the budget given to every other model in the study.

Needs ANTHROPIC_API_KEY in the environment (or an `ant auth login` profile).

Run:  python prompt_study/generate_claude.py            # submit and wait
      python prompt_study/generate_claude.py --limit 5  # small trial first
      python prompt_study/generate_claude.py --batch msgbatch_...   # resume a poll
"""

import argparse
import json
import os
import sys
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts.jsonl")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "1024"))   # same budget for every model
SYSTEM = ("Return only the Python code, in a single ```python code block. "
          "Do not explain the code or add commentary outside the block.")


def load_prompts(limit=None):
    rows = [json.loads(line) for line in open(PROMPTS, encoding="utf-8")]
    return rows[:limit] if limit else rows


def extract_code(text):
    """The code out of the reply -- prefer a fenced block, fall back to the text."""
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def submit(client, rows):
    batch = client.messages.batches.create(requests=[
        Request(
            custom_id=r["task_id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                thinking={"type": "disabled"},      # see the note at the top
                messages=[{"role": "user", "content": r["prompt"]}],
            ),
        )
        for r in rows
    ])
    print(f"submitted batch {batch.id} ({len(rows)} prompts, model {MODEL})")
    return batch.id


def wait(client, batch_id):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  {batch.processing_status}: {counts.succeeded} done, "
              f"{counts.processing} processing, {counts.errored} errored", flush=True)
        if batch.processing_status == "ended":
            return batch
        time.sleep(30)


def collect(client, batch_id, rows, out_path):
    by_id = {r["task_id"]: r for r in rows}
    n_ok = n_bad = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            row = by_id.get(result.custom_id)
            if row is None:                                  # not from this submission
                continue
            if result.result.type != "succeeded":
                n_bad += 1
                print(f"  [{result.custom_id}] {result.result.type}")
                continue
            msg = result.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            n_ok += 1
            f.write(json.dumps({
                **row,
                "model": MODEL,
                "generated_code": extract_code(text),
                "raw_output": text,
                "n_output_tokens": msg.usage.output_tokens,
                "n_prompt_tokens": msg.usage.input_tokens,
                "stop_reason": msg.stop_reason,
            }) + "\n")
    print(f"\nwrote {n_ok} generations to {os.path.basename(out_path)}"
          + (f" ({n_bad} failed)" if n_bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N prompts (trial run)")
    ap.add_argument("--batch", help="poll an existing batch instead of submitting a new one")
    ap.add_argument("--out", default=os.path.join(HERE, "generations_claude.jsonl"))
    args = ap.parse_args()

    client = anthropic.Anthropic()          # key from env, or an `ant auth login` profile
    rows = load_prompts(args.limit)

    batch_id = args.batch or submit(client, rows)
    batch = wait(client, batch_id)
    if batch.request_counts.errored:
        print(f"note: {batch.request_counts.errored} requests errored", file=sys.stderr)
    collect(client, batch_id, rows, args.out)


if __name__ == "__main__":
    main()
