"""
Generate solutions to the 664 benchmark tasks with Claude, via the Batch API.

The open models run this workload on an ARC GPU node (generate.py). Claude is an
API model, so it needs its own runner -- but the *task* must be identical or the
comparison is meaningless. This script therefore imports load_tasks() from
generate.py rather than re-deriving the prompts: same datasets, same instruction
text, same task ids. Only the execution path differs.

Two settings keep it comparable with the open models rather than optimal for
Claude:

  * thinking is DISABLED. Sonnet 5 thinks by default; the open models cannot, so
    leaving it on would compare a reasoning model against two non-reasoning ones
    and confound the model with the mode.
  * greedy-equivalent decoding. Sonnet 5 rejects temperature/top_p entirely, so
    there is nothing to set -- but nothing is sampled at a different temperature
    either, which is the property we actually need.

The token budget deliberately does NOT match. The open models ran at 512, which
binds on 0.3% (Qwen) and 1.1% (DeepSeek) of tasks -- close enough to non-binding
for them, but Claude writes several times more per task, so an equal cap would
truncate it and manufacture the very verbosity difference we are trying to
measure. The cap here is 4096, and the run asserts afterwards that it bound on
nobody. A budget that binds on one model and not another is not a control.

Needs ANTHROPIC_API_KEY in the environment.

Run:  python arc_qwen/generate_claude_bench.py --limit 3   # trial, ~6 tasks
      python arc_qwen/generate_claude_bench.py             # full 664
      python arc_qwen/generate_claude_bench.py --batch msgbatch_...   # resume
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
sys.path.insert(0, HERE)

from generate import extract_code, load_tasks        # noqa: E402  -- identical prompts

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "4096"))   # see the note above

# Verbosity conditions for the token experiment (Section 8). These vary one thing:
# how much code the task is asked for. They deliberately avoid saying anything
# about structure or quality -- an instruction like "write clean code" would
# confound the amount written with an explicit quality request, and the question
# is precisely whether writing more costs quality on its own.
#
# All three are present as system prompts, including the neutral one. Running the
# neutral arm with no system prompt at all would confound the verbosity level with
# whether a system prompt exists.
SYSTEMS = {
    "terse":   "Write the shortest solution that works. No docstring, no comments, "
               "no input validation.",
    "neutral": "Write a solution to the task.",
    "verbose": "Write a complete and defensive solution: validate the inputs, handle "
               "edge cases explicitly, and document the function with a docstring and "
               "comments.",
}
SYSTEM_MODE = os.environ.get("GEN_SYSTEM", "neutral")


def submit(client, tasks, system):
    batch = client.messages.batches.create(requests=[
        Request(
            custom_id=t["task_id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                thinking={"type": "disabled"},        # comparability, not preference
                messages=[{"role": "user", "content": t["instruction"]}],
            ),
        )
        for t in tasks
    ])
    print(f"submitted batch {batch.id} ({len(tasks)} tasks, model {MODEL}, "
          f"system '{SYSTEM_MODE}', max_tokens {MAX_TOKENS})", flush=True)
    return batch.id


def wait(client, batch_id):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  {batch.processing_status}: {c.succeeded} done, {c.processing} processing, "
              f"{c.errored} errored", flush=True)
        if batch.processing_status == "ended":
            return batch
        time.sleep(30)


def collect(client, batch_id, tasks, out_path):
    by_id = {t["task_id"]: t for t in tasks}
    n_ok = n_bad = n_trunc = 0
    tok_in = tok_out = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            task = by_id.get(result.custom_id)
            if task is None:
                continue
            if result.result.type != "succeeded":
                n_bad += 1
                print(f"  [{result.custom_id}] {result.result.type}", file=sys.stderr)
                continue
            msg = result.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            n_ok += 1
            tok_in += msg.usage.input_tokens
            tok_out += msg.usage.output_tokens
            if msg.stop_reason == "max_tokens":
                n_trunc += 1
            f.write(json.dumps({
                "task_id": task["task_id"], "source": task["source"], "model": MODEL,
                "system_mode": SYSTEM_MODE,
                "generated_code": extract_code(text),
                "raw_output": text,
                "canonical_code": task["canonical"],      # the similarity panel needs this
                "n_prompt_tokens": msg.usage.input_tokens,
                "n_output_tokens": msg.usage.output_tokens,
                "stop_reason": msg.stop_reason,
            }) + "\n")

    print(f"\nwrote {n_ok} generations to {os.path.basename(out_path)}"
          + (f" ({n_bad} failed)" if n_bad else ""))
    print(f"tokens: {tok_in:,} in, {tok_out:,} out "
          f"(mean {tok_out / n_ok:.0f} out/task)" if n_ok else "")
    # Batch pricing, Sonnet 5 introductory rate through 2026-08-31: $1/$5 per MTok.
    print(f"cost at batch intro rate: ~${tok_in / 1e6 * 1.0 + tok_out / 1e6 * 5.0:.2f}")
    if n_trunc:
        print(f"\n!! {n_trunc}/{n_ok} generations hit the {MAX_TOKENS}-token cap. "
              f"The budget BINDS on Claude and does not bind on the open models, so the "
              f"comparison is confounded. Re-run with a higher GEN_MAX_TOKENS.",
              file=sys.stderr)
    else:
        print(f"cap check: 0/{n_ok} truncated -- the {MAX_TOKENS}-token budget binds on "
              f"nobody, so it cannot explain any difference between models.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="N tasks per source (trial run)")
    ap.add_argument("--batch", help="poll an existing batch instead of submitting")
    ap.add_argument("--out", default=None,
                    help="defaults to generations_claude<suffix>.jsonl for the condition")
    args = ap.parse_args()
    if args.out is None:
        # Always suffixed by condition, including neutral. generations_claude.jsonl
        # is Section 7's run, which carried no system prompt at all; the three
        # conditions here each carry one, so they form their own comparable set and
        # must not overwrite it.
        args.out = os.path.join(HERE, f"generations_claude_{SYSTEM_MODE}.jsonl")

    client = anthropic.Anthropic()
    tasks = load_tasks(args.limit)
    print(f"{len(tasks)} tasks "
          f"({sum(t['source'] == 'humaneval' for t in tasks)} humaneval, "
          f"{sum(t['source'] == 'mbpp' for t in tasks)} mbpp)", flush=True)

    batch_id = args.batch or submit(client, tasks, SYSTEMS[SYSTEM_MODE])
    batch = wait(client, batch_id)
    if batch.request_counts.errored:
        print(f"note: {batch.request_counts.errored} requests errored", file=sys.stderr)
    collect(client, batch_id, tasks, args.out)


if __name__ == "__main__":
    main()
