"""
Regenerate the 664 benchmark solutions a second time, to measure how much the
sampled model's structural output moves between runs.

Why this exists. The two open models decode greedily, so re-running them returns
byte-identical code and their numbers carry no run-to-run uncertainty. Sonnet 5
is run with no decoding temperature set, so it answers at the API default,
which samples: the figures in
Section 7 are one draw per task, not the modal one. That is stated in Section 9.4,
but stating it is not the same as bounding it -- without a second run there is no
way to say whether Claude's 1.90 smells per 100 SLOC would come back as 1.90 or
as 2.4, and the "capability does not buy structural quality" result leans on the
three models' densities being close.

This script produces that second draw. It must match the Section 7 run exactly or
the difference between them is not run-to-run variance but a change of setup, so:

  * NO system prompt. generations_claude.jsonl was produced without one --
    generate_claude_bench.py always sends one, which is why this cannot simply be
    that script re-run.
  * same prompts, via load_tasks() from generate.py.
  * same model, same 4096-token budget, thinking disabled.

Everything that could differ is held; only the sampling differs, which is the
thing being measured.

The Batch API would be half price, but its queue was not moving when this was
needed, so this goes through the normal endpoint with a thread pool. The whole run
is roughly 122k input and 75k output tokens -- about a dollar at standard rates.

Run:  python arc_qwen/replicate_claude_bench.py --limit 3   # trial
      python arc_qwen/replicate_claude_bench.py             # full 664
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate import extract_code, load_tasks        # noqa: E402  -- identical prompts

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "4096"))
WORKERS = int(os.environ.get("GEN_WORKERS", "12"))

_print_lock = threading.Lock()


def one(client, task, done, total):
    """Generate a single solution, retrying through transient API failures."""
    for attempt in range(6):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": task["instruction"]}],
            )
            break
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            # 429 and 529 are worth waiting out; a 400 never becomes a 200.
            status = getattr(e, "status_code", None)
            if status not in (429, 500, 502, 503, 529) or attempt == 5:
                with _print_lock:
                    print(f"  [{task['task_id']}] gave up: {e}", file=sys.stderr)
                return None
            time.sleep((2 ** attempt) + random.random())
    else:
        return None

    text = "".join(b.text for b in msg.content if b.type == "text")
    with _print_lock:
        done[0] += 1
        if done[0] % 50 == 0 or done[0] == total:
            print(f"  {done[0]}/{total}", flush=True)
    return {
        "task_id": task["task_id"], "source": task["source"], "model": MODEL,
        "run": "replication",
        "generated_code": extract_code(text),
        "raw_output": text,
        "canonical_code": task["canonical"],
        "n_prompt_tokens": msg.usage.input_tokens,
        "n_output_tokens": msg.usage.output_tokens,
        "stop_reason": msg.stop_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="N tasks per source (trial run)")
    ap.add_argument("--out", default=os.path.join(HERE, "generations_claude_rep.jsonl"))
    args = ap.parse_args()

    client = anthropic.Anthropic(max_retries=0)      # retries are handled above
    tasks = load_tasks(args.limit)
    print(f"{len(tasks)} tasks, {WORKERS} workers, model {MODEL}, no system prompt",
          flush=True)

    done = [0]
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(lambda t: one(client, t, done, len(tasks)), tasks))

    rows = [r for r in rows if r]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tok_in = sum(r["n_prompt_tokens"] for r in rows)
    tok_out = sum(r["n_output_tokens"] for r in rows)
    n_trunc = sum(r["stop_reason"] == "max_tokens" for r in rows)
    print(f"\nwrote {len(rows)}/{len(tasks)} generations to {os.path.basename(args.out)} "
          f"in {time.time() - started:.0f}s")
    print(f"tokens: {tok_in:,} in, {tok_out:,} out")
    print(f"cost at standard rate: ~${tok_in / 1e6 * 2 + tok_out / 1e6 * 10:.2f}")
    if n_trunc:
        print(f"!! {n_trunc} hit the {MAX_TOKENS}-token cap", file=sys.stderr)
    else:
        print(f"cap check: 0/{len(rows)} truncated")


if __name__ == "__main__":
    main()
