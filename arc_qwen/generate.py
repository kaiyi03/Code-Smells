"""
Generate code for the benchmark tasks with a code LLM (runs on an ARC GPU node).

Loads the model named by GEN_MODEL (default Qwen2.5-Coder-1.5B-Instruct), prompts
it to solve each MBPP / HumanEval task, and records the generated code plus
generation cost. The output (generations*.jsonl) is copied back to the laptop and
scored with the eval panel. Every model is run through this same script -- same
prompts, same greedy decoding, same batch size -- so the results are comparable.

Runs on a compute node (real memory + internet); the HF cache lives on the
shared /data area so the model is downloaded once and reused.

Usage (inside a SLURM GPU job -- see gen.slurm):
    python generate.py --limit 5 --out ~/generations_test.jsonl   # quick test
    python generate.py --out ~/generations.jsonl                  # full run
"""

import argparse
import json
import os
import re
import time

from datasets import load_dataset

# torch and transformers are imported inside main(): they are only needed to run a
# model, and importing them at module level would make this file unimportable off
# a GPU node. generate_claude_bench.py imports load_tasks() from here so that the
# API model gets byte-identical prompts -- that only works if the import is cheap.

MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_NEW = 512

# Verbosity conditions for the token experiment (Section 8), kept character-for-
# character identical to the ones in generate_claude_bench.py -- a difference in
# wording between models would confound the model with the instruction. Empty
# string means no system prompt, which is what the Section 7 runs used.
SYSTEMS = {
    "": "",
    "terse":   "Write the shortest solution that works. No docstring, no comments, "
               "no input validation.",
    "neutral": "Write a solution to the task.",
    "verbose": "Write a complete and defensive solution: validate the inputs, handle "
               "edge cases explicitly, and document the function with a docstring and "
               "comments.",
}
SYSTEM_MODE = os.environ.get("GEN_SYSTEM", "")
# Keep this the same for every model: it sets how much work shares a forward pass,
# so the tokens/second figure is only comparable across models at a fixed batch.
BATCH = int(os.environ.get("GEN_BATCH", "8"))


# The token experiment (RQ3) varies how much the model is asked to write while
# holding the task fixed. The directive is APPENDED to the same instruction every
# model already receives, and "neutral" appends nothing -- so the neutral condition
# is byte-identical to the plain benchmark run and does not need generating twice.
#
# Neither directive names a measure we score. "Shortest" and "explain it" are
# instructions about the answer, not about lines of code or comment density. The
# exception is unavoidable: asking for a docstring does raise comment density
# directly, so that measure is manipulated under `verbose` and cannot be read as
# an outcome there. Length, defect count and correctness can.
STYLES = {
    "neutral": "",
    "terse": "\n\nKeep the solution as short as you can while still being correct.",
    "verbose": ("\n\nWrite the solution out fully and explain it: give the function a "
                "docstring, comment any step that is not obvious, check the inputs, "
                "and handle edge cases explicitly."),
}


def load_tasks(limit=None, style="neutral"):
    """Return the benchmark tasks as {task_id, source, instruction, canonical}.

    `style` selects a verbosity directive from STYLES, appended to the instruction.
    """
    suffix = STYLES[style]
    tasks = []
    for ex in load_dataset("openai/openai_humaneval", split="test"):
        tasks.append({
            "task_id": ex["task_id"].replace("/", "_"),
            "source": "humaneval",
            "instruction": ("Complete the following Python function. Return only the "
                            "complete function in a single ```python code block.\n\n"
                            + ex["prompt"]),
            "canonical": ex["prompt"] + ex["canonical_solution"],
        })
    for ex in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        tests = "\n".join(ex["test_list"])
        tasks.append({
            "task_id": f"mbpp_{ex['task_id']}",
            "source": "mbpp",
            "instruction": ("Write a Python function for the task below. Return only the "
                            "code in a single ```python code block.\n\n"
                            f"Task: {ex['text']}\n\nIt must pass these tests:\n{tests}"),
            "canonical": ex["code"],
        })
    if limit:                                   # small, balanced slice for a quick test
        he = [t for t in tasks if t["source"] == "humaneval"][:limit]
        mb = [t for t in tasks if t["source"] == "mbpp"][:limit]
        return he + mb
    return tasks


def extract_code(text):
    """Pull the code out of the reply: prefer a fenced ```python block."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="tasks per source (test runs)")
    ap.add_argument("--out", default=os.path.expanduser("~/generations.jsonl"))
    args = ap.parse_args()

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"                    # decoder-only batching needs left pad
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.eval()

    tasks = load_tasks(args.limit)
    print(f"{len(tasks)} tasks; generating (batch {BATCH}, max_new {MAX_NEW}, greedy) ...",
          flush=True)

    n_done, n_tokens, t0 = 0, 0, time.time()
    with open(args.out, "w", encoding="utf-8") as fout:    # write incrementally
        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i + BATCH]
            sysmsg = SYSTEMS[SYSTEM_MODE]
            prompts = [tok.apply_chat_template(
                           ([{"role": "system", "content": sysmsg}] if sysmsg else [])
                           + [{"role": "user", "content": t["instruction"]}],
                           tokenize=False, add_generation_prompt=True)
                       for t in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            n_in = enc["input_ids"].shape[1]
            st = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            dt = time.time() - st
            gen = out[:, n_in:]                              # strip the prompt tokens
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for t, g, ids in zip(batch, texts, gen):
                n_out = int((ids != tok.pad_token_id).sum())
                n_tokens += n_out
                fout.write(json.dumps({
                    "task_id": t["task_id"], "source": t["source"], "model": MODEL,
                    "system_mode": SYSTEM_MODE,
                    "batch_index": i // BATCH, "batch_size": BATCH,
                    "generated_code": extract_code(g),
                    "raw_output": g,
                    "canonical_code": t["canonical"],
                    "n_prompt_tokens": int(n_in),
                    "n_output_tokens": n_out,
                    "batch_seconds": round(dt, 2),
                }) + "\n")
                fout.flush()
            n_done += len(batch)
            print(f"  {n_done}/{len(tasks)}  ({dt:.1f}s/batch)", flush=True)
            del enc, out, gen                    # the KV cache of a full 512-token
            torch.cuda.empty_cache()             # generation is what fills the card

    total = time.time() - t0
    print(f"wrote {n_done} generations to {args.out} in {total:.0f}s "
          f"({n_tokens / total:.0f} output tok/s overall)", flush=True)


if __name__ == "__main__":
    main()
