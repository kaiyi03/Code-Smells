"""
Generate solutions to the smell-inducing prompts with a local code model (ARC GPU).

The control arm for the Claude run: same prompts, same output budget, same code
extraction, so a difference in the generated code is the model rather than the
setup. Mirrors arc_qwen/generate.py, but the tasks come from prompts.jsonl and
there is no canonical solution to compare against -- these prompts have neither
reference implementations nor tests, so only the reference-free measures apply.

Usage (inside a SLURM GPU job -- see gen_prompts.slurm):
    GEN_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct python generate_local.py --out gen_qwen.jsonl
"""

import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts.jsonl")

MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_NEW = int(os.environ.get("GEN_MAX_TOKENS", "1024"))    # same budget as the Claude run
BATCH = int(os.environ.get("GEN_BATCH", "8"))
SYSTEM = ("Return only the Python code, in a single ```python code block. "
          "Do not explain the code or add commentary outside the block.")


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=os.path.join(HERE, "generations_local.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(PROMPTS, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.eval()

    print(f"{len(rows)} prompts; generating (batch {BATCH}, max_new {MAX_NEW}, greedy) ...",
          flush=True)
    t0, n_tok = time.time(), 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            n_in = enc["input_ids"].shape[1]
            st = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            dt = time.time() - st
            gen = out[:, n_in:]
            for r, g, ids in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True), gen):
                n_out = int((ids != tok.pad_token_id).sum())
                n_tok += n_out
                fout.write(json.dumps({
                    **r, "model": MODEL,
                    "generated_code": extract_code(g), "raw_output": g,
                    "n_prompt_tokens": int(n_in), "n_output_tokens": n_out,
                    "batch_index": i // BATCH, "batch_size": BATCH,
                    "batch_seconds": round(dt, 2),
                }) + "\n")
                fout.flush()
            print(f"  {min(i + BATCH, len(rows))}/{len(rows)}  ({dt:.1f}s/batch)", flush=True)
            del enc, out, gen
            torch.cuda.empty_cache()

    total = time.time() - t0
    print(f"wrote {len(rows)} generations to {args.out} in {total:.0f}s "
          f"({n_tok / total:.0f} output tok/s)", flush=True)


if __name__ == "__main__":
    main()
