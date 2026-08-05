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

# Two system prompts, differing in one instruction. The first run used `minimal`
# and produced solutions in which nearly half the functions had `pass` bodies --
# structurally clean because the work was never done. `full` asks for a complete
# implementation, so the structural measures score written code rather than a
# skeleton. Both are kept: the pair is a controlled comparison of one instruction,
# and every generation records which it used.
SYSTEMS = {
    "minimal": ("Return only the Python code, in a single ```python code block. "
                "Do not explain the code or add commentary outside the block."),
    "full": ("Return only the Python code, in a single ```python code block. "
             "Do not explain the code or add commentary outside the block. "
             "Implement every function fully -- do not leave `pass`, `...`, or "
             "placeholder bodies, and do not stub out work as a TODO."),
    # A third, stronger condition. `full` asks for complete functions and barely
    # moved the stub rate; this adds the two things it left implicit -- an explicit
    # licence to write at length, and a stated preference for a long complete
    # answer over a short sketch. It also names every stub form we detect, since
    # the models substitute one for another when only `pass` is forbidden.
    "emphatic": ("Return only the Python code, in a single ```python code block. "
                 "Do not explain the code or add commentary outside the block. "
                 "Write a complete, working implementation. Every function you "
                 "define must have a real body that does the work: never `pass`, "
                 "never `...`, never a docstring alone, never `raise "
                 "NotImplementedError`, never a TODO. There is no length limit -- "
                 "write as much code as the task genuinely needs, and always prefer "
                 "a longer complete implementation over a shorter sketch."),
}
SYSTEM_MODE = os.environ.get("GEN_SYSTEM", "full")
SYSTEM = SYSTEMS[SYSTEM_MODE]


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

    print(f"system prompt: {SYSTEM_MODE}", flush=True)
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
                    **r, "model": MODEL, "system_mode": SYSTEM_MODE,
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
