"""
Mine real smelly examples of the smells that the reused datasets do NOT cover.

CodeSmellData 1.0/2.0 and PySmell are Pylint-era datasets, so they contain none
of our ruff/magic-value smells: magic_number, inefficient_loop, inefficient_copy,
perf_try_in_loop (and duplicate_code). That left those smells validated on the
injected data alone. But they DO occur in real code -- we just have to find them.

This runs our own detectors (the pylint magic-value extension + ruff PERF rules,
via detect_many) over the same CodeSmellData 2.0 raw corpus we mine the clean pool
from, and keeps real methods that carry each uncovered smell. The result feeds the
real-world validation so those smells get a real smelly-vs-clean detection strength
too. Same operational definition as everywhere else; not circular (independent
structural measures are what's tested).

duplicate_code is an across-method clone, so a single mined method almost never
contains one -- it is expected to stay under-sampled and is reported as such.

Writes realworld_smelly.jsonl (schema matches reused.jsonl).

Run:  python smell_injection/build_realworld_smelly.py [--target 250]
"""

import argparse
import ast
import os
import subprocess
import sys


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if not os.path.exists(venv_py):
        venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "bin", "python"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

import json
import random
from collections import Counter

from build_injected import detect_many

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, os.pardir, "_datasets_eval", "CodeSmellExt",
                   "dataset", "extracted", "CodeSmellData_2.0.json")
OUT = os.path.join(HERE, "realworld_smelly.jsonl")
UNCOVERED = {"magic_number", "inefficient_loop", "inefficient_copy", "perf_try_in_loop"}
SCAN_CAP = 9000
MIN_LINES = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250, help="examples per uncovered smell")
    args = ap.parse_args()

    raw = json.load(open(RAW, encoding="utf-8"))
    order = list(range(len(raw)))
    random.Random(1).shuffle(order)                 # seed 1 (clean pool used seed 0)
    print(f"loaded {len(raw)} raw methods; mining {sorted(UNCOVERED)} (target {args.target} each) ...")

    kept, kept_ids, count, scanned, i = [], set(), Counter(), 0, 0
    while (any(count[s] < args.target for s in UNCOVERED)
           and scanned < SCAN_CAP and i < len(order)):
        batch = {}
        while len(batch) < 300 and i < len(order):
            idx = order[i]
            i += 1
            code = (raw[idx].get("code") or "").strip()
            if not code or len(code.splitlines()) < MIN_LINES:
                continue
            try:
                ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            batch[str(idx)] = code
        if not batch:
            break
        scanned += len(batch)
        found = detect_many(batch)
        for k, smells in found.items():
            for s in (smells & UNCOVERED):
                if count[s] < args.target and (k, s) not in kept_ids:
                    rec = raw[int(k)]
                    kept.append({
                        "id": f"cs2mined_{k}_{s}", "source": "codesmelldata2_mined",
                        "smell": s, "label": "yes", "code": batch[k],
                        "label_origin": "pylint" if s == "magic_number" else "ruff",
                        "meta": {"repo": rec.get("repo"), "fun": rec.get("fun_name")},
                    })
                    kept_ids.add((k, s))
                    count[s] += 1
        print(f"  scanned {scanned}: " + ", ".join(f"{s}={count[s]}" for s in sorted(UNCOVERED)),
              flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(kept)} mined real smelly methods to {OUT}")
    print("per smell:", dict(count))
    short = [s for s in UNCOVERED if count[s] < args.target]
    if short:
        print(f"under target (rarer in real single methods): {short}")


if __name__ == "__main__":
    main()
