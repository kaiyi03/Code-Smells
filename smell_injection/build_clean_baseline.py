"""
Build a real-clean baseline for the real-world validation layer.

The reused smell datasets (reused.jsonl) are all POSITIVE -- real methods that
Pylint flagged as smelly. To ask whether the structural measures separate real
smelly code from real CLEAN code, we need real clean code too. CodeSmellData 2.0
ships the full raw method corpus (CodeSmellData_2.0.json, ~254k mined methods,
UNLABELLED) from the SAME GitHub repos as the smelly examples. We sample it, run
the same detector we use everywhere, and keep the methods that carry NONE of our
tracked smells -> a real-clean pool from the same distribution as the smelly side.

This is not circular: the detectors (Pylint/Ruff) define clean-vs-smelly (the
operational definition), and the INDEPENDENT structural measures are what get
tested against that split -- exactly as on the injected benchmark.

Writes realworld_clean.jsonl (schema matches reused.jsonl).

Run:  python smell_injection/build_clean_baseline.py [--target 1000]
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

from build_injected import detect_many          # same detector as the rest of the project

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, os.pardir, "_datasets_eval", "CodeSmellExt",
                   "dataset", "extracted", "CodeSmellData_2.0.json")
OUT = os.path.join(HERE, "realworld_clean.jsonl")
SCAN_CAP = 4000      # bound on how many raw methods to detect while hunting for clean ones
MIN_LINES = 2        # skip degenerate one-liners so the baseline is representative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000, help="clean methods to collect")
    args = ap.parse_args()

    raw = json.load(open(RAW, encoding="utf-8"))
    order = list(range(len(raw)))
    random.Random(0).shuffle(order)                 # deterministic sample
    print(f"loaded {len(raw)} raw CS-2.0 methods; hunting for {args.target} clean ones ...")

    kept, scanned = [], 0
    i = 0
    while len(kept) < args.target and scanned < SCAN_CAP and i < len(order):
        # assemble a batch of parseable, non-trivial candidates
        batch = {}
        while len(batch) < 250 and i < len(order):
            rec = raw[order[i]]
            i += 1
            code = (rec.get("code") or "").strip()
            if not code or len(code.splitlines()) < MIN_LINES:
                continue
            try:
                ast.parse(code)                     # parseable only (else 'no smell' is meaningless)
            except (SyntaxError, ValueError):
                continue
            batch[str(order[i - 1])] = code
        if not batch:
            break
        scanned += len(batch)
        found = detect_many(batch)                  # same pylint+ruff detector
        for k, code in batch.items():
            if not found.get(k):                    # no tracked smell -> clean
                rec = raw[int(k)]
                kept.append({
                    "id": f"cs2clean_{k}", "source": "codesmelldata2_clean",
                    "smell": "clean", "label": "no", "code": code,
                    "label_origin": "pylint-clean",
                    "meta": {"repo": rec.get("repo"), "fun": rec.get("fun_name"),
                             "path": rec.get("path")},
                })
                if len(kept) >= args.target:
                    break
        print(f"  scanned {scanned}: {len(kept)} clean kept", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(kept)} clean methods to {OUT} "
          f"(scanned {scanned}, clean rate {len(kept) / max(scanned, 1) * 100:.0f}%)")


if __name__ == "__main__":
    main()
