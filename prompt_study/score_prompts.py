"""
Score the prompt-study generations with the reference-free half of the panel.

These prompts have no tests and no reference solutions, so correctness and the
similarity family cannot run -- what applies is exactly the half the trust table
validated on real code: the detectors and the structural measures.

Two things this answers that the MBPP/HumanEval evaluation cannot:

  * whether the longer-structure smells appear at all, once the task is big
    enough to host them;
  * whether a prompt written to induce a named smell actually induces it. Only
    six of the prompt set's 25 categories overlap with the twelve our detectors
    confirm, so the induction rate is measured on that subset and the rest are
    reported as structure only.

Smells are counted twice -- over everything the model emitted, and over the
definitions alone -- because models differ in how much they volunteer around the
answer, and literals inside an appended test case are not a smell in the code.

Run:  python prompt_study/score_prompts.py
"""

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
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__),
                                         *sys.argv[1:]]).returncode)


_bootstrap()

import argparse                                              # noqa: E402
import ast                                                   # noqa: E402
import csv                                                   # noqa: E402
import glob                                                  # noqa: E402
import json                                                  # noqa: E402
import statistics                                            # noqa: E402
from collections import Counter                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "eval_tool"))
sys.path.insert(0, os.path.join(ROOT, "smell_injection"))
sys.path.insert(0, os.path.join(ROOT, "arc_qwen"))

from measures import PANEL                                   # noqa: E402
from build_injected import detect_many, has_duplicate, JSCPD, INJECTORS   # noqa: E402
from evaluate_generations import definitions_only            # noqa: E402

STRUCT = [m for m in PANEL if not m.needs_ref]
ALL_SMELLS = list(INJECTORS)
LABELS = {"qwen": "Qwen2.5-Coder-1.5B", "deepseek": "DeepSeek-Coder-1.3B",
          "claude": "Claude Sonnet 5"}


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _is_stub(fn):
    """A function whose body is only `pass`, `...`, or a docstring."""
    body = [n for n in fn.body
            if not (isinstance(n, ast.Expr)
                    and isinstance(getattr(n, "value", None), ast.Constant)
                    and isinstance(n.value.value, str))]              # drop the docstring
    if not body:
        return True
    return all(isinstance(n, ast.Pass)
               or (isinstance(n, ast.Expr)
                   and isinstance(getattr(n, "value", None), ast.Constant)
                   and n.value.value is Ellipsis)
               for n in body)


def completeness(code):
    """(functions, stub functions, longest body in statements).

    A structural measure cannot tell a well-decomposed solution from a skeleton
    of empty functions -- both look small and simple. Code that was never written
    cannot carry a smell, so every structural number here is reported next to how
    much of the work was actually done."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, None, None
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return 0, 0, 0
    return len(fns), sum(1 for f in fns if _is_stub(f)), max(len(f.body) for f in fns)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else float("nan")


def score(rows, label, dup=True):
    """Detectors twice + the structural measures, per generation.

    duplicate_code is the one smell pylint and ruff cannot see -- it needs jscpd,
    which runs per file. Twenty of the prompts target it, so skipping it would
    report a zero we never measured; when jscpd is absent we say so instead."""
    print(f"\n=== {label}  ({len(rows)} generations) ===")
    print("detecting smells (pylint + ruff, batched) ...")
    emitted = detect_many({r["task_id"]: r["generated_code"] for r in rows})
    defs = {r["task_id"]: definitions_only(r["generated_code"]) for r in rows}
    only = detect_many({tid: code for tid, (code, _) in defs.items()})

    if dup and JSCPD:
        print(f"running jscpd for duplicate_code over {len(rows)} generations (slow) ...")
        for r in rows:
            if has_duplicate(r["generated_code"]):
                emitted.setdefault(r["task_id"], set()).add("duplicate_code")
            if has_duplicate(defs[r["task_id"]][0]):
                only.setdefault(r["task_id"], set()).add("duplicate_code")
    elif dup:
        print("[warn] jscpd not found -- duplicate_code is NOT measured, not absent")

    print("scoring the structural measures ...")
    out = []
    for r in rows:
        tid = r["task_id"]
        n_fn, n_stub, longest = completeness(r["generated_code"])
        out.append({
            **{k: r.get(k) for k in ("task_id", "complexity", "domain",
                                     "action_keyword", "n_output_tokens",
                                     "system_mode")},
            "intended": r.get("intended_tracked") or [],
            "n_functions": n_fn, "n_stub": n_stub, "longest_body": longest,
            "smells": sorted(emitted.get(tid, set())),
            "smells_defs": sorted(only.get(tid, set())),
            "n_extra": defs[tid][1],
            "struct": {m.name: m.fn(r["generated_code"]) for m in STRUCT},
        })
    return out


def report(rows, label):
    n = len(rows)
    any_e = sum(1 for r in rows if r["smells"])
    any_d = sum(1 for r in rows if r["smells_defs"])
    extra = sum(1 for r in rows if r["n_extra"])
    print(f"\nSmell rate: as emitted {any_e}/{n} = {100 * any_e / n:.1f}%   "
          f"definitions only {any_d}/{n} = {100 * any_d / n:.1f}%   "
          f"({extra} add statements outside the definitions)")

    # How much of the work was actually done. A skeleton of empty functions scores
    # well on every structural measure, so this qualifies all of them.
    ok = [r for r in rows if r["n_functions"] is not None]
    fns = sum(r["n_functions"] for r in ok)
    stubs = sum(r["n_stub"] for r in ok)
    if fns:
        allstub = sum(1 for r in ok if r["n_functions"] and r["n_stub"] == r["n_functions"])
        print(f"\nCompleteness: {stubs}/{fns} functions are stubs "
              f"({100 * stubs / fns:.1f}%);  {allstub}/{len(ok)} generations are "
              f"entirely stubs;  longest function body: median "
              f"{statistics.median([r['longest_body'] for r in ok])} statements, "
              f"max {max(r['longest_body'] for r in ok)}")
        print(f"              {len(rows) - len(ok)} generations did not parse")

    ce = Counter(s for r in rows for s in r["smells"])
    cd = Counter(s for r in rows for s in r["smells_defs"])
    print(f"\n  {'smell':22}{'as emitted':>12}{'definitions':>13}")
    for s in ALL_SMELLS:
        if ce.get(s) or cd.get(s):
            print(f"  {s:22}{ce.get(s, 0):>12}{cd.get(s, 0):>13}")
    print(f"  {'distinct smells seen':22}{len([s for s in ALL_SMELLS if ce.get(s)]):>12}"
          f"{len([s for s in ALL_SMELLS if cd.get(s)]):>13}  of 12")

    # Did a prompt written to induce a smell induce it? Only measurable where the
    # prompt set's category overlaps the twelve our detectors confirm.
    targeted = [r for r in rows if r["intended"]]
    if targeted:
        print(f"\nInduction rate (prompts whose target smell we can confirm: {len(targeted)}):")
        by_smell = {}
        for r in targeted:
            for s in r["intended"]:
                hit, tot = by_smell.get(s, (0, 0))
                by_smell[s] = (hit + (s in r["smells_defs"]), tot + 1)
        for s, (hit, tot) in sorted(by_smell.items(), key=lambda kv: -kv[1][1]):
            print(f"  {s:22}{hit:>4}/{tot:<4} = {100 * hit / tot:5.1f}%")
        hit = sum(h for h, _ in by_smell.values())
        tot = sum(t for _, t in by_smell.values())
        print(f"  {'overall':22}{hit:>4}/{tot:<4} = {100 * hit / tot:5.1f}%")

    print("\nStructure and cost (mean), by stated complexity:")
    keys = ["sloc", "cyclomatic", "cognitive", "maintainability", "comment_density"]
    print(f"  {'complexity':14}{'n':>5}" + "".join(f"{k:>17}" for k in keys) + f"{'out tokens':>12}")
    for c in ["basic", "intermediate", "advanced"]:
        rs = [r for r in rows if r["complexity"] == c]
        if not rs:
            continue
        vals = "".join(f"{mean([r['struct'][k] for r in rs]):>17.2f}" for k in keys)
        print(f"  {c:14}{len(rs):>5}{vals}{mean([r['n_output_tokens'] for r in rs]):>12.0f}")


def write_csv(rows, path):
    cols = (["task_id", "system_mode", "complexity", "domain", "action_keyword", "intended",
             "n_smells", "smells", "n_smells_defs", "smells_defs", "n_extra_stmts",
             "n_functions", "n_stub", "longest_body"]
            + [m.name for m in STRUCT] + ["n_output_tokens"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["task_id"], r.get("system_mode"), r["complexity"], r["domain"],
                        r["action_keyword"],
                        ";".join(r["intended"]), len(r["smells"]), ";".join(r["smells"]),
                        len(r["smells_defs"]), ";".join(r["smells_defs"]), r["n_extra"],
                        r["n_functions"], r["n_stub"], r["longest_body"]]
                       + ["" if r["struct"][m.name] is None else f"{r['struct'][m.name]:.3f}"
                          for m in STRUCT]
                       + [r["n_output_tokens"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="score just one model tag (qwen, deepseek, claude)")
    ap.add_argument("--no-dup", action="store_true",
                    help="skip jscpd (fast, but duplicate_code goes unmeasured)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(HERE, "generations_*.jsonl")))
    if args.only:
        paths = [p for p in paths if p.endswith(f"_{args.only}.jsonl")]
    if not paths:
        raise SystemExit("no generations_*.jsonl found -- generate first")

    for path in paths:
        tag = os.path.basename(path)[len("generations_"):-len(".jsonl")]
        rows = score(load(path), LABELS.get(tag, tag), dup=not args.no_dup)
        report(rows, tag)
        out = os.path.join(HERE, f"summary_{tag}.csv")
        write_csv(rows, out)
        print(f"\nwrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
