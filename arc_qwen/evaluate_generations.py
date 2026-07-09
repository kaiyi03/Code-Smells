"""
Evaluate the Qwen generations with the evaluation tool (deployment mode).

Where run_panel.py runs the measure panel over the INJECTED benchmark to learn
which measures detect which smells, this points the SAME tools at a real model's
output (arc_qwen/generations.jsonl) to answer the applied question: what does
Qwen2.5-Coder's code actually look like?

For each generated solution it reports:
  * smell rate  -- run the smell detectors (pylint + ruff) on the generated code.
                   Legitimate here, not circular: the model's output was never
                   labelled by them, so this is a fresh measurement.
  * correctness -- run the solution against its MBPP / HumanEval tests (pass@1),
                   reusing the sandbox from correctness.py.
  * similarity  -- the reference-based measures (BLEU, CodeBLEU, ...) vs the
                   task's canonical solution.
  * structure   -- the reference-free structural measures on the generated code,
                   shown next to the canonical solution's for comparison.
  * cost        -- output tokens and generation time (already recorded).

HumanEval similarity is measured on the function BODY only. For HumanEval the
model is given the signature + docstring and asked to complete it, so its output
echoes that prompt and the canonical is prompt + solution -- the shared
signature/docstring would inflate every similarity score regardless of how well
the task was solved. We therefore strip the signature + docstring from both sides
and compare bodies. MBPP has no shared boilerplate (the model writes the whole
function from a text description), so MBPP is scored on the full function. Results
are reported per source; similarity is never pooled across the two.

Writes evaluation_summary.csv (one row per generation) and evaluation_report.html.

Run:  python arc_qwen/evaluate_generations.py            (auto-switches to venv)
      python arc_qwen/evaluate_generations.py --dup      (also run jscpd, slow)
"""

import argparse
import os
import subprocess
import sys


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if not os.path.exists(venv_py):  # linux / ARC layout
        venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "bin", "python"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

import ast
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "eval_tool"))
sys.path.insert(0, os.path.join(ROOT, "smell_injection"))

from measures import PANEL                                             # noqa: E402
from correctness import load_tests, build_program, run_program        # noqa: E402
from build_injected import detect_many, has_duplicate, INJECTORS      # noqa: E402

GENERATIONS = os.path.join(HERE, "generations.jsonl")
OUT_CSV = os.path.join(HERE, "evaluation_summary.csv")
OUT_HTML = os.path.join(HERE, "evaluation_report.html")
BATCH = 16          # generate.py's batch size, for reconstructing per-batch time
WORKERS = 8

STRUCT = [m for m in PANEL if not m.needs_ref]
SIM = [m for m in PANEL if m.needs_ref]
ALL_SMELLS = list(INJECTORS)          # the 12 tracked smell names, in order


def load_generations():
    return [json.loads(line) for line in open(GENERATIONS, encoding="utf-8")]


def load_humaneval_bodies():
    """task_id -> (prompt, canonical_solution), for the body-only HumanEval fix."""
    from datasets import load_dataset
    out = {}
    for ex in load_dataset("openai/openai_humaneval", split="test"):
        out[ex["task_id"].replace("/", "_")] = (ex["prompt"], ex["canonical_solution"])
    return out


def _body_only(code):
    """The first function's body as normalised source, with the signature and any
    docstring removed. None if the code doesn't parse into a function."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return None
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                                   # drop the docstring
    return "\n".join(ast.unparse(s) for s in body)


def _strip_prompt_lcp(gen, prompt):
    """Fallback: strip the longest common prefix that gen shares with the prompt."""
    i, n = 0, min(len(gen), len(prompt))
    while i < n and gen[i] == prompt[i]:
        i += 1
    return gen[i:].lstrip("\n")


def sim_inputs(g, he):
    """(candidate, reference) for the reference-based similarity measures.
    MBPP: whole generated function vs whole canonical (no shared text).
    HumanEval: body only -- the model echoes the given signature + docstring, so
    the full text would score as inflated. Compare the bodies instead, falling
    back to a longest-common-prefix strip if the generated code won't parse."""
    if g["source"] != "humaneval":
        return g["generated_code"], g["canonical_code"]
    cand, ref = _body_only(g["generated_code"]), _body_only(g["canonical_code"])
    if cand is not None and ref is not None:
        return cand, ref
    prompt, canon_sol = he.get(g["task_id"], ("", g["canonical_code"]))
    return _strip_prompt_lcp(g["generated_code"], prompt), canon_sol


def score_correctness(gens, tests):
    """pass@1: run each generation against its task's tests, in parallel."""
    jobs = {g["task_id"]: build_program(g["generated_code"], tests[g["task_id"]])
            for g in gens if g["task_id"] in tests}
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(run_program, prog): tid for tid, prog in jobs.items()}
        for fut in futs:
            out[futs[fut]] = fut.result()
    return out                          # task_id -> 'pass' | 'fail' | 'timeout'


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup", action="store_true",
                    help="also run jscpd for duplicate_code (per-file, slow)")
    args = ap.parse_args()

    gens = load_generations()
    print(f"loaded {len(gens)} generations from {os.path.basename(GENERATIONS)}")

    # --- smells: batched pylint + ruff over all generated code (11 smells) ---
    print("detecting smells (pylint + ruff, batched) ...")
    smells = detect_many({g["task_id"]: g["generated_code"] for g in gens})
    if args.dup:
        print("running jscpd per generation for duplicate_code (slow) ...")
        for g in gens:
            if has_duplicate(g["generated_code"]):
                smells.setdefault(g["task_id"], set()).add("duplicate_code")

    # --- correctness: run the tests (pass@1) ---
    print("loading MBPP + HumanEval tests ...")
    tests = load_tests()
    he = load_humaneval_bodies()
    testable = [g for g in gens if g["task_id"] in tests]
    print(f"running {len(testable)} solutions against their tests "
          f"({WORKERS} at a time) ...")
    passed = score_correctness(testable, tests)

    # --- similarity + structure: the panel, per generation ---
    # similarity uses body-only inputs for HumanEval (see sim_inputs); structural
    # measures are reference-free and always run on the full generated code.
    print("scoring the measure panel (structural + similarity) ...")
    rows = []
    batch_time = {}
    for i, g in enumerate(gens):
        batch_time[i // BATCH] = g.get("batch_seconds") or 0.0
        cand, ref = sim_inputs(g, he)
        gen_m = {m.name: m.fn(g["generated_code"]) for m in STRUCT}
        gen_m.update({m.name: m.fn(cand, ref) for m in SIM})
        can_struct = {m.name: m.fn(g["canonical_code"]) for m in STRUCT}
        rows.append({
            "task_id": g["task_id"], "source": g["source"],
            "smells": sorted(smells.get(g["task_id"], set())),
            "result": passed.get(g["task_id"], "no-test"),
            "n_output_tokens": g.get("n_output_tokens"),
            "gen": gen_m, "can_struct": can_struct,
        })

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    SRCS = sorted(by_src)                     # e.g. ['humaneval', 'mbpp']

    def subset(src):
        return rows if src == "overall" else by_src[src]

    report_console(rows, by_src, SRCS, subset, batch_time)
    counts = Counter(s for r in rows for s in r["smells"])
    write_csv(rows)
    write_html(rows, by_src, SRCS, subset, counts, batch_time)
    print(f"\nwrote {os.path.basename(OUT_CSV)} and {os.path.basename(OUT_HTML)}")


def report_console(rows, by_src, SRCS, subset, batch_time):
    n = len(rows)
    print("\n=== Qwen2.5-Coder evaluation ===\n")

    print("Correctness (pass@1):")
    for src in ["overall"] + SRCS:
        rs = [r for r in subset(src) if r["result"] != "no-test"]
        ok = sum(r["result"] == "pass" for r in rs)
        print(f"  {src:10} {ok:>4}/{len(rs):<4} = {ok / len(rs) * 100:5.1f}%" if rs else f"  {src:10} n/a")

    print("\nSmell rate (share of generations flagged):")
    for src in ["overall"] + SRCS:
        rs = subset(src)
        a = sum(1 for r in rs if r["smells"])
        print(f"  {src:10} any smell {a:>4}/{len(rs):<4} = {a / len(rs) * 100:.1f}%")
    per_src = {src: Counter(s for r in by_src[src] for s in r["smells"]) for src in SRCS}
    total = Counter(s for r in rows for s in r["smells"])
    if total:
        print(f"  {'per smell':22}" + "".join(f"{src:>12}" for src in SRCS) + f"{'total':>8}")
        for smell in ALL_SMELLS:
            if total.get(smell):
                print(f"    {smell:20}" + "".join(f"{per_src[src].get(smell, 0):>12}" for src in SRCS)
                      + f"{total[smell]:>8}")

    print("\nSimilarity to canonical (0-100; HumanEval = body only, MBPP = full):")
    print(f"  {'':10}" + "".join(f"{m.name:>10}" for m in SIM))
    for src in SRCS:
        print(f"  {src:10}" + "".join(f"{mean([r['gen'][m.name] for r in by_src[src]]):>10.1f}" for m in SIM))

    print("\nStructure (mean, generated vs canonical), by source:")
    for src in SRCS:
        print(f"  [{src}]")
        for m in STRUCT:
            gm = mean([r["gen"][m.name] for r in by_src[src]])
            cm = mean([r["can_struct"][m.name] for r in by_src[src]])
            print(f"    {m.name:22} gen {gm:9.2f}   canon {cm:9.2f}")

    print("\nGeneration cost:")
    for src in ["overall"] + SRCS:
        toks = [r["n_output_tokens"] for r in subset(src) if r["n_output_tokens"] is not None]
        if toks:
            print(f"  {src:10} output tokens mean {mean(toks):.0f}, median {statistics.median(toks):.0f}")
    total_time = sum(batch_time.values())
    all_toks = [r["n_output_tokens"] for r in rows if r["n_output_tokens"] is not None]
    if total_time:
        print(f"  total generation time {total_time:.0f}s for {n} solutions "
              f"({sum(all_toks) / total_time:.0f} tok/s)")


def write_csv(rows):
    cols = (["task_id", "source", "result", "n_smells", "smells"]
            + [m.name for m in STRUCT] + [m.name for m in SIM] + ["n_output_tokens"])
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = [r["task_id"], r["source"], r["result"], str(len(r["smells"])),
                    ";".join(r["smells"])]
            for m in STRUCT + SIM:
                v = r["gen"][m.name]
                vals.append("" if v is None else f"{v:.3f}")
            vals.append("" if r["n_output_tokens"] is None else str(r["n_output_tokens"]))
            f.write(",".join(vals) + "\n")


def _pct(part, whole):
    return f"{part / whole * 100:.1f}%" if whole else "n/a"


def write_html(rows, by_src, SRCS, subset, counts, batch_time):
    import html
    n = len(rows)
    total_time = sum(batch_time.values())
    tokens = [r["n_output_tokens"] for r in rows if r["n_output_tokens"] is not None]

    tested = [r for r in rows if r["result"] != "no-test"]
    passed_n = sum(r["result"] == "pass" for r in tested)
    any_smell = sum(1 for r in rows if r["smells"])
    cb = {src: mean([r["gen"]["codebleu"] for r in by_src[src]]) for src in SRCS}
    cb_tile = " / ".join(f"{cb[src]:.0f}" for src in SRCS)
    tiles = [
        ("pass@1", _pct(passed_n, len(tested)), f"{passed_n} of {len(tested)} pass their tests"),
        ("clean rate", _pct(n - any_smell, n), f"{n - any_smell} of {n} have no tracked smell"),
        ("CodeBLEU " + " / ".join(SRCS), cb_tile, "similarity to canonical (HE = body only)"),
        ("tokens / solution", f"{mean(tokens):.0f}", "mean output length"),
    ]

    p = []
    p.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Qwen2.5-Coder &mdash; evaluation</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a;line-height:1.5}}
 .wrap{{max-width:1000px;margin:0 auto;padding:32px 26px 80px}}
 h1{{font-size:25px;margin:0 0 2px}} .sub{{color:#666;margin:0 0 24px;font-size:14px}}
 h2{{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:#555;
     border-bottom:2px solid #e2e2e2;padding-bottom:6px;margin:36px 0 8px}}
 .note{{color:#777;font-size:12.5px;margin:0 0 12px}}
 .tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
 .tile{{background:#fff;border:1px solid #e4e4e4;border-radius:10px;padding:16px 18px}}
 .tile .v{{font-size:25px;font-weight:700}} .tile .k{{font-size:12.5px;color:#555;margin-top:2px}}
 .tile .d{{font-size:11.5px;color:#999;margin-top:4px}}
 table{{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;
        border:1px solid #e4e4e4;border-radius:8px;overflow:hidden}}
 th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid #eee}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f4f4f6;font-weight:600}}
 .bar{{height:9px;background:#e5e7eb;border-radius:5px;overflow:hidden;display:inline-block;width:120px;vertical-align:middle}}
 .bar>span{{display:block;height:100%;background:#b91c1c}}
 .scroll{{max-height:460px;overflow:auto;border:1px solid #e4e4e4;border-radius:8px}}
 .scroll table{{border:0;border-radius:0}} .scroll th{{position:sticky;top:0}}
 .pass{{color:#15803d;font-weight:600}} .fail{{color:#b91c1c}}
 code{{background:#f0f0f2;padding:1px 5px;border-radius:4px;font-size:12px}}
 @media(max-width:720px){{.tiles{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="wrap">
<h1>Qwen2.5-Coder &mdash; evaluation</h1>
<p class="sub">The evaluation tool applied to {n} generated solutions
(164 HumanEval + 500 MBPP). Smells via pylint/ruff; correctness by execution;
similarity vs the canonical solution. HumanEval and MBPP are reported separately;
HumanEval similarity is scored on the function body only (the model echoes the
given signature + docstring, which would otherwise inflate it).</p>
<div class="tiles">""")
    for k, v, d in tiles:
        p.append(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div><div class="d">{d}</div></div>')
    p.append("</div>")

    # correctness
    p.append("<h2>Correctness (pass@1)</h2><table><tr><th>set</th><th>passed</th><th>tested</th><th>pass@1</th></tr>")
    for src in ["overall"] + SRCS:
        rs = [r for r in subset(src) if r["result"] != "no-test"]
        ok = sum(r["result"] == "pass" for r in rs)
        p.append(f"<tr><td>{src}</td><td>{ok}</td><td>{len(rs)}</td><td>{_pct(ok, len(rs))}</td></tr>")
    p.append("</table>")

    # smell rate, per source
    p.append("<h2>Smell rate</h2>")
    p.append('<p class="note">Share of generations carrying each tracked smell, by source '
             '(11 detector smells; duplicate_code needs <code>--dup</code>).</p>')
    p.append("<table><tr><th>smell</th>" + "".join(f"<th>{s}</th>" for s in SRCS)
             + "<th>total</th><th></th></tr>")
    per_src = {src: Counter(s for r in by_src[src] for s in r["smells"]) for src in SRCS}
    a_by_src = {src: sum(1 for r in by_src[src] if r["smells"]) for src in SRCS}
    any_row = "".join(f"<td>{a_by_src[s]} ({_pct(a_by_src[s], len(by_src[s]))})</td>" for s in SRCS)
    p.append(f'<tr><td><b>any smell</b></td>{any_row}<td>{any_smell}</td><td></td></tr>')
    for smell, c in counts.most_common():
        cells = "".join(f"<td>{per_src[s].get(smell, 0)}</td>" for s in SRCS)
        p.append(f'<tr><td><code>{smell}</code></td>{cells}<td>{c}</td>'
                 f'<td style="text-align:left"><span class="bar"><span style="width:{c / n * 100:.0f}%"></span></span></td></tr>')
    if not counts:
        p.append(f'<tr><td colspan="{len(SRCS) + 3}" style="text-align:left;color:#999">no tracked smells found</td></tr>')
    p.append("</table>")

    # similarity, per source (never pooled)
    p.append("<h2>Similarity to canonical (0&ndash;100, higher = closer)</h2>")
    p.append('<p class="note">HumanEval scored on the function body only; MBPP on the full function. '
             'Not pooled &mdash; the two measure different things.</p>')
    p.append("<table><tr><th>set</th>" + "".join(f"<th>{m.name}</th>" for m in SIM) + "</tr>")
    for src in SRCS:
        p.append(f"<tr><td>{src}</td>" +
                 "".join(f"<td>{mean([r['gen'][m.name] for r in by_src[src]]):.1f}</td>" for m in SIM) + "</tr>")
    p.append("</table>")

    # structure per source
    p.append("<h2>Structure &mdash; generated vs canonical (mean)</h2>")
    p.append("<table><tr><th>measure</th>"
             + "".join(f"<th>{s} gen</th><th>{s} canon</th>" for s in SRCS) + "<th>blurb</th></tr>")
    for m in STRUCT:
        cells = ""
        for s in SRCS:
            gm = mean([r["gen"][m.name] for r in by_src[s]])
            cm = mean([r["can_struct"][m.name] for r in by_src[s]])
            cells += f"<td>{gm:.2f}</td><td>{cm:.2f}</td>"
        p.append(f'<tr><td>{m.name}</td>{cells}'
                 f'<td style="text-align:left;color:#888;font-size:12px">{html.escape(m.blurb)}</td></tr>')
    p.append("</table>")

    # cost
    p.append("<h2>Generation cost</h2><table><tr><th>set</th><th>mean tokens</th><th>median tokens</th></tr>")
    for src in ["overall"] + SRCS:
        toks = [r["n_output_tokens"] for r in subset(src) if r["n_output_tokens"] is not None]
        if toks:
            p.append(f"<tr><td>{src}</td><td>{mean(toks):.0f}</td><td>{statistics.median(toks):.0f}</td></tr>")
    p.append("</table>")
    if total_time:
        p.append(f'<p class="note">Total generation time {total_time:.0f}s for {n} solutions '
                 f'({sum(tokens) / total_time:.0f} tok/s on the V100).</p>')

    # per-generation table
    p.append("<h2>Every generation</h2><div class='scroll'><table><tr>"
             "<th>task</th><th>source</th><th>result</th><th># smells</th>"
             "<th>smells</th><th>CodeBLEU</th><th>sloc</th><th>tokens</th></tr>")
    for r in sorted(rows, key=lambda r: (r["result"] != "fail", r["source"], r["task_id"])):
        cbv, sloc = r["gen"]["codebleu"], r["gen"]["sloc"]
        cls = "pass" if r["result"] == "pass" else ("fail" if r["result"] == "fail" else "")
        p.append(f'<tr><td>{html.escape(r["task_id"])}</td><td>{r["source"]}</td>'
                 f'<td class="{cls}">{r["result"]}</td><td>{len(r["smells"])}</td>'
                 f'<td style="text-align:left">{html.escape(", ".join(r["smells"]))}</td>'
                 f'<td>{"" if cbv is None else f"{cbv:.1f}"}</td>'
                 f'<td>{"" if sloc is None else f"{sloc:.0f}"}</td>'
                 f'<td>{r["n_output_tokens"] or ""}</td></tr>')
    p.append("</table></div>")

    p.append("</div></body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


if __name__ == "__main__":
    main()
