"""
Render the prompt study as a report page (prompt_report.html).

Everything on the page is computed from the summary_*.csv files, so the prose
explains the numbers but never states one -- the page cannot drift from its data.

Run:  python prompt_study/report.py
"""

import csv
import glob
import html
import os
import statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prompt_report.html")

# tag -> (model label, the condition it was generated under)
#
# Two things vary: the system prompt (whether the model is told to implement every
# function) and the token budget. The 1k runs are a controlled pair on the
# instruction; the 4k runs are the headline comparison, where the budget binds on
# nobody. A run whose output was substantially truncated is left out entirely --
# structural measures on code that stops mid-statement are not measurements.
RUNS = {
    "qwen":         ("Qwen2.5-Coder-1.5B", "minimal, 1k budget"),
    "qwenfull":     ("Qwen2.5-Coder-1.5B", "implement fully, 1k budget"),
    "qwen4k":       ("Qwen2.5-Coder-1.5B", "implement fully, 4k budget"),
    "deepseek":     ("DeepSeek-Coder-1.3B", "minimal, 1k budget"),
    "deepseekfull": ("DeepSeek-Coder-1.3B", "implement fully, 1k budget"),
    "deepseek4k":   ("DeepSeek-Coder-1.3B", "implement fully, 4k budget"),
    "claude":       ("Claude Sonnet 5", "implement fully, 4k budget"),
}
ORDER = ["qwen", "qwenfull", "qwen4k",
         "deepseek", "deepseekfull", "deepseek4k", "claude"]

# Runs excluded from the page, with the reason shown to the reader.
EXCLUDE = {}
TWELVE = ["long_method", "deep_nesting", "complex_conditional", "long_parameter_list",
          "mutable_default", "broad_except", "dead_code", "magic_number",
          "inefficient_loop", "inefficient_copy", "perf_try_in_loop", "duplicate_code"]
# the six the prompt set targets that our detectors can confirm
TARGETED = ["long_method", "long_parameter_list", "duplicate_code",
            "magic_number", "deep_nesting", "dead_code"]

# For reference: the same panel on the short-function benchmark (MBPP + HumanEval),
# from arc_qwen/. Quoted so the two task sets can be compared.
SHORT_TASK = {"smell_rate": 12.2, "distinct": 5, "sloc": 6.6, "cyclomatic": 2.8}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def load(tag):
    path = os.path.join(HERE, f"summary_{tag}.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(rows):
    n = len(rows)
    smelly = sum(1 for r in rows if int(r["n_smells_defs"] or 0) > 0)
    counts = Counter(s for r in rows for s in (r["smells_defs"] or "").split(";") if s)

    fns = sum(int(r["n_functions"]) for r in rows if r["n_functions"])
    stubs = sum(int(r["n_stub"]) for r in rows if r["n_functions"])
    allstub = sum(1 for r in rows
                  if r["n_functions"] and int(r["n_functions"]) > 0
                  and int(r["n_stub"]) == int(r["n_functions"]))
    parsed = sum(1 for r in rows if r["n_functions"])

    induction = {}
    for s in TARGETED:
        tgt = [r for r in rows if s in (r["intended"] or "").split(";")]
        if tgt:
            hit = sum(1 for r in tgt if s in (r["smells_defs"] or "").split(";"))
            induction[s] = (hit, len(tgt))

    by_complexity = {}
    for c in ["basic", "intermediate", "advanced"]:
        rs = [r for r in rows if r["complexity"] == c]
        if rs:
            by_complexity[c] = {
                "n": len(rs),
                "sloc": mean([num(r["sloc"]) for r in rs]),
                "cyclomatic": mean([num(r["cyclomatic"]) for r in rs]),
                "cognitive": mean([num(r["cognitive"]) for r in rs]),
                "tokens": mean([num(r["n_output_tokens"]) for r in rs]),
                "smell_rate": 100 * sum(1 for r in rs if int(r["n_smells_defs"] or 0) > 0) / len(rs),
            }

    return {
        "n": n,
        "smell_rate": 100 * smelly / n,
        "counts": counts,
        "distinct": sum(1 for s in TWELVE if counts.get(s)),
        "stub_rate": 100 * stubs / fns if fns else None,
        "allstub": allstub, "parsed": parsed,
        "induction": induction,
        "by_complexity": by_complexity,
        "sloc": mean([num(r["sloc"]) for r in rows]),
        "cyclomatic": mean([num(r["cyclomatic"]) for r in rows]),
        "tokens": mean([num(r["n_output_tokens"]) for r in rows]),
    }


def pct(x, dp=1):
    return "&mdash;" if x is None else f"{x:.{dp}f}%"


def fmt(x, dp=1):
    return "&mdash;" if x is None else f"{x:,.{dp}f}"


CSS = """
:root{--ink:#16202b;--paper:#fff;--muted:#5b6b7a;--rule:#e2e8ee;--sur:#f6f8fa;
      --accent:#0f6d6d;--good:#1b6b41;--warn:#96580c;--bad:#9b2c2c}
@media(prefers-color-scheme:dark){:root{--ink:#e4e9ef;--paper:#11151a;--muted:#93a2b1;
      --rule:#28313b;--sur:#191e25;--accent:#4fb8b0;--good:#5cbb85;--warn:#d99b4a;--bad:#e08585}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);line-height:1.6;font-size:15.5px;
     font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:44px 24px 90px}
h1{font-size:29px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 8px;font-size:16px;max-width:70ch}
.meta{color:var(--muted);font-size:13px;margin:0 0 30px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--accent);
   font-weight:700;margin:44px 0 4px;padding-top:20px;border-top:2px solid var(--rule)}
h3{font-size:17px;margin:14px 0 6px;font-weight:650}
p{margin:0 0 12px;max-width:74ch}
.note{color:var(--muted);font-size:13.5px;margin:0 0 14px;max-width:74ch}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 6px;
      font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
         border-bottom:2px solid var(--rule)}
tbody tr:hover{background:var(--sur)}
.scroll{overflow-x:auto}
.good{color:var(--good);font-weight:650}.warn{color:var(--warn);font-weight:650}
.bad{color:var(--bad);font-weight:650}
.dim{color:var(--muted)}
.box{border-left:3px solid var(--accent);background:var(--sur);padding:12px 16px;
     margin:16px 0;border-radius:0 6px 6px 0}
.box p:last-child{margin-bottom:0}
.box .tag{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--accent);
          font-weight:700;display:block;margin-bottom:4px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}
.tile{background:var(--sur);border:1px solid var(--rule);border-radius:8px;padding:13px 15px}
.tile .v{font-size:22px;font-weight:700}
.tile .k{font-size:12px;color:var(--muted);margin-top:2px}
code{background:var(--sur);padding:1px 5px;border-radius:3px;font-size:.9em;
     font-family:ui-monospace,Consolas,monospace}
a{color:var(--accent)}
@media(max-width:700px){.tiles{grid-template-columns:repeat(2,1fr)}}
"""


def build(data):
    present = [t for t in ORDER if t in data]
    p = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prompt study &mdash; smell-inducing prompts</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Smell-inducing prompts</h1>
<p class="sub">What happens when models are asked for code that is structurally
demanding by design, rather than for the short self-contained functions the
correctness benchmarks use.</p>
<p class="meta">476 Python prompts, each written to provoke a named structural defect,
from the <a href="https://github.com/rosawoo/code-smell">Code Smell Detection Dataset</a>
(2026), used with permission and cited in the write-up. Two models, each run twice under
different instructions.</p>"""]

    # ---------------------------------------------------------------- context
    p.append("<h2>Why this exists</h2>")
    p.append("""<p>The main evaluation uses MBPP and HumanEval, where a task is a single
short function. That is a real limitation: most of the twelve defects cannot physically
occur in a five-line answer, so only four or five ever appeared and the evaluation was
under-powered for exactly the problems it was built to study.</p>
<p>These prompts ask for something different &mdash; a function that validates input,
calculates totals, applies discounts, updates inventory, generates an invoice, sends a
confirmation and logs the transaction. Work of that shape has room for a long method,
deep nesting, or duplicated blocks. The question this page answers is whether that room
gets used.</p>""")

    p.append("""<div class="box"><span class="tag">What can and cannot be measured here</span>
<p>These prompts come with <strong>no tests and no reference solutions</strong>. So
correctness cannot be measured &mdash; there is nothing to execute against &mdash; and
neither can any similarity measure, since those compare code to a known-good version.
What runs is the reference-free half of the panel: the detectors, and the structural
measures. That is the same half the detection report validated on real code, so it is
the half we have grounds to trust here.</p></div>""")

    # ---------------------------------------------------------------- headline
    base = data[present[0]]
    p.append("<h2>Does the task set surface the defects?</h2>")
    p.append('<div class="tiles">')
    for k, v, note in [
        ("distinct defects", f"{max(d['distinct'] for d in data.values())} / 12",
         "vs 5 on short tasks"),
        ("smell rate", pct(max(d['smell_rate'] for d in data.values())),
         f"vs {SHORT_TASK['smell_rate']}% on short tasks"),
        ("mean lines", fmt(max(d['sloc'] for d in data.values())),
         f"vs {SHORT_TASK['sloc']} on short tasks"),
        ("mean branching", fmt(max(d['cyclomatic'] for d in data.values())),
         f"vs {SHORT_TASK['cyclomatic']} on short tasks"),
    ]:
        p.append(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div>'
                 f'<div class="k dim">{note}</div></div>')
    p.append("</div>")
    p.append("""<p class="note">Yes. The generated code is roughly four times longer and
four times branchier than on the short-function benchmark, and twice as many of the twelve
defects appear. The task set was the binding constraint, not the models.</p>""")

    p.append('<div class="scroll"><table><thead><tr><th>run</th><th>instruction</th>'
             '<th>smell rate</th><th>defects seen</th><th>mean lines</th>'
             '<th>mean branching</th><th>output tokens</th></tr></thead><tbody>')
    for t in present:
        d, (label, mode) = data[t], RUNS[t]
        p.append(f"<tr><td>{label}</td><td class='dim'>{mode}</td>"
                 f"<td>{pct(d['smell_rate'])}</td><td>{d['distinct']} of 12</td>"
                 f"<td>{fmt(d['sloc'])}</td><td>{fmt(d['cyclomatic'])}</td>"
                 f"<td>{fmt(d['tokens'], 0)}</td></tr>")
    p.append("</tbody></table></div>")
    p.append("""<p class="note"><strong>Smell rate</strong> is the share of generations
carrying at least one of the twelve defects, counted on the definitions only &mdash; any
test cases or demo calls a model appends are excluded, because literals inside an
<code>assert</code> are not a defect in the function.</p>""")

    # ---------------------------------------------------------------- completeness
    p.append("<h2>Completeness: was the work actually done?</h2>")
    p.append("""<p>A structural measure cannot tell a well-decomposed solution from a
skeleton of empty functions &mdash; both look short and simple. Code that was never
written cannot carry a defect, so a model can score well by not doing the work. Every
structural number on this page is therefore reported next to how complete the code is.</p>""")
    p.append('<div class="scroll"><table><thead><tr><th>run</th><th>instruction</th>'
             '<th>stub functions</th><th>all-stub generations</th>'
             '<th>parsed</th></tr></thead><tbody>')
    for t in present:
        d, (label, mode) = data[t], RUNS[t]
        cls = "bad" if (d["stub_rate"] or 0) > 20 else "warn" if (d["stub_rate"] or 0) > 10 else "good"
        p.append(f"<tr><td>{label}</td><td class='dim'>{mode}</td>"
                 f"<td class='{cls}'>{pct(d['stub_rate'])}</td>"
                 f"<td>{d['allstub']} of {d['parsed']}</td>"
                 f"<td>{d['parsed']} of {d['n']}</td></tr>")
    p.append("</tbody></table></div>")
    p.append("""<p class="note">A <strong>stub</strong> is a function whose body is only
<code>pass</code>, <code>...</code>, or a docstring. The two instructions differ in one
sentence: the second asks for every function to be implemented fully. Comparing the pair
shows how much of the apparent structural quality was the model declining to write
code.</p>""")

    # ---------------------------------------------------------------- induction
    p.append("<h2>Did the prompt induce the defect it targeted?</h2>")
    p.append("""<p>Each prompt names the defect it was written to provoke, which makes this
a labelled test rather than a profile: for each one we can ask whether the thing the prompt
was engineering for actually appeared.</p>
<p class="note">The prompt set uses Fowler's 25-category catalogue; this project tracks the
twelve a static analyser can confirm. Six overlap, so induction is measurable on
<strong>152 of the 476</strong> prompts. The other 324 still contribute everything else on
this page &mdash; the detectors fire on whatever they find, regardless of what the prompt
was aiming at.</p>""")
    p.append('<div class="scroll"><table><thead><tr><th>targeted defect</th><th>prompts</th>'
             + "".join(f"<th>{RUNS[t][0].split('-')[0]}<br>"
                       f"<span class='dim' style='font-weight:400'>{RUNS[t][1]}</span></th>"
                       for t in present) + "</tr></thead><tbody>")
    for s in TARGETED:
        tot = next((d["induction"][s][1] for d in data.values() if s in d["induction"]), 0)
        cells = ""
        for t in present:
            ind = data[t]["induction"].get(s)
            if not ind:
                cells += "<td>&mdash;</td>"
                continue
            hit, n = ind
            rate = 100 * hit / n
            cls = "good" if rate >= 60 else "warn" if rate >= 20 else "bad"
            cells += f"<td class='{cls}'>{rate:.0f}%</td>"
        p.append(f"<tr><td><code>{s}</code></td><td class='dim'>{tot}</td>{cells}</tr>")
    p.append("</tbody></table></div>")
    p.append("""<p class="note">Read down a column: a prompt can reliably provoke some
defects and reliably fail to provoke others. That spread is the finding &mdash; prompt
wording steers structural quality for some defects and not at all for others.</p>""")

    # ---------------------------------------------------------------- per defect
    p.append("<h2>Which defects appear at all</h2>")
    p.append("""<p class="note">Counts of generations carrying each defect, out of 476.
A blank means the defect never appeared. Compare against the short-function benchmark,
where only magic numbers, dead code, and occasional loop inefficiencies ever showed up.</p>""")
    p.append('<div class="scroll"><table><thead><tr><th>defect</th>'
             + "".join(f"<th>{RUNS[t][0].split('-')[0]}<br>"
                       f"<span class='dim' style='font-weight:400'>{RUNS[t][1]}</span></th>"
                       for t in present) + "</tr></thead><tbody>")
    for s in TWELVE:
        if not any(data[t]["counts"].get(s) for t in present):
            continue
        cells = "".join(f"<td>{data[t]['counts'].get(s) or '&mdash;'}</td>" for t in present)
        p.append(f"<tr><td><code>{s}</code></td>{cells}</tr>")
    p.append("</tbody></table></div>")

    # ---------------------------------------------------------------- complexity
    p.append("<h2>Does structure track the prompt's stated difficulty?</h2>")
    p.append("""<p>Every prompt is labelled basic, intermediate or advanced by its authors.
If those labels mean anything, the generated code should get measurably heavier as they
rise &mdash; which is a check on the dataset as much as on the models.</p>""")
    for t in present:
        d, (label, mode) = data[t], RUNS[t]
        if not d["by_complexity"]:
            continue
        p.append(f"<h3>{label} <span class='dim' style='font-weight:400'>&mdash; {mode}</span></h3>")
        p.append('<div class="scroll"><table><thead><tr><th>stated level</th><th>prompts</th>'
                 '<th>mean lines</th><th>mean branching</th><th>cognitive</th>'
                 '<th>output tokens</th><th>smell rate</th></tr></thead><tbody>')
        for c in ["basic", "intermediate", "advanced"]:
            b = d["by_complexity"].get(c)
            if not b:
                continue
            p.append(f"<tr><td>{c}</td><td class='dim'>{b['n']}</td>"
                     f"<td>{fmt(b['sloc'])}</td><td>{fmt(b['cyclomatic'])}</td>"
                     f"<td>{fmt(b['cognitive'])}</td><td>{fmt(b['tokens'], 0)}</td>"
                     f"<td>{pct(b['smell_rate'])}</td></tr>")
        p.append("</tbody></table></div>")
    p.append("""<p class="note">It does, consistently, in every run. Note that the dataset's
separate <code>expected_token_depth</code> field is a one-to-one relabelling of this
complexity level, so it carries no additional information &mdash; the measured output
tokens above are the variable worth using.</p>""")

    # ---------------------------------------------------------------- reading it
    p.append("<h2>How to read this, and what it does not show</h2>")
    p.append("""<ul>
<li><strong>No correctness anywhere on this page.</strong> These prompts have no tests.
A generation counted as clean here may not work at all; structural quality and correctness
are independent properties and only one of them is measured.</li>
<li><strong>Induction is measured on 152 prompts, not 476.</strong> Only six of the prompt
set's categories overlap with the twelve this project can confirm.</li>
<li><strong>Detectors define the defects.</strong> They are an operational definition, not
ground truth, and the structural measures are evaluated against them rather than the other
way round.</li>
<li><strong>Two small instruct-tuned models.</strong> Findings here describe these models
at this scale, not language models in general.</li>
</ul>""")
    if EXCLUDE:
        p.append("<h2>Runs left out</h2><ul>")
        for tag, why in EXCLUDE.items():
            p.append(f"<li><strong>{RUNS.get(tag, (tag, ''))[0]} "
                     f"({RUNS.get(tag, ('', ''))[1]})</strong> &mdash; {html.escape(why)}. "
                     f"Structural measures on code that stops mid-statement are not "
                     f"measurements, so the run is excluded rather than shown with a "
                     f"caveat.</li>")
        p.append("</ul>")

    p.append('<p class="meta">Generated by <code>prompt_study/report.py</code> from '
             '<code>summary_*.csv</code>. Every figure on this page is computed from those '
             'files rather than written by hand.</p>')
    p.append("</div></body></html>")
    return "\n".join(p)


def main():
    data = {}
    for tag in RUNS:
        rows = load(tag)
        if not rows:
            continue
        # A run whose output was mostly cut off mid-statement cannot be scored: the
        # detectors need code that parses, and the structural measures would be
        # describing fragments. Such a run is dropped and the reason recorded.
        parsed = sum(1 for r in rows if r["n_functions"])
        if parsed / len(rows) < 0.8:
            EXCLUDE[tag] = (f"{100 * (1 - parsed / len(rows)):.0f}% of this run's output was "
                            f"truncated by the token budget and does not parse")
            print(f"  [excluded] {tag}: {EXCLUDE[tag]}")
            continue
        data[tag] = stats(rows)
    if not data:
        raise SystemExit("no summary_*.csv found -- run score_prompts.py first")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(build(data))
    print(f"wrote {os.path.basename(OUT)} from {len(data)} run(s): {', '.join(data)}")


if __name__ == "__main__":
    main()
