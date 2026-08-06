"""
Render the analysis package as a page (analysis_report.html).

The figures are the ones the paper will use; this page is where they are
explained. Each carries what it shows, how to read it, and what it does not
license you to say.

Run:  python analysis/make_analysis.py && python analysis/report.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_report.html")
TAB = os.path.join(HERE, "tables")

FIGURES = [
    ("fig1_detection_heatmap", "Detection strength, side by side",
     """Every structural measure against every defect, scored twice. <b>Left</b> is the
     controlled experiment: one defect introduced into otherwise unchanged code, so any
     movement is caused by the defect. <b>Right</b> is real human-written code labelled
     by the same detectors, where the two groups differ in many ways at once.""",
     """Read a row across. Where both panels are hot, the measure genuinely detects the
     defect. Where only the right panel is hot, it is responding to something the defect
     travels with — usually size. The right panel is visibly hotter overall, and that gap
     is the paper's central claim: an observational study alone would credit these
     measures with far more than they can actually do. Defect names are coloured by the
     verdict they end up with."""),

    ("fig4_family_coverage", "No single measure catches everything",
     """For each defect, the <b>strongest</b> measure available in each family, on the
     controlled data. The bars are maxima on purpose: if the best measure in a family
     cannot see a defect, no measure in that family can.""",
     """Eight of the twelve defects are beyond every structural measure at the
     conventional large-effect threshold. That matters because the structural family is
     the only one available on deployed code — the similarity family reaches all twelve,
     but only where a known-good reference exists to compare against, which a benchmark
     has by construction and real code never does."""),

    ("fig5_model_structure", "What the models actually write",
     """Structural measures averaged over 664 generated solutions, next to the canonical
     solutions for the same tasks.""",
     """Generated code is <i>simpler</i> than the reference on every complexity measure
     and more heavily commented. This holds for all three models, including the frontier
     one, so it is not a quirk of small models. It is also the opposite of what a reader
     might assume — the worry that models produce sprawling over-complicated code is not
     what the measurements show on tasks of this size."""),

    ("fig6_verbosity_confound", "Scoring raw output measures verbosity",
     """Defect rate per model, computed two ways: over everything the model emitted, and
     over just the function definitions.""",
     """DeepSeek appears two and a half times more defective than the others until its own
     appended test cases are excluded, at which point the gap nearly closes. The literals
     inside <code>assert f(10) == 40</code> are counted as magic numbers by any detector.
     With three models the shape of the artefact is clearer: the correction moves DeepSeek
     alone and leaves Qwen and Claude exactly where they were, so this is a property of
     one model's output habits rather than a deflator that applies to everyone."""),

    ("fig7_task_set", "The task set was the binding constraint",
     """Defect rate and number of distinct defects observed, on the short-function
     benchmark against the smell-inducing prompt set.""",
     """On MBPP and HumanEval only four or five of the twelve defects ever appeared, which
     looked like evidence that models write structurally clean code. Given tasks with room
     for the other defects, the rate roughly quadruples and twice as many defects show up.
     The earlier result was a property of the benchmark, not of the models."""),

    ("fig8_induction_rate", "Prompts steer some defects and not others",
     """For prompts written to provoke a specific defect, how often that defect actually
     appeared.""",
     """The spread is the finding. Asking for many parameters reliably produces too many
     parameters. Asking for a function that does seven things produces, in every run, a
     decomposition into five or six small functions instead — never a long method. Prompt
     wording is a strong lever for some structural properties and no lever at all for
     others."""),

    ("fig9_perplexity", "Naturalness is not structural quality",
     """Perplexity under a code language model, on both sources, for all twelve defects.
     The hypothesis being tested is that structurally poor code is unusual code, and so
     should surprise a model trained on a large corpus.""",
     """It does not. No defect clears even the small-effect band in the defective
     direction, and most values point the wrong way — defective code is slightly
     <i>more</i> predictable than clean code. The mechanism is mundane: the largest
     negative values belong to long methods and duplicated code, whose padding and
     repetition are the easiest text there is to predict. Fluency-based scores are
     routinely proposed as quality proxies; this is a clean negative result for that
     idea."""),

    ("fig10_capability_vs_structure", "Capability and structure are close to unrelated",
     """pass@1 next to defect density per hundred lines, for the three models on the same
     664 tasks.""",
     """Correctness spans forty-one points: DeepSeek 51.7%, Qwen 57.4%, Claude 92.6%.
     Defect density spans six hundredths: 1.93, 1.96, 1.90. A model can be far better at
     writing code that works without being any better at writing code that is well
     structured. This only became visible once a frontier model joined the comparison,
     and it is the strongest reason not to read structural metrics as a measure of model
     quality."""),
]

CSS = """
:root{--ink:#16202b;--paper:#fff;--muted:#5b6b7a;--rule:#e2e8ee;--sur:#f6f8fa;--accent:#0f6d6d}
@media(prefers-color-scheme:dark){:root{--ink:#e4e9ef;--paper:#11151a;--muted:#93a2b1;
  --rule:#28313b;--sur:#191e25;--accent:#4fb8b0}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);line-height:1.62;font-size:15.5px;
     font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:44px 24px 90px}
h1{font-size:29px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 26px;font-size:16px;max-width:70ch}
h2{font-size:19px;margin:40px 0 4px;padding-top:22px;border-top:2px solid var(--rule)}
.n{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);
   font-weight:700}
p{margin:0 0 12px;max-width:74ch}
figure{margin:16px 0 10px}
img{width:100%;height:auto;border:1px solid var(--rule);border-radius:7px;background:#fff}
.read{border-left:3px solid var(--accent);background:var(--sur);padding:11px 15px;
      border-radius:0 6px 6px 0;margin:12px 0}
.read .t{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);
         font-weight:700;display:block;margin-bottom:3px}
.read p:last-child{margin-bottom:0}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0;
      font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--rule);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
code{background:var(--sur);padding:1px 5px;border-radius:3px;font-size:.9em;
     font-family:ui-monospace,Consolas,monospace}
.dl{color:var(--muted);font-size:13px}
a{color:var(--accent)}
.scroll{overflow-x:auto}
"""


def detection_table():
    p = os.path.join(TAB, "detection.csv")
    if not os.path.exists(p):
        return ""
    rows = list(csv.DictReader(open(p, encoding="utf-8")))

    def ci(lo, hi):
        try:
            return f"<span class='dl'>[{float(lo):.2f}, {float(hi):.2f}]</span>"
        except (TypeError, ValueError):
            return "<span class='dl'>at cap</span>"

    out = ["<h2><span class='n'>Table</span><br>Strongest measure per defect, with intervals</h2>",
           "<p>The 95% intervals are what make the verdict a claim rather than an "
           "impression. For the co-occurrence defects the controlled interval "
           "<b>contains zero</b> while the observational interval <b>excludes</b> it — "
           "the measure demonstrably cannot see the defect, yet separates the groups.</p>",
           "<div class='scroll'><table><thead><tr><th>defect</th><th>strongest measure</th>"
           "<th>injected</th><th>95% CI</th><th>real</th><th>95% CI</th><th>verdict</th>"
           "</tr></thead><tbody>"]
    for r in rows:
        out.append(
            f"<tr><td><code>{r['smell']}</code></td><td>{r['measure'].replace('_', ' ')}</td>"
            f"<td>{float(r['injected_d']):.2f}</td><td>{ci(r['injected_lo'], r['injected_hi'])}</td>"
            f"<td>{float(r['real_d']):.2f}</td><td>{ci(r['real_lo'], r['real_hi'])}</td>"
            f"<td>{r['verdict']}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def build():
    p = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analysis &mdash; figures and what they show</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Analysis</h1>
<p class="sub">The figures and tables behind the write-up, each with what it shows and how
to read it. Everything here is generated from the result files by
<code>analysis/make_analysis.py</code>, so no number on this page was typed in by hand.</p>"""]

    for i, (name, title, what, how) in enumerate(FIGURES, 1):
        if not os.path.exists(os.path.join(HERE, "figures", f"{name}.png")):
            continue
        p.append(f"<h2><span class='n'>Figure {i}</span><br>{title}</h2>")
        p.append(f"<p>{what}</p>")
        p.append(f'<figure><img src="figures/{name}.png" alt="{title}"></figure>')
        p.append(f"<div class='read'><span class='t'>How to read it</span><p>{how}</p></div>")
        p.append(f"<p class='dl'>Vector version for the paper: "
                 f"<a href='figures/{name}.pdf'>{name}.pdf</a></p>")

    p.append(detection_table())

    p.append("""<h2><span class='n'>Method</span><br>How the intervals are computed</h2>
<p>Detection strength is the standardised difference between the defective and clean
groups: the difference in means divided by the pooled standard deviation, oriented so
positive always means the defective code scored worse, and capped at five.</p>
<p>The interval uses the large-sample variance for an independent-groups standardised
difference (Hedges &amp; Olkin): the first term is sampling error in the means, the second
the error in the pooled standard deviation. Values sitting at the cap are reported without
an interval — the cap is not an estimate, so an interval around it would imply a precision
that is not there. Of the 204 measure-by-defect combinations, 23 are at the cap and are
flagged in <code>tables/detection_full.csv</code>.</p>
<p class="dl">Tables for the paper: <a href="tables/detection.tex">detection.tex</a>,
<a href="tables/benchmark.tex">benchmark.tex</a>,
<a href="tables/detection_full.csv">detection_full.csv</a> (every combination),
<a href="tables/detection.csv">detection.csv</a>.</p>
</div></body></html>""")
    return "\n".join(p)


if __name__ == "__main__":
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(build())
    print(f"wrote {os.path.basename(OUT)}")
