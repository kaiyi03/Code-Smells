"""
Compare the models side by side (model_comparison.html).

evaluate_generations.py produces one summary per model. Read on its own, each
answers "what does this model's code look like?". The question that needs two
models is different -- "does the finding hold for more than one model?" -- and
that is a difference, not two separate pictures, so it belongs in one table
rather than two pages. This builds that table, and keeps the per-model reports
as the detail behind it.

Reads arc_qwen/evaluation_summary*.csv (stdlib only -- no measures import, so it
runs anywhere, and it picks up the GPU measures automatically when they are in
the file). Writes arc_qwen/model_comparison.html.

Run:  python arc_qwen/compare_models.py
"""

import csv
import glob
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(HERE, "model_comparison.html")

# tag (the --tag passed to evaluate_generations.py; "" = the first model) -> label
LABELS = {
    "": "Qwen2.5-Coder-1.5B",
    "deepseek": "DeepSeek-Coder-1.3B",
}

# Display order. Intersected with the CSV header, so a measure that wasn't
# available on the machine that scored the run is simply left out.
STRUCT_SHOW = ["sloc", "cyclomatic", "cognitive", "maintainability",
               "halstead_volume", "halstead_difficulty", "halstead_effort",
               "comment_density", "api_calls", "perplexity"]
SIM_SHOW = ["bleu", "chrf", "rouge_l", "codebleu", "ast_similarity",
            "codebert_score", "bertscore"]
# worse=down for these, worse=up for the rest -- used only for the arrow hints
HIGHER_IS_BETTER = {"maintainability", "comment_density"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row, col):
    v = row.get(col, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, dp=2):
    return "&mdash;" if v is None else f"{v:,.{dp}f}"


def pct(a, b):
    return None if not b else 100.0 * a / b


def discover():
    """[(tag, label, rows, columns)] for every model summary present."""
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, "evaluation_summary*.csv"))):
        base = os.path.basename(path)[len("evaluation_summary"):-len(".csv")]
        tag = base.lstrip("_")
        rows = load(path)
        if not rows:
            continue
        out.append({"tag": tag, "label": LABELS.get(tag, tag or "model"),
                    "rows": rows, "cols": set(rows[0])})
    # the unsuffixed (first) model leads
    out.sort(key=lambda m: (m["tag"] != "", m["tag"]))
    return out


# ---------------------------------------------------------------- statistics

def by_source(rows):
    srcs = {}
    for r in rows:
        srcs.setdefault(r["source"], []).append(r)
    return srcs


def stats(m):
    """Everything the page shows, for one model."""
    rows = m["rows"]
    srcs = by_source(rows)
    s = {"n": len(rows), "srcs": sorted(srcs), "by_src": srcs}

    def block(rs):
        tested = [r for r in rs if r["result"] != "no-test"]
        smelly = [r for r in rs if int(r["n_smells"] or 0) > 0]
        smelly_defs = [r for r in rs if int(r.get("n_smells_defs") or 0) > 0]
        extra = [r for r in rs if int(r.get("n_extra_stmts") or 0) > 0]
        toks = [num(r, "n_output_tokens") for r in rs]
        return {
            "n": len(rs),
            "pass1": pct(sum(r["result"] == "pass" for r in tested), len(tested)),
            "smell_rate": pct(len(smelly), len(rs)),
            # None (not 0) when the column predates the definitions-only split
            "smell_rate_defs": pct(len(smelly_defs), len(rs)) if "n_smells_defs" in rs[0] else None,
            "extra_rate": pct(len(extra), len(rs)) if "n_extra_stmts" in rs[0] else None,
            "tokens": mean(toks),
        }

    s["overall"] = block(rows)
    s["per_src"] = {src: block(rs) for src, rs in srcs.items()}

    def tally(col):
        out = {}
        for r in rows:
            for name in (r.get(col) or "").split(";"):
                if name:
                    out[name] = out.get(name, 0) + 1
        return out

    s["smells"] = tally("smells")
    s["smells_defs"] = tally("smells_defs")

    s["sim"] = {src: {c: mean([num(r, c) for r in rs]) for c in SIM_SHOW if c in m["cols"]}
                for src, rs in srcs.items()}
    s["struct"] = {c: mean([num(r, c) for r in rows]) for c in STRUCT_SHOW if c in m["cols"]}
    s["canon"] = {c: mean([num(r, f"canon_{c}") for r in rows])
                  for c in STRUCT_SHOW if f"canon_{c}" in m["cols"]}
    return s


def replication(models, S):
    """The claims the second model exists to test. Each is (claim, per-model value,
    agree?) -- computed, not asserted, so the page can't drift from the data."""
    checks = []

    def add(claim, values, same):
        checks.append((claim, values, same))

    # 1. which smell dominates
    tops = []
    for m in models:
        c = S[m["tag"]]["smells"]
        tops.append(max(c, key=c.get) if c else "none")
    add("The most common smell in the generated code", tops, len(set(tops)) == 1)

    # 2. how much of the 12-smell space shows up at all
    seen = [len(S[m["tag"]]["smells"]) for m in models]
    add("Distinct smells appearing at all (of 12)", [str(v) for v in seen],
        max(seen) - min(seen) <= 1)

    # 3. is the smell rate about the code, or about extra test code the model added
    dirs, vals = [], []
    for m in models:
        o = S[m["tag"]]["overall"]
        a, b = o["smell_rate"], o["smell_rate_defs"]
        if a is None or b is None or not a:
            dirs.append(None); vals.append("&mdash;")
        else:
            dirs.append(b >= 0.75 * a); vals.append(f"{a:.1f}% &rarr; {b:.1f}%")
    add("Smell rate survives rescoring the definitions alone (keeps &ge;&nbsp;75%)", vals,
        None if None in dirs else all(dirs))

    # 4. is HumanEval smellier than MBPP
    dirs, vals = [], []
    for m in models:
        p = S[m["tag"]]["per_src"]
        he, mb = p.get("humaneval", {}).get("smell_rate"), p.get("mbpp", {}).get("smell_rate")
        if he is None or mb is None:
            dirs.append(None); vals.append("&mdash;")
        else:
            dirs.append(he > mb); vals.append(f"{he:.1f}% vs {mb:.1f}%")
    add("HumanEval code is smellier than MBPP code", vals,
        None if None in dirs else len(set(dirs)) == 1)

    # 4. commenting habit vs the canonical solutions
    dirs, vals = [], []
    for m in models:
        s = S[m["tag"]]
        g, c = s["struct"].get("comment_density"), s["canon"].get("comment_density")
        if g is None or c is None:
            dirs.append(None); vals.append("&mdash;")
        else:
            dirs.append(g > c); vals.append(f"{g:.1f} vs {c:.1f}")
    add("Generated code is more heavily commented than the canonical solution", vals,
        None if None in dirs else len(set(dirs)) == 1)

    # 5. perplexity of the model's own output vs the canonical
    dirs, vals = [], []
    for m in models:
        s = S[m["tag"]]
        g, c = s["struct"].get("perplexity"), s["canon"].get("perplexity")
        if g is None or c is None:
            dirs.append(None); vals.append("&mdash;")
        else:
            dirs.append(g < c); vals.append(f"{g:.2f} vs {c:.2f}")
    add("Generated code is less perplexing than the canonical solution", vals,
        None if None in dirs else len(set(dirs)) == 1)

    return checks


# ---------------------------------------------------------------- rendering

CSS = """
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a;line-height:1.5}
 .wrap{max-width:1000px;margin:0 auto;padding:32px 26px 80px}
 h1{font-size:25px;margin:0 0 2px} .sub{color:#666;margin:0 0 24px;font-size:14px}
 h2{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:#555;
    border-bottom:2px solid #e2e2e2;padding-bottom:6px;margin:36px 0 8px}
 .note{color:#777;font-size:12.5px;margin:0 0 12px}
 table{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;
       border:1px solid #e4e4e4;border-radius:8px;overflow:hidden}
 th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #eee;
       font-variant-numeric:tabular-nums}
 th:first-child,td:first-child{text-align:left}
 th{background:#f4f4f6;font-weight:600}
 td.canon{color:#888}
 .yes{color:#15803d;font-weight:600} .no{color:#b45309;font-weight:600}
 .tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:6px}
 .tile{background:#fff;border:1px solid #e4e4e4;border-radius:10px;padding:16px 18px}
 .tile .m{font-size:12.5px;color:#555;font-weight:600;margin-bottom:8px}
 .tile .v{font-size:23px;font-weight:700} .tile .k{font-size:12px;color:#777}
 .tile .row{display:flex;gap:22px}
 .links{margin-top:10px;font-size:13.5px}
 .links a{color:#2563eb;text-decoration:none;margin-right:18px}
 .scrollx{overflow-x:auto}
 @media(max-width:720px){.tiles{grid-template-columns:1fr}}
"""


def th_models(models, first=""):
    return f"<tr><th>{first}</th>" + "".join(f"<th>{m['label']}</th>" for m in models) + "</tr>"


def write_html(models, S):
    checks = replication(models, S)
    p = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Model comparison &mdash; evaluation</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Model comparison</h1>
<p class="sub">The same evaluation applied to {len(models)} models. Both were run through
one generation script (same prompts, greedy decoding, same batch size) and scored with
one measure panel, so every difference below is the model rather than the setup.
HumanEval and MBPP are reported separately wherever the two behave differently;
similarity is never pooled across them.</p>
<div class="tiles">"""]

    for m in models:
        s = S[m["tag"]]
        o = s["overall"]
        p.append(f"""<div class="tile"><div class="m">{m['label']}</div><div class="row">
<div><div class="v">{fmt(o['pass1'], 1)}%</div><div class="k">pass@1</div></div>
<div><div class="v">{fmt(o['smell_rate'], 1)}%</div><div class="k">smell rate</div></div>
<div><div class="v">{fmt(o['tokens'], 0)}</div><div class="k">tokens / solution</div></div>
</div></div>""")
    p.append("</div>")

    # --- does it replicate ---
    p.append("<h2>Does the finding hold for both models?</h2>")
    p.append('<p class="note">Each row is a claim the Qwen-only results supported. '
             'The values are computed from the summaries, and the verdict just compares '
             'them &mdash; so this table cannot drift from the data behind it.</p>')
    p.append('<div class="scrollx"><table>' + th_models(models, "claim") + "<th>holds?</th>")
    for claim, vals, same in checks:
        cells = "".join(f"<td>{v}</td>" for v in vals)
        verdict = ('<td style="color:#aaa">not measured</td>' if same is None
                   else '<td class="yes">both</td>' if same
                   else '<td class="no">differs</td>')
        p.append(f"<tr><td>{claim}</td>{cells}{verdict}</tr>")
    p.append("</table></div>")

    # --- headline, by source ---
    p.append("<h2>Correctness, smell rate and cost</h2>")
    srcs = sorted({s for m in models for s in S[m["tag"]]["srcs"]})
    p.append('<div class="scrollx"><table>' + th_models(models, ""))
    for key, label, dp, suffix in [("pass1", "pass@1", 1, "%"),
                                   ("smell_rate", "smell rate (as emitted)", 1, "%"),
                                   ("smell_rate_defs", "smell rate (definitions only)", 1, "%"),
                                   ("extra_rate", "adds statements outside the definitions", 1, "%"),
                                   ("tokens", "output tokens / solution", 0, "")]:
        p.append(f'<tr><td><b>{label}</b> &mdash; overall</td>'
                 + "".join(f"<td>{fmt(S[m['tag']]['overall'][key], dp)}{suffix}</td>"
                           for m in models) + "</tr>")
        for src in srcs:
            p.append(f'<tr><td style="padding-left:22px">{src}</td>'
                     + "".join(f"<td>{fmt(S[m['tag']]['per_src'].get(src, {}).get(key), dp)}{suffix}</td>"
                               for m in models) + "</tr>")
    p.append("</table></div>")

    # --- smells ---
    p.append("<h2>Which smells actually appear</h2>")
    p.append('<p class="note">Counts of generations flagged, out of '
             + " / ".join(f"{S[m['tag']]['n']}" for m in models)
             + '. Blank means the smell never appeared &mdash; on single short functions '
             'most of the twelve cannot physically occur.</p>')
    all_smells = sorted({k for m in models for k in S[m["tag"]]["smells"]})
    p.append('<div class="scrollx"><table><tr><th>smell</th>'
             + "".join(f'<th>{m["label"]}<br>'
                       '<span style="font-weight:400;color:#888">as emitted / definitions</span></th>'
                       for m in models) + "</tr>")
    for name in all_smells:
        cells = ""
        for m in models:
            a = S[m["tag"]]["smells"].get(name)
            b = S[m["tag"]]["smells_defs"].get(name)
            cells += (f"<td>{a or '&mdash;'} <span style='color:#888'>/ {b or 0}</span></td>"
                      if S[m["tag"]]["smells_defs"] else f"<td>{a or '&mdash;'}</td>")
        p.append(f"<tr><td>{name}</td>{cells}</tr>")
    p.append("</table></div>")

    # --- similarity ---
    p.append("<h2>Similarity to the canonical solution</h2>")
    p.append('<p class="note">0&ndash;100, higher = closer to the reference. HumanEval is '
             'scored on the function body only (the model echoes the given signature and '
             'docstring, which would otherwise inflate every score).</p>')
    sim_cols = [c for c in SIM_SHOW if any(c in S[m["tag"]]["sim"].get(src, {})
                                           for m in models for src in srcs)]
    p.append('<div class="scrollx"><table><tr><th>measure</th>'
             + "".join(f"<th>{m['label']}<br><span style='font-weight:400;color:#888'>{src}</span></th>"
                       for src in srcs for m in models) + "</tr>")
    for c in sim_cols:
        p.append(f"<tr><td>{c}</td>"
                 + "".join(f"<td>{fmt(S[m['tag']]['sim'].get(src, {}).get(c), 1)}</td>"
                           for src in srcs for m in models) + "</tr>")
    p.append("</table></div>")

    # --- structure ---
    p.append("<h2>Structure of the generated code</h2>")
    p.append('<p class="note">Mean over all generations, with the canonical solutions as '
             'the shared reference point (identical tasks, so the canonical column is the '
             'same for both models). Reference-free, so these pool across HumanEval and MBPP. '
             '&uarr; = higher is worse, except maintainability and comment density.</p>')
    struct_cols = [c for c in STRUCT_SHOW if any(c in S[m["tag"]]["struct"] for m in models)]
    canon = next((S[m["tag"]]["canon"] for m in models if S[m["tag"]]["canon"]), {})
    p.append('<div class="scrollx"><table>' + th_models(models, "measure").replace(
        "</tr>", "<th>canonical</th></tr>"))
    for c in struct_cols:
        arrow = "&darr;" if c in HIGHER_IS_BETTER else "&uarr;"
        p.append(f"<tr><td>{c} <span style='color:#aaa'>{arrow}</span></td>"
                 + "".join(f"<td>{fmt(S[m['tag']]['struct'].get(c))}</td>" for m in models)
                 + f'<td class="canon">{fmt(canon.get(c))}</td></tr>')
    p.append("</table></div>")

    p.append('<h2>Detail</h2><div class="links">')
    for m in models:
        suffix = f"_{m['tag']}" if m["tag"] else ""
        p.append(f'<a href="evaluation_report{suffix}.html">{m["label"]} &mdash; full report &rarr;</a>')
    p.append("</div>")

    p.append("</div></body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def main():
    models = discover()
    if not models:
        raise SystemExit("no evaluation_summary*.csv found -- run evaluate_generations.py first")
    S = {m["tag"]: stats(m) for m in models}
    print(f"comparing {len(models)} model(s): " + ", ".join(m["label"] for m in models))
    for claim, vals, same in replication(models, S):
        verdict = "n/a" if same is None else "both" if same else "differs"
        print(f"  [{verdict}] {claim}: " + " | ".join(vals))
    write_html(models, S)
    print(f"\nwrote {os.path.basename(OUT_HTML)}")


if __name__ == "__main__":
    main()
