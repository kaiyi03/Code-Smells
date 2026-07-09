"""
Build a human-readable reference guide for the 12 injected smells.

For each smell it writes: a plain-English description, how the injector changes
the code, which tool/rule confirms it, whether it is behaviour-preserving, and
one or two REAL clean->smelly examples pulled from samples.jsonl (shortest first,
so the diff is easy to read). The added/removed lines are highlighted.

Output: smell_guide.html  (open in a browser).

Run:  python smell_injection/smell_guide.py
"""

import difflib
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples.jsonl")
OUT = os.path.join(HERE, "smell_guide.html")

# Group each smell by HOW natural the injection looks, so the guide can be
# honest about the synthetic-looking ones (magic_number etc.).
CATEGORIES = {
    "deoptimiser": ("De-optimisers",
        "These take a real, well-written code shape and rewrite it as a worse "
        "one. They look the most like genuine bad code because they ARE genuine "
        "bad code -- just applied deliberately."),
    "extender": ("Extenders",
        "These bolt a realistic-looking bad construct onto working code "
        "(an extra parameter, a dead import, a copy-pasted function). Natural "
        "enough to pass as something a tired programmer might actually write."),
    "tripwire": ("Minimal trip-wires",
        "These add the SMALLEST inert construct that trips the detector's rule, "
        "while guaranteeing the code still behaves identically. They look "
        "synthetic on purpose -- a 'real' version of these smells would be "
        "load-bearing, and changing behaviour would break the clean/smelly pair."),
}

# name -> everything needed to explain it. `tool` is the operational definition.
SMELLS = {
    "mutable_default": dict(
        cat="extender", tool="pylint", rule="W0102 (dangerous-default-value)",
        what="A function parameter whose default value is a mutable object, e.g. "
             "<code>def f(x, cache=[])</code>. Python builds that list ONCE and "
             "shares it across every call, so it silently accumulates state "
             "between calls -- a classic source of bugs.",
        how="The injector appends a new parameter <code>cache=[]</code> to the "
            "function's signature.",
        preserved="Yes. The added parameter is never referenced in the body, so "
                  "the function computes exactly what it did before."),
    "broad_except": dict(
        cat="extender", tool="pylint", rule="W0718 / W0703 (broad-except)",
        what="Catching bare <code>Exception</code> swallows every error -- "
             "including bugs you never meant to hide -- which makes failures "
             "silent and debugging painful.",
        how="The injector wraps the whole body (keeping any docstring outside) in "
            "<code>try: &lt;body&gt; except Exception: return None</code>.",
        preserved="Yes, for our tested code. If the original body raised nothing, "
                  "the try runs identically. If it WOULD have raised, the clean "
                  "twin fails its test too -- so the pair stays matched."),
    "dead_code": dict(
        cat="extender", tool="pylint", rule="W0611 (unused-import) or W0612 (unused-variable)",
        what="Code that is present but has no effect: an <b>unused import</b> or an "
             "<b>unused local variable</b>. It clutters the file and misleads "
             "readers about what the function actually needs.",
        how="The injector adds one of two forms (it alternates, so the dataset "
            "contains both): an unused standard-library import "
            "(<code>import os</code>), or an unused local variable assigned from "
            "the first parameter (<code>leftover = nums</code>).",
        preserved="Yes. An unused import has no effect on the output, and the "
                  "unused variable is assigned but never read."),
    "long_parameter_list": dict(
        cat="extender", tool="pylint", rule="R0913 (too-many-arguments, &gt;5)",
        what="A function with too many parameters is hard to call correctly and "
             "usually a sign it is doing too much.",
        how="The injector appends six keyword parameters "
            "<code>extra_0=None ... extra_5=None</code>, pushing the count over "
            "pylint's limit of five.",
        preserved="Yes. The extra parameters default to None and are never used."),
    "deep_nesting": dict(
        cat="tripwire", tool="pylint", rule="R1702 (too-many-nested-blocks, &gt;5)",
        what="Code nested many levels deep (loops inside loops inside ifs...) is "
             "hard to follow and hard to test.",
        how="The injector wraps the body in six nested <code>for _n in range(1)</code> "
            "loops. Each loop runs exactly once, so the real code still executes "
            "a single time -- just buried six levels deep.",
        preserved="Yes. <code>range(1)</code> means every loop iterates once, so "
                  "the body runs exactly as before."),
    "complex_conditional": dict(
        cat="tripwire", tool="pylint", rule="R0916 (too-many-boolean-expressions, &gt;5)",
        what="A single condition chaining many boolean terms (a tangled compound "
             "conditional) is hard to read and easy to get subtly wrong.",
        how="The injector inserts <code>if p and p and p ... (8 times): pass</code> "
            "using the first parameter <code>p</code>. Eight terms clears pylint's "
            "limit of five.",
        preserved="Yes. The <code>if</code> body is just <code>pass</code> and the "
                  "condition has no side effects, so nothing changes."),
    "long_method": dict(
        cat="tripwire", tool="pylint", rule="R0915 (too-many-statements, &gt;50)",
        what="A function with dozens of statements is hard to understand, review, "
             "and test as one unit.",
        how="The injector pads the body with ~55 trivial <code>_acc = _acc + i</code> "
            "statements, pushing it past pylint's 50-statement limit.",
        preserved="Yes. <code>_acc</code> is a throwaway accumulator that the real "
                  "logic never reads."),
    "magic_number": dict(
        cat="tripwire", tool="pylint", rule="R2004 (magic-value-comparison)",
        what="An unexplained literal number in the logic -- the reader has no idea "
             "what it means or why it is that value.",
        how="The injector inserts <code>_ = p == 42</code> (comparing the first "
            "parameter against the bare literal 42).",
        preserved="Yes. The comparison's result is discarded into <code>_</code>.",
        caveat="This is the one that looks most like dead code, and that is "
               "unavoidable. pylint's rule R2004 only fires on a magic number "
               "used in a COMPARISON (it would ignore <code>price * 1.08</code>), "
               "so the injector must produce a comparison. And a REAL magic number "
               "is load-bearing -- it sits inside the logic and drives the output -- "
               "so making it real would change behaviour and break the clean/smelly "
               "pair. The only way to be both detector-confirmed AND "
               "behaviour-preserving is an inert comparison, which necessarily "
               "reads as dead code. It is correctly labelled; it just looks "
               "synthetic by design."),
    "perf_try_in_loop": dict(
        cat="deoptimiser", tool="ruff", rule="PERF203 (try-except-in-loop)",
        what="Putting a <code>try/except</code> inside a loop adds per-iteration "
             "overhead; the handler is usually better hoisted outside the loop.",
        how="The injector finds an existing loop and wraps its body in "
            "<code>try: &lt;body&gt; except ValueError: pass</code>. It uses "
            "<code>ValueError</code> (not <code>Exception</code>) on purpose, so "
            "it does not ALSO count as a broad_except.",
        preserved="Yes. If the loop body raises no ValueError, the try is "
                  "transparent. (Applies only where a loop already exists.)"),
    "inefficient_loop": dict(
        cat="deoptimiser", tool="ruff", rule="PERF401 (manual-list-comprehension)",
        what="An inefficient loop: building a list with a manual <code>append</code> "
             "loop where a list comprehension would be clearer and faster.",
        how="The injector finds a transforming comprehension "
            "<code>xs = [f(t) for t in it]</code> and rewrites it as "
            "<code>xs = []</code> then <code>for t in it: xs.append(f(t))</code>.",
        preserved="Yes. It builds the identical list, just the slow way."),
    "inefficient_copy": dict(
        cat="deoptimiser", tool="ruff", rule="PERF402 (manual-list-copy)",
        what="Copying a list with a manual loop where <code>list(it)</code> or "
             "<code>it.copy()</code> would do.",
        how="The injector finds a list copy (<code>[x for x in it]</code>, "
            "<code>list(it)</code>, <code>it[:]</code>, or <code>it.copy()</code>) "
            "and rewrites it as an empty-list-plus-append loop. A guard "
            "(<code>_provably_list</code>) only allows this when <code>it</code> is "
            "provably a list, so a dict/set copy is never mis-rewritten.",
        preserved="Yes -- and the provably-a-list guard is exactly the fix for an "
                  "earlier bug where a dict copy got turned into a broken list loop."),
    "duplicate_code": dict(
        cat="extender", tool="jscpd", rule="clone ≥ 5 lines / 25 tokens",
        what="Copy-pasted code. Two near-identical blocks mean every future fix "
             "has to be made in two places -- and one will be forgotten.",
        how="The injector appends a renamed copy of the whole function "
            "(<code>foo</code> plus <code>foo_copy</code>), creating a duplicated "
            "block that jscpd detects.",
        preserved="Yes. The copy is a separate, never-called function, so the "
                  "original behaviour is untouched."),
}

ORDER = list(SMELLS)  # keep dict order for the page


def pick_examples(rows, smell, k=2, max_clean_lines=14):
    """Shortest clean twins first, so the diff is easy to read."""
    g = [r for r in rows if r["smell"] == smell]
    g.sort(key=lambda r: len(r["clean_code"].splitlines()))
    short = [r for r in g if len(r["clean_code"].splitlines()) <= max_clean_lines]
    return (short or g)[:k]


def diff_html(clean, smelly):
    """Side-by-side-ish: render smelly with added/changed lines highlighted,
    clean with removed/changed lines highlighted."""
    cl, sl = clean.splitlines(), smelly.splitlines()
    sm = difflib.SequenceMatcher(a=cl, b=sl)
    clean_marks = [""] * len(cl)
    smelly_marks = [""] * len(sl)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                clean_marks[i] = "del"
        if tag in ("replace", "insert"):
            for j in range(j1, j2):
                smelly_marks[j] = "add"

    def block(lines, marks):
        out = []
        for line, mark in zip(lines, marks):
            cls = f' class="{mark}"' if mark else ""
            out.append(f"<div{cls}>{html.escape(line) or '&nbsp;'}</div>")
        return "\n".join(out)

    return block(cl, clean_marks), block(sl, smelly_marks)


def main():
    rows = [json.loads(l) for l in open(SAMPLES, encoding="utf-8")]

    parts = []
    parts.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Injected smells &mdash; reference guide</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         color: #1a1a1a; background: #fafafa; line-height: 1.5; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 32px 28px 80px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  .sub {{ color: #666; margin: 0 0 24px; }}
  h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: .06em;
        color: #555; border-bottom: 2px solid #e2e2e2; padding-bottom: 6px;
        margin: 40px 0 6px; }}
  .catnote {{ color: #666; font-size: 13.5px; margin: 0 0 20px; }}
  .card {{ background: #fff; border: 1px solid #e4e4e4; border-radius: 10px;
           padding: 20px 22px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,.03); }}
  .card h3 {{ margin: 0 0 2px; font-size: 19px; font-family: ui-monospace, Consolas, monospace; }}
  .tool {{ font-size: 12.5px; color: #fff; background: #4b5563; border-radius: 5px;
           padding: 2px 8px; margin-left: 8px; vertical-align: middle; }}
  .tool.pylint {{ background: #2563eb; }} .tool.ruff {{ background: #7c3aed; }}
  .tool.jscpd {{ background: #0891b2; }}
  .rule {{ color: #777; font-size: 13px; font-family: ui-monospace, monospace; margin: 2px 0 14px; }}
  .lbl {{ font-weight: 600; font-size: 12px; text-transform: uppercase;
          letter-spacing: .04em; color: #888; margin: 12px 0 3px; }}
  .txt {{ margin: 2px 0 4px; }}
  code {{ background: #f0f0f2; padding: 1px 5px; border-radius: 4px;
          font-family: ui-monospace, Consolas, monospace; font-size: 13px; }}
  .keep {{ color: #15803d; font-weight: 600; }}
  .caveat {{ background: #fff8e6; border-left: 4px solid #f0b429; padding: 10px 14px;
             border-radius: 0 6px 6px 0; margin: 12px 0 2px; font-size: 14px; }}
  .ex {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 8px 0; }}
  .ex .col {{ border: 1px solid #e4e4e4; border-radius: 8px; overflow: hidden; }}
  .ex .head {{ font-size: 12px; font-weight: 600; padding: 5px 12px; color: #fff; }}
  .ex .clean .head {{ background: #4b5563; }}
  .ex .smelly .head {{ background: #b91c1c; }}
  .ex pre {{ margin: 0; padding: 10px 12px; font-family: ui-monospace, Consolas, monospace;
             font-size: 12.5px; overflow-x: auto; background: #fff; }}
  .ex pre div {{ white-space: pre; padding: 0 4px; border-radius: 3px; }}
  .ex pre .add {{ background: #dcfce7; }}
  .ex pre .del {{ background: #fee2e2; }}
  .src {{ font-size: 11.5px; color: #999; padding: 3px 12px; border-top: 1px solid #eee; }}
  .toc {{ background: #fff; border: 1px solid #e4e4e4; border-radius: 10px; padding: 14px 20px; }}
  .toc a {{ color: #2563eb; text-decoration: none; font-family: ui-monospace, monospace;
            font-size: 13px; }}
  .toc a:hover {{ text-decoration: underline; }}
  .toc span {{ display: inline-block; margin: 3px 14px 3px 0; }}
  @media (max-width: 720px) {{ .ex {{ grid-template-columns: 1fr; }} }}
</style></head><body><div class="wrap">
<h1>Injected smells &mdash; reference guide</h1>
<p class="sub">The 12 code smells in the benchmark, how each is injected, and real
clean&rarr;smelly examples from the dataset. Green = added by the injector, red = removed.</p>
<div class="toc"><b>Jump to:</b><br>""")

    # table of contents
    for s in ORDER:
        parts.append(f'<span><a href="#{s}">{s}</a></span>')
    parts.append("</div>")

    # a short framing note
    parts.append("""<div class="card" style="margin-top:22px">
    <b>How to read this.</b> Every smell is defined <i>operationally</i>: it is
    whatever its detector flags (pylint, ruff, or jscpd). Every injection is
    <b>behaviour-preserving</b> &mdash; the clean and smelly halves pass exactly the
    same tests &mdash; which is what makes each a valid labelled pair. The three
    groups below differ only in how natural the result looks, which is itself a
    consequence of those two rules.</div>""")

    # group by category, in a sensible order
    for cat in ("deoptimiser", "extender", "tripwire"):
        title, note = CATEGORIES[cat]
        parts.append(f"<h2>{title}</h2><p class='catnote'>{note}</p>")
        for s in ORDER:
            info = SMELLS[s]
            if info["cat"] != cat:
                continue
            parts.append(f'<div class="card" id="{s}">')
            parts.append(f'<h3>{s}<span class="tool {info["tool"]}">{info["tool"]}</span></h3>')
            parts.append(f'<div class="rule">confirmed by: {info["rule"]}</div>')
            parts.append(f'<div class="lbl">What it is</div><div class="txt">{info["what"]}</div>')
            parts.append(f'<div class="lbl">How it is injected</div><div class="txt">{info["how"]}</div>')
            parts.append(f'<div class="lbl">Behaviour preserved?</div>'
                         f'<div class="txt"><span class="keep">{info["preserved"]}</span></div>')
            if info.get("caveat"):
                parts.append(f'<div class="caveat"><b>Worth knowing:</b> {info["caveat"]}</div>')

            for r in pick_examples(rows, s):
                cl_html, sl_html = diff_html(r["clean_code"], r["smelly_code"])
                parts.append('<div class="ex">')
                parts.append(f'<div class="col clean"><div class="head">clean</div><pre>{cl_html}</pre>'
                             f'<div class="src">{html.escape(r["source_task"])}</div></div>')
                parts.append(f'<div class="col smelly"><div class="head">smelly &mdash; {s}</div>'
                             f'<pre>{sl_html}</pre>'
                             f'<div class="src">{html.escape(r["id"])}</div></div>')
                parts.append('</div>')
            parts.append("</div>")

    parts.append("</div></body></html>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
