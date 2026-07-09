"""
Real-world validation layer (the benchmark's second source).

run_panel.py evaluates the measures on the INJECTED benchmark -- synthetic
clean/smelly pairs where we control exactly which smell is present. This script
is the complement the project scope calls for: it runs the same structural
measures on REAL, naturally-occurring labelled code (reused.jsonl, drawn from
CodeSmellData 1.0 / 2.0 and PySmell) to check whether the findings generalise
beyond the artificial samples.

Two limits are inherent to real code and are handled honestly here:
  * No clean twin. Real smelly code has no matched clean version, so the
    reference-based similarity measures (BLEU, CodeBLEU, ...) cannot run -- only
    the reference-free STRUCTURAL measures apply.
  * Clean negatives are scarce. The reused data has clean examples for only two
    smells (long_method, long_parameter_list, from PySmell's human labels). So a
    real smelly-vs-clean "detection strength" is computed for those two; for the
    other five smells we report the structural profile of the real smelly code
    (there is no clean group to compare against).

Where a real detection strength can be computed it is shown next to the injected
one, so "does the measure separate smelly from clean on synthetic pairs, and does
it still separate them on real code?" can be read directly.

Detection strength here is the UNPAIRED Cohen's d (the injected version is
paired): (mean_smelly - mean_clean) / pooled_std, oriented so positive = worse,
capped at +/-5.

Run:  python eval_tool/run_realworld.py            (auto-switches to the venv)
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
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

import csv
import json
import math
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
REUSED = os.path.join(ROOT, "smell_injection", "reused.jsonl")
INJECTED_CSV = os.path.join(HERE, "panel_results.csv")
OUT_CSV = os.path.join(HERE, "realworld_results.csv")
OUT_HTML = os.path.join(HERE, "realworld_report.html")
D_CAP = 5.0

from measures import PANEL                                    # noqa: E402
STRUCT = [m for m in PANEL if not m.needs_ref]                # reference-free only


def load_reused():
    rows = [json.loads(l) for l in open(REUSED, encoding="utf-8") if l.strip()]
    smelly, clean = defaultdict(list), defaultdict(list)
    for r in rows:
        (smelly if r.get("label") == "yes" else clean)[r["smell"]].append(r["code"])
    return smelly, clean


def load_injected_d():
    """(smell, measure) -> injected paired cohen_d, from panel_results.csv."""
    out = {}
    if not os.path.exists(INJECTED_CSV):
        return out
    for row in csv.DictReader(open(INJECTED_CSV, encoding="utf-8")):
        try:
            out[(row["smell"], row["measure"])] = float(row["cohen_d"])
        except (KeyError, ValueError):
            pass
    return out


def measure_all(codes, m):
    return [v for v in (m.fn(c) for c in codes) if v is not None]


def cohens_d(smelly, clean, worse):
    """Unpaired Cohen's d, oriented positive = worse, capped at +/-D_CAP."""
    if len(smelly) < 2 or len(clean) < 2:
        return float("nan")
    ms, mc = statistics.fmean(smelly), statistics.fmean(clean)
    n1, n2 = len(smelly), len(clean)
    s1, s2 = statistics.stdev(smelly), statistics.stdev(clean)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        d = 0.0 if ms == mc else math.copysign(D_CAP, ms - mc)
    else:
        d = (ms - mc) / pooled
    d = d if worse == "up" else -d
    return max(-D_CAP, min(D_CAP, d))


def main():
    smelly, clean = load_reused()
    injected_d = load_injected_d()
    smells = sorted(smelly)
    with_clean = sorted(s for s in smells if clean.get(s))
    without_clean = [s for s in smells if not clean.get(s)]

    print(f"loaded reused.jsonl: {sum(len(v) for v in smelly.values())} smelly + "
          f"{sum(len(v) for v in clean.values())} clean across {len(smells)} smells")
    print(f"structural measures: {', '.join(m.name for m in STRUCT)} "
          f"(similarity measures need a clean twin -> not applicable)\n")

    # precompute measure values per (smell, group)
    sm_vals = {(s, m.name): measure_all(smelly[s], m) for s in smells for m in STRUCT}
    cl_vals = {(s, m.name): measure_all(clean[s], m) for s in with_clean for m in STRUCT}

    # ---- real detection strength + generalisation (smells with clean negatives) ----
    print("Real detection strength (real smelly vs real clean), with the injected value for comparison:")
    print(f"  {'smell / measure':30}{'injected d':>12}{'real d':>10}{'clean med':>11}{'smelly med':>12}")
    rows_out = []
    for s in with_clean:
        n_s, n_c = len(smelly[s]), len(clean[s])
        print(f"  {s}  (real: {n_s} smelly, {n_c} clean)")
        for m in STRUCT:
            sv, cv = sm_vals[(s, m.name)], cl_vals[(s, m.name)]
            d = cohens_d(sv, cv, m.worse)
            inj = injected_d.get((s, m.name), float("nan"))
            cm = statistics.median(cv) if cv else float("nan")
            sm = statistics.median(sv) if sv else float("nan")
            print(f"    {m.name:26}{inj:>12.2f}{d:>10.2f}{cm:>11.1f}{sm:>12.1f}")
            rows_out.append([s, m.name, n_s, n_c, f"{inj:.3f}", f"{d:.3f}",
                             f"{cm:.3f}", f"{sm:.3f}"])

    # ---- structural profile of real smelly code (smells without clean negatives) ----
    print("\nStructural profile of real smelly code (no clean group in the reused data):")
    print(f"  {'smell / measure':30}{'n':>7}{'median':>10}{'mean':>10}")
    for s in without_clean:
        print(f"  {s}  ({len(smelly[s])} smelly)")
        for m in STRUCT:
            sv = sm_vals[(s, m.name)]
            med = statistics.median(sv) if sv else float("nan")
            mn = statistics.fmean(sv) if sv else float("nan")
            print(f"    {m.name:26}{len(sv):>7}{med:>10.1f}{mn:>10.1f}")
            rows_out.append([s, m.name, len(smelly[s]), 0, "", "",
                             f"{med:.3f}", f"{mn:.3f}"])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smell", "measure", "n_smelly", "n_clean",
                    "injected_cohen_d", "real_cohen_d", "clean_median", "smelly_median"])
        w.writerows(rows_out)

    write_html(smells, with_clean, without_clean, smelly, clean, sm_vals, cl_vals, injected_d)
    print(f"\nwrote {os.path.basename(OUT_CSV)} and {os.path.basename(OUT_HTML)}")


def _colour(d):
    """Match the injected report's heatmap feel: stronger = deeper."""
    if d != d:  # nan
        return "#f5f5f5"
    a = min(abs(d) / 3.0, 1.0)
    return f"rgba(185,28,28,{0.10 + 0.6 * a:.2f})"


def write_html(smells, with_clean, without_clean, smelly, clean, sm_vals, cl_vals, injected_d):
    import statistics as st
    p = []
    p.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Real-world validation &mdash; structural measures</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a;line-height:1.5}}
 .wrap{{max-width:1000px;margin:0 auto;padding:32px 26px 80px}}
 h1{{font-size:24px;margin:0 0 2px}} .sub{{color:#666;margin:0 0 22px;font-size:14px}}
 h2{{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:#555;
     border-bottom:2px solid #e2e2e2;padding-bottom:6px;margin:34px 0 6px}}
 .note{{color:#777;font-size:12.5px;margin:0 0 12px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff;
        border:1px solid #e4e4e4;border-radius:8px;overflow:hidden;margin-bottom:8px}}
 th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #eee}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f4f4f6;font-weight:600}}
 code{{background:#f0f0f2;padding:1px 5px;border-radius:4px;font-size:12px}}
 .prov{{font-size:12px;color:#888}}
</style></head><body><div class="wrap">
<h1>Real-world validation &mdash; structural measures</h1>
<p class="sub">The benchmark's second source: the structural measures run on real,
naturally-occurring labelled code (CodeSmellData 1.0 / 2.0 and PySmell), to check
whether the findings generalise beyond the injected samples. Similarity measures
need a clean twin and so do not apply to real code.</p>""")

    # generalisation table
    p.append("<h2>Real vs injected detection strength</h2>")
    p.append('<p class="note">Real smelly code vs real clean code (unpaired Cohen&rsquo;s d), '
             'next to the injected value (paired). Only <code>long_method</code> and '
             '<code>long_parameter_list</code> have real clean negatives. Positive = worse; capped at &plusmn;5.</p>')
    for s in with_clean:
        p.append(f'<p style="margin:14px 0 4px"><b>{s}</b> '
                 f'<span class="prov">({len(smelly[s])} real smelly, {len(clean[s])} real clean)</span></p>')
        p.append("<table><tr><th>measure</th><th>injected d (paired)</th><th>real d (unpaired)</th>"
                 "<th>real clean median</th><th>real smelly median</th></tr>")
        for m in STRUCT:
            sv, cv = sm_vals[(s, m.name)], cl_vals[(s, m.name)]
            d = cohens_d(sv, cv, m.worse)
            inj = injected_d.get((s, m.name), float("nan"))
            cm = st.median(cv) if cv else float("nan")
            sm = st.median(sv) if sv else float("nan")
            inj_s = "" if inj != inj else f"{inj:.2f}"
            d_s = "" if d != d else f"{d:.2f}"
            p.append(f'<tr><td>{m.name}</td><td>{inj_s}</td>'
                     f'<td style="background:{_colour(d)}">{d_s}</td>'
                     f'<td>{"" if cm != cm else f"{cm:.1f}"}</td>'
                     f'<td>{"" if sm != sm else f"{sm:.1f}"}</td></tr>')
        p.append("</table>")

    # structural profile for smells without negatives
    p.append("<h2>Real smelly-code structural profile (no clean group)</h2>")
    p.append('<p class="note">For these five smells the reused data has smelly examples but no clean '
             'ones, so no detection strength can be computed &mdash; only the profile of the real smelly code.</p>')
    p.append("<table><tr><th>smell</th><th>n</th>"
             + "".join(f"<th>{m.name}</th>" for m in STRUCT) + "</tr>")
    for s in without_clean:
        cells = "".join(
            f'<td>{("" if not sm_vals[(s, m.name)] else f"{st.median(sm_vals[(s, m.name)]):.1f}")}</td>'
            for m in STRUCT)
        p.append(f"<tr><td><code>{s}</code></td><td>{len(smelly[s])}</td>{cells}</tr>")
    p.append("</table><p class='note'>Values are medians of the real smelly code.</p>")

    # provenance
    p.append("<h2>Data provenance</h2><table><tr><th>smell</th><th>real smelly</th><th>real clean</th></tr>")
    for s in smells:
        p.append(f"<tr><td><code>{s}</code></td><td>{len(smelly[s])}</td><td>{len(clean.get(s, []))}</td></tr>")
    p.append("</table>")
    p.append('<p class="prov">Sources: CodeSmellData 1.0, CodeSmellData 2.0 (Pylint-labelled real '
             'GitHub methods), and PySmell (human-labelled; the only source of clean negatives).</p>')

    p.append("</div></body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


if __name__ == "__main__":
    main()
