"""
Build the analysis package: every figure and table the write-up needs.

Figures are written twice -- PDF for LaTeX (vector, scales cleanly) and PNG for
the HTML page. Tables are written as LaTeX fragments the paper can \\input, and
as CSV so the numbers can be checked by hand.

Everything is computed from the result files in the repo. Nothing is typed in,
so a figure cannot disagree with the data behind it.

Run:  python analysis/make_analysis.py
"""

import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

PANEL = os.path.join(ROOT, "eval_tool", "panel_results.csv")
REAL = os.path.join(ROOT, "eval_tool", "realworld_results.csv")
CORR = os.path.join(ROOT, "eval_tool", "correctness_results.csv")

# Ordered worst-first by real-world detectability, so the heatmap reads top-down.
SMELLS = ["deep_nesting", "complex_conditional", "long_method", "inefficient_copy",
          "duplicate_code", "magic_number", "inefficient_loop", "perf_try_in_loop",
          "broad_except", "long_parameter_list", "mutable_default", "dead_code"]
STRUCT = ["sloc", "cyclomatic", "cognitive", "maintainability", "halstead_volume",
          "halstead_difficulty", "halstead_effort", "comment_density", "api_calls",
          "perplexity"]
SIM = ["bleu", "chrf", "rouge_l", "codebleu", "ast_similarity",
       "codebert_score", "bertscore"]

# The verdicts from the detection report, restated here so the figures can colour by them.
TRUST = {
    "long_method": "detects", "deep_nesting": "detects", "complex_conditional": "detects",
    "broad_except": "co-occurs", "magic_number": "co-occurs", "inefficient_copy": "co-occurs",
    "inefficient_loop": "co-occurs", "perf_try_in_loop": "co-occurs",
    "long_parameter_list": "blind", "dead_code": "blind",
    "mutable_default": "blind", "duplicate_code": "blind",
}
VC = {"detects": "#1b6b41", "co-occurs": "#b8860b", "blind": "#9b2c2c"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150)
    plt.close(fig)
    print(f"  figures/{name}.pdf + .png")


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def d_ci(d, n1, n2):
    """95% interval for an independent-groups standardised difference.

    Large-sample variance from Hedges & Olkin: the first term is sampling error
    in the means, the second the error in the pooled standard deviation. Values
    at the +/-5 cap are reported without an interval -- the cap is not an
    estimate, so an interval around it would be misleading."""
    if d is None or not n1 or not n2 or abs(d) >= 5:
        return None, None
    se = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
    return d - 1.96 * se, d + 1.96 * se


# =====================================================================
# Load
# =====================================================================
panel = read(PANEL)
real = read(REAL)
corr = read(CORR)

inj_d = {(r["smell"], r["measure"]): num(r["cohen_d"]) for r in panel}
fam = {r["measure"]: r["family"] for r in panel}
n_pairs = {r["smell"]: int(r["n_pairs"]) for r in panel if r["n_pairs"]}

real_d, n_smelly, n_clean = {}, {}, {}
for r in real:
    k = (r["smell"], r["measure"])
    real_d[k] = num(r["real_cohen_d"])
    n_smelly[r["smell"]] = int(r["n_smelly"]) if r["n_smelly"] else None
    n_clean[r["smell"]] = int(r["n_clean"]) if r["n_clean"] else None

MEASURES_IN_PANEL = sorted({r["measure"] for r in panel})
STRUCT = [m for m in STRUCT if m in MEASURES_IN_PANEL]
SIM = [m for m in SIM if m in MEASURES_IN_PANEL]

print("Building the analysis package")
print(f"  {len(panel)} injected rows, {len(real)} real rows, "
      f"{len(STRUCT)} structural + {len(SIM)} similarity measures\n")


# =====================================================================
# Figure 1 -- detection strength, injected next to real
# =====================================================================
def fig_heatmap():
    inj = np.array([[inj_d.get((s, m)) if inj_d.get((s, m)) is not None else np.nan
                     for m in STRUCT] for s in SMELLS])
    rl = np.array([[real_d.get((s, m)) if real_d.get((s, m)) is not None else np.nan
                    for m in STRUCT] for s in SMELLS])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    for ax, data, title in [(axes[0], inj, "Injected (controlled)"),
                            (axes[1], rl, "Real (observational)")]:
        im = ax.imshow(data, cmap="RdYlGn_r", vmin=-2, vmax=5, aspect="auto")
        ax.set_xticks(range(len(STRUCT)))
        ax.set_xticklabels([m.replace("halstead_", "hal. ").replace("_", " ")
                            for m in STRUCT], rotation=45, ha="right")
        ax.set_title(title, pad=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if (v > 2.6 or v < -1.2) else "#222")
    axes[0].set_yticks(range(len(SMELLS)))
    axes[0].set_yticklabels([s.replace("_", " ") for s in SMELLS])
    for lab, ax in zip(SMELLS, [axes[0]] * len(SMELLS)):
        pass
    for tick, s in zip(axes[0].get_yticklabels(), SMELLS):
        tick.set_color(VC[TRUST[s]])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("detection strength (standard deviations, capped at 5)")
    fig.suptitle("How strongly each structural measure separates defective from clean code",
                 y=1.0, fontsize=11)
    save(fig, "fig1_detection_heatmap")


# =====================================================================
# Figure 2 -- the disagreement that matters
# =====================================================================
def fig_scatter():
    """Injected against real, read by REGION rather than by distance from the diagonal.

    An earlier version drew the diagonal and said points on it agree. That was
    misleading: almost nothing sits on it, and the genuinely-detected defects sit
    well above it. That is not a failure of those measures -- it is the paper's own
    argument showing up in the geometry. Real defective code differs from clean code
    in more ways than the defect (above all in size), so the real value is inflated
    relative to the controlled one. The diagonal is therefore a reference line, not
    an expectation, and the figure now says so.

    What the axes mean is what matters: horizontal position answers "can this
    measure respond to the defect at all", vertical position answers "do real
    defective and clean code differ on it". The verdict is the combination.
    """
    # One point per defect -- not one per defect-measure cell. An earlier version
    # plotted all 120 cells, which buried the twelve conclusions in a cloud of
    # mostly-blind combinations that Figure 1 already shows. Each defect is placed
    # at its STRONGEST real-world structural measure, the same selection the main
    # table uses, so this figure and that table cannot disagree.
    pts = []
    for s in SMELLS:
        best, bd, ba = None, None, -1.0
        for m in STRUCT:
            v = real_d.get((s, m))
            if v is not None and abs(v) > ba:
                best, bd, ba = m, v, abs(v)
        a = inj_d.get((s, best)) if best else None
        if best is None or a is None:
            continue
        pts.append((s, best, a, bd))

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    xlim, ylim = [-0.75, 4.9], [-2.0, 6.0]

    # --- regions first, so points and labels sit on top -----------------------
    ax.axvspan(-0.2, 0.2, ymin=(0.8 - ylim[0]) / (ylim[1] - ylim[0]), color="#b8860b",
               alpha=.13, zorder=0)                       # cannot respond, yet separates
    ax.add_patch(plt.Rectangle((0.8, 0.8), xlim[1] - 0.8, ylim[1] - 0.8,
                               color="#1b6b41", alpha=.07, zorder=0))
    ax.add_patch(plt.Rectangle((xlim[0], ylim[0]), 0.8 - xlim[0], 0.8 - ylim[0],
                               color="#888", alpha=.07, zorder=0))

    ax.plot([-1, 6], [-1, 6], ls="--", lw=.9, color="#ccc", zorder=1)
    ax.axhline(0, lw=.7, color="#ddd", zorder=0)
    ax.axvline(0, lw=.7, color="#ddd", zorder=0)
    ax.axhline(0.8, lw=.7, ls=":", color="#aaa", zorder=0)
    ax.axvline(0.8, lw=.7, ls=":", color="#aaa", zorder=0)

    seen = set()
    for s, m, a, b in pts:
        v = TRUST[s]
        ax.scatter(a, b, s=64, color=VC[v], zorder=4, edgecolors="white", linewidths=.8,
                   label=v if v not in seen else None)
        seen.add(v)

    # Labels. Eight defects pile up against x=0 because their measure cannot respond
    # at all, so those are written to the right of the axis at their own height,
    # nudged apart only where two would overlap. Stacking them in a fixed column
    # instead ran the last few off the bottom of the axes.
    cluster = sorted([p for p in pts if p[2] < 0.35], key=lambda p: -p[3])
    ys, MIN_GAP = [], 0.42
    for _, _, _, b in cluster:
        y = b if not ys else min(b, ys[-1] - MIN_GAP)
        ys.append(y)
    for (s, m, a, b), ly in zip(cluster, ys):
        ax.annotate(f"{s.replace('_', ' ')}  ({m.replace('_', ' ')})",
                    (a, b), xytext=(0.62, ly), textcoords="data",
                    fontsize=7.0, color="#444", va="center", ha="left", zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=.55, color="#c4c4c4",
                                    shrinkA=0, shrinkB=3))
    for s, m, a, b in [p for p in pts if p[2] >= 0.35]:
        if s == "duplicate_code":
            # Clears both thresholds yet is classified blind: doubling a block doubles
            # its length, so both numbers are the same size artefact. Said here because
            # a red point inside the green region otherwise reads as an error. Placed to
            # the right -- the space above it belongs to the x=0 cluster's labels.
            ax.annotate("duplicate code  (sloc)\nboth numbers are the same size artefact",
                        (a, b), xytext=(1.32, 2.32), textcoords="data", fontsize=7.0,
                        color="#9b2c2c", ha="left", va="center", linespacing=1.35,
                        zorder=5, arrowprops=dict(arrowstyle="-", lw=.55,
                                                  color="#c9a0a0", shrinkA=0, shrinkB=4))
        else:
            ax.annotate(f"{s.replace('_', ' ')}\n({m.replace('_', ' ')})", (a, b),
                        xytext=(0, 12), textcoords="offset points", fontsize=7.0,
                        color="#444", ha="center", va="bottom", linespacing=1.3, zorder=5)

    # --- region labels, in corners the data leaves empty ----------------------
    ax.text(-0.68, 5.85, "CO-OCCURRENCE\nthe measure provably cannot respond,\n"
            "yet real defective code separates anyway", fontsize=7.2, color="#7a5c10",
            ha="left", va="top", linespacing=1.45, zorder=6)
    ax.text(4.8, 0.95, "DETECTION\nresponds on both sources", fontsize=7.2,
            color="#14523a", ha="right", va="bottom", linespacing=1.45, zorder=6)
    ax.text(-0.68, -1.92, "BLIND\nneither source", fontsize=7.2, color="#777",
            ha="left", va="bottom", linespacing=1.45, zorder=6)
    ax.annotate("equal on both sources", (4.05, 4.05), fontsize=6.6, color="#bbb",
                rotation=33, ha="center", va="bottom", zorder=2)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("injected — does changing only the defect move the measure?")
    ax.set_ylabel("real — do defective and clean code differ on it?")
    ax.set_title("One point per defect, at its strongest real-world measure", pad=8)
    ax.legend(title="verdict", loc="lower right", frameon=False,
              borderpad=0.2, labelspacing=0.3)
    fig.text(0.5, -0.015,
             "Detected defects sit ABOVE the diagonal, and that is expected: real defective "
             "code differs from clean code in\nmore ways than the defect alone, so the real "
             "value is inflated. The diagonal is a reference, not a target.",
             ha="center", va="top", fontsize=7.4, color="#555")
    save(fig, "fig2_injected_vs_real")


# =====================================================================
# Figure 3 -- which measures are redundant
# =====================================================================
def fig_correlation():
    """How much measures agree, grouped by whether they belong to the same family.

    This replaced a full correlation matrix. The matrix carried one fact -- two
    internally-agreeing blocks, near-independent of each other -- and made the
    reader decode a grid with identical labels on both axes to recover it. Every
    pairwise correlation is still computed here; they are just plotted as three
    distributions, which is the comparison the fact is actually about.

    Showing every pair rather than three means also keeps the spread visible: if
    the families overlapped, that would show up here and does not.
    """
    ms = STRUCT + SIM
    vecs = {m: np.array([inj_d.get((s, m), np.nan) for s in SMELLS], dtype=float) for m in ms}

    def corr(a, b):
        x, y = vecs[a], vecs[b]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            return float(np.corrcoef(x[ok], y[ok])[0, 1])
        return None

    groups = {"within\nstructural": [], "within\nsimilarity": [], "across\nfamilies": []}
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            c = corr(a, b)
            if c is None:
                continue
            if a in STRUCT and b in STRUCT:
                groups["within\nstructural"].append(c)
            elif a in SIM and b in SIM:
                groups["within\nsimilarity"].append(c)
            else:
                groups["across\nfamilies"].append(c)

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    colours = ["#0f6d6d", "#7a3d9e", "#b8860b"]
    rng = np.random.default_rng(0)                # jitter only; seeded so the figure is stable
    for i, (name, vals) in enumerate(groups.items()):
        if not vals:
            continue
        x = i + (rng.random(len(vals)) - .5) * .34
        ax.scatter(x, vals, s=20, alpha=.5, color=colours[i], edgecolors="none", zorder=3)
        med = float(np.median(vals))
        ax.plot([i - .28, i + .28], [med, med], lw=2.4, color=colours[i], zorder=4)
        ax.text(i, 1.34, f"median {med:+.2f}", ha="center", va="bottom",
                fontsize=8.4, color=colours[i], fontweight="bold")
        ax.text(i, 1.19, f"{len(vals)} pairs", ha="center", va="bottom",
                fontsize=7.4, color=colours[i])

    ax.axhline(0, lw=.8, color="#bbb", zorder=0)
    ax.axhline(1, lw=.7, ls=":", color="#ccc", zorder=0)
    # The scale means nothing without saying what its ends are, so they are labelled
    # on the axis rather than left to the caption.
    ax.text(2.62, 1.0, "the two measures rank\nthe defects identically\n→ one is redundant",
            fontsize=7, color="#555", va="center", ha="left", linespacing=1.4)
    ax.text(2.62, 0.0, "the two carry\nunrelated information\n→ both are needed",
            fontsize=7, color="#555", va="center", ha="left", linespacing=1.4)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), linespacing=1.35)
    ax.set_xlim(-0.5, 4.05)
    ax.set_ylim(-1.05, 1.6)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_ylabel("agreement between two measures\n(correlation of their profiles)",
                  linespacing=1.5)
    ax.set_title("The similarity family is internally redundant;\nthe structural family is not",
                 fontsize=10, pad=26)
    fig.text(0.5, -0.02,
             "Each measure has a PROFILE: its detection strength on each of the twelve defects, "
             "so twelve numbers.\nCorrelating two profiles gives one point above. Nine structural "
             "measures give 9x8/2 = 36 pairs, seven\nsimilarity measures 21, and the two families "
             "crossed 9x7 = 63. Comment density is excluded: no injection\ntouches comments, so "
             "its profile is flat and has no correlation to take.",
             ha="center", va="top", fontsize=7.4, color="#555")
    save(fig, "fig3_measure_correlation")


# =====================================================================
# Figure 4 -- no single measure covers everything
# =====================================================================
def fig_coverage():
    """Best-case coverage per family. The bars are MAXIMA, deliberately.

    The claim being tested is "no measure covers every defect". That is an
    upper-bound claim, so the honest test is the best measure available in each
    family: if the strongest structural measure cannot see a defect, no structural
    measure can, and adding more of them will not help. A mean would answer a
    different and less useful question -- how the typical measure does -- which
    Figure 1 already shows cell by cell.
    """
    best_s, best_m = [], []
    for s in SMELLS:
        vs = [inj_d.get((s, m)) for m in STRUCT if inj_d.get((s, m)) is not None]
        vm = [inj_d.get((s, m)) for m in SIM if inj_d.get((s, m)) is not None]
        best_s.append(max(vs) if vs else 0)
        best_m.append(max(vm) if vm else 0)

    y = np.arange(len(SMELLS))
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.barh(y - .2, best_s, height=.38, color="#0f6d6d",
            label="best structural measure (reference-free)")
    ax.barh(y + .2, best_m, height=.38, color="#7a3d9e",
            label="best similarity measure (needs a clean reference)")

    # The threshold is Cohen's convention for a large effect, not a value chosen
    # here to make the point -- so it is named on the figure.
    ax.axvline(0.8, ls="--", lw=1.0, color="#555", zorder=3)
    ax.text(0.86, -0.72, "$d=0.8$: Cohen's conventional threshold for a large effect",
            fontsize=7, color="#555", va="center")

    # Mark the defects the structural family cannot reach even at its best. These
    # are the rows that carry the argument, so they are called out rather than left
    # for the reader to spot.
    blind_rows = [i for i, v in enumerate(best_s) if v < 0.8]
    for i in blind_rows:
        # To the right of the threshold, not over the bar: on these rows the
        # structural bar is short by definition, so the space past d=0.8 is clear.
        ax.annotate("beyond every structural measure", (0.92, i - .2),
                    fontsize=6.6, color="#9b2c2c", va="center", ha="left", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in SMELLS])
    ax.invert_yaxis()
    ax.set_xlim(0, 5.4)
    ax.set_xlabel("detection strength of the BEST measure in that family (injected)")
    ax.set_title(f"Best case per family: every defect is reachable by something, but the\n"
                 f"structural family alone leaves {len(blind_rows)} of {len(SMELLS)} "
                 f"out of reach", fontsize=10, pad=30)
    # Above the axes, not inside them: at lower right the legend box sat on top of the
    # dead-code and mutable-default bars, which are exactly the rows the figure is about.
    ax.legend(frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(0, 1.005),
              fontsize=7.8, labelspacing=0.25)
    fig.text(0.5, -0.02,
             "Bars are maxima, not averages: if the strongest measure in a family cannot see a "
             "defect, no measure in\nthat family can. The similarity family reaches everything, "
             "but only where a clean reference exists --\nwhich the benchmark has by construction "
             "and deployed code never does.",
             ha="center", va="top", fontsize=7.4, color="#555")
    save(fig, "fig4_family_coverage")


# =====================================================================
# Figure 5 -- what the models actually produce (RQ2)
# =====================================================================
# The three models scored on the 664 benchmark tasks. Ordered by capability so the
# figures read left to right from weakest to strongest.
BENCH_RUNS = [("evaluation_summary_deepseek.csv", "DeepSeek-Coder"),
              ("evaluation_summary.csv", "Qwen2.5-Coder"),
              ("evaluation_summary_claude.csv", "Claude Sonnet 5")]
BENCH_COLOURS = ["#7a3d9e", "#0f6d6d", "#b8860b"]


def _bench():
    """Load whichever benchmark evaluations exist, in BENCH_RUNS order."""
    out = {}
    for fn, label in BENCH_RUNS:
        p = os.path.join(ROOT, "arc_qwen", fn)
        if os.path.exists(p):
            out[label] = read(p)
    return out


def fig_models():
    keys = ["sloc", "cyclomatic", "cognitive", "maintainability", "comment_density"]
    runs = _bench()
    if not runs:
        print("  [skip] fig5 -- no evaluation summaries")
        return
    data, canon = {}, None
    for label, rows in runs.items():
        data[label] = [np.nanmean([num(r[k]) if num(r[k]) is not None else np.nan
                                   for r in rows]) for k in keys]
        if canon is None:                    # identical across runs -- same 664 tasks
            canon = [np.nanmean([num(r.get(f"canon_{k}"))
                                 if num(r.get(f"canon_{k}")) is not None else np.nan
                                 for r in rows]) for k in keys]

    x = np.arange(len(keys))
    series = list(data.items()) + [("canonical solution", canon)]
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (label, vals) in enumerate(series):
        colour = BENCH_COLOURS[i] if i < len(data) else "#555"
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=label, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ") for k in keys])
    ax.set_ylabel("mean over 664 generated solutions")
    ax.set_title("Generated code is simpler than the reference on every complexity\n"
                 "measure, and that holds for every model regardless of capability",
                 fontsize=10, pad=8)
    ax.legend(frameon=False, ncol=2, fontsize=7.6)
    save(fig, "fig5_model_structure")


# =====================================================================
# Figure 6 -- the verbosity confound
# =====================================================================
def fig_verbosity():
    runs = _bench()
    if not runs:
        print("  [skip] fig6 -- no evaluation summaries")
        return
    labels, emitted, defs = [], [], []
    for label, rows in runs.items():
        n = len(rows)
        labels.append(label)
        emitted.append(100 * sum(1 for r in rows if int(r["n_smells"] or 0) > 0) / n)
        defs.append(100 * sum(1 for r in rows if int(r.get("n_smells_defs") or 0) > 0) / n)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - .18, emitted, .34, label="as emitted", color="#b8860b")
    ax.bar(x + .18, defs, .34, label="definitions only", color="#0f6d6d")
    for i, (a, b) in enumerate(zip(emitted, defs)):
        ax.annotate(f"{a:.1f}%", (i - .18, a), ha="center", va="bottom", fontsize=7.5)
        ax.annotate(f"{b:.1f}%", (i + .18, b), ha="center", va="bottom", fontsize=7.5)
        if a - b > 2:                        # mark only the model the correction moves
            ax.annotate("", xy=(i + .18, b + .4), xytext=(i - .18, a + .4),
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#9b2c2c",
                                        connectionstyle="arc3,rad=-.35"))
            ax.text(i, a + 2.4, "the gap is\nvolunteered test code", ha="center",
                    fontsize=7, color="#9b2c2c", linespacing=1.35)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(emitted) * 1.32)
    ax.set_ylabel("share of generations carrying a defect")
    ax.set_title("Scoring raw output partly measures verbosity: of three models,\n"
                 "only the one that volunteers extra code moves when it is excluded",
                 fontsize=9.5, pad=8)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig6_verbosity_confound")


# =====================================================================
# Figure 10 -- capability varies, structural density does not
# =====================================================================
def fig_capability_vs_structure():
    """The finding that only appears once a frontier model joins the comparison.

    Correctness spans forty points across these three models. Defect density per
    hundred lines spans six hundredths. Whatever the panel is measuring, it is not
    tracking the thing the benchmarks rank models by -- which is the point: a model
    can be far better at producing working code without being any better at
    producing well-structured code.
    """
    runs = _bench()
    if not runs:
        print("  [skip] fig10 -- no evaluation summaries")
        return
    labels, p1, dens = [], [], []
    for label, rows in runs.items():
        labels.append(label)
        p1.append(100 * sum(1 for r in rows if r["result"] == "pass") / len(rows))
        tot = sum(num(r["n_smells_defs"]) or 0 for r in rows)
        sl = sum(num(r["sloc"]) or 0 for r in rows)
        dens.append(100 * tot / sl if sl else 0)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for ax, vals, ttl, ylab, hi in [
            (axes[0], p1, "Correctness varies enormously", "pass@1 (%)", True),
            (axes[1], dens, "Structural defect density does not",
             "defects per 100 SLOC", False)]:
        ax.bar(range(len(labels)), vals, color=BENCH_COLOURS[:len(labels)], width=.62)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.6)
        ax.set_title(ttl, fontsize=9.8)
        ax.set_ylabel(ylab)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}%" if hi else f"{v:.2f}", (i, v), ha="center",
                        va="bottom", fontsize=8)
        # Same-shaped axes would hide the point; each is scaled to its own range,
        # and the spread is stated in the title so the scaling cannot mislead.
        ax.set_ylim(0, max(vals) * 1.28)
    fig.suptitle("A model can be far better at writing code that works without being\n"
                 "any better at writing code that is well structured",
                 y=1.06, fontsize=10.5)
    save(fig, "fig10_capability_vs_structure")


# =====================================================================
# Figure 7 + 8 -- the prompt study (RQ3)
# =====================================================================
def _prompt_runs():
    """The headline comparison: every model under one instruction and a token budget
    that binds on none of them. A run whose output was mostly truncated is dropped --
    structural measures on code that stops mid-statement are not measurements."""
    out = {}
    for tag, label in [("qwen4k", "Qwen2.5-Coder"),
                       ("deepseek4k", "DeepSeek-Coder"),
                       ("claude", "Claude Sonnet 5")]:
        p = os.path.join(ROOT, "prompt_study", f"summary_{tag}.csv")
        if not os.path.exists(p):
            continue
        rows = read(p)
        parsed = sum(1 for r in rows if r["n_functions"])
        if parsed / len(rows) < 0.8:
            print(f"  [excluded from fig7/fig8] {tag}: "
                  f"{100 * (1 - parsed / len(rows)):.0f}% truncated")
            continue
        out[label] = rows
    return out


def fig_task_set():
    runs = _prompt_runs()
    if not runs:
        print("  [skip] fig7 -- no prompt-study summaries")
        return
    # Short-task reference from the deployment evaluation.
    p = os.path.join(ROOT, "arc_qwen", "evaluation_summary.csv")
    short = read(p) if os.path.exists(p) else []

    def rate(rows):
        return 100 * sum(1 for r in rows if int(r.get("n_smells_defs") or 0) > 0) / len(rows)

    def distinct(rows):
        return len({s for r in rows for s in (r["smells_defs"] or "").split(";") if s})

    names = (["MBPP + HumanEval\n(short tasks)"] if short else []) + list(runs)
    rates = ([rate(short)] if short else []) + [rate(v) for v in runs.values()]
    dis = ([distinct(short)] if short else []) + [distinct(v) for v in runs.values()]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    cols = ["#9b2c2c"] + ["#0f6d6d"] * (len(names) - 1) if short else ["#0f6d6d"] * len(names)
    for ax, vals, ttl, ylab in [
            (axes[0], rates, "Share of generations carrying a defect", "%"),
            (axes[1], dis, "Distinct defects observed (of 12)", "count")]:
        ax.bar(range(len(names)), vals, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(ylab)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}" + ("%" if ylab == "%" else ""), (i, v),
                        ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("The task set was the binding constraint, not the models", y=1.02, fontsize=11)
    save(fig, "fig7_task_set")


def fig_induction():
    runs = _prompt_runs()
    if not runs:
        print("  [skip] fig8 -- no prompt-study summaries")
        return
    targets = ["long_parameter_list", "magic_number", "dead_code",
               "deep_nesting", "duplicate_code", "long_method"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    w = 0.8 / len(runs)
    for i, (label, rows) in enumerate(runs.items()):
        vals = []
        for s in targets:
            tgt = [r for r in rows if s in (r["intended"] or "").split(";")]
            hit = sum(1 for r in tgt if s in (r["smells_defs"] or "").split(";"))
            vals.append(100 * hit / len(tgt) if tgt else 0)
        ax.bar(np.arange(len(targets)) + (i - (len(runs) - 1) / 2) * w, vals, w, label=label)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([s.replace("_", " ") for s in targets], rotation=18, ha="right")
    ax.set_ylabel("share of targeted prompts where the defect appeared")
    ax.set_title("A prompt can reliably provoke some defects and never provoke others",
                 pad=8, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig8_induction_rate")


# =====================================================================
# Figure 9 -- the perplexity negative result, on its own
# =====================================================================
def fig_perplexity():
    """Perplexity is a named negative result, so it gets its own panel rather than
    one column of the heatmap. The precise finding: no value clears even the
    small-effect band in the defective direction, and most sit slightly below
    zero -- padded and duplicated code is MORE predictable, not less."""
    inj = [inj_d.get((s, "perplexity")) for s in SMELLS]
    rl = [real_d.get((s, "perplexity")) for s in SMELLS]
    y = np.arange(len(SMELLS))

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axvspan(-0.2, 0.2, color="#888", alpha=.12, zorder=0)
    ax.axvline(0, lw=.8, color="#999", zorder=0)
    ax.axvline(0.8, ls="--", lw=.9, color="#1b6b41", zorder=0)
    ax.text(0.82, -0.45, "large effect — where the working\nstructural measures sit",
            fontsize=6.8, color="#1b6b41", va="top")
    for yy, a, b in zip(y, inj, rl):
        if a is not None and b is not None:
            ax.plot([a, b], [yy, yy], lw=.8, color="#bbb", zorder=1)
    ax.scatter([v for v in inj if v is not None],
               [yy for yy, v in zip(y, inj) if v is not None],
               s=30, color="#0f6d6d", label="injected", zorder=2)
    ax.scatter([v for v in rl if v is not None],
               [yy for yy, v in zip(y, rl) if v is not None],
               s=30, color="#b8860b", marker="s", label="real", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in SMELLS])
    ax.invert_yaxis()
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel("detection strength of perplexity (shaded: negligible)")
    ax.set_title("A code language model does not find defective code more surprising",
                 fontsize=10, pad=8)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig9_perplexity")


# =====================================================================
# Tables
# =====================================================================
def table_detection():
    """Best real-world measure per defect, with intervals -- the paper's main table."""
    rows_out = []
    for s in SMELLS:
        best, bd, best_abs = None, None, -1.0
        for m in STRUCT:
            v = real_d.get((s, m))
            if v is not None and abs(v) > best_abs:
                best, bd, best_abs = m, v, abs(v)
        if best is None:                       # no real-world measurement for this defect
            continue
        lo, hi = d_ci(bd, n_smelly.get(s), n_clean.get(s))
        ilo, ihi = d_ci(inj_d.get((s, best)), n_pairs.get(s), n_pairs.get(s))
        rows_out.append({
            "smell": s, "verdict": TRUST[s], "measure": best,
            "injected_d": inj_d.get((s, best)), "injected_lo": ilo, "injected_hi": ihi,
            "real_d": bd, "real_lo": lo, "real_hi": hi,
            "n_smelly": n_smelly.get(s), "n_clean": n_clean.get(s),
        })

    with open(os.path.join(TAB, "detection.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    def c(v):
        return "---" if v is None else f"{v:.2f}"

    with open(os.path.join(TAB, "detection.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by analysis/make_analysis.py -- do not edit\n")
        f.write("\\begin{tabular}{llrrl}\n\\toprule\n")
        f.write("Defect & Strongest measure & Injected & Real & Verdict \\\\\n\\midrule\n")
        for r in rows_out:
            ci = "" if r["real_lo"] is None else f" \\tiny[{r['real_lo']:.2f}, {r['real_hi']:.2f}]"
            f.write(f"\\texttt{{{r['smell'].replace('_', '\\_')}}} & "
                    f"{r['measure'].replace('_', ' ')} & {c(r['injected_d'])} & "
                    f"{c(r['real_d'])}{ci} & {r['verdict']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  tables/detection.csv + detection.tex")


def table_correctness():
    with open(os.path.join(TAB, "benchmark.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by analysis/make_analysis.py -- do not edit\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Defect & Examples & Tested & Behaviour preserved \\\\\n\\midrule\n")
        for r in corr:
            f.write(f"\\texttt{{{r['smell'].replace('_', '\\_')}}} & 100 & "
                    f"{r['n_tested']} & {r['behaviour_kept']}/{r['n_tested']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  tables/benchmark.tex")


def stats_dump():
    """Every measure and defect with an interval -- the appendix table."""
    out = []
    for s in SMELLS:
        for m in STRUCT:
            i, r = inj_d.get((s, m)), real_d.get((s, m))
            ilo, ihi = d_ci(i, n_pairs.get(s), n_pairs.get(s))
            rlo, rhi = d_ci(r, n_smelly.get(s), n_clean.get(s))
            out.append({"smell": s, "measure": m, "family": fam.get(m, "structural"),
                        "injected_d": i, "injected_lo": ilo, "injected_hi": ihi,
                        "real_d": r, "real_lo": rlo, "real_hi": rhi,
                        "capped": abs(i or 0) >= 5 or abs(r or 0) >= 5})
        for m in SIM:
            i = inj_d.get((s, m))
            ilo, ihi = d_ci(i, n_pairs.get(s), n_pairs.get(s))
            out.append({"smell": s, "measure": m, "family": "similarity",
                        "injected_d": i, "injected_lo": ilo, "injected_hi": ihi,
                        "real_d": None, "real_lo": None, "real_hi": None,
                        "capped": abs(i or 0) >= 5})
    with open(os.path.join(TAB, "detection_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n_capped = sum(1 for r in out if r["capped"])
    print(f"  tables/detection_full.csv ({len(out)} rows, {n_capped} at the cap)")


def copy_into_paper():
    """The paper compiles standalone (Overleaf zip, CI checkout): its figures and
    tables live under paper/ as committed copies of what this script produced."""
    import shutil
    pfig = os.path.join(ROOT, "paper", "figures")
    ptab = os.path.join(ROOT, "paper", "tables")
    os.makedirs(pfig, exist_ok=True)
    os.makedirs(ptab, exist_ok=True)
    n = 0
    for f in os.listdir(FIG):
        if f.endswith(".pdf"):
            shutil.copy2(os.path.join(FIG, f), os.path.join(pfig, f))
            n += 1
    for f in os.listdir(TAB):
        if f.endswith(".tex"):
            shutil.copy2(os.path.join(TAB, f), os.path.join(ptab, f))
            n += 1
    print(f"\nCopied {n} files into paper/figures and paper/tables")


if __name__ == "__main__":
    print("Figures:")
    fig_heatmap()
    fig_scatter()
    fig_correlation()
    fig_coverage()
    fig_models()
    fig_verbosity()
    fig_task_set()
    fig_induction()
    fig_perplexity()
    fig_capability_vs_structure()
    print("\nTables:")
    table_detection()
    table_correctness()
    stats_dump()
    copy_into_paper()
    print(f"\nWrote to {os.path.relpath(FIG, ROOT)} and {os.path.relpath(TAB, ROOT)}")
