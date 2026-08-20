"""
The token experiment: does asking a model for more code cost structural quality,
or correctness, or neither?

Three system prompts vary one thing -- how much code the task asks for -- while
the task prompts, the decoding, the token cap and the 664 tasks stay fixed:

  terse    "Write the shortest solution that works. No docstring, no comments,
            no input validation."
  neutral  "Write a solution to the task."
  verbose  "Write a complete and defensive solution: validate the inputs, handle
            edge cases explicitly, and document the function with a docstring and
            comments."

None of them mentions structure or quality. An instruction like "write clean
code" would confound the amount written with an explicit request for quality, and
whether volume alone carries a cost is the question being asked.

The benchmark tasks are used rather than the smell-inducing prompt set because
they ship with unit tests, so correctness moves alongside structure and the two
can be read against each other. That is the whole point: a verbosity setting that
improved structure while breaking the code would not be an improvement.

Reads arc_qwen/evaluation_summary_<model>_<condition>.csv, writes a table and a
figure. Run score_token_experiment.py first to produce those summaries.

Run:  python analysis/token_experiment.py
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")

MODELS = [("deepseek", "DeepSeek-Coder"), ("qwen", "Qwen2.5-Coder"),
          ("claude", "Claude Sonnet 5")]
CONDITIONS = ["terse", "neutral", "verbose"]
COLOURS = {"deepseek": "#7a3d9e", "qwen": "#0f6d6d", "claude": "#b8860b"}


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def load(tag, cond):
    p = os.path.join(ROOT, "arc_qwen", f"evaluation_summary_{tag}_{cond}.csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(rows):
    """Everything the section needs from one run, in one pass."""
    n = len(rows)
    tok = [num(r.get("n_output_tokens")) or 0 for r in rows]
    sloc = [num(r.get("sloc")) or 0 for r in rows]
    defects = [num(r.get("n_smells_defs")) or 0 for r in rows]
    return {
        "n": n,
        "pass1": 100 * sum(1 for r in rows if r["result"] == "pass") / n,
        "tokens": sum(tok) / n,
        "sloc": sum(sloc) / n,
        "defects_total": sum(defects),
        # Per 100 source lines, computed over the pooled totals rather than as a
        # mean of per-file ratios: a five-line file with one defect would otherwise
        # count as heavily as a fifty-line file with ten.
        "per100": 100 * sum(defects) / sum(sloc) if sum(sloc) else 0,
        "with_defect": 100 * sum(1 for d in defects if d > 0) / n,
        "comments": sum(num(r.get("comment_density")) or 0 for r in rows) / n,
        "cyclomatic": sum(num(r.get("cyclomatic")) or 0 for r in rows) / n,
    }


def main():
    grid = {}
    for tag, label in MODELS:
        for cond in CONDITIONS:
            rows = load(tag, cond)
            if rows:
                grid[(tag, cond)] = stats(rows)

    if not grid:
        raise SystemExit("no evaluation summaries found -- run score_token_experiment.py")

    # ---- table -------------------------------------------------------------
    hdr = (f"{'model':<16}{'condition':<10}{'tokens':>8}{'SLOC':>7}{'defects':>9}"
           f"{'/100 SLOC':>11}{'with defect':>13}{'comments':>10}{'pass@1':>8}")
    print(hdr)
    print("-" * len(hdr))
    for tag, label in MODELS:
        for cond in CONDITIONS:
            s = grid.get((tag, cond))
            if not s:
                continue
            print(f"{label:<16}{cond:<10}{s['tokens']:>8.0f}{s['sloc']:>7.1f}"
                  f"{s['defects_total']:>9.0f}{s['per100']:>11.2f}"
                  f"{s['with_defect']:>12.1f}%{s['comments']:>10.2f}{s['pass1']:>7.1f}%")
        print()

    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, "token_experiment.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "n", "mean_output_tokens", "mean_sloc",
                    "defects_total", "defects_per_100_sloc", "pct_with_defect",
                    "mean_comment_density", "pass_at_1"])
        for (tag, cond), s in grid.items():
            w.writerow([tag, cond, s["n"], f"{s['tokens']:.1f}", f"{s['sloc']:.2f}",
                        int(s["defects_total"]), f"{s['per100']:.3f}",
                        f"{s['with_defect']:.2f}", f"{s['comments']:.3f}",
                        f"{s['pass1']:.2f}"])
    print(f"wrote tables/token_experiment.csv ({len(grid)} runs)")

    # ---- figure ------------------------------------------------------------
    # Three panels, one per outcome, condition on the x-axis. Lines rather than
    # bars because the conditions are ordered -- terse to verbose is a scale, and
    # the shape of each model's line is the result.
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.3))
    # Raw count and per-line rate get separate panels on purpose: they move in
    # opposite directions, and a figure showing only one of them would support
    # whichever conclusion its author preferred.
    panels = [
        ("tokens", "output tokens per task", "How much was written"),
        ("defects_total", "smells found (all 664)", "Smells, counted"),
        ("per100", "smells per 100 SLOC", "Smells, per line of code"),
        ("pass1", "pass@1 (%)", "Correctness"),
    ]
    x = range(len(CONDITIONS))
    for ax, (key, ylab, title) in zip(axes, panels):
        for tag, label in MODELS:
            ys = [grid[(tag, c)][key] for c in CONDITIONS if (tag, c) in grid]
            if len(ys) != len(CONDITIONS):
                continue
            ax.plot(x, ys, marker="o", lw=1.8, ms=6, color=COLOURS[tag], label=label)
        ax.set_xticks(list(x))
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Asking for more code: all three write more and all three gain smells, "
                 "but only one dilutes its rate", y=1.06, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig11_token_experiment.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote figures/fig11_token_experiment.pdf + .png")


if __name__ == "__main__":
    main()
