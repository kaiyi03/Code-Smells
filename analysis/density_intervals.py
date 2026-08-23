"""N14: put an interval on the three smell densities, and test equivalence.

Section 7.4 asserts 1.93 / 1.96 / 1.90 are "the same number" and, until now,
supported that only with the replication in 9.4. A replication bounds one model's
run-to-run movement; it says nothing about whether the gap BETWEEN models is
distinguishable from zero. This computes that directly.

Density is a pooled ratio (total smells / total SLOC), not a mean of per-task
ratios, so the interval comes from a task-level bootstrap: resample the 664 tasks
with replacement, recompute the ratio, repeat. Pairs are then compared on the
bootstrap distribution of their difference, and checked against an equivalence
margin.
"""
import csv
import os
import random

ROOT = r"C:\KY_D\KY 2025 - 2026\Summer 26 Research\arc_qwen"
RUNS = [("evaluation_summary_deepseek.csv", "DeepSeek-Coder-1.3B"),
        ("evaluation_summary.csv", "Qwen2.5-Coder-1.5B"),
        ("evaluation_summary_claude.csv", "Claude Sonnet 5")]
B = 20000
random.seed(20260821)          # fixed so the paper's numbers are reproducible


def load(fn):
    rows = []
    with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["n_smells_defs"]), float(r["sloc"])))
            except (ValueError, KeyError):
                pass
    return rows


def density(rows):
    s = sum(x[1] for x in rows)
    return 100.0 * sum(x[0] for x in rows) / s if s else 0.0


data = {label: load(fn) for fn, label in RUNS}
for label, rows in data.items():
    print("%-22s %d tasks, %d smells, %.0f sloc, density %.3f"
          % (label, len(rows), sum(r[0] for r in rows),
             sum(r[1] for r in rows), density(rows)))

# One shared set of resample index draws per model, reused for the differences,
# so the paired comparisons are internally consistent.
n = {label: len(rows) for label, rows in data.items()}
boots = {label: [] for label in data}
for b in range(B):
    for label, rows in data.items():
        idx = [random.randrange(n[label]) for _ in range(n[label])]
        boots[label].append(density([rows[i] for i in idx]))


def pct(xs, p):
    ys = sorted(xs)
    return ys[max(0, min(len(ys) - 1, int(round(p / 100.0 * (len(ys) - 1)))))]


print("\n95pct bootstrap CI for smell density (per 100 SLOC), " + str(B) + " resamples:")
for label in data:
    print("  %-22s %.2f  [%.2f, %.2f]"
          % (label, density(data[label]), pct(boots[label], 2.5), pct(boots[label], 97.5)))

print("\npairwise differences:")
labels = [l for _, l in RUNS]
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b_ = labels[i], labels[j]
        diffs = [boots[a][k] - boots[b_][k] for k in range(B)]
        lo, hi = pct(diffs, 2.5), pct(diffs, 97.5)
        obs = density(data[a]) - density(data[b_])
        crosses = lo <= 0 <= hi
        print("  %-22s - %-22s  %+.3f  [%+.3f, %+.3f]  %s"
              % (a, b_, obs, lo, hi, "CI includes 0" if crosses else "excludes 0"))

# Equivalence: is the whole interval inside a margin small enough to be
# practically nothing? 0.25 smells per 100 SLOC is one smell per 400 lines.
MARGIN = 0.25
print("\nequivalence at +/- %.2f smells per 100 SLOC:" % MARGIN)
worst = 0.0
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b_ = labels[i], labels[j]
        diffs = [boots[a][k] - boots[b_][k] for k in range(B)]
        lo, hi = pct(diffs, 2.5), pct(diffs, 97.5)
        worst = max(worst, abs(lo), abs(hi))
        print("  %-22s vs %-22s  %s"
              % (a, b_, "EQUIVALENT" if -MARGIN < lo and hi < MARGIN else "not shown equivalent"))
print("\nwidest bound across all three pairs: %.3f" % worst)
