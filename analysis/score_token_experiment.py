"""
Score every run of the token experiment through the same panel.

Nine runs -- three models by three verbosity conditions -- each scored by
evaluate_generations.py exactly as the Section 7 runs were, including jscpd for
duplicate_code. Consistency matters more than speed here: a defect rate measured
with one detector set is not comparable to a rate measured with another, and the
whole experiment is a comparison.

Skips any run whose generations file is missing, so it can be run while the
cluster jobs are still landing and re-run to pick up the rest.

Run:  python analysis/score_token_experiment.py
      python analysis/score_token_experiment.py --no-dup   # faster, drops jscpd
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
ARC = os.path.join(ROOT, "arc_qwen")

MODELS = [("deepseek", "DeepSeek-Coder"), ("qwen", "Qwen2.5-Coder"),
          ("claude", "Claude Sonnet 5")]
CONDITIONS = ["terse", "neutral", "verbose"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dup", action="store_true",
                    help="skip jscpd; faster, but duplicate_code goes unmeasured")
    ap.add_argument("--force", action="store_true",
                    help="rescore runs that already have a summary")
    args = ap.parse_args()

    done, skipped, missing = [], [], []
    for tag, label in MODELS:
        for cond in CONDITIONS:
            run = f"{tag}_{cond}"
            gen = os.path.join(ARC, f"generations_{run}.jsonl")
            out = os.path.join(ARC, f"evaluation_summary_{run}.csv")
            if not os.path.exists(gen):
                missing.append(run)
                continue
            if os.path.exists(out) and not args.force:
                skipped.append(run)
                continue

            cmd = [sys.executable, os.path.join(ARC, "evaluate_generations.py"),
                   "--tag", run, "--label", f"{label} ({cond})"]
            if not args.no_dup:
                cmd.append("--dup")
            print(f"\n=== scoring {run} ===", flush=True)
            r = subprocess.run(cmd, cwd=ARC)
            (done if r.returncode == 0 else missing).append(run)

    print(f"\nscored {len(done)}: {', '.join(done) if done else '-'}")
    if skipped:
        print(f"already had summaries ({len(skipped)}): {', '.join(skipped)}")
    if missing:
        print(f"not yet available ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
