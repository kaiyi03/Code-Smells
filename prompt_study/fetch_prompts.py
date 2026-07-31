"""
Fetch the smell-inducing prompt set and convert it to the schema the rest of the
pipeline already reads.

The dataset (Woo, 2026 -- https://github.com/rosawoo/code-smell) is 476 Python
generation prompts, each written to provoke a named code smell, tagged with the
smell it targets, a complexity level, a domain, and an action keyword. The action
keyword and token-depth fields come from the Green Prompting methodology it
follows, which studied how prompt wording drives token cost -- they are the
independent variables for the token-usage question.

It is fetched at run time rather than committed here, so the copy in this repo
cannot drift from theirs and we are not redistributing their data.

Writes prompt_study/prompts.jsonl. Run:  python prompt_study/fetch_prompts.py
"""

import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prompts.jsonl")
BASE = "https://raw.githubusercontent.com/rosawoo/code-smell/main/dataset"
FILES = ["prompts_core.json", "prompts_synthetic.json"]

# Their taxonomy is Fowler's 25; ours is the twelve a tool can confirm mechanically.
# Only these six overlap -- for the rest our detectors stay silent, so the
# induction rate can only be measured on this subset. Stated, not hidden.
SHARED = {
    "Long Method": "long_method",
    "Long Parameter List": "long_parameter_list",
    "Dead Code": "dead_code",
    "Duplicated Code": "duplicate_code",
    "Magic Numbers/Strings": "magic_number",
    "Deep Nesting": "deep_nesting",
}


def fetch(name):
    url = f"{BASE}/{name}"
    print(f"fetching {url} ...")
    with urllib.request.urlopen(url) as r:                      # noqa: S310 (fixed host)
        return json.loads(r.read().decode("utf-8"))


def normalise(rec):
    """One prompt, in the fields the generation and scoring scripts expect."""
    smells = rec.get("code_smells") or []
    if isinstance(smells, str):                                 # the CSV form is a bare string
        smells = [s.strip() for s in smells.split(";") if s.strip()]
    return {
        "task_id": rec["id"],
        "source": "promptset",
        "prompt": rec["prompt"],
        "intended_smells": smells,
        "intended_tracked": sorted({SHARED[s] for s in smells if s in SHARED}),
        "complexity": rec.get("complexity"),
        "domain": rec.get("domain"),
        "action_keyword": (rec.get("action_keywords") or [None])[0]
        if isinstance(rec.get("action_keywords"), list) else rec.get("action_keywords"),
        "expected_token_depth": rec.get("expected_token_depth"),
        "synthetic": bool(rec.get("synthetic")),
    }


def main():
    rows, seen = [], set()
    for name in FILES:
        for rec in fetch(name):
            row = normalise(rec)
            if row["task_id"] in seen:
                continue
            seen.add(row["task_id"])
            rows.append(row)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tracked = sum(1 for r in rows if r["intended_tracked"])
    print(f"\nwrote {len(rows)} prompts to {os.path.basename(OUT)}")
    print(f"  {tracked} target a smell our detectors cover; "
          f"{len(rows) - tracked} target one they do not")
    for field in ["complexity", "expected_token_depth", "action_keyword"]:
        counts = {}
        for r in rows:
            counts[r[field]] = counts.get(r[field], 0) + 1
        shown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  {field}: {shown}")


if __name__ == "__main__":
    main()
