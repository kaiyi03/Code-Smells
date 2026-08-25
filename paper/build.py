#!/usr/bin/env python3
"""Build both versions of the paper, so the anonymous one is never something to
remember at submission time.

    python paper/build.py

Produces, in paper/:
    main.pdf        the readable version, with the author block and both URLs
    main-anon.pdf   the same paper with the author block blank and the URLs
                    withheld, for a double-blind submission

The two differ only in what \\ifanon guards. Everything else, including page
count and layout, is identical, so proofreading either one covers both.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")

# The identifying strings that must not survive into the anonymous build. The
# check is on the rendered text, not the source, because the source keeps them
# behind \else.
IDENTIFYING = ["Kai Yi Ng", "kaiyi.ng", "University of Oxford",
               "kaiyi03", "code-smell-dashboard"]


def set_anon(on):
    src = open(TEX, encoding="utf-8", newline="").read()
    lines = src.split("\n")
    hits = [i for i, l in enumerate(lines) if l.strip() in ("\\anonfalse", "\\anontrue")]
    if len(hits) != 1:
        sys.exit("expected one \\anonfalse/\\anontrue line, found %d" % len(hits))
    lines[hits[0]] = "\\anontrue" if on else "\\anonfalse"
    open(TEX, "w", encoding="utf-8", newline="").write("\n".join(lines))


def latex():
    """pdflatex, bibtex, pdflatex, pdflatex -- the usual four passes."""
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    log = open(os.path.join(HERE, "main.log"), encoding="utf-8",
               errors="replace").read()
    errs = len(re.findall(r"^! ", log, re.M))
    pages = re.findall(r"\((\d+) pages", log)
    return errs, (pages[-1] if pages else "?")


def text_of(pdf):
    out = subprocess.run(["pdftotext", pdf, "-"], cwd=HERE,
                         capture_output=True, text=True, errors="replace")
    return out.stdout


def main():
    print("readable build ...")
    set_anon(False)
    errs, pages = latex()
    print("  %s pages, %d errors" % (pages, errs))

    print("anonymous build ...")
    set_anon(True)
    errs, pages = latex()
    shutil.copy(os.path.join(HERE, "main.pdf"), os.path.join(HERE, "main-anon.pdf"))
    print("  %s pages, %d errors" % (pages, errs))

    body = text_of("main-anon.pdf")
    leaks = [s for s in IDENTIFYING if s.lower() in body.lower()]
    if leaks:
        print("  !! main-anon.pdf still contains: %s" % ", ".join(leaks))
    else:
        print("  anonymity check: none of %d identifying strings appear"
              % len(IDENTIFYING))

    # Leave the working copy readable and main.pdf matching it.
    print("restoring the readable build ...")
    set_anon(False)
    latex()
    print("\nmain.pdf       readable")
    print("main-anon.pdf  for double-blind submission")


if __name__ == "__main__":
    main()
