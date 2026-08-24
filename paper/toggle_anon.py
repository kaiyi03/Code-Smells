"""Flip the anonymisation switch. Usage: toggle_anon.py on|off"""
import io
import sys

P = r"C:\KY_D\KY 2025 - 2026\Summer 26 Research\paper\main.tex"
want = sys.argv[1] if len(sys.argv) > 1 else "on"
src, dst = ("\\anonfalse", "\\anontrue") if want == "on" else ("\\anontrue", "\\anonfalse")

t = io.open(P, encoding="utf-8", newline="").read()
# Only the declaration line, not the two mentions inside the comment above it.
lines = t.split("\n")
hits = [i for i, l in enumerate(lines) if l.strip() in ("\\anonfalse", "\\anontrue")]
assert len(hits) == 1, "found %d declaration lines" % len(hits)
lines[hits[0]] = dst
io.open(P, "w", encoding="utf-8", newline="").write("\n".join(lines))
print("anonymisation: %s (line %d is now %s)" % (want, hits[0] + 1, dst))
