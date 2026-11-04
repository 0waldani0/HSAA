#!/usr/bin/env python
"""Claims hygiene gate.

Fails if a documentation file repeats a capability claim that this project has
measured to be false, vacuous or unreproducible, unless the surrounding lines
explicitly mark it as such. The canonical measured numbers live in
docs/METRICS.md; docs must not invent new ones without a measurement behind
them.

Usage:  python tests/check_claims.py
Exit:   0 = all claim mentions are qualified; 1 = at least one unqualified.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BANNED = [
    (r"zero[\s-]*shot synthesis", "zero-shot synthesis is a fixed substring template (bench_zeroshot_falsify)"),
    (r"OOD open[\s-]*ended extrapolation", "OOD extrapolation is a corpus-sentence collage (bench_claim_falsify)"),
    (r"template[\s-]*free relation inference", "relation inference passes on an untrained brain (bench_claim_falsify)"),
    (r"52[ ,.]?000 natural", "the 52k vocabulary is 59,904 synthetic affix strings (METRICS section 4)"),
]

ALLOW = re.compile(
    r"(FALSIF|REFUT|VACUOUS|CORPUS[\s-]*INERT|NOT REPRODUCIBLE|UNPROVEN|"
    r"NO EXECUTABLE|FALSE AS|SYNTHETIC|overstated|unsupported|not supported|"
    r"опроверг|не подтвержд|фальсифиц)",
    re.IGNORECASE,
)

SKIP_DIRS = {".git", "build", "opencode.json"}
DOC_FILES = [
    "README.md",
    "PROJECT.md",
    "TEST_READY.md",
    "TEST_INFRA.md",
    "second_brain/README.md",
    "second_brain/measur/README.md",
]


def doc_files():
    files = [os.path.join(ROOT, f) for f in DOC_FILES]
    docs_dir = os.path.join(ROOT, "docs")
    if os.path.isdir(docs_dir):
        for name in sorted(os.listdir(docs_dir)):
            if name.lower().endswith(".md"):
                files.append(os.path.join(docs_dir, name))
    return [f for f in files if os.path.isfile(f)]


def main():
    problems = []
    for path in doc_files():
        try:
            lines = io.open(path, encoding="utf-8").read().split("\n")
        except OSError as exc:
            problems.append("%s: cannot read (%s)" % (path, exc))
            continue
        for i, line in enumerate(lines):
            for pattern, why in BANNED:
                if not re.search(pattern, line, re.IGNORECASE):
                    continue
                window = " ".join(lines[max(0, i - 2):i + 3])
                if ALLOW.search(window):
                    continue
                problems.append(
                    "%s:%d: unqualified claim `%s`; %s" %
                    (os.path.relpath(path, ROOT), i + 1, line.strip(), why))

    metrics = os.path.join(ROOT, "docs", "METRICS.md")
    if not os.path.isfile(metrics):
        problems.append("docs/METRICS.md is missing (single source of truth)")

    if problems:
        print("CLAIMS: FAILED")
        for p in problems:
            print("  " + p)
        return 1
    print("CLAIMS: PASSED (%d docs scanned)" % len(doc_files()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
