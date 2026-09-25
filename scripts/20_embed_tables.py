#!/usr/bin/env python3
"""Embed the generated LaTeX tables into the manuscript so it compiles on its own.

Every standalone `\\gentable{NAME}` line in the manuscript is replaced by the contents of
tables/NAME.tex, between marker comments. Re-running the script refreshes the text
between existing markers, so run it again whenever 06, 10 or 19 regenerate a table.

The robustness table is embedded only once tables/robustness_summary.tex exists;
until then its \\IfFileExists line (which shows the red PENDING note) is left as is.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from campaign import location

REPO = Path(__file__).resolve().parents[1]
TABLES = location("tables_dir")
ROBUST_LINE = re.compile(r"^\\IfFileExists\{tables/robustness_summary\.tex\}.*$", re.M)
GENTABLE_LINE = re.compile(r"^\\gentable\{([A-Za-z0-9_]+)\}[ \t]*$", re.M)
BLOCK = re.compile(r"^% >>> generated: ([A-Za-z0-9_]+)\.tex .*?^% <<< \1\.tex[ \t]*$", re.M | re.S)


def block(name: str) -> str:
    src = TABLES / f"{name}.tex"
    if not src.exists():
        raise SystemExit(f"missing generated table: {src}")
    body = src.read_text(encoding="utf-8").rstrip("\n")
    return (f"% >>> generated: {name}.tex (scripts/20_embed_tables.py) -- regenerate, do not edit\n"
            f"{body}\n% <<< {name}.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=str(REPO / "revision.tex"))
    args = ap.parse_args()
    path = Path(args.tex)
    s = path.read_text(encoding="utf-8")

    names = []
    s = BLOCK.sub(lambda m: (names.append(m.group(1)), block(m.group(1)))[1], s)
    s = GENTABLE_LINE.sub(lambda m: (names.append(m.group(1)), block(m.group(1)))[1], s)
    if (TABLES / "robustness_summary.tex").exists():
        s, n = ROBUST_LINE.subn(lambda m: block("robustness_summary"), s)
        names += ["robustness_summary"] * n

    path.write_text(s, encoding="utf-8")
    print(f"embedded {len(names)} tables into {path.name}: {', '.join(names)}")
    if "robustness_summary" not in names:
        print("robustness_summary.tex not present yet; PENDING note kept")


if __name__ == "__main__":
    main()
