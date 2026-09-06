"""Generate docs/reference/ports-catalog.md from the port catalog.

Run with --write to regenerate the committed doc, or --check to fail closed when the doc has drifted
from PORT_CATALOG. The ports_catalog gate runs the --check path so the reference can never lie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from refaat.devkit.catalog import render_markdown

DOC_PATH = Path("docs/reference/ports-catalog.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render_markdown()
    if args.write:
        DOC_PATH.write_text(rendered, encoding="utf-8")
        print(f"Wrote {DOC_PATH}")
        return 0

    current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
    if current != rendered:
        print(f"{DOC_PATH} is stale; run: python scripts/generate_ports_catalog_doc.py --write")
        return 1
    print(f"{DOC_PATH} matches the port catalog.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
