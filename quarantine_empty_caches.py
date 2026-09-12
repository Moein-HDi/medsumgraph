"""Move empty ([]) per-entity cache files to a quarantine folder.

The new empty-cache policy (don't write [] for failed/no-triple extractions)
takes effect from now on. This script deals with the historical mess: there
are currently thousands of `[]` v2 cache files from the earlier noisy runs.
Moving them (instead of deleting) keeps an audit trail and lets you restore
them if you ever need to debug a particular entity.

After running this, subsequent builds will treat those CUIs as uncached and
re-extract them with the new (strict) prompt -- leaving the old noisy results
out of the graph.

Usage: python quarantine_empty_caches.py [--dry-run]
"""
import argparse
import shutil
from pathlib import Path

import config

QUARANTINE_DIR = config.KG_CACHE_DIR / "_quarantine_empty"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count and report; do not move files.",
    )
    args = parser.parse_args()

    cache_dir = config.KG_CACHE_DIR
    empty_files: list[Path] = []
    total = 0
    for path in cache_dir.glob("*.v2.json"):
        total += 1
        if path.read_text(encoding="utf-8").strip() in ("[]", ""):
            empty_files.append(path)

    print(f"Total v2 cache files: {total}")
    print(f"Empty ([]) cache files: {len(empty_files)}")

    if not empty_files:
        return

    if args.dry_run:
        for p in empty_files[:10]:
            print(f"  would move: {p.name}")
        if len(empty_files) > 10:
            print(f"  ... and {len(empty_files) - 10} more")
        print("Dry run only -- no files moved.")
        return

    QUARANTINE_DIR.mkdir(exist_ok=True)
    moved = 0
    for path in empty_files:
        dest = QUARANTINE_DIR / path.name
        shutil.move(str(path), str(dest))
        moved += 1
    print(f"Moved {moved} empty cache files to {QUARANTINE_DIR}")


if __name__ == "__main__":
    main()
