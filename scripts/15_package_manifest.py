"""Generate or check package inventory; generated results/images are excluded."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def inventory():
    files = [p for folder in ("src", "scripts", "configs", "docs", "tests", "archives")
             for p in (REPO/folder).rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files += [REPO/p for p in ("README.md", "requirements.txt", "requirements-lock.txt", "CITATION.cff", "LICENSE", ".gitignore") if (REPO/p).exists()]
    return [{"path":p.relative_to(REPO).as_posix(), "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
             "bytes":p.stat().st_size} for p in sorted(files)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    path = REPO/"PACKAGE_MANIFEST.json"
    current = inventory()
    if args.write:
        path.write_text(json.dumps(current, indent=2)+"\n")
    recorded = json.loads(path.read_text())
    if recorded != current:
        old = {r["path"]:r for r in recorded}
        new = {r["path"]:r for r in current}
        raise SystemExit(f"Inventory differs: missing={sorted(old.keys()-new.keys())}, "
                         f"added={sorted(new.keys()-old.keys())}, "
                         f"changed={sorted(k for k in old.keys() & new.keys() if old[k]!=new[k])}")
    print(f"PASS: {len(current)} entries; 0 missing, 0 changed")


if __name__ == "__main__":
    main()
