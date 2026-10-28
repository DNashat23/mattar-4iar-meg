"""Check the uploaded _nr .sqd files against a manifest (sizes taken from Box). Prints missing / wrong-size files.

    python verify_sqd.py /scratch/$USER/meg_rtpad/sqd                                  (main task: sqd_manifest.csv)
    python verify_sqd.py /scratch/$USER/meg_project/sqd sqd_manifest_localizer.csv     (localizer)
"""
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(__file__).with_name(sys.argv[2] if len(sys.argv) > 2 else "sqd_manifest.csv")
rows = list(csv.DictReader(open(manifest)))
bad = []
for r in rows:
    f = root / r["meg_id"] / r["file"]
    if not f.exists():
        bad.append(f"MISSING  {r['meg_id']}/{r['file']}")
    elif f.stat().st_size != int(r["bytes"]):
        bad.append(f"SIZE     {r['meg_id']}/{r['file']}: {f.stat().st_size} vs expected {r['bytes']}")
print(f"{len(rows)} files expected, {len(rows) - len(bad)} OK")
print("\n".join(bad) if bad else "all present with the expected sizes")
sys.exit(1 if bad else 0)
