"""Rename raw EDF files to the pseudonymised scheme the loader expects.

    python scripts/rename_edf.py SRC_DIR DST_DIR mapping.csv

mapping.csv is YOUR private file (keep it out of version control) with two columns, a substring that identifies
a participant in the original file names and the ID to use:

    surname_or_substring,subject_id
    Dupont,S01
    Martin,S02

The leading session tag of the original name (e.g. "3_", "5d_", "test1_", "ft2_") is kept: "3_Dupont_xx.edf" -> "3_S01.edf".
EDF headers may still contain patient information; this script does not rewrite them.
"""
import csv
import re
import shutil
import sys
from pathlib import Path

TAG = re.compile(r"^(?P<tag>(?:test\d*|ft\d|\d+d?))?[-_ ]*")


def main(src, dst, mapping):
    with open(mapping, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0] != "surname_or_substring"]
    Path(dst).mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(src).glob("*.edf")):
        sid = next((s for key, s in rows if key.lower() in p.name.lower()), None)
        if sid is None:
            print("no mapping for", p.name, "-> skipped")
            continue
        tag = TAG.match(p.name).group("tag")
        out = Path(dst) / (f"{tag}_{sid}.edf" if tag else f"{sid}.edf")
        shutil.copyfile(p, out)
        print(f"{p.name} -> {out.name}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
