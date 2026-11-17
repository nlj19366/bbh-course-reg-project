#!/usr/bin/env python3
"""
Build the data files for the catalogue demo page (demo/index.html).

The page itself has no degree logic. Every verdict it shows is computed here,
in Python, with the same solver check_bip.py uses (bip.py), and written out
as JSON. The students are made up (demo/students.json).

Writes demo/data/:
  index.json      coverage numbers, the programme list, which chunk holds each one
  p-NN.json       per programme: the catalogue lines, the tree, unclassified, review
  students.json   per showcase programme: each student's courses and the verdict
                  for every node of the tree

Usage
    python demo/build_demo.py                      # reads out_v2 via the combined file
    python demo/build_demo.py --check              # also compare golden students with golden/

The verdicts come from finder.evaluate; its docstring explains what met, adviser,
unknown and needed mean and how one course filling one slot is handled. The page says
so when a programme fails as a whole although each requirement looks met on its own.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARSER = HERE.parent
sys.path.insert(0, str(PARSER))

from check_results import PROGRAMS_DIR  # noqa: E402
from finder import evaluate  # noqa: E402

CATALOGUE = PARSER / "catalogue" / "usc_degree_requirements_2026_2027.json"
SUMMARY = PARSER / "out_v2" / "summary.csv"
STUDENTS = HERE / "students.json"
OUT = HERE / "data"
CHUNK = 40


def catalogue_lines(stem: str) -> list[str]:
    """Same numbering as run_classifier.number_lines: line 1 is the file's first line."""
    lines = (PROGRAMS_DIR / f"{stem}.txt").read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def coverage(cat: dict) -> dict:
    rows = list(csv.DictReader(SUMMARY.open(encoding="utf-8")))
    progs = cat["programmes"]
    full = [p for p in progs if p["requirements"]["status"] == "ok"
            and not p["review"]["needs_data"] and not p["review"]["not_checked_by_code"]]
    return {
        "scraped": cat["counts"]["programmes_in_catalogue"],
        "included": cat["counts"]["included"],
        "not_included": cat["counts"]["not_included"],
        "fully_checkable": len(full),
        "partial": sum(1 for r in rows if r["status"] == "partial"),
        "with_unresolved": sum(1 for r in rows if int(r["unresolved"])),
        "unresolved_terms": sum(int(r["unresolved"]) for r in rows),
        "no_fit_sentences": sum(int(r["unc_no_fit"]) for r in rows),
        "with_placeholders": sum(1 for r in rows if int(r["course_reuse"]) + int(r["sequencing"])
                                 + int(r["manual_review"])),
        "model": cat["model"],
        "generated_at": cat["generated_at"],
        "run_started_at": next(csv.DictReader((PARSER / "out_v2" / "run_meta.csv").open(encoding="utf-8")))["started_at"],
        "not_included_stems": [x["stem"] for x in cat["not_included"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare golden students with golden/*.json")
    args = ap.parse_args()

    cat = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    by_stem = {p["stem"]: p for p in cat["programmes"]}
    students = json.loads(STUDENTS.read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()

    index, chunks = [], {}
    for i, p in enumerate(cat["programmes"]):
        n = i // CHUNK
        r = p["requirements"]
        index.append({"stem": p["stem"], "name": p.get("name") or r["program"],
                      "credential": p.get("credential"), "status": r["status"], "chunk": n,
                      "showcase": p["stem"] in students["programmes"]})
        chunks.setdefault(n, {})[p["stem"]] = {
            "name": p.get("name") or r["program"],
            "url": p.get("catalogue_url"),
            "status": r["status"],
            "lines": catalogue_lines(p["stem"]),
            "root": r["root"],
            "unclassified": r["unclassified"],
            "review": p["review"],
        }
    for n, data in chunks.items():
        (OUT / f"p-{n:02d}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                                             encoding="utf-8")

    show, wrong = {}, 0
    for stem, people in students["programmes"].items():
        program = by_stem[stem]["requirements"]
        golden_path = PARSER / "golden" / f"{stem}.json"
        golden = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.exists() else {"cases": []}
        cases = {c["student"]: c for c in golden["cases"]}
        show[stem] = []
        for s in people:
            if "golden" in s:
                case = cases[s["golden"]]
                ev = evaluate(program, case["courses"], golden.get("default_units", 4))
                if args.check:
                    ok = ev["result"] == case["expect"] and (
                        "unmet" not in case or len(ev["unmet"]) == len(case["unmet"]))
                    wrong += not ok
                    print(f"{'ok  ' if ok else 'WRONG'} {stem} {s['name']}: {ev['result']} "
                          f"(golden: {case['expect']}), unmet {ev['unmet']}")
            else:
                ev = evaluate(program, s["courses"], 4)
                if args.check:
                    print(f"     {stem} {s['name']}: {ev['result']}, unmet {ev['unmet']}")
            show[stem].append({"name": s["name"], "summary": s["summary"],
                               "golden": s.get("golden"), **ev})

    (OUT / "index.json").write_text(json.dumps({"coverage": coverage(cat), "programmes": index},
                                               ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT / "students.json").write_text(json.dumps({"about": students["about"], "programmes": show},
                                                  ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.glob("*.json"))
    print(f"Wrote {len(list(OUT.glob('*.json')))} files to {OUT} ({size / 1e6:.1f} MB).")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
