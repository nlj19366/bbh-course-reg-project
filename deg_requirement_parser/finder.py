#!/usr/bin/env python3
"""
Check one student's parsed STARS report against the whole catalogue file.

This is the Python the finder app runs in the browser (through Pyodide, with
pulp_shim standing in for PuLP). It has no I/O: everything comes in as data.

    run(report, catalogue, include_in_progress=False) -> dict
    run_json(report_json, catalogue_json, options_json) -> str     # the browser bridge
    detail(report, catalogue, stem, include_in_progress=False) -> dict

  report     the STARS parser's output (RegCheck's gui/stars-parser shape:
             completedCourses, inProgressCourses, equivalents, major, degree, minors).
             The older hand-written shape in fixtures/stars/mock_stars_report.json
             also works.
  catalogue  catalogue/usc_degree_requirements_2026_2027.json

It returns:
  courses    which courses were counted, which are in progress, and which weren't
             counted and why
  declared   the student's own major and minors, matched to programmes by name, each
             with a verdict for every node of its tree
  ranked     the closest other majors and minors, with top-level verdicts only.
             detail() gives the full tree for one of them.

Course codes. STARS prints "MATH 125" and "CHEM 105A"; the catalogue trees say
"MATH 125g" and "CHEM 105aLg". canonical_code() keeps the prefix, the number and any
sequence letter (lower case) and drops the designation letters G, L, X, M, P and W
(docs/reference/01-reading-a-stars-report.md, "Course suffixes"). Both the trees and
the student's courses are canonicalised before solving. This is a heuristic: it
assumes no two real courses differ only by a designation letter.

Node verdicts (evaluate):
  met      the node's subtree is satisfiable with placeholder leaves counted as NOT met
           and unresolved pools counted as NOT matching
  adviser  satisfiable only if the placeholder leaves (manual_review, sequencing,
           course_reuse) are met. Code can't check those.
  unknown  satisfiable only if an unresolved pool ("approved by your adviser") matches
  needed   not satisfiable even when everything unknown goes the student's way
Each node is solved on its own, but with the courses that other top-level
requirements need taken out of its pool, because one course fills one slot:
  - if the whole tree can be satisfied (possibly with adviser items), the pool for a
    top-level requirement is every course the solver's assignment doesn't put elsewhere;
  - if it can't, the pool leaves out courses named as specific_course leaves in the
    other top-level requirements (so CSCI 310 counts toward the CS core, not toward the
    CS electives).
That second rule is a heuristic for display only. The programme verdict ("result") is
bip.solve over the whole tree, unchanged.

Ranking: courses still needed. For each programme, the solver may add stand-in
courses: any course the tree names that the student hasn't taken, plus generic
stand-ins that fit any count leaf. It minimises how many it adds. That number is the
fewest more courses (of 4 units each) that could complete the programme, if every one
were the right course and every adviser item and adviser-approved pool went the
student's way, so it is a best case. The top-level requirements whose leaves got a
stand-in are the ones listed as still needing courses. Programmes none of the
student's courses fit (ignoring "any course" leaves) are left out. Order: fewest
courses still needed, then most courses that fit.
"""

from __future__ import annotations

import copy
import json
import re

import bip
import pulp
from eligibility import is_eligible

IMPOSSIBLE = {"family": "specific_course", "course": "ZZZZ 999"}  # a course no student has
STAND_IN = "ZZST"  # prefix of the generic stand-in courses used to count what's left
STAND_INS = 24     # enough for the largest unit total in the catalogue at 4 units each
DESIGNATIONS = set("GLXMPW")
USC_CODE = re.compile(r"^([A-Z]{2,5}) ?(\d{3})([A-Za-z]*)$")


# --- Course codes ---------------------------------------------------------------

def canonical_code(code: str) -> str | None:
    """'MATH 125g' -> 'MATH 125', 'CHEM 105aLg' -> 'CHEM 105a', 'CHEM 105A' -> 'CHEM 105a'.
    None for codes that aren't USC-shaped ('AP32:COMP', 'TR-ASL-1')."""
    m = USC_CODE.match(code.strip())
    if not m:
        return None
    letters = "".join(ch.lower() for ch in m[3] if ch.upper() not in DESIGNATIONS)
    return f"{m[1]} {m[2]}{letters}"


def canonical_tree(node):
    """A copy of a requirement tree with every course code canonicalised."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "course" and isinstance(v, str):
                out[k] = canonical_code(v) or v
            elif k == "courses" and isinstance(v, list):
                out[k] = [canonical_code(c) or c for c in v]
            else:
                out[k] = canonical_tree(v)
        return out
    if isinstance(node, list):
        return [canonical_tree(x) for x in node]
    return node


# --- Tree helpers -----------------------------------------------------------------

def children(node: dict) -> list[tuple[str, dict]]:
    key = "all_of" if "all_of" in node else ("from" if "from" in node else None)
    return [(f"/{key}/{i}", c) for i, c in enumerate(node[key])] if key else []


def walk(node: dict, path: str = "root"):
    yield path, node
    for suffix, child in children(node):
        yield from walk(child, path + suffix)


def with_path_ids(node: dict, path: str = "root") -> dict:
    """A copy where every node's id is its path, so the solver's assignment is keyed by path."""
    out = copy.deepcopy(node)
    for p, n in walk(out, path):
        n["id"] = p
    return out


def placeholders_failed(node: dict) -> dict:
    """A copy where every placeholder leaf becomes a course nobody has taken."""
    out = copy.deepcopy(node)

    def fix(n: dict) -> dict:
        if bip.node_kind(n) in bip.PLACEHOLDER_FAMILIES:
            return dict(IMPOSSIBLE)
        for key in ("all_of", "from"):
            if key in n:
                n[key] = [fix(c) for c in n[key]]
        return n

    return fix(out)


def without_of_which(node: dict) -> dict:
    out = dict(node)
    out.pop("of_which", None)
    return out


def feasible(node: dict, taken: dict[str, float], unknown_as: bool) -> bool:
    return bip._solve_once(node, taken, unknown_as) is not None


def node_verdict(node: dict, taken: dict[str, float]) -> str:
    if bip.node_kind(node) in bip.PLACEHOLDER_FAMILIES:
        return "adviser"
    if feasible(placeholders_failed(node), taken, unknown_as=False):
        return "met"
    if feasible(node, taken, unknown_as=False):
        return "adviser"
    if feasible(node, taken, unknown_as=True):
        return "unknown"
    return "needed"


def taken_courses(courses: list, default_units: float = 4) -> dict[str, float]:
    taken = {}
    for c in courses:
        code, units = (c[0], float(c[1])) if isinstance(c, list) else (c, float(default_units))
        taken[code] = units
    return taken


# --- Verdicts for one programme -----------------------------------------------------

def evaluate(program: dict, courses: list, default_units: float = 4) -> dict:
    """A verdict for every node of one programme's tree."""
    taken = taken_courses(courses, default_units)
    root = program["root"]
    verdict = bip.solve(program, taken)  # the real programme verdict, unchanged
    tagged = with_path_ids(root)
    # Prefer the strict assignment: the lenient one may park a real elective in a free
    # slot and fill the elective from an "approved by your adviser" pool instead.
    solved = None
    if verdict.result != "fail":
        solved = bip._solve_once(tagged, taken, unknown_as=False) or bip._solve_once(tagged, taken, unknown_as=True)
    assignment = {}
    if solved:
        for leaf, xs in solved.leaf_x.items():
            used = sorted(c for c, x in xs.items() if (x.value() or 0) > 0.5)
            if used:
                assignment[leaf] = used
    nodes, checks_failed = {}, {}
    tops = children(root) if "all_of" in root else []
    for suffix, top in tops:
        top_path = "root" + suffix
        if solved:
            elsewhere = {c for leaf, used in assignment.items()
                         if not (leaf == top_path or leaf.startswith(top_path + "/")) for c in used}
        else:
            mine = {n["course"] for _, n in walk(top) if n.get("family") == "specific_course"}
            elsewhere = {n["course"] for s2, other in tops if s2 != suffix
                         for _, n in walk(other) if n.get("family") == "specific_course"} - mine
        pool = {c: u for c, u in taken.items() if c not in elsewhere}
        for path, n in walk(top, top_path):
            nodes[path] = node_verdict(n, pool)
            if n.get("of_which") and nodes[path] != "met":
                checks_failed[path] = node_verdict(without_of_which(n), pool) != nodes[path]
    nodes["root"] = node_verdict(root, taken) if verdict.result != "fail" else "needed"
    if root.get("of_which") and nodes["root"] != "met":
        checks_failed["root"] = node_verdict(without_of_which(root), taken) != nodes["root"]
    return {
        "courses": [[c, u] for c, u in taken.items()],
        "result": verdict.result,
        "unmet": verdict.unmet,
        "nodes": nodes,
        "checks_failed": [p for p, failed in checks_failed.items() if failed],
        "assignment": assignment,
    }


# --- The student's courses --------------------------------------------------------------

def select_courses(report: dict, include_in_progress: bool = False) -> dict:
    """Which of the report's courses count, in canonical form.

    Counted: completed USC courses that passed, and transfer or exam credit that the
    report maps to a USC course (equivalents). Not counted, with a reason: failed or
    deleted courses, transfer credit with no USC equivalent, and codes that aren't
    USC-shaped. In-progress courses are counted only with include_in_progress.
    """
    equivalents = {e["from"]: e["to"] for e in report.get("equivalents") or []}
    counted: dict[str, dict] = {}
    not_counted, in_progress = [], []

    def add(code: str, raw: str, units: float, how: str) -> None:
        canon = canonical_code(code)
        if canon is None:
            not_counted.append({"code": raw, "reason": "Not a USC course code"})
            return
        prev = counted.get(canon)
        if prev is None or units > prev["units"]:  # a retake counts once
            counted[canon] = {"code": canon, "printed": raw, "units": units, "how": how}

    for c in report.get("completedCourses") or []:
        raw = c.get("code", "")
        units = float(c.get("units") or 0)
        source = c.get("source", "usc")
        if c.get("passed") is False:
            not_counted.append({"code": raw, "reason": "Not passed, or the credit was removed"})
        elif source == "usc":
            add(raw, raw, units, "completed")
        elif raw in equivalents:
            add(equivalents[raw], raw, units, f"counts as {equivalents[raw]}")
        elif source == "transfer_generic":
            not_counted.append({"code": raw, "reason": "General transfer credit: ask your adviser what it counts as"})
        else:
            not_counted.append({"code": raw, "reason": "Transfer or exam credit with no USC equivalent on the report"})

    for c in report.get("inProgressCourses") or []:
        raw = c.get("code", "")
        canon = canonical_code(raw)
        in_progress.append({"code": canon or raw, "printed": raw, "units": float(c.get("units") or 0)})
        if include_in_progress and canon:
            add(raw, raw, float(c.get("units") or 0), "in progress")

    return {"counted": sorted(counted.values(), key=lambda x: x["code"]),
            "in_progress": in_progress, "not_counted": not_counted}


# --- Programmes ------------------------------------------------------------------------

def clean_name(name: str) -> str:
    return name.replace("*", "").strip()


def is_minor(name: str) -> bool:
    return clean_name(name).lower().endswith("minor")


def match_programme(names: dict[str, str], wanted: str | None, degree: str | None = None,
                    minor: bool = False) -> dict:
    """Match a STARS major or minor name to programme stems.

    {"stem": ..., "candidates": [...]}: stem is set only when exactly one programme
    matches; otherwise candidates lists the near matches for the student to pick from.
    """
    if not wanted:
        return {"stem": None, "candidates": []}
    w = re.sub(r"\s+", " ", wanted).strip().lower()
    exact, near = [], []
    for stem, name in names.items():
        n = clean_name(name).lower()
        if minor != is_minor(n):
            continue
        if minor:
            base = n[: -len("minor")].strip()
        else:
            m = re.match(r"^(.*)\(([^()]*)\)$", n)
            base, cred = (m[1].strip(), m[2].strip()) if m else (n, "")
            if degree and cred and cred != degree.lower():
                continue
        if base == w:
            exact.append(stem)
        elif base.startswith(w) or w in base:
            near.append(stem)
    if len(exact) == 1:
        return {"stem": exact[0], "candidates": exact + near}
    return {"stem": None, "candidates": exact + near}


def fit_count(root: dict, taken: dict[str, float]) -> int:
    """How many of the student's courses fit at least one leaf, ignoring "any course" leaves."""
    fits = set()
    for _, n in walk(root):
        fam = n.get("family")
        if fam == "specific_course" and n["course"] in taken:
            fits.add(n["course"])
        elif fam == "count":
            cond = n.get("eligible_if")
            if cond == {"level_min": 0}:
                continue
            for c in taken:
                if c not in fits and is_eligible(cond, c) is True:
                    fits.add(c)
    return len(fits)


def programmes(catalogue: dict) -> dict[str, dict]:
    """stem -> {"name", "program"} with canonicalised trees."""
    out = {}
    for p in catalogue["programmes"]:
        req = p["requirements"]
        out[p["stem"]] = {"name": p.get("name") or req["program"],
                          "url": p.get("catalogue_url"),
                          "program": {"root": canonical_tree(req["root"])}}
    return out


class _WithStandIns(bip._Builder):
    """bip's model, where generic stand-in courses fit every count leaf."""

    def eligible(self, cond, course):
        if course.startswith(STAND_IN + " "):
            return True
        return super().eligible(cond, course)


def still_needed(root: dict, taken: dict[str, float]) -> dict | None:
    """The fewest stand-in courses that complete the tree, and where they go.

    {"courses": n, "by_requirement": {top-level path: n}}, or None if even the stand-ins
    can't complete it.
    """
    named = {n["course"] for _, n in walk(root) if n.get("family") == "specific_course"}
    stand_ins = {c: 4.0 for c in named if c not in taken}
    stand_ins.update({f"{STAND_IN} {900 + i}": 4.0 for i in range(STAND_INS)})
    b = _WithStandIns({**taken, **stand_ins}, unknown_as=True)
    b.build(with_path_ids(root))
    b.prob.objective = pulp.lpSum(x for c, xs in b.slots.items() if c in stand_ins for x in xs)
    if not b.feasible():
        return None
    by_req: dict[str, int] = {}
    for leaf, xs in b.leaf_x.items():
        n = sum(1 for c, x in xs.items() if c in stand_ins and (x.value() or 0) > 0.5)
        if n:
            top = "/".join(leaf.split("/")[:3]) if leaf.startswith("root/") else "root"
            by_req[top] = by_req.get(top, 0) + n
    return {"courses": sum(by_req.values()), "by_requirement": by_req}


def summarise(stem: str, entry: dict, gap: dict | None, fit: int) -> dict:
    root = entry["program"]["root"]
    remaining = []
    if gap:
        for path, n in sorted(gap["by_requirement"].items(), key=lambda kv: -kv[1]):
            node = root if path == "root" else root["all_of"][int(path.split("/")[2])]
            remaining.append({"path": path, "courses": n,
                              "description": node.get("description") or node.get("id") or "A requirement"})
    return {
        "stem": stem,
        "name": clean_name(entry["name"]),
        "url": entry["url"],
        "minor": is_minor(entry["name"]),
        "courses_needed": gap["courses"] if gap else None,
        "remaining": remaining,
        "courses_that_fit": fit,
    }


def run(report: dict, catalogue: dict, include_in_progress: bool = False, progress=None) -> dict:
    courses = select_courses(report, include_in_progress)
    taken = {c["code"]: c["units"] for c in courses["counted"]}
    as_list = [[c, u] for c, u in taken.items()]
    progs = programmes(catalogue)
    names = {s: e["name"] for s, e in progs.items()}

    declared = []
    major = match_programme(names, report.get("major"), report.get("degree"))
    declared.append({"kind": "major", "printed": report.get("major"), **major})
    for m in report.get("minors") or []:
        name = m.get("name") if isinstance(m, dict) else m
        declared.append({"kind": "minor", "printed": name, **match_programme(names, name, minor=True)})
    for d in declared:
        if d["stem"]:
            entry = progs[d["stem"]]
            d["name"] = clean_name(entry["name"])
            d["url"] = entry["url"]
            d["evaluation"] = evaluate(entry["program"], as_list)
            d["courses_needed"] = (still_needed(entry["program"]["root"], taken) or {}).get("courses")
    own = {d["stem"] for d in declared if d["stem"]}

    ranked = []
    stems = list(progs)
    for i, stem in enumerate(stems):
        if progress:
            progress(i, len(stems))
        if stem in own:
            continue
        entry = progs[stem]
        fit = fit_count(entry["program"]["root"], taken)
        if fit == 0:
            continue
        ranked.append(summarise(stem, entry, still_needed(entry["program"]["root"], taken), fit))
    unknown = 10 ** 6
    ranked.sort(key=lambda r: (unknown if r["courses_needed"] is None else r["courses_needed"],
                               -r["courses_that_fit"], r["name"]))
    return {
        "courses": courses,
        "declared": declared,
        "ranked": {"majors": [r for r in ranked if not r["minor"]],
                   "minors": [r for r in ranked if r["minor"]]},
        "checked": len(stems),
    }


def detail(report: dict, catalogue: dict, stem: str, include_in_progress: bool = False) -> dict:
    """The full node-by-node verdict for one programme, as for a declared one."""
    courses = select_courses(report, include_in_progress)
    entry = programmes(catalogue)[stem]
    return {"stem": stem, "name": clean_name(entry["name"]), "url": entry["url"],
            "evaluation": evaluate(entry["program"], [[c["code"], c["units"]] for c in courses["counted"]])}


def run_json(report_json: str, catalogue_json: str, options_json: str = "{}") -> str:
    """The browser bridge: JSON strings in, a JSON string out."""
    opts = json.loads(options_json)
    report, catalogue = json.loads(report_json), json.loads(catalogue_json)
    if opts.get("stem"):
        out = detail(report, catalogue, opts["stem"], bool(opts.get("include_in_progress")))
    else:
        out = run(report, catalogue, bool(opts.get("include_in_progress")))
    return json.dumps(out)
