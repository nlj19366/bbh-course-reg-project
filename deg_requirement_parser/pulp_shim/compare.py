#!/usr/bin/env python3
"""
Check pulp_shim against real PuLP + CBC: same workload, run once with each, results diffed.

    python pulp_shim/compare.py            # from deg_requirement_parser/; needs pulp and scipy

Each run happens in its own subprocess (one with this folder first on sys.path, one
without), so the two "pulp" modules never meet. The workload uses only files in this
repo:
  golden   every golden/ case against examples/ and against out_v2/ trees: bip.solve
  demo     every student in demo/students.json: demo/build_demo.evaluate (every node's
           verdict, the of_which checks that fail, result, unmet)
  sweep    six course lists (fixtures/stars/mock_stars_report.json and five demo
           students) against all programmes in the combined catalogue file: bip.solve

Compared: result, unmet, node verdicts, failed checks, and the number of course slots
the solver used (its objective). Which courses fill which slot is not compared: when
several assignments use equally few courses, the two solvers may pick different ones.
Exits non-zero on any difference.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHIM = Path(__file__).resolve().parent
PARSER = SHIM.parent
REPO = PARSER.parent


def workload() -> dict:
    sys.path.insert(0, str(PARSER))
    sys.path.insert(0, str(PARSER / "demo"))
    import bip
    import build_demo
    from check_bip import taken_courses

    def slots(v) -> int:
        return sum(len(cs) for cs in v.assignment.values())

    out = {}
    goldens = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((PARSER / "golden").glob("*.json"))}
    for folder in ("examples", "out_v2"):
        for stem, g in goldens.items():
            program = json.loads((PARSER / folder / f"{stem}.json").read_text(encoding="utf-8"))
            for case in g["cases"]:
                v = bip.solve(program, taken_courses(case, g.get("default_units", 4)))
                out[f"golden|{folder}|{stem}|{case['student']}"] = [v.result, sorted(v.unmet), slots(v)]

    cat = json.loads((PARSER / "catalogue" / "usc_degree_requirements_2026_2027.json").read_text(encoding="utf-8"))
    by_stem = {p["stem"]: p["requirements"] for p in cat["programmes"]}
    students = json.loads((PARSER / "demo" / "students.json").read_text(encoding="utf-8"))["programmes"]
    lists = []
    for stem, people in students.items():
        g = goldens.get(stem, {"cases": []})
        cases = {c["student"]: c for c in g["cases"]}
        for s in people:
            courses = cases[s["golden"]]["courses"] if "golden" in s else s["courses"]
            ev = build_demo.evaluate(by_stem[stem], courses, g.get("default_units", 4))
            out[f"demo|{stem}|{s['name']}"] = [ev["result"], sorted(ev["unmet"]), ev["nodes"], sorted(ev["checks_failed"]),
                                              sum(len(cs) for cs in ev["assignment"].values())]
            if "golden" not in s and len(lists) < 5:
                lists.append((s["name"], build_demo.taken_courses(courses)))

    mock = json.loads((REPO / "fixtures" / "stars" / "mock_stars_report.json").read_text(encoding="utf-8"))
    lists.insert(0, ("mock_stars_report", {c["code"]: float(c["units"]) for c in mock["completedCourses"]}))
    for name, taken in lists:
        for stem, program in by_stem.items():
            v = bip.solve(program, taken)
            out[f"sweep|{name}|{stem}"] = [v.result, sorted(v.unmet), slots(v)]
    return out


def run(with_shim: bool) -> tuple[dict, float]:
    env_path = [str(SHIM)] if with_shim else []
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out.json"
        code = (f"import sys; sys.path[:0] = {env_path!r}; import pulp; "
                f"assert ('pulp_shim' in pulp.__file__) == {with_shim}, pulp.__file__; "
                f"sys.path.insert(0, {str(SHIM)!r}); import compare, json; "
                f"open({str(dest)!r}, 'w').write(json.dumps(compare.workload()))")
        t = time.perf_counter()
        subprocess.run([sys.executable, "-c", code], check=True, cwd=PARSER)
        return json.loads(dest.read_text()), time.perf_counter() - t


def main() -> int:
    real, t_real = run(with_shim=False)
    shim, t_shim = run(with_shim=True)
    keys = sorted(set(real) | set(shim))
    diffs = [k for k in keys if real.get(k) != shim.get(k)]
    groups = {}
    for k in keys:
        groups[k.split("|")[0]] = groups.get(k.split("|")[0], 0) + 1
    print("Compared " + ", ".join(f"{n} {g}" for g, n in groups.items()) + " results.")
    print(f"Real PuLP + CBC: {t_real:.1f} s. pulp_shim + scipy: {t_shim:.1f} s.")
    sweep = [v[0] for k, v in real.items() if k.startswith("sweep|")]
    print("Sweep verdicts (real): " + ", ".join(f"{r} {sweep.count(r)}" for r in ("pass", "unknown", "fail")))
    for k in diffs[:20]:
        print(f"DIFFERENT {k}\n  real: {json.dumps(real.get(k))[:300]}\n  shim: {json.dumps(shim.get(k))[:300]}")
    print(f"{len(diffs)} difference(s).")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
