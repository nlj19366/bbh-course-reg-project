#!/usr/bin/env python3
"""
Tests for finder.py.

    python -m unittest test_finder
    PYTHONPATH=pulp_shim python -m unittest test_finder     # the same, on the browser's solver

The reports here are made up. They use the STARS parser's output shape
(RegCheck's gui/stars-parser) or the older mock in fixtures/stars/.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import bip
import finder

HERE = Path(__file__).resolve().parent
CATALOGUE = json.loads((HERE / "catalogue" / "usc_degree_requirements_2026_2027.json").read_text(encoding="utf-8"))
MOCK = json.loads((HERE.parent / "fixtures" / "stars" / "mock_stars_report.json").read_text(encoding="utf-8"))


def row(code, units=4, source="usc", passed=True):
    return {"term": "20253", "code": code, "rawCode": code.replace(" ", ""), "units": units,
            "suffixes": [], "flags": [], "source": source, "passed": passed}


class CanonicalCode(unittest.TestCase):
    def test_designations_dropped(self):
        self.assertEqual(finder.canonical_code("MATH 125g"), "MATH 125")
        self.assertEqual(finder.canonical_code("MATH 118gx"), "MATH 118")
        self.assertEqual(finder.canonical_code("AMST 101gmw"), "AMST 101")
        self.assertEqual(finder.canonical_code("CSCI 103L"), "CSCI 103")

    def test_sequence_letters_kept_in_lower_case(self):
        self.assertEqual(finder.canonical_code("CHEM 105aLg"), "CHEM 105a")
        self.assertEqual(finder.canonical_code("CHEM 105A"), "CHEM 105a")
        self.assertEqual(finder.canonical_code("PPD 497a"), "PPD 497a")

    def test_both_spellings_meet(self):
        self.assertEqual(finder.canonical_code("PHED 120A"), finder.canonical_code("PHED 120a"))

    def test_not_usc_shaped(self):
        self.assertIsNone(finder.canonical_code("AP32:COMP"))
        self.assertIsNone(finder.canonical_code("TR-ASL-1"))


class SelectCourses(unittest.TestCase):
    def setUp(self):
        self.report = {
            "completedCourses": [
                row("MATH 125"), row("WRIT 150"), row("ECON 203", passed=False),
                row("CMPR 120", source="transfer"), row("ANTH 101", 3, source="transfer"),
                row("AP32:COMP", source="exam"), row("TR-ASL-1", source="transfer_generic"),
                row("CHEM 105A"),
            ],
            "inProgressCourses": [row("CSCI 310", passed=None)],
            "equivalents": [{"from": "CMPR 120", "to": "CSCI 113"}],
        }

    def test_counted(self):
        got = [c["code"] for c in finder.select_courses(self.report)["counted"]]
        self.assertEqual(got, ["CHEM 105a", "CSCI 113", "MATH 125", "WRIT 150"])

    def test_not_counted_has_reasons(self):
        reasons = {c["code"]: c["reason"] for c in finder.select_courses(self.report)["not_counted"]}
        self.assertEqual(set(reasons), {"ECON 203", "ANTH 101", "AP32:COMP", "TR-ASL-1"})
        self.assertIn("adviser", reasons["TR-ASL-1"])

    def test_in_progress_only_on_request(self):
        self.assertNotIn("CSCI 310", [c["code"] for c in finder.select_courses(self.report)["counted"]])
        with_ip = finder.select_courses(self.report, include_in_progress=True)
        self.assertIn("CSCI 310", [c["code"] for c in with_ip["counted"]])

    def test_old_mock_shape(self):
        got = finder.select_courses(MOCK)
        self.assertEqual(len(got["counted"]), 9)
        self.assertEqual(got["not_counted"], [])


class MatchProgramme(unittest.TestCase):
    names = {p["stem"]: p.get("name") or p["requirements"]["program"] for p in CATALOGUE["programmes"]}

    def test_major_with_degree(self):
        self.assertEqual(finder.match_programme(self.names, "Computer Science", "BS")["stem"], "086_computer_science_bs")

    def test_case_and_spacing(self):
        self.assertEqual(finder.match_programme(self.names, "computer  science", "BS")["stem"], "086_computer_science_bs")

    def test_minor(self):
        got = finder.match_programme(self.names, "Computer Science", minor=True)
        self.assertTrue(got["stem"] and got["stem"].endswith("_minor"))

    def test_no_match_gives_no_stem(self):
        got = finder.match_programme(self.names, "Underwater Basket Weaving", "BA")
        self.assertIsNone(got["stem"])
        self.assertEqual(got["candidates"], [])


class Run(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = finder.run(MOCK, CATALOGUE)

    def test_own_major_checked_with_the_real_verdict(self):
        major = self.out["declared"][0]
        self.assertEqual(major["stem"], "086_computer_science_bs")
        program = finder.programmes(CATALOGUE)[major["stem"]]["program"]
        taken = {c["code"]: c["units"] for c in self.out["courses"]["counted"]}
        self.assertEqual(major["evaluation"]["result"], bip.solve(program, taken).result)
        self.assertGreater(major["courses_needed"], 0)

    def test_suffixes_matched(self):
        # The mock says MATH 125; the CS tree says MATH 125g. They must meet.
        major = self.out["declared"][0]
        used = {c for cs in major["evaluation"]["assignment"].values() for c in cs}
        self.assertTrue(used or major["evaluation"]["result"] == "fail")
        root = finder.programmes(CATALOGUE)[major["stem"]]["program"]["root"]
        self.assertGreater(finder.fit_count(root, {"MATH 125": 4.0}), 0)

    def test_ranked_lists_are_sorted_and_exclude_own_major(self):
        for kind in ("majors", "minors"):
            rows = self.out["ranked"][kind]
            self.assertTrue(rows)
            keys = [(r["courses_needed"] if r["courses_needed"] is not None else 10 ** 6, -r["courses_that_fit"])
                    for r in rows]
            self.assertEqual(keys, sorted(keys))
            self.assertNotIn("086_computer_science_bs", [r["stem"] for r in rows])
            self.assertTrue(all(r["courses_that_fit"] > 0 for r in rows))

    def test_remaining_adds_up(self):
        for r in self.out["ranked"]["minors"]:
            if r["courses_needed"] is not None:
                self.assertEqual(sum(x["courses"] for x in r["remaining"]), r["courses_needed"])

    def test_complete_programme_needs_nothing(self):
        # A student who has every course a small tree names needs 0 more.
        root = {"all_of": [{"family": "specific_course", "course": "MATH 125"},
                           {"family": "count", "eligible_if": {"prefix": ["CSCI"]},
                            "thresholds": [{"count_by": "courses", "op": "min", "value": 2}]}]}
        self.assertEqual(finder.still_needed(root, {"MATH 125": 4.0, "CSCI 103": 4.0, "CSCI 104": 4.0})["courses"], 0)
        self.assertEqual(finder.still_needed(root, {"CSCI 103": 4.0})["courses"], 2)

    def test_json_bridge(self):
        out = json.loads(finder.run_json(json.dumps(MOCK), json.dumps(CATALOGUE),
                                         json.dumps({"stem": "086_computer_science_bs"})))
        self.assertEqual(out["stem"], "086_computer_science_bs")
        self.assertIn("root", out["evaluation"]["nodes"])


if __name__ == "__main__":
    unittest.main()
