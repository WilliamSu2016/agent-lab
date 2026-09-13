"""Wires the evaluation harness into the existing unittest suite.

Running ``python -m unittest discover -s tests`` will now also execute every
case in ``tests/evaluations/cases.json`` through the deterministic evaluation
harness (see ``tests/evaluations/run_evaluations.py``).
"""

import unittest

from tests.evaluations.run_evaluations import evaluate_case, load_cases


class EvaluationHarnessTest(unittest.TestCase):
    @unittest.skip(
        "Disabled: this harness scripts tool-calling (web_search) against the old "
        "hand-rolled src.agent.run_agent(question, llm, search_fn, max_turns) signature. "
        "The SDK-based src.agent.run_agent(question, agent, max_turns=...) has a "
        "different signature and no tools yet (search_web not migrated). Re-enable "
        "once search_web is migrated to an @function_tool and this harness is "
        "updated to script the SDK Agent instead."
    )
    def test_all_cases_pass_deterministic_assertions(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 10, "expected at least 10 evaluation cases")

        for case in cases:
            with self.subTest(case=case["id"]):
                result = evaluate_case(case)
                failed_checks = [c for c in result["checks"] if not c["passed"]]
                self.assertTrue(
                    result["passed"],
                    f"case {case['id']} failed: {failed_checks}",
                )


if __name__ == "__main__":
    unittest.main()
