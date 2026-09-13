"""Production Evaluation Pipeline.

Answers, for every code or prompt change, the question production teams
actually need answered before shipping: *did this change make the agent
worse, and in which specific dimension?* -- never just "does the final
answer string match a golden string" (that check is both too strict, since
a correct answer can be phrased many ways, and too weak, since a string
that happens to match can still have been produced by the wrong tool, the
wrong route, an unauthorized side effect, or a hallucinated, uncited
claim).

Package layout:

    evals/
    ├── dataset/
    │   └── tasks.py     # >= 20 EvalTask definitions (input + expected
    │                      behavior + success criteria + risk level)
    ├── runner.py         # SystemUnderTest protocol, AgentRunResult,
    │                      run_offline_evaluation(), report rendering
    ├── metrics.py         # the 11 required evaluation dimensions, each a
    │                      structured, non-string-equality check
    ├── baseline.py         # persist/load an EvaluationRun as a baseline
    └── regression.py       # compare a new run against a baseline,
                             per-dimension, per-task -- detect_regressions()

See ``docs/06-EVALUATION.md`` for the full design write-up and
``tests/test_evaluation.py`` for the worked "good agent vs regressed
agent" example that exercises the full
run -> save baseline -> re-run -> detect regression pipeline.
"""
