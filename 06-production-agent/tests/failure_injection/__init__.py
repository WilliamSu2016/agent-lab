"""Failure Injection / Chaos Testing.

This package deliberately adds **no new business functionality**. Every
test here injects a failure into code that already exists elsewhere in this
repository (``src/reliability``, ``src/specialists/llm.py``,
``src/multi_agent_research``, ``src/durable``, ``src/api``) and asserts on
the *already-implemented* handling behaviour -- retry/backoff/limit,
timeout, structured fallback, checkpoint/resume, idempotency, rate
limiting. Where a real gap is discovered (see
``test_llm_failure.py``'s Planner "invalid structured output" tests), the
test documents and pins down the *current* behaviour rather than adding a
new fix, per the experiment's explicit instructions.

See ``docs/09-FAILURE-MATRIX.md`` for the full Failure -> Detection ->
Recovery -> User Experience matrix these tests exist to prove.

Separate from ``tests/test_failure_injection.py`` (an earlier, unrelated
Reliability-experiment file that already existed before this experiment --
left untouched).
"""
