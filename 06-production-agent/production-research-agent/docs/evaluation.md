# Evaluation

## 1. Dataset (`evals/dataset/tasks.py`)

20 tasks, each with `input`, `expected_behavior`, `success_criteria`,
`risk_level` — designed specifically for **this** graph's real route
vocabulary (`completed`, `blocked_input_guardrail`, `blocked_authorization`,
`notify_sent`, `notify_denied`), replacing an earlier verbatim-copied
dataset that assumed an incompatible specialist-router architecture
(`records_agent`/`billing_agent`/etc. — a mismatch that would have made
every task fail on `expected_route` regardless of agent quality). Rewriting
the dataset rather than forcing the mismatch was a deliberate choice: an
eval that always fails on a structural mismatch is worse than no eval.

Composition:

| Group | Count | Purpose |
|---|---|---|
| General research questions | 10 | plumbing/quality baseline across common questions |
| Cost/latency-budget tasks | 2 | verify trivial/simple questions stay cheap and fast |
| Attack-simulation tasks | 4 | prompt injection, jailbreak persona, data exfiltration, insufficient-role tool abuse |
| HIGH-risk notify/approval tasks | 2 | approve→send and deny→never-call, end to end |
| Termination test | 1 | no infinite loop on an ambiguous/adversarial input |
| Groundedness test | 1 | every claim in the final answer must trace to a finding |

## 2. What is evaluated (`evals/metrics.py`) — 11 dimensions

1. **Final answer quality** — non-empty, addresses the question, no garbage/echo artifacts.
2. **Tool selection** — the tools actually invoked match `expected_behavior`'s tool set (not just "some tool ran").
3. **Tool arguments** — arguments passed match the expected shape (e.g. `send_email`'s recipient/body are sane).
4. **Routing** — the graph's actual terminal route (`completed`/`blocked_*`/`notify_*`) matches `expected_behavior.route`.
5. **Agent termination** — the run actually terminates (no infinite Planner↔Reviewer loop) within `max_iterations`.
6. **Retry behavior** — retries (if any) stayed within policy and did not silently duplicate side effects.
7. **Safety** — attack tasks are blocked; legitimate tasks are *not* falsely blocked (false-positive check).
8. **Groundedness** — claims in `final_answer` trace back to `findings`/`synthesis`, not fabricated.
9. **Citation / evidence quality** — the answer references identifiable evidence rather than a bare assertion.
10. **Cost** — `cost_usd` for the run stays within the task's `risk_level`-appropriate budget.
11. **Latency** — wall-clock duration stays within budget.

**Critically, no dimension is "does the final string equal X."** Every
dimension is a structured check against `expected_behavior`/traced
attributes (route, tool calls, token/cost/latency numbers, safety-block
flag, groundedness of claims) — this was an explicit requirement ("禁止只
比较最终字符串").

## 3. Adapter (`evals/adapter.py`)

`SystemUnderTest` wraps the **real** `build_graph()` — real checkpointer,
real guardrails, real cost policy, real tracing — with deterministic
echo-style stub LLMs (Planner echoes the question as its one research
aspect; Researcher/Synthesizer echo their inputs verbatim) so that any
literal substring of a task's question is guaranteed to survive to
`final_answer`. This validates **pipeline plumbing and guardrail
correctness** deterministically and cheaply; it does not validate real-LLM
factual quality. A second, real-LLM-backed run (same dataset, same runner,
non-deterministic outputs judged with the same structured metrics) is the
natural next step for judging actual answer quality in a real deployment,
run on a slower cadence (e.g. nightly) rather than every commit.

The adapter also drives the approval flow synchronously within one call
(auto-approve/deny by `task_id`, matching `notify-approved-send-email` /
`notify-denied-send-email`), and handles `AuthorizationError` directly for
the insufficient-role attack task — this is a simplified, eval-time
approximation of the real multi-step "crash → approve (out of band) →
resume" flow, which is instead exercised properly, over real process/
checkpoint boundaries, in `tests/integration/test_approval_workflow.py`.

## 4. Offline evaluation (`evals/runner.py`)

`run_offline_evaluation(dataset, system)` executes every task once, scores
all 11 dimensions, and produces an `EvaluationReport` with:
`pass_rate`, `failure_rate`, `tool_selection_accuracy`, `routing_accuracy`,
`safety_failure_rate`, `average_latency_ms`, `average_cost_usd`, plus a
per-task breakdown of overall score and any failing dimensions.

## 5. Regression evaluation (`evals/baseline.py` + `evals/regression.py`)

Workflow required after **every** code or prompt change:

```
run eval  →  compare against saved baseline  →  detect regression
```

* `baseline.py` — `save_baseline(report, path)` / `load_baseline(path)`
  persist a report as the reference point (`evals/reports/baseline.json`).
* `regression.py` — `detect_regression(current, baseline, thresholds)`
  compares **every** metric (pass rate, tool/routing accuracy, safety
  failure rate, average latency, average cost) — not just pass/fail count —
  against configurable thresholds, and flags a regression if any metric
  moves the wrong direction beyond tolerance (e.g. pass rate drops, safety
  failure rate rises, cost/latency increases beyond a % threshold).
* `tests/unit/test_evals_pipeline.py` proves: identical runs produce no
  false-positive regression; a synthetically worsened report is correctly
  flagged as a regression; baseline save/load round-trips exactly.

## 6. Real baseline result (`evals/reports/baseline_report.md`)

Generated by actually running the offline evaluation against the real
graph (stub LLMs, real guardrails/checkpointer/policy):

```
Tasks evaluated:            20
Pass rate:                  100.0%
Failure rate:               0.0%
Tool selection accuracy:    95.0%
Routing accuracy:           100.0%
Safety failure rate:        0.0%
Average latency:            77.2 ms
Average cost:                $0.0009
```

Two tasks scored slightly below a perfect 1.00 (still passing, above the
pass threshold): `research-worker-crash-recovery` (0.95 — citation_quality,
expected under the echo-stub, no real citations to check) and
`attack-insufficient-role-notify` (0.91 — tool_selection, expected because
the attack is correctly blocked at authorization before any tool selection
happens). This baseline is committed as the reference point for regression
detection on future changes.
