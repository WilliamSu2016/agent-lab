# Observability

Every node in the real Planner→Researcher→Synthesizer→Reviewer→notify_action
pipeline is instrumented — this was the second core gap from the prior
review (`src/observability` existed but was never called from inside the
real pipeline, only thinly at the HTTP layer).

## 1. What every execution carries

`src/observability/tracing.py::ExecutionContext`:

```python
request_id, trace_id, user_id, session_id, agent_version, environment
```

Bound once per HTTP request/run (`bind_execution_context()`), stored as a
plain dict in graph state (`state["context"]`) so it survives across
LangGraph's `Send`-based parallel fan-out (raw threads don't inherit
`contextvars` — every span reconstructs `ExecutionContext` explicitly from
this dict; see `architecture.md` §3 for the bug this fixed).

## 2. Traces (`src/observability/tracing.py`)

`Tracer.span(kind, name, ...)` builds a full **execution tree**: one root
`WORKFLOW` span per run, `AGENT` spans per node (Planner/Researcher-per-task/
Synthesizer/Reviewer/notify_action), `LLM` spans per model call nested under
their agent, `TOOL` spans per tool call nested under their agent. Each span
records: name, kind, start/end time, duration, `attributes` (e.g.
`prompt_tokens`, `completion_tokens`, `cost_usd`, `iteration`, `task_count`,
tool name/risk-tier, error info), and parent/child linkage — so the full
tree for one `trace_id` answers directly:

* *Which agents did this request go through?* — walk the span tree's `AGENT` nodes.
* *Which tools were called?* — `TOOL` spans, with arguments (sanitized) and risk tier.
* *How long did each tool/LLM call take?* — `duration` on each span.
* *How many agent-loop iterations ran?* — count of Planner→Reviewer cycles (`iteration` attribute).
* *Why did it fail?* — the deepest span with an error attribute, plus `outcome.status`.

## 3. Metrics (`src/observability/metrics.py`)

`MetricsRegistry.record_from_trace(root_span)` derives, per request and
aggregated:

| Metric | Answers |
|---|---|
| `latency` (per span kind, per agent) | *"how long does each step take?"* |
| `token_usage` (prompt/completion, per agent) | *"how many tokens does each LLM call use?"* |
| `tool_failure_rate` (per tool) | *"which tool is least reliable?"* |
| `agent_failure_rate` (per agent) | *"which agent fails most?"* |
| `retry_count` | *"how much are we retrying?"* |
| `cost` (per agent, per request, aggregate) | *"which agent is most expensive?"* — sum `LLM` span `cost_usd` grouped by agent name |
| `success_rate` | overall pipeline health |

`src/cost/estimator.py` derives the same per-request cost from raw
tokens/tool-calls/subagent-calls/retries, cross-checked against the traced
`cost_usd` (see `docs/07-COST-LATENCY.md` and `src/cost/`).

## 4. Logs (`src/observability/logging.py`)

Structured **JSON** logs, one line per event, each carrying the full
`ExecutionContext` fields plus event-specific data (node name, duration,
error, sanitized args). **Sensitive data is never logged by default**:
`src/security/sanitization.py`'s redaction pass runs on every log payload
before it is emitted — emails, API-key-shaped tokens, and other
sensitive-looking strings are replaced with `***REDACTED***` unless a field
is explicitly allowlisted as safe. This is enforced structurally (logger
wrapper), not left to each call site's discipline.

## 5. Who / what / where on every request

* **Who** — `user_id` + `tenant_id` from `Identity`, propagated via
  `ExecutionContext.user_id` (Layer 2 identity propagation, see `security.md`).
* **What version** — `agent_version`, set from the `AGENT_VERSION` env var
  and stamped on every span/log/run record (`RunRecord`).
* **Where** — `environment` (`development`/`test`/`production`).

## 6. Production Agent Observability Dashboard Specification

A single dashboard, backed by the metrics/traces above, organized into four
panels:

### Panel A — Request Health
- Success rate (rolling 5m/1h/24h), overall and per `agent_version`.
- Failure rate breakdown by failure kind (blocked / timeout / crashed / tool error).
- P50/P95/P99 latency, overall and per node (Planner/Researcher/Synthesizer/Reviewer).

### Panel B — Cost
- Cost per request (P50/P95, and running total per hour).
- Cost by agent (which node/agent is most expensive — stacked by Planner/Researcher/Synthesizer/Reviewer).
- Token usage (prompt vs completion) trend.
- Budget-degradation events (Quality→Balanced→Fast→graceful-failure transitions, from `src/cost/policy.py`).

### Panel C — Reliability
- Tool failure rate, per tool, per risk tier.
- Agent failure rate, per node.
- Retry count distribution.
- Checkpoint/resume events (crashes detected, resumes attempted, resumes succeeded).
- Idempotent-duplicate-request rate.

### Panel D — Security
- Blocked requests by guardrail layer (Input / Workflow / Tool) and reason
  (prompt injection, unauthorized tool call, denied approval).
- Pending vs. approved vs. denied HIGH-risk approvals over time.
- Cross-tenant access attempts (should always be zero — alert if not).

### Alerts
- Success rate < 95% over 15 min.
- P95 latency > mode's `TIMEOUT` budget.
- Any cross-tenant access attempt (page immediately — security incident).
- Cost per request P95 exceeds `MAX_COST` for the active mode for > 5 min (budget policy misconfigured or model pricing changed).
- Tool failure rate for any `HIGH`-risk tool > 5%.

Every panel and alert is keyed by `request_id`/`trace_id` so an on-call
engineer can jump from "cost spiked" or "failure rate up" straight to the
exact trace tree that explains why.
