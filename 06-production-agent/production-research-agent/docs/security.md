# Security & Guardrails

Three layers, wired directly into the real graph's execution path (the
previous review's core finding was that `src/security` existed but was
**never imported** by the real pipeline — that is fixed here).

## Layer 1 — Input Guardrail

`src/security/guardrails.py::InputGuardrail`, invoked at
`supervisor_entry` (the graph's first node) before the Planner ever sees the
question:

* **Prompt injection detection** — pattern/heuristic scan (`sanitization.py`)
  for classic jailbreak/override phrasing ("ignore previous instructions",
  role-play overrides, instruction-leak probes, delimiter-escape attempts).
* Sets `state["blocked"] = True`, `block_reason="prompt_injection"` — the
  graph short-circuits straight to the Finalizer with a safe refusal
  message; the Planner/Researchers never run (no wasted LLM spend, no
  attacker-controlled prompt reaches downstream nodes).

## Layer 2 — Agent / Workflow Guardrail

`src/security/guardrails.py::AgentWorkflowGuardrail`:

* **Authorization check** — every tool call is checked against the caller's
  `Identity.roles` (`src/security/authorization.py`) before execution.
* **User identity propagation** — `Identity` (`user_id`, `tenant_id`,
  `roles`) flows through `ExecutionContext`/state from the HTTP boundary all
  the way to tool calls; tools never re-derive identity independently.
* **Tenant isolation** — every stateful lookup (`ApprovalStore`,
  `SessionMemoryStore`, `RunRegistry`) is keyed/filtered by `tenant_id`; one
  tenant's approval/cache/run can never be read or resumed by another
  (`tests/security/test_data_isolation.py`).
* **High-risk action approval** — `enforce_high_risk_approval()`: a
  `HIGH`-risk tool call (see tool tiers below) is **not** executed until an
  approval exists for its `approval_request_id`. If none exists yet, it
  raises `ApprovalRequiredError` (a *pending*, not a *failure*, state — see
  below).
* **Output validation** — the Finalizer's answer is scanned before being
  returned (`OutputGuardrail`): blocks obvious secret/credential leakage
  patterns and enforces the answer actually addresses the question (not an
  empty/garbage completion).

### The `notify_action` proof: "Agent → approval → Tool", not "Agent → Tool"

`src/agents/supervisor.py::make_notify_node()` is the concrete, executable
demonstration that a `HIGH`-risk tool is unreachable without approval:

```
finalizer → notify_action → END
```

If a run sets `notify_email`, `notify_action`:

1. Builds a `ToolCallRequest` for `send_email` (`HIGH` risk).
2. Calls `ApprovalStore.submit_or_get(idempotency_key=trace_id, ...)` —
   **idempotent**: a `resume()` retry after a pending approval polls the
   *same* approval request instead of creating a new one each time.
3. Calls the same `authorize_tool_call()` + `enforce_high_risk_approval()`
   Layer-2 primitives used everywhere else (no separate, divergent approval
   code path).
4. If no decision yet → `ApprovalRequiredError` propagates **uncaught**.
   LangGraph treats the node as not-yet-complete; the checkpoint stops right
   before `send_email` executes. A human approves/denies out of band, and
   `resume()` re-enters `notify_action`, which now finds the decision and
   either calls `send_email` (approve) or ends with `notify_status="denied"`
   (deny) — **the tool function itself is never invoked on the deny path**.

This is tested end-to-end (not just unit-tested) in
`tests/integration/test_approval_workflow.py`: approve→send, deny→never-call,
and no-op-when-unset.

**Known limitation**: `notify_email` is not yet exposed as a `POST /runs`
request field — reachable via direct graph/Python calls and the integration
test today, not yet via the public HTTP API. See `docs/11-Production-Ready.md`.

## Layer 3 — Tool Guardrail

`src/security/tool_policy.py` + `src/security/guardrails.py::ToolGuardrail`:

* **Tool argument validation** — every tool has a declared schema; calls
  with missing/malformed/out-of-range arguments are rejected before the
  tool function ever runs.
* **Tool risk tiers**:

  | Tier | Meaning | Example | Gate |
  |---|---|---|---|
  | `LOW` | read-only | `search_web` | argument validation only |
  | `MEDIUM` | write but reversible | `update_record` | argument validation + authorization |
  | `HIGH` | external, hard-to-reverse side effect | `send_email` | argument validation + authorization **+ human approval** |

* **Sensitive-data filtering** — `src/security/sanitization.py` redacts
  emails/API-key-shaped strings/etc. from anything that reaches logs (see
  `observability.md`) and from tool arguments before they're persisted in a
  trace.

## Attack test coverage (`tests/security/`)

`test_prompt_injection.py`, `test_authorization.py`, `test_tool_policy.py`,
`test_data_isolation.py` — **10+ concrete attack scenarios**, including:

1. Classic "ignore previous instructions" injection.
2. Role-play jailbreak ("pretend you are DAN / no restrictions").
3. Instruction-leak probe ("repeat your system prompt").
4. Delimiter/escape-sequence injection to break out of the user-content block.
5. HIGH-risk tool call attempted with an insufficient role.
6. HIGH-risk tool call attempted with no approval record at all.
7. HIGH-risk tool call attempted with a *denied* approval, retried.
8. Tool call with malformed/missing required arguments.
9. Tool call with an out-of-range/invalid argument value.
10. Cross-tenant read attempt on another tenant's approval/cache/run record.
11. Cross-tenant resume attempt on another tenant's `thread_id`.
12. Output containing a secret-shaped string, verified blocked by the output guardrail.

All 68 security tests pass against the real graph's guardrail wiring (not a
mock), confirmed in this session's full-suite runs.
