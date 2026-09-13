# Architecture

## 1. What this is

`production-research-agent` is the **single, real** Agent — the Planner →
Researcher(s) → Synthesizer → Reviewer research pipeline — upgraded so that
it is the thing actually served by the API, with every production capability
(guardrails, checkpointing, observability, cost control) wired directly into
its execution path instead of living next to it as a separate demo.

This directly closes the core finding of `docs/10-Final-Review.md`: the old
repo deployed a placeholder "durable demo" graph while the real research
pipeline sat unintegrated. Here there is only **one** graph, and it is both
real and production-capable.

## 2. Agent orchestration

```
supervisor_entry → planner → fan_out (Send) → researcher (parallel, N workers)
                                                     │
                                                     ▼
                                              join → synthesizer → reviewer
                                                     │
                                        approved? ───┴─── not approved & iteration < max
                                          │                          │
                                          ▼                          ▼
                                      finalizer ◄───────────── planner (loop)
                                          │
                                          ▼
                                   notify_action (optional, HIGH-risk gate)
                                          │
                                          ▼
                                         END
```

* `src/graph/graph.py::build_graph()` builds a `langgraph.graph.StateGraph`
  over `ProductionResearchState` (`src/graph/state.py`).
* **Planner** (`src/agents/supervisor.py::make_planner_node`) decomposes the
  question into up to `max_workers` research aspects (`ResearchTask`), or
  refines aspects using the Reviewer's `missing_aspects` feedback on a retry
  loop.
* **Fan-out** (`fan_out_to_workers`) uses LangGraph's `Send` API to run one
  `researcher` node invocation per task, in parallel worker threads.
* **Researcher** (`src/agents/researcher.py`) performs one task: sanitizes
  input (Layer 1), calls the LLM under a timeout + retry policy
  (`src/reliability`), and returns a `Finding`. A single crashed/timed-out
  worker never corrupts the others' results (see `reliability.md`).
* **Synthesizer** (`src/agents/synthesizer.py`) merges all findings into one
  report.
* **Reviewer** (`src/agents/reviewer.py`) grades the synthesis on five axes
  (completeness / factual consistency / evidence quality / logical
  consistency / missing aspects) and either approves or requests another
  Planner iteration (bounded by `max_iterations`).
* **Finalizer** (`make_finalizer_node`) produces the `final_answer` — either
  the approved synthesis, or a clearly-labeled best-effort answer if the
  iteration budget is exhausted without approval.
* **notify_action** (`make_notify_node`) is an optional last step that
  demonstrates the **HIGH-risk tool approval gate end-to-end**: if
  `notify_email` is set on the run, it requests human approval before
  calling the `send_email` tool (see `security.md`).

## 3. State

A single `ProductionResearchState` (`TypedDict`, `src/graph/state.py`) flows
through every node. Each node owns and returns only the keys it is
responsible for (LangGraph merges partial updates), which keeps the ownership
contract explicit and auditable:

| Field | Owner |
|---|---|
| `question`, `identity`, `mode`, `context` | set once at `initial_state()` |
| `tasks`, `iteration` | Planner |
| `findings` | Researcher (accumulated via a reducer, one per worker) |
| `synthesis` | Synthesizer |
| `review` | Reviewer |
| `final_answer`, `blocked`, `block_reason` | Finalizer / guardrails |
| `notify_status` | notify_action |
| `tokens_used`, `cost_usd`, `trace` | every node (additive) |

`context` carries the full `ExecutionContext` (`request_id`, `trace_id`,
`user_id`, `session_id`, `agent_version`, `environment`) as a plain dict —
**not** a Python `contextvar` object — because LangGraph's `Send`-based
fan-out runs each worker in a raw thread that does not inherit
`contextvars`. Every node reconstructs `ExecutionContext` explicitly from
`state["context"]` before starting a trace span. This was a real bug found
and fixed during this upgrade (traces silently lost their parent/session
linkage under parallel fan-out until this was fixed).

## 4. Tools

`src/tools/` holds the two concrete example tools used to prove the tool-risk
model end-to-end:

* `search.py` — `search_web` (**LOW** risk, read-only).
* `actions.py` — `update_record` (**MEDIUM**, write but reversible) and
  `send_email` (**HIGH**, external side effect, gated by human approval).

Every tool call in this codebase is wrapped by `src/security/tool_policy.py`
+ `src/security/guardrails.py` (Layer 3), so risk tier, argument validation,
and authorization are enforced structurally rather than left to each tool's
own discipline. See `security.md` for the full three-layer model.

## 5. MCP

Not implemented. Same honest limitation as the prior review: there is no
MCP client/server/tool-discovery layer in this repository. If external tools
need to be sourced dynamically (rather than the two example tools wired
directly into the graph), that is future scope, not silently pretended to
exist.

## 6. Memory

`src/memory/store.py::SessionMemoryStore` is a minimal, explicit,
**tenant-isolated question→answer cache** (not a vector store, not long-term
user memory). It is wired as an **optional** parameter of
`run_production_research()` (`src/graph/graph.py`): a cache hit returns the
previous answer immediately with `tokens_used=0, cost_usd=0.0`, entirely
bypassing the graph; a cache miss runs the full pipeline and remembers the
result on successful, non-blocked completion. This is deliberately kept at
the wrapper level (not baked into a graph node) so it never interferes with
checkpointer/durability semantics.

**Known limitation**: not yet exposed through the HTTP API (`POST /runs`
does not currently accept an opt-in "check memory first" flag) — this is a
documented scope boundary, see `docs/11-Production-Ready.md`.

## 7. RAG

Not implemented — no ingestion/embedding/vector index/retriever. The
Researcher relies on the LLM's own knowledge and states this as a limitation
in its trace rather than fabricating citations. `evals/metrics.py` still
scores *groundedness* and *citation quality* heuristically, so a future RAG
layer has an evaluation harness ready to plug into.

## 8. Multi-Agent design

Planner/Researcher/Synthesizer/Reviewer are true, separately-defined LangGraph
nodes with narrow state ownership — not one monolithic prompt. Parallel
Researcher fan-out is real (`Send`), bounded by `max_workers`, and each
worker's failure is isolated (a crashed worker becomes one missing/failed
`Finding`, not a whole-run crash, unless the failure is a simulated hard
crash used to test checkpoint/resume — see `reliability.md`). All
orchestration is in-process (single LangGraph runtime, single machine) — no
cross-process/cross-machine task distribution, same honest limitation as
before.
