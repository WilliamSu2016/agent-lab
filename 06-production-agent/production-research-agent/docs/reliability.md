# Reliability

This document covers everything needed to answer: *"what happens when a
worker crashes, an LLM call times out, or two requests race each other?"* —
and shows it is now wired into the **real** research pipeline, not a
separate toy graph.

## 1. Checkpointing (durable execution)

* `src/graph/checkpointer.py` provides `sqlite_checkpointer(path)` — a
  **production-capable** LangGraph `SqliteSaver`, backed by a real file (or
  `:memory:` only in tests). `InMemorySaver` is never used in production
  code paths.
* Every run is identified by a `thread_id` (one per HTTP `/runs` request —
  see `deployment.md`). LangGraph checkpoints the full `ProductionResearchState`
  after every graph super-step (Planner done, each Researcher done, Synthesizer
  done, Reviewer done, Finalizer done).
* `src/graph/recovery.py`:
  - `run_or_crash(graph, state, thread_id)` — invokes the graph normally;
    used for the first, non-recovery attempt.
  - `resume(graph, thread_id)` — invokes the graph with `None` input against
    an existing `thread_id`; LangGraph resumes from the last successfully
    checkpointed super-step, **not from the beginning**.
  - `get_execution_history(graph, thread_id)` — returns the full checkpoint
    history for a thread (see §5).
  - `get_pending_tasks(graph, thread_id)` — introspects which graph node(s)
    would run next, so an operator/API caller can see *where* a resume will
    continue.

## 2. Verified crash/resume scenario

`tests/failure_injection/test_worker_crash.py` (and the earlier
`06-production-agent/tests/test_recovery.py` this was built on) proves the
exact required scenario:

```
Research A → succeeds
Research B → succeeds
Research C → raises WorkerCrash(BaseException)   # simulates a process crash,
                                                   # not a normal exception
```

`WorkerCrash` deliberately subclasses `BaseException` (not `Exception`) so it
is **not** swallowed by the Researcher node's normal resilience
`except Exception` handler — this mirrors a real crash (segfault, `SIGKILL`,
`SystemExit`), which no in-process `try/except` can catch either.

After restart + `resume()`:
* Findings for A and B are read back from the checkpoint — **their
  Researcher LLM calls are never re-invoked** (call-count assertions in the
  test enforce this).
* Only C's Researcher node re-executes, from a fresh worker invocation for
  that one task — not from the Planner, not from A/B's already-committed
  state.

## 3. Retry & timeout (`src/reliability/`)

* `retry.py` — exponential backoff with jitter, retryable-status-code
  allowlist (`408/409/429/500/502/503/504` by default), hard retry limit.
  Applied around every LLM call in `src/agents/llm.py::TextLLMCall`.
* `timeout.py` — `run_with_timeout()` wraps LLM calls (per-worker) and whole
  runs (`src/api/runs.py`'s background task) with a hard deadline; a timeout
  raises `ToolTimeoutError`, which the API surfaces as `status="interrupted"`
  (resumable) rather than an opaque 500.
* `errors.py` — typed reliability errors (`ToolTimeoutError`, retry
  exhaustion, etc.) so callers can branch on failure *kind*, not string
  matching.

Expected behavior per failure kind (full matrix in
`06-production-agent/docs/09-FAILURE-MATRIX.md`):

| Failure | Expected behavior |
|---|---|
| LLM timeout | retry → backoff → retry limit → structured failure (never a silent hang) |
| LLM 5xx | retry (same policy) → structured failure after limit |
| Tool timeout / 5xx | same retry/backoff, then a typed error surfaces to the caller |
| Worker crash | checkpoint → restart → resume (see §2) |
| Checkpoint failure | write is retried; if it cannot be persisted, the run is reported failed rather than silently losing state |

## 4. Idempotency & concurrency

* `src/reliability/idempotency.py::IdempotencyStore` — a request submitted
  twice with the same idempotency key returns the **same** stored result
  instead of re-running (and re-charging) the pipeline. Used by
  `src/security/authorization.py::ApprovalStore.submit_or_get()` so that a
  `resume()` retry after a pending `ApprovalRequiredError` polls the *same*
  approval request rather than creating a duplicate one each time.
* Concurrent requests against the same `thread_id` are exercised in
  `tests/failure_injection/test_concurrency.py` and `test_duplicate_request.py`
  — duplicate submissions are detected and short-circuited to the existing
  result; concurrent *distinct* runs do not interfere (`checkpoint_ns`/
  `thread_id` isolation).

## 5. Execution history vs. memory vs. database

| | **Checkpoint** | **In-memory state (a live Python object)** | **Database (business data)** | **Execution history** |
|---|---|---|---|---|
| What it stores | A serialized snapshot of `ProductionResearchState` after each super-step | The current run's variables while the process is alive | Durable *business* records (e.g. what `update_record`/`send_email` actually changed) | The ordered list of all checkpoints for a `thread_id` |
| Survives a crash? | Yes (it's on disk/DB) | No — lost the instant the process dies | Yes | Yes (it *is* the checkpoints) |
| Purpose | Resume execution exactly where it left off | Fast access during one live run | Record of effects on the world, independent of the agent's control flow | Audit / debugging: "what did this run actually do, step by step?" |
| API | `sqlite_checkpointer()` + LangGraph's `checkpointer.put/get` | plain Python state dict | tool implementations (`src/tools/actions.py`) | `recovery.get_execution_history()` |

`get_execution_history()` is what lets an operator answer *"why did run X
fail?"* by replaying every intermediate state, not just the final one.

## 6. Known limitation

`RunRegistry` (`src/api/runs.py`) — the HTTP-visible `status`/`final_answer`
per run — is an **in-memory index**, not itself checkpointed. If the API
process restarts, the underlying LangGraph checkpoint (the real durable
state) survives and `resume()` still works correctly at the graph level, but
the HTTP-visible run record is lost unless a caller already has the
`thread_id` and re-registers it. This is called out explicitly rather than
silently glossed over — see `docs/11-Production-Ready.md`.
