# Production Research Agent

A real, working Planner → Researcher(s) → Synthesizer → Reviewer research
Agent (LangGraph), upgraded end-to-end to be production-ready: three-layer
security guardrails, durable checkpointed execution, full observability,
cost/latency budget policy with automatic degradation, a real offline/
regression evaluation harness, and a deployable FastAPI service.

This package is the direct answer to `docs/10-Final-Review.md` in the parent
repo (`06-production-agent/`): every capability that previously existed as
an isolated, unintegrated module (durable execution, security, observability,
cost, evaluation) is now wired directly into the one graph that is actually
served.

## Architecture at a glance

```
supervisor_entry (Layer 1 input guardrail)
        │
        ▼
     planner ──► fan_out (Send) ──► researcher × N (parallel, Layer 2/3 guardrails)
        ▲                                  │
        │                                  ▼
        └────────── reviewer ◄──────── synthesizer
                        │ approved
                        ▼
                    finalizer (output guardrail)
                        │
                        ▼
                  notify_action (HIGH-risk approval gate, optional)
                        │
                        ▼
                       END
```

See `docs/architecture.md` for the full breakdown.

## Repository layout

```
src/
  agents/         # Planner, Researcher, Synthesizer, Reviewer, notify_action
  tools/          # search_web (LOW), update_record (MEDIUM), send_email (HIGH)
  graph/          # StateGraph wiring, state schema, checkpointer, recovery
  memory/         # optional tenant-isolated question -> answer cache
  security/       # 3-layer guardrails, tool risk tiers, authorization, sanitization
  reliability/    # retry, timeout, idempotency
  observability/  # structured logging, tracing (execution tree), metrics
  cost/           # budget policy (fast/balanced/quality), cost estimator
  api/            # FastAPI app: /runs, /approvals, /health, /ready, /metrics
  config/         # environment-driven settings

evals/            # 20-task dataset + offline/regression evaluation pipeline
tests/            # unit / integration / security / failure_injection
deployment/       # Dockerfile, docker-compose.yml
docs/             # architecture, reliability, security, observability,
                  # evaluation, deployment
```

## Quickstart

```powershell
# From the repo's shared venv (see repo root AGENTS.md for setup):
..\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run the full test suite (unit + integration + security + failure_injection):
..\.venv\Scripts\python.exe -m pytest tests -q

# Run the offline evaluation and regenerate the baseline report:
..\.venv\Scripts\python.exe -m evals.runner   # or see evals/adapter.py for programmatic use

# Start the API locally (requires .env, copied from .env.example):
..\.venv\Scripts\python.exe -m src.api.main
```

Local LangGraph dev server (no auth, localhost only):

```bash
docker compose -f deployment/docker-compose.yml --profile dev up langgraph-dev
```

Production container:

```bash
docker compose -f deployment/docker-compose.yml --profile prod up -d agent-api
```

## Tests

153 tests, all passing:

| Suite | Covers |
|---|---|
| `tests/unit/` | state schema, tool tiers, cost policy, checkpointer, memory cache, evaluation pipeline |
| `tests/integration/` | full API lifecycle (health/ready/auth/validation/run/resume/tenant isolation), HIGH-risk approval workflow end to end |
| `tests/security/` | prompt injection, authorization, tool policy, tenant/data isolation — 10+ attack scenarios |
| `tests/failure_injection/` | LLM failure, tool failure, MCP unavailability, worker crash + resume, duplicate requests, concurrency |

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — orchestration, state, tools, MCP/RAG/memory scope, multi-agent design
- [`docs/reliability.md`](docs/reliability.md) — checkpointing, crash/resume, retry/timeout, idempotency, checkpoint vs memory vs database vs execution history
- [`docs/security.md`](docs/security.md) — three-layer guardrails, tool risk tiers, the `notify_action` approval gate, attack test coverage
- [`docs/observability.md`](docs/observability.md) — tracing/metrics/logging, dashboard specification
- [`docs/evaluation.md`](docs/evaluation.md) — 20-task dataset, 11 evaluation dimensions, baseline/regression workflow
- [`docs/deployment.md`](docs/deployment.md) — `langgraph.json`, Docker, API surface, auth/rate-limit/health/ready/graceful-shutdown

See also, at the parent repo root: `docs/10-Final-Review.md` (the review this
package answers) and `docs/11-Production-Ready.md` (summary of what changed
and known remaining limitations).

## Known, documented limitations

- `RunRegistry` (HTTP-visible run status) is in-memory per process; the
  underlying LangGraph checkpoint is durable and correct, but a process
  restart loses the HTTP-visible run index unless the caller retains the
  `thread_id`. See `docs/reliability.md` §6.
- `notify_email` (the HIGH-risk approval demo) and `memory_store` (the
  answer cache) are wired at the Python/graph level and proven in
  integration tests, but not yet exposed as `POST /runs` request fields.
  See `docs/security.md` and `docs/architecture.md` §6.
- No MCP layer, no RAG/retrieval layer — honestly out of scope, not silently
  assumed. See `docs/architecture.md` §5/§7.
- SQLite checkpointer/in-memory rate-limiter/approval-store are
  single-instance; horizontal scaling needs a shared backend. See
  `docs/deployment.md` §6.
