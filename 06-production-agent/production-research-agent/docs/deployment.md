# Deployment

## 1. `langgraph.json`

```json
{
  "dependencies": ["."],
  "graphs": {
    "production_research_agent": "./src/graph/entrypoint.py:graph"
  },
  "env": ".env",
  "python_version": "3.10"
}
```

* **Graph entrypoint**: `src/graph/entrypoint.py:graph` — the exact same
  `build_graph()` used by the FastAPI app and by every test, so `langgraph
  dev` (local dev server) exercises the identical, real pipeline — not a
  separate demo graph (the core defect this whole upgrade closes).
* **Dependencies**: `["."]` — this package itself, installed editable so the
  dev server picks up local source changes without a rebuild.
* **Environment variables**: sourced from `.env` (see `.env.example` for the
  full documented list) — never hard-coded in `langgraph.json` itself.

## 2. Development — local LangGraph server

```bash
docker compose -f deployment/docker-compose.yml --profile dev up langgraph-dev
```

Runs `langgraph dev` against `langgraph.json` on `:8123`, repo mounted
read-write for live reload. No auth/rate limiting — a local,
single-developer tool, never exposed beyond localhost.

## 3. Production — this project's own FastAPI app

```bash
docker compose -f deployment/docker-compose.yml --profile prod up -d agent-api
```

* **Persistent storage**: the production SQLite checkpointer
  (`src/graph/checkpointer.py::sqlite_checkpointer`) writes to
  `/app/traces/production_checkpoints.sqlite3`, mounted on a **named volume**
  (`checkpoints:`) — survives `docker compose down` (only `down -v` removes
  it) and any container replacement/redeploy resumes exactly where the
  previous instance left off.
* **Production checkpointer**: never `InMemorySaver` in this profile.
* **Production environment variables**: supplied only via `env_file: ../.env`
  at container run time — `deployment/Dockerfile` copies no `.env`, no
  `tests/`, no `docs/`, no `evals/` into the image (kept out via
  `.dockerignore`), and no secret is ever set as a Dockerfile `ARG`/`ENV`
  (both get baked into image layers permanently). `src/api/main.py` reads
  every setting/secret from the environment at start-up and **fails loudly
  before binding a socket** if anything required (`OPENAI_API_KEY`,
  `API_AUTH_TOKENS`, etc.) is missing — never a container that "starts" with
  broken configuration.

## 4. API surface (`src/api/app.py`)

| Endpoint | Purpose |
|---|---|
| `POST /runs` | Start a new run (`thread_id`-per-run); returns immediately (`202`, `status="running"`) |
| `GET /runs/{id}` | Current run summary (status/final_answer/error) |
| `GET /runs/{id}/state` | Full execution history + pending next tasks (see `reliability.md` §5) |
| `POST /runs/{id}/resume` | Resume an `interrupted`/`failed` run from its last checkpoint |
| `GET /approvals/{id}` | Inspect a pending/decided HIGH-risk tool approval |
| `POST /approvals/{id}/decide` | Approve/deny a HIGH-risk tool call (closes the "approval store exists but is unreachable" gap from the Final Review) |
| `GET /health` | Liveness — process is up |
| `GET /ready` | Readiness — not accepting traffic during shutdown drain, plus any injected readiness probes |
| `GET /metrics` | Dashboard-ready summary of every metric recorded so far (auth-protected) |

### Cross-cutting requirements, all implemented in `src/api/app.py`/`auth.py`/`rate_limit.py`/`errors.py`

1. **Authentication** — `src/api/auth.py::make_verify_bearer_token`; every
   `/runs*` and `/approvals*` route requires a bearer token from
   `API_AUTH_TOKENS`. The app refuses to start "open" if that env var is empty.
2. **Request validation** — Pydantic schemas (`src/api/schemas.py`) reject
   malformed bodies with a structured `422` before any handler code runs.
3. **Rate limiting** — `src/api/rate_limit.py::RateLimiter`, per-`user_id`,
   configurable window/max requests (`RATE_LIMIT_*`).
4. **Timeout** — every run is bounded by its mode's
   `ExecutionPolicy.budget.timeout_seconds` (`src/cost/policy.py`); a timeout
   is reported as `status="interrupted"` (resumable), not a hung connection.
5. **Error handling** — `src/api/errors.py::EXCEPTION_HANDLERS` maps every
   known exception type (auth, validation, not-found, invalid-state,
   guardrail-blocked, timeout) to a structured JSON error response with a
   stable shape — no bare tracebacks leak to callers.
6. **Request ID** — `request_context_middleware` assigns/propagates
   `X-Request-ID` on every request/response and threads it into
   `ExecutionContext` for that request's whole trace tree.
7. **Health check** — `GET /health` (liveness).
8. **Readiness check** — `GET /ready` (returns `503` while shutting down or
   while any injected readiness probe reports unhealthy).
9. **Graceful shutdown** — the FastAPI `lifespan` context flips a
   `shutting_down` flag (so `/ready` starts failing immediately, letting a
   load balancer stop sending new traffic) and then `await
   registry.drain(shutdown_drain_timeout_seconds)` waits for in-flight runs
   to finish (bounded by `API_GRACEFUL_SHUTDOWN_SECONDS`) before the process exits.

## 5. Secrets policy

* No secret is ever written into the Docker image (`Dockerfile` comment
  block spells this out explicitly; `.dockerignore` excludes `.env`).
* `.env.example` documents every variable's name and purpose with a **blank**
  or clearly-non-secret placeholder value — never a real key.
* Real values live only in a local, git-ignored `.env`, supplied to the
  container via `env_file:`.

## 6. Known limitation

`RunRegistry` is an in-memory index of run metadata (see `reliability.md`
§6) — a horizontally-scaled, multi-replica deployment needs this (and
`RateLimiter`/`ApprovalStore`) backed by a shared store (Redis/Postgres)
instead of per-process memory. The underlying LangGraph checkpointer
(SQLite here) would also need to move to a shared backend (e.g. Postgres)
for true multi-replica writes — SQLite is a single-writer file, adequate for
a single-instance production deployment as shipped here, not for horizontal
scaling. This is a conscious, documented scope boundary (see
`docs/11-Production-Ready.md`), not an oversight.
