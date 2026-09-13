"""Production API layer for the Deployment experiment (docs/08-DEPLOYMENT.md).

This package is the "Production" half of the Deployment experiment's
Development-vs-Production split:

* **Development** = the stock LangGraph CLI dev server (``langgraph dev``),
  configured by ``deployment/langgraph.json``. It gives fast local
  iteration against the graph itself, with no auth/rate-limiting/etc. --
  appropriate for a single developer on their own machine, never exposed
  to real traffic.
* **Production** = this package: a hand-written FastAPI application that
  wraps the *same* durable graph (``src.durable``) behind a small,
  deliberately explicit HTTP contract (``POST /runs``, ``GET /runs/{id}``,
  ``GET /runs/{id}/state``, ``POST /runs/{id}/resume``, ``/health``,
  ``/ready``) and enforces every cross-cutting requirement the experiment
  lists (auth, validation, rate limiting, timeout, error handling, request
  id, health/readiness, graceful shutdown) -- none of which the stock dev
  server is responsible for.

Modules:

    schemas.py     Pydantic request/response models (Requirement 2).
    auth.py        Bearer-token authentication + identity propagation
                   (Requirements 1 and, via ``src.security.authorization``,
                   part of the Security experiment's Requirement 4).
    rate_limit.py  In-memory sliding-window rate limiter (Requirement 3).
    errors.py      Structured error responses (Requirement 5).
    runs.py        The run registry: thread_id-per-run lifecycle on top of
                   ``src.durable`` (crash/resume, execution history).
    app.py         FastAPI app factory: routes, middleware, lifespan
                   (request id, timeout, health/ready, graceful shutdown).
    main.py        Process entry point (``python -m src.api.main``).
"""
