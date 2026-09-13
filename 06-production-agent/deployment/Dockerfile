# syntax=docker/dockerfile:1
# Production image for the Deployment experiment's FastAPI app
# (src/api/main.py). Build context is the repository root:
#
#   docker build -f deployment/Dockerfile -t production-agent:latest .
#
# Secrets policy (required by the experiment): this image NEVER contains a
# secret value. OPENAI_API_KEY / API_AUTH_TOKENS / etc. are supplied only
# at *container run time* via `docker run --env-file .env` or
# `docker-compose`'s `env_file:` -- never as a Dockerfile `ARG`/`ENV`
# (both get baked into image layers and `docker history`/`docker inspect`
# would leak them), and `.env` itself is excluded from the build context by
# `.dockerignore` so a stray `COPY . .` can never copy it in either.

# ---------------------------------------------------------------------------
# Stage 1: build a venv with dependencies only (better layer caching: this
# stage only re-runs when requirements.txt changes, not on every source edit).
# ---------------------------------------------------------------------------
FROM python:3.10.8-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime image -- copy the venv + source only, no build tooling.
# ---------------------------------------------------------------------------
FROM python:3.10.8-slim AS runtime

# Never run the app as root -- limits blast radius of a compromised
# dependency/process.
RUN groupadd --system app && useradd --system --gid app --home /app app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY config ./config
COPY src ./src
# NOTE: no `.env`, no `requirements.txt` build artifacts, no tests/ or
# docs/ -- keeps the image small and guarantees no local secret file can
# ever be present even by accident (see .dockerignore).

# Persistent checkpointer storage -- a real deployment mounts this path as
# a volume (see docker-compose.yml's `checkpoints` volume); without a
# mounted volume this directory is still writable so the container works
# standalone, but its contents do not survive a container replacement.
RUN mkdir -p /app/traces && chown -R app:app /app
VOLUME ["/app/traces"]

USER app
EXPOSE 8000

# Requirement 7 (health check): Docker's own HEALTHCHECK, independent of
# whatever orchestrator-level probe (Kubernetes liveness/readiness) also
# calls the same endpoints -- useful for plain `docker run`/docker-compose
# deployments that have no separate orchestrator.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# `main.py` reads every setting/secret from the environment at start-up
# (see config/settings.py) and fails loudly before binding a socket if
# anything required is missing -- never a container that "starts" with
# broken configuration.
CMD ["python", "-m", "src.api.main"]
