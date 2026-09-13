# syntax=docker/dockerfile:1
# Production image for the Production Research Agent API (src/api/main.py).
# Build context is the `production-research-agent/` directory:
#
#   docker build -f deployment/Dockerfile -t production-research-agent:latest .
#
# Secrets policy: this image NEVER contains a secret value. OPENAI_API_KEY /
# API_AUTH_TOKENS / etc. are supplied only at *container run time* via
# `docker run --env-file .env` or docker-compose's `env_file:` -- never as a
# Dockerfile ARG/ENV (both get baked into image layers), and `.env` is
# excluded from the build context by `.dockerignore` so a stray `COPY . .`
# can never copy it in either.

FROM python:3.10.8-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.10.8-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home /app app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src
# NOTE: no `.env`, no tests/, no docs/, no evals/ -- keeps the image small
# and guarantees no local secret file can ever be baked in by accident
# (see .dockerignore).

RUN mkdir -p /app/traces && chown -R app:app /app
VOLUME ["/app/traces"]

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# main.py reads every setting/secret from the environment at start-up and
# fails loudly before binding a socket if anything required is missing --
# never a container that "starts" with broken configuration.
CMD ["python", "-m", "src.api.main"]
