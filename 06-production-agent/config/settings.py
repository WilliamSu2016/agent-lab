"""Configuration and secrets loading (Requirement: Configuration & Secrets).

Design rules enforced by this module:

1. Every secret (currently: the LLM API key) is read *exclusively* from
   environment variables -- never hard-coded in this module or anywhere else
   in source, and never given a non-empty default. ``.env`` files are only
   ever loaded by the process entry point (via ``python-dotenv``'s
   ``load_dotenv()``) *before* calling :func:`load_settings`; this module
   itself never touches the filesystem.
2. A missing required secret (or any other invalid required configuration)
   fails loudly and immediately with :class:`ConfigurationError` -- never a
   bare ``KeyError``/``AttributeError``/``TypeError`` raised deep inside an
   HTTP client after a confusing amount of unrelated work has already run.
3. Model name, agent limits (``max_workers``/``max_iterations``), timeouts,
   and retry policy are all configurable via environment variables, with
   safe, explicit defaults for these *non-secret* operational knobs only.
4. ``ENVIRONMENT`` distinguishes at least ``development``/``test``/
   ``production``. ``production`` additionally requires an explicit
   ``OPENAI_BASE_URL`` (no silent fallback to the public OpenAI endpoint --
   see ``src/tracing.py``'s docstring for why sending a private gateway's
   traffic to an unconfigured public endpoint would be an unintended
   data-exfiltration path).

See ``docs/01-CONFIGURATION.md`` for the full reference of every environment
variable this module reads, and ``.env.example`` for a safe template with no
real secret values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ENV_DEVELOPMENT = "development"
ENV_TEST = "test"
ENV_PRODUCTION = "production"
VALID_ENVIRONMENTS: tuple[str, ...] = (ENV_DEVELOPMENT, ENV_TEST, ENV_PRODUCTION)

# Secrets read by this module. NEVER give any of these a non-empty default --
# that would defeat the "fail loudly when a secret is missing" requirement.
REQUIRED_SECRET_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY",)

_PUBLIC_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ConfigurationError(RuntimeError):
    """Raised for any missing/invalid required configuration or secret.

    Every failure this module can produce raises exactly this type (never a
    bare ``KeyError``/``ValueError``), so callers -- and the automated test
    suite (``tests/test_config.py``) -- can assert on configuration failures
    deterministically, and a missing secret always fails the same,
    unambiguous way regardless of which entry point loads it.
    """


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_str(source: Mapping[str, str], name: str, *, default: str | None = None) -> str:
    value = _clean(source.get(name))
    if value:
        return value
    if default is not None:
        return default
    raise ConfigurationError(
        f"Missing required configuration: {name!r}. Set it as an environment "
        f"variable (see .env.example / docs/01-CONFIGURATION.md)."
    )


def _optional_int(source: Mapping[str, str], name: str, default: int, *, minimum: int = 1) -> int:
    raw = _clean(source.get(name))
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name!r} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ConfigurationError(f"{name!r} must be >= {minimum}, got {value!r}.")
    return value


def _optional_float(source: Mapping[str, str], name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = _clean(source.get(name))
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name!r} must be a number, got {raw!r}.") from exc
    if value <= minimum:
        raise ConfigurationError(f"{name!r} must be > {minimum}, got {value!r}.")
    return value


def _optional_int_tuple(source: Mapping[str, str], name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = _clean(source.get(name))
    if not raw:
        return default
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ConfigurationError(
                f"{name!r} must be a comma-separated list of integers, got {raw!r}."
            ) from exc
    if not values:
        raise ConfigurationError(f"{name!r} must contain at least one integer, got {raw!r}.")
    return tuple(values)


@dataclass(frozen=True)
class RetryPolicy:
    """Retry behaviour for outbound LLM calls.

    Configurable, never a secret. ``max_retries=0`` disables retries
    entirely (the first attempt is made and any failure is raised
    immediately) -- useful for ``test`` environments that want deterministic,
    fast failures.
    """

    max_retries: int = 2
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 20.0
    # Only these HTTP status codes are treated as retryable transient errors;
    # everything else (401/403/400/404/...) fails immediately -- see
    # docs/00-PRODUCTION-ARCHITECTURE.md, Reliability Q2/Q3.
    retry_on_status_codes: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)


@dataclass(frozen=True)
class AgentLimits:
    """Hard caps on agent execution, independent of what any LLM proposes."""

    max_workers: int = 6
    max_iterations: int = 3
    worker_timeout_seconds: float = 30.0
    run_timeout_seconds: float = 300.0


@dataclass(frozen=True)
class Settings:
    """Fully resolved, validated configuration for one process run."""

    environment: str
    api_key: str  # secret -- never logged, never included in __repr__/str
    model_name: str
    base_url: str
    limits: AgentLimits
    retry: RetryPolicy

    def __repr__(self) -> str:  # pragma: no cover - defensive; never leak the key
        return (
            f"Settings(environment={self.environment!r}, api_key='***redacted***', "
            f"model_name={self.model_name!r}, base_url={self.base_url!r}, "
            f"limits={self.limits!r}, retry={self.retry!r})"
        )

    __str__ = __repr__


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load and validate all configuration from environment variables.

    ``env`` defaults to ``os.environ`` (the normal case for every real entry
    point). Tests pass an explicit mapping instead, so they can simulate a
    missing/invalid secret or setting without mutating the real process
    environment or depending on whatever ``.env`` happens to exist on disk.

    Raises :class:`ConfigurationError` -- and only that -- for any missing or
    invalid required configuration, including every required secret. This
    function never returns a ``Settings`` with an empty ``api_key``, and it
    never reads a ``.env`` file itself (that is the caller's job, via
    ``dotenv.load_dotenv()``, before this function runs).
    """
    if env is None:
        import os

        source: Mapping[str, str] = os.environ
    else:
        source = env

    environment = _clean(source.get("ENVIRONMENT")).lower() or ENV_DEVELOPMENT
    if environment not in VALID_ENVIRONMENTS:
        raise ConfigurationError(
            f"Invalid ENVIRONMENT={environment!r}. Must be one of {VALID_ENVIRONMENTS}."
        )

    # Requirement 1 & 2: secrets come only from the environment, and a
    # missing one fails immediately and explicitly -- never silently
    # defaulted, never discovered later as a confusing HTTP 401.
    missing_secrets = [name for name in REQUIRED_SECRET_ENV_VARS if not _clean(source.get(name))]
    if missing_secrets:
        raise ConfigurationError(
            "Missing required secret(s): "
            + ", ".join(missing_secrets)
            + ". Set them as environment variables (see .env.example); "
              "secrets must never be hard-coded in source or committed to git."
        )
    api_key = _clean(source.get("OPENAI_API_KEY"))

    # Requirement 6: model name must be configurable (no hard-coded default
    # model anywhere in source -- an empty/missing value fails loudly).
    model_name = _require_str(source, "OPENAI_MODEL")

    # Production requires an explicit endpoint: silently falling back to the
    # public OpenAI backend in production could send traffic intended for a
    # private/self-hosted gateway to the wrong place.
    base_url_default = None if environment == ENV_PRODUCTION else _PUBLIC_OPENAI_BASE_URL
    base_url = _require_str(source, "OPENAI_BASE_URL", default=base_url_default)

    # Requirement 7/8/9: agent limits, timeouts, and max iterations are all
    # independently configurable via environment variables.
    limits = AgentLimits(
        max_workers=_optional_int(source, "AGENT_MAX_WORKERS", AgentLimits.max_workers, minimum=1),
        max_iterations=_optional_int(source, "AGENT_MAX_ITERATIONS", AgentLimits.max_iterations, minimum=1),
        worker_timeout_seconds=_optional_float(
            source, "AGENT_WORKER_TIMEOUT_SECONDS", AgentLimits.worker_timeout_seconds
        ),
        run_timeout_seconds=_optional_float(
            source, "AGENT_RUN_TIMEOUT_SECONDS", AgentLimits.run_timeout_seconds
        ),
    )

    # Requirement 10: retry policy is fully configurable.
    retry = RetryPolicy(
        max_retries=_optional_int(source, "LLM_RETRY_MAX_RETRIES", RetryPolicy.max_retries, minimum=0),
        backoff_base_seconds=_optional_float(
            source, "LLM_RETRY_BACKOFF_BASE_SECONDS", RetryPolicy.backoff_base_seconds
        ),
        backoff_max_seconds=_optional_float(
            source, "LLM_RETRY_BACKOFF_MAX_SECONDS", RetryPolicy.backoff_max_seconds
        ),
        retry_on_status_codes=_optional_int_tuple(
            source, "LLM_RETRY_STATUS_CODES", RetryPolicy.retry_on_status_codes
        ),
    )
    if retry.backoff_max_seconds < retry.backoff_base_seconds:
        raise ConfigurationError(
            "LLM_RETRY_BACKOFF_MAX_SECONDS must be >= LLM_RETRY_BACKOFF_BASE_SECONDS "
            f"(got max={retry.backoff_max_seconds!r} < base={retry.backoff_base_seconds!r})."
        )

    return Settings(
        environment=environment,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        limits=limits,
        retry=retry,
    )
