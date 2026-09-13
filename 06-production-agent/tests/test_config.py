"""Automated tests for ``config.settings`` (Configuration & Secrets).

Run with:

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_config.py -v

These tests never read the real process environment or any real ``.env``
file: every test passes an explicit, in-memory mapping to
``load_settings(env=...)`` so the outcome is fully deterministic regardless
of what secrets happen to be configured on the machine running the suite.
No real secret value appears anywhere in this file.
"""

from __future__ import annotations

import pytest

from config import (
    AgentLimits,
    ConfigurationError,
    RetryPolicy,
    Settings,
    load_settings,
)

# A minimal, complete, *fake* configuration -- not a real credential -- used
# as a baseline that individual tests mutate to exercise one failure at a
# time.
BASE_ENV: dict[str, str] = {
    "OPENAI_API_KEY": "test-fake-key-not-a-real-secret",
    "OPENAI_MODEL": "fake-model-for-tests",
    "OPENAI_BASE_URL": "https://example.invalid/v1",
    "ENVIRONMENT": "test",
}


def _env(**overrides: str) -> dict[str, str]:
    """Build a fresh env mapping from ``BASE_ENV`` with the given overrides.

    Passing ``None`` for a key removes it entirely (simulates "not set").
    """
    merged = dict(BASE_ENV)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Core requirement: the app must fail clearly when a required secret is missing.
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_configuration_error() -> None:
    """The single most important test: no ``OPENAI_API_KEY`` -> immediate,
    specific failure -- never a silent default, never a generic exception
    raised later deep inside an HTTP client."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env=_env(OPENAI_API_KEY=None))

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "Missing required secret" in message


def test_blank_api_key_is_treated_as_missing() -> None:
    """Whitespace-only values must not slip through as "configured"."""
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_settings(env=_env(OPENAI_API_KEY="   "))


def test_empty_environment_mapping_fails_on_missing_secret() -> None:
    """An empty environment (nothing configured at all) must fail on the
    missing secret, not crash with an unrelated KeyError/AttributeError."""
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_settings(env={})


def test_configuration_error_never_leaks_a_secret_value() -> None:
    """Even when a secret *is* present but something else is invalid, the
    raised error message must never echo back the secret value."""
    real_looking_secret = "sk-should-never-appear-in-any-error-message"
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env=_env(OPENAI_API_KEY=real_looking_secret, ENVIRONMENT="not-a-real-environment"))
    assert real_looking_secret not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Model name must be configurable and required.
# ---------------------------------------------------------------------------


def test_missing_model_name_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_MODEL"):
        load_settings(env=_env(OPENAI_MODEL=None))


def test_model_name_is_read_from_environment() -> None:
    settings = load_settings(env=_env(OPENAI_MODEL="my-custom-model-v2"))
    assert settings.model_name == "my-custom-model-v2"


# ---------------------------------------------------------------------------
# Environment must distinguish at least development/test/production.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["development", "test", "production"])
def test_valid_environments_are_accepted(environment: str) -> None:
    overrides: dict[str, str | None] = {"ENVIRONMENT": environment}
    if environment == "production":
        # Production requires an explicit base_url (no public-endpoint fallback).
        overrides["OPENAI_BASE_URL"] = "https://private-gateway.example.invalid/v1"
    settings = load_settings(env=_env(**overrides))
    assert settings.environment == environment


def test_invalid_environment_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="ENVIRONMENT"):
        load_settings(env=_env(ENVIRONMENT="staging-typo"))


def test_missing_environment_defaults_to_development() -> None:
    settings = load_settings(env=_env(ENVIRONMENT=None))
    assert settings.environment == "development"


def test_production_requires_explicit_base_url() -> None:
    """Production must never silently fall back to the public OpenAI
    endpoint -- that could send a private gateway's traffic to the wrong
    place (see docs/01-CONFIGURATION.md)."""
    with pytest.raises(ConfigurationError, match="OPENAI_BASE_URL"):
        load_settings(env=_env(ENVIRONMENT="production", OPENAI_BASE_URL=None))


def test_non_production_defaults_base_url_to_public_endpoint() -> None:
    settings = load_settings(env=_env(OPENAI_BASE_URL=None, ENVIRONMENT="development"))
    assert settings.base_url == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Agent limits (max_workers/max_iterations) and timeouts must be configurable.
# ---------------------------------------------------------------------------


def test_agent_limits_use_defaults_when_unset() -> None:
    settings = load_settings(env=_env())
    assert settings.limits == AgentLimits()


def test_agent_limits_are_overridable_via_environment() -> None:
    settings = load_settings(
        env=_env(
            AGENT_MAX_WORKERS="10",
            AGENT_MAX_ITERATIONS="5",
            AGENT_WORKER_TIMEOUT_SECONDS="45.5",
            AGENT_RUN_TIMEOUT_SECONDS="600",
        )
    )
    assert settings.limits.max_workers == 10
    assert settings.limits.max_iterations == 5
    assert settings.limits.worker_timeout_seconds == 45.5
    assert settings.limits.run_timeout_seconds == 600.0


@pytest.mark.parametrize(
    "name,value",
    [
        ("AGENT_MAX_WORKERS", "0"),
        ("AGENT_MAX_WORKERS", "not-a-number"),
        ("AGENT_MAX_ITERATIONS", "-1"),
        ("AGENT_WORKER_TIMEOUT_SECONDS", "0"),
        ("AGENT_WORKER_TIMEOUT_SECONDS", "-5"),
        ("AGENT_RUN_TIMEOUT_SECONDS", "not-a-number"),
    ],
)
def test_invalid_agent_limits_raise_configuration_error(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        load_settings(env=_env(**{name: value}))


# ---------------------------------------------------------------------------
# Retry policy must be configurable.
# ---------------------------------------------------------------------------


def test_retry_policy_uses_defaults_when_unset() -> None:
    settings = load_settings(env=_env())
    assert settings.retry == RetryPolicy()


def test_retry_policy_is_overridable_via_environment() -> None:
    settings = load_settings(
        env=_env(
            LLM_RETRY_MAX_RETRIES="5",
            LLM_RETRY_BACKOFF_BASE_SECONDS="2",
            LLM_RETRY_BACKOFF_MAX_SECONDS="30",
            LLM_RETRY_STATUS_CODES="429,503",
        )
    )
    assert settings.retry.max_retries == 5
    assert settings.retry.backoff_base_seconds == 2.0
    assert settings.retry.backoff_max_seconds == 30.0
    assert settings.retry.retry_on_status_codes == (429, 503)


def test_retry_max_retries_of_zero_disables_retries() -> None:
    settings = load_settings(env=_env(LLM_RETRY_MAX_RETRIES="0"))
    assert settings.retry.max_retries == 0


def test_retry_negative_max_retries_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="LLM_RETRY_MAX_RETRIES"):
        load_settings(env=_env(LLM_RETRY_MAX_RETRIES="-1"))


def test_retry_backoff_max_below_base_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="LLM_RETRY_BACKOFF_MAX_SECONDS"):
        load_settings(
            env=_env(LLM_RETRY_BACKOFF_BASE_SECONDS="10", LLM_RETRY_BACKOFF_MAX_SECONDS="1")
        )


def test_retry_status_codes_must_be_integers() -> None:
    with pytest.raises(ConfigurationError, match="LLM_RETRY_STATUS_CODES"):
        load_settings(env=_env(LLM_RETRY_STATUS_CODES="429,not-a-code"))


# ---------------------------------------------------------------------------
# Settings must never leak the secret via repr/str (e.g. accidental logging).
# ---------------------------------------------------------------------------


def test_settings_repr_redacts_the_api_key() -> None:
    secret = "sk-real-looking-secret-value-should-be-redacted"
    settings = load_settings(env=_env(OPENAI_API_KEY=secret))
    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert "***redacted***" in repr(settings)


def test_settings_is_a_frozen_dataclass_instance() -> None:
    settings = load_settings(env=_env())
    assert isinstance(settings, Settings)
    with pytest.raises(Exception):
        settings.api_key = "mutated"  # type: ignore[misc]
