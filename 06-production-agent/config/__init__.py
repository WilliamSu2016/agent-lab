"""Centralized configuration and secrets loading for this project.

Every process entry point (CLI ``main()`` functions under ``src/``) must load
its configuration exclusively through :func:`load_settings` in
``config.settings`` -- never by reading ``os.environ`` directly, and never by
hard-coding a model name, API key, timeout, or limit inline.

Public API:

    from config import load_settings, ConfigurationError, Settings

    settings = load_settings()  # raises ConfigurationError if misconfigured
"""

from __future__ import annotations

from config.settings import (
    AgentLimits,
    ConfigurationError,
    ENV_DEVELOPMENT,
    ENV_PRODUCTION,
    ENV_TEST,
    RetryPolicy,
    Settings,
    VALID_ENVIRONMENTS,
    load_settings,
)

__all__ = [
    "AgentLimits",
    "ConfigurationError",
    "ENV_DEVELOPMENT",
    "ENV_PRODUCTION",
    "ENV_TEST",
    "RetryPolicy",
    "Settings",
    "VALID_ENVIRONMENTS",
    "load_settings",
]
