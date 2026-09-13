"""Requirement 1 (Authentication) + Requirement 4 from the Security
experiment (user identity propagation), applied at the HTTP boundary.

Two independent checks, deliberately not conflated:

* **Authentication** ("is this caller allowed to talk to this API at
  all?") -- a bearer token compared against a configured allow-list. A
  request with a missing/invalid token never reaches a route handler.
* **Identity propagation** ("who is the authenticated caller, for
  authorization/tenant-isolation/observability purposes?") -- read from
  explicit ``X-User-Id``/``X-Tenant-Id``/``X-User-Roles`` headers into an
  ``src.security.authorization.Identity`` and passed explicitly to every
  downstream call (the run registry, the tracer, the logger) -- never a
  thread-local/global "current user" (see
  ``src/security/authorization.py``'s module docstring for why that
  matters for a concurrent, multi-tenant server).

A valid bearer token proves *authentication*; it does not by itself prove
*which* identity is calling -- a gateway/token-issuer is expected to have
already verified that the ``X-User-Id``/``X-Tenant-Id`` headers it
forwards are genuine (this API trusts its network perimeter for that, the
same trust boundary assumption most internal services make once they sit
behind an authenticated gateway).
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from src.security.authorization import Identity


class AuthenticationError(HTTPException):
    def __init__(self, detail: str = "Missing or invalid bearer token."):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


@dataclass(frozen=True)
class AuthConfig:
    """Configured once at process start-up (``app.py``'s lifespan) from
    ``API_AUTH_TOKENS`` (comma-separated). An empty allow-list is refused
    outright -- see :func:`from_env` -- so the API can never accidentally
    start up "open" because a deployer forgot to set the variable."""

    valid_tokens: frozenset[str]

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AuthConfig":
        source = env if env is not None else os.environ
        raw = (source.get("API_AUTH_TOKENS") or "").strip()
        tokens = frozenset(t.strip() for t in raw.split(",") if t.strip())
        if not tokens:
            raise ValueError(
                "API_AUTH_TOKENS must be set to at least one comma-separated bearer token; "
                "refusing to start an API with no authentication configured."
            )
        return cls(valid_tokens=tokens)

    def is_valid(self, token: str) -> bool:
        # constant-time compare against every configured token: avoids
        # leaking token length/prefix via timing even though this is an
        # allow-list of a handful of tokens, not a single secret.
        return any(hmac.compare_digest(token, candidate) for candidate in self.valid_tokens)


def make_verify_bearer_token(auth_config: AuthConfig):
    """Returns a FastAPI dependency bound to ``auth_config`` (so tests can
    inject their own token set instead of reading real process env vars)."""

    def verify_bearer_token(authorization: str = Header(default="")) -> None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token or not auth_config.is_valid(token):
            raise AuthenticationError()

    return verify_bearer_token


def get_identity(
    x_user_id: str = Header(..., min_length=1, description="Caller identity, propagated explicitly."),
    x_tenant_id: str = Header(default="default"),
    x_user_roles: str = Header(default="user"),
) -> Identity:
    """FastAPI dependency: builds an explicit :class:`Identity` from
    request headers for this one request only -- never cached/reused
    across requests, so concurrent requests for different tenants can
    never cross-contaminate (see module docstring)."""
    roles = frozenset(r.strip() for r in x_user_roles.split(",") if r.strip())
    return Identity(user_id=x_user_id, tenant_id=x_tenant_id, roles=roles)


__all__ = ["AuthenticationError", "AuthConfig", "make_verify_bearer_token", "get_identity"]
