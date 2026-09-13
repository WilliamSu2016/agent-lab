"""Layer 2 support: identity propagation, role-based authorization, tenant
isolation, and the approval gate that HIGH-risk tools must pass through.

Requirements covered here:

3. Authorization check       -- :func:`check_role_authorization`
4. User identity propagation -- :class:`Identity` is an explicit,
   caller-supplied value threaded through every call (never a
   thread-local/global "current user" -- see the module docstring note
   below for why that distinction matters).
5. Tenant isolation          -- :func:`check_tenant_isolation`
7. High-risk action approval -- :class:`ApprovalStore` +
   :class:`ApprovalRequiredError`/:class:`ApprovalDeniedError`, wired into
   ``guardrails.AgentWorkflowGuardrail`` so a HIGH-risk tool call always
   goes ``Agent -> approval -> Tool``, never ``Agent -> Tool`` directly.

Why identity is an explicit value, not a thread-local
------------------------------------------------------
A thread-local/global "current user" is a classic source of
identity-leak bugs in concurrent agent systems (e.g. this project's own
fan-out research workers, experiment 4/5/14): if two requests for two
different tenants are ever handled concurrently on a shared
thread-pool/event loop, a thread-local can leak one request's identity
into another's tool calls. Every function in this module instead takes
an explicit :class:`Identity` argument, and every guardrail call site
(``guardrails.py``) must pass one through explicitly -- there is no
"ambient" identity anywhere in this package.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.security.tool_policy import ToolRiskLevel, ToolSpec


class AuthorizationError(PermissionError):
    """Base class for every Layer 2 authorization failure."""


class InsufficientRoleError(AuthorizationError):
    def __init__(self, identity: "Identity", tool_name: str, required_roles: frozenset[str]):
        super().__init__(
            f"user_id={identity.user_id!r} (roles={sorted(identity.roles)!r}) lacks a required "
            f"role {sorted(required_roles)!r} to call tool {tool_name!r}."
        )
        self.identity = identity
        self.tool_name = tool_name
        self.required_roles = required_roles


class TenantIsolationError(AuthorizationError):
    """Raised when an identity from one tenant attempts to touch a
    resource that belongs to a different tenant. Never bypassed, even for
    otherwise-privileged roles -- tenant isolation and role authorization
    are independent checks (see ``guardrails.AgentWorkflowGuardrail
    .authorize_tool_call``, which always runs both)."""

    def __init__(self, identity: "Identity", resource_tenant_id: str):
        super().__init__(
            f"user_id={identity.user_id!r} belongs to tenant {identity.tenant_id!r} and may not "
            f"access a resource belonging to tenant {resource_tenant_id!r}."
        )
        self.identity = identity
        self.resource_tenant_id = resource_tenant_id


@dataclass(frozen=True)
class Identity:
    """A caller's identity: who they are, which tenant they belong to, and
    which roles they hold. Immutable and always passed explicitly --
    Requirement 4, "user identity propagation"."""

    user_id: str
    tenant_id: str
    roles: frozenset[str] = field(default_factory=frozenset)

    def has_any_role(self, roles: frozenset[str]) -> bool:
        if not roles:
            return True
        return bool(self.roles & roles)


# Minimum role(s) required to invoke a tool at each risk tier at all (HIGH
# risk still additionally requires the approval gate below, even for a
# caller who holds the "admin" role -- role authorization and approval are
# two separate, both-required controls).
DEFAULT_MINIMUM_ROLES: dict[ToolRiskLevel, frozenset[str]] = {
    ToolRiskLevel.LOW: frozenset(),  # any authenticated identity
    ToolRiskLevel.MEDIUM: frozenset({"editor", "admin"}),
    ToolRiskLevel.HIGH: frozenset({"admin"}),
}


def check_role_authorization(
    identity: Identity,
    tool: ToolSpec,
    *,
    minimum_roles: Mapping[ToolRiskLevel, frozenset[str]] | None = None,
) -> None:
    """Requirement 3: authorization check. Raises
    :class:`InsufficientRoleError` if ``identity`` does not hold at least
    one of the roles required for ``tool``'s risk tier."""
    table = minimum_roles or DEFAULT_MINIMUM_ROLES
    required = table.get(tool.risk_level, frozenset())
    if not identity.has_any_role(required):
        raise InsufficientRoleError(identity, tool.name, required)


def check_tenant_isolation(identity: Identity, resource_tenant_id: str) -> None:
    """Requirement 5: tenant isolation. Raises :class:`TenantIsolationError`
    if the resource a tool call would touch belongs to a different tenant
    than the caller."""
    if resource_tenant_id != identity.tenant_id:
        raise TenantIsolationError(identity, resource_tenant_id)


# ---------------------------------------------------------------------------
# High-risk action approval (Requirement 7).
# ---------------------------------------------------------------------------


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass
class ApprovalRequest:
    """One pending (or decided) approval for a single HIGH-risk tool call.
    Carries enough information for a human/out-of-band approver to decide
    without re-deriving anything from agent state."""

    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_by: Identity
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str | None = None
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "requested_by": self.requested_by.user_id,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
        }


class ApprovalRequiredError(RuntimeError):
    """Raised in place of executing a HIGH-risk tool call. This is the
    concrete mechanism behind "Agent -> approval -> Tool" instead of
    "Agent -> Tool": the caller receives a pending :class:`ApprovalRequest`
    instead of a tool result, and must route it through an approval
    workflow (:meth:`ApprovalStore.decide`) before the *same* tool call can
    be retried and actually reach the tool function."""

    def __init__(self, request: ApprovalRequest):
        super().__init__(
            f"Tool {request.tool_name!r} is HIGH risk and requires approval "
            f"before it can run (request_id={request.request_id})."
        )
        self.request = request


class ApprovalDeniedError(RuntimeError):
    def __init__(self, request: ApprovalRequest):
        super().__init__(f"Approval request {request.request_id} for tool {request.tool_name!r} was denied.")
        self.request = request


class ApprovalNotFoundError(KeyError):
    def __init__(self, request_id: str):
        super().__init__(f"No approval request found with request_id={request_id!r}.")
        self.request_id = request_id


class ApprovalAlreadyDecidedError(RuntimeError):
    def __init__(self, request: ApprovalRequest):
        super().__init__(
            f"Approval request {request.request_id} was already decided (status={request.status.value})."
        )
        self.request = request


class ApprovalStore:
    """In-memory approval queue.

    NOT durable across a process restart and not shared across replicas --
    exactly the same caveat as
    ``reliability.idempotency.InMemoryIdempotencyStore``. A real deployment
    must back this with a durable, auditable store (e.g. a database table
    recording who approved what, when, for compliance/audit purposes) so an
    approval decision is never lost and is always attributable to a real
    approver.
    """

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._idempotency_keys: dict[str, str] = {}

    def submit(self, tool_name: str, arguments: Mapping[str, Any], requested_by: Identity) -> ApprovalRequest:
        request = ApprovalRequest(
            request_id=uuid.uuid4().hex,
            tool_name=tool_name,
            arguments=dict(arguments),
            requested_by=requested_by,
        )
        self._requests[request.request_id] = request
        return request

    def submit_or_get(
        self, idempotency_key: str, tool_name: str, arguments: Mapping[str, Any], requested_by: Identity
    ) -> ApprovalRequest:
        """Like :meth:`submit`, but deduplicated by ``idempotency_key`` --
        a HIGH-risk call retried multiple times for the *same* logical
        request (e.g. a graph node re-invoked on ``resume()`` after an
        earlier attempt raised :class:`ApprovalRequiredError`) must keep
        polling the *same* pending approval, not spawn a new one on every
        retry."""
        existing_id = self._idempotency_keys.get(idempotency_key)
        if existing_id is not None:
            return self.get(existing_id)
        request = self.submit(tool_name, arguments, requested_by)
        self._idempotency_keys[idempotency_key] = request.request_id
        return request

    def get(self, request_id: str) -> ApprovalRequest:
        try:
            return self._requests[request_id]
        except KeyError:
            raise ApprovalNotFoundError(request_id) from None

    def decide(self, request_id: str, *, approved: bool, approved_by: str) -> ApprovalRequest:
        request = self.get(request_id)
        if request.status is not ApprovalStatus.PENDING:
            raise ApprovalAlreadyDecidedError(request)
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.approved_by = approved_by
        request.decided_at = time.time()
        return request

    def require_approved(self, request_id: str) -> ApprovalRequest:
        """Returns the request if (and only if) it has been APPROVED;
        raises :class:`ApprovalRequiredError` while still PENDING, or
        :class:`ApprovalDeniedError` if it was DENIED. Never silently lets
        a non-approved request through."""
        request = self.get(request_id)
        if request.status is ApprovalStatus.PENDING:
            raise ApprovalRequiredError(request)
        if request.status is ApprovalStatus.DENIED:
            raise ApprovalDeniedError(request)
        return request


__all__ = [
    "AuthorizationError",
    "InsufficientRoleError",
    "TenantIsolationError",
    "Identity",
    "DEFAULT_MINIMUM_ROLES",
    "check_role_authorization",
    "check_tenant_isolation",
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalRequiredError",
    "ApprovalDeniedError",
    "ApprovalNotFoundError",
    "ApprovalAlreadyDecidedError",
    "ApprovalStore",
]
