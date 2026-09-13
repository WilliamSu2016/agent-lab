"""Layer 2 (Agent/Workflow Guardrail): authorization, identity propagation,
tenant isolation, and the HIGH-risk approval gate.
"""

from __future__ import annotations

import pytest

from src.security.authorization import (
    ApprovalAlreadyDecidedError,
    ApprovalDeniedError,
    ApprovalNotFoundError,
    ApprovalRequiredError,
    ApprovalStore,
    Identity,
    InsufficientRoleError,
    TenantIsolationError,
    check_role_authorization,
    check_tenant_isolation,
)
from src.security.guardrails import AgentWorkflowGuardrail, ToolCallRequest, ToolGuardrail, secure_tool_call
from src.security.tool_policy import build_default_registry


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def approvals():
    return ApprovalStore()


@pytest.fixture
def workflow_guardrail(registry, approvals):
    return AgentWorkflowGuardrail(registry, approvals)


@pytest.fixture
def tool_guardrail(registry):
    return ToolGuardrail(registry)


class TestRoleBasedAuthorization:
    """Requirement 3: authorization check."""

    def test_any_identity_can_call_a_low_risk_tool(self, registry):
        viewer = Identity(user_id="u1", tenant_id="t1", roles=frozenset())
        tool = registry.get("search_web")
        check_role_authorization(viewer, tool)  # does not raise

    def test_viewer_without_editor_role_cannot_call_medium_risk_tool(self, registry):
        viewer = Identity(user_id="u1", tenant_id="t1", roles=frozenset())
        tool = registry.get("update_record")
        with pytest.raises(InsufficientRoleError):
            check_role_authorization(viewer, tool)

    def test_editor_can_call_medium_risk_tool(self, registry):
        editor = Identity(user_id="u2", tenant_id="t1", roles=frozenset({"editor"}))
        tool = registry.get("update_record")
        check_role_authorization(editor, tool)  # does not raise

    def test_editor_without_admin_role_cannot_call_high_risk_tool(self, registry):
        editor = Identity(user_id="u2", tenant_id="t1", roles=frozenset({"editor"}))
        tool = registry.get("send_email")
        with pytest.raises(InsufficientRoleError):
            check_role_authorization(editor, tool)

    def test_admin_can_call_high_risk_tool_role_check(self, registry):
        admin = Identity(user_id="u3", tenant_id="t1", roles=frozenset({"admin"}))
        tool = registry.get("send_email")
        check_role_authorization(admin, tool)  # does not raise (approval is separate)

    def test_end_to_end_secure_tool_call_rejects_insufficient_role(self, workflow_guardrail, tool_guardrail):
        viewer = Identity(user_id="u1", tenant_id="t1", roles=frozenset())
        request = ToolCallRequest(
            tool_name="update_record", arguments={"record_id": "r1", "field": "status", "value": "closed"}, identity=viewer
        )
        with pytest.raises(InsufficientRoleError):
            secure_tool_call(
                request=request,
                tool_fn=lambda **kw: "updated",
                workflow_guardrail=workflow_guardrail,
                tool_guardrail=tool_guardrail,
            )


class TestTenantIsolation:
    """Requirement 5: tenant isolation -- a user must never be able to
    touch another tenant's resources, regardless of role."""

    def test_same_tenant_resource_access_is_allowed(self):
        identity = Identity(user_id="u1", tenant_id="tenant-a", roles=frozenset({"admin"}))
        check_tenant_isolation(identity, "tenant-a")  # does not raise

    def test_cross_tenant_resource_access_is_rejected(self):
        identity = Identity(user_id="u1", tenant_id="tenant-a", roles=frozenset({"admin"}))
        with pytest.raises(TenantIsolationError):
            check_tenant_isolation(identity, "tenant-b")

    def test_admin_role_does_not_bypass_tenant_isolation(self, workflow_guardrail, tool_guardrail):
        """A cross-tenant attack attempt: an admin from tenant-a tries to
        update a record that (per its record_id/tenant lookup) belongs to
        tenant-b. Role authorization alone must not be sufficient."""
        cross_tenant_admin = Identity(user_id="attacker", tenant_id="tenant-a", roles=frozenset({"admin"}))
        request = ToolCallRequest(
            tool_name="update_record",
            arguments={"record_id": "victim-record-1", "field": "balance", "value": "0"},
            identity=cross_tenant_admin,
            resource_tenant_id="tenant-b",
        )
        with pytest.raises(TenantIsolationError):
            secure_tool_call(
                request=request,
                tool_fn=lambda **kw: "updated",
                workflow_guardrail=workflow_guardrail,
                tool_guardrail=tool_guardrail,
            )


class TestUserIdentityPropagation:
    """Requirement 4: identity must be an explicit, caller-supplied value
    -- never implicit/ambient. These tests demonstrate that two different
    identities calling the exact same tool with the exact same arguments
    produce different authorization outcomes purely based on the
    explicitly-passed ``Identity``, proving no hidden/global state is
    involved."""

    def test_same_call_different_identity_different_outcome(self, workflow_guardrail, tool_guardrail):
        request_args = {"record_id": "r1", "field": "status", "value": "closed"}

        viewer_request = ToolCallRequest(
            tool_name="update_record", arguments=request_args, identity=Identity("u1", "t1", frozenset())
        )
        editor_request = ToolCallRequest(
            tool_name="update_record", arguments=request_args, identity=Identity("u2", "t1", frozenset({"editor"}))
        )

        with pytest.raises(InsufficientRoleError):
            secure_tool_call(
                request=viewer_request,
                tool_fn=lambda **kw: "updated",
                workflow_guardrail=workflow_guardrail,
                tool_guardrail=tool_guardrail,
            )
        result = secure_tool_call(
            request=editor_request,
            tool_fn=lambda **kw: "updated",
            workflow_guardrail=workflow_guardrail,
            tool_guardrail=tool_guardrail,
        )
        assert result.output == "updated"


class TestHighRiskApprovalGate:
    """Requirement 7: HIGH-risk tools go Agent -> approval -> Tool, never
    Agent -> Tool directly."""

    def test_low_risk_tool_never_requires_approval(self, workflow_guardrail, tool_guardrail):
        identity = Identity("u1", "t1", frozenset())
        request = ToolCallRequest(tool_name="search_web", arguments={"query": "langgraph"}, identity=identity)
        result = secure_tool_call(
            request=request, tool_fn=lambda **kw: "results", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
        )
        assert result.output == "results"

    def test_high_risk_tool_call_without_approval_never_reaches_the_tool_function(
        self, workflow_guardrail, tool_guardrail
    ):
        admin = Identity("u3", "t1", frozenset({"admin"}))
        tool_was_called = {"value": False}

        def send_email(**kwargs):
            tool_was_called["value"] = True
            return "sent"

        request = ToolCallRequest(
            tool_name="send_email",
            arguments={"to": "a@b.com", "subject": "hi", "body": "hello"},
            identity=admin,
        )
        with pytest.raises(ApprovalRequiredError) as exc_info:
            secure_tool_call(request=request, tool_fn=send_email, workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail)

        # The whole point of this gate: Agent -> Tool must NOT have happened.
        assert tool_was_called["value"] is False
        assert exc_info.value.request.status.value == "pending"

    def test_high_risk_tool_call_succeeds_after_explicit_approval(self, workflow_guardrail, tool_guardrail, approvals):
        admin = Identity("u3", "t1", frozenset({"admin"}))
        arguments = {"to": "a@b.com", "subject": "hi", "body": "hello"}
        request = ToolCallRequest(tool_name="send_email", arguments=arguments, identity=admin)

        with pytest.raises(ApprovalRequiredError) as exc_info:
            secure_tool_call(
                request=request, tool_fn=lambda **kw: "sent", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
            )
        request_id = exc_info.value.request.request_id

        approvals.decide(request_id, approved=True, approved_by="security-manager")
        approved_request = ToolCallRequest(
            tool_name="send_email", arguments=arguments, identity=admin, approval_request_id=request_id
        )
        result = secure_tool_call(
            request=approved_request,
            tool_fn=lambda **kw: "sent",
            workflow_guardrail=workflow_guardrail,
            tool_guardrail=tool_guardrail,
        )
        assert result.output == "sent"

    def test_denied_approval_permanently_blocks_that_request(self, workflow_guardrail, tool_guardrail, approvals):
        admin = Identity("u3", "t1", frozenset({"admin"}))
        arguments = {"to": "a@b.com", "subject": "hi", "body": "hello"}
        request = ToolCallRequest(tool_name="send_email", arguments=arguments, identity=admin)

        with pytest.raises(ApprovalRequiredError) as exc_info:
            secure_tool_call(
                request=request, tool_fn=lambda **kw: "sent", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
            )
        request_id = exc_info.value.request.request_id
        approvals.decide(request_id, approved=False, approved_by="security-manager")

        denied_request = ToolCallRequest(
            tool_name="send_email", arguments=arguments, identity=admin, approval_request_id=request_id
        )
        with pytest.raises(ApprovalDeniedError):
            secure_tool_call(
                request=denied_request,
                tool_fn=lambda **kw: "sent",
                workflow_guardrail=workflow_guardrail,
                tool_guardrail=tool_guardrail,
            )

    def test_approval_request_can_only_be_decided_once(self, approvals):
        admin = Identity("u3", "t1", frozenset({"admin"}))
        request = approvals.submit("send_email", {"to": "a@b.com"}, admin)
        approvals.decide(request.request_id, approved=True, approved_by="mgr1")
        with pytest.raises(ApprovalAlreadyDecidedError):
            approvals.decide(request.request_id, approved=False, approved_by="mgr2")

    def test_unknown_approval_request_id_raises(self, approvals):
        with pytest.raises(ApprovalNotFoundError):
            approvals.require_approved("does-not-exist")

    def test_an_attacker_cannot_forge_approval_by_guessing_a_request_id(self, workflow_guardrail, tool_guardrail):
        """Simulated attack: the caller supplies a made-up
        ``approval_request_id`` hoping the approval gate treats "any
        non-None id" as approved. It must not."""
        admin = Identity("attacker", "t1", frozenset({"admin"}))
        request = ToolCallRequest(
            tool_name="send_email",
            arguments={"to": "a@b.com", "subject": "hi", "body": "hello"},
            identity=admin,
            approval_request_id="forged-request-id-12345",
        )
        from src.security.authorization import ApprovalNotFoundError

        with pytest.raises(ApprovalNotFoundError):
            secure_tool_call(
                request=request, tool_fn=lambda **kw: "sent", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
            )
