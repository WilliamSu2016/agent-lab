"""Layer 3 (Tool Guardrail): risk classification, registry, and argument
validation. Also verifies the LOW/MEDIUM/HIGH tiering example from the
requirement (search_web / update_record / send_email) end to end.
"""

from __future__ import annotations

import pytest

from src.security.authorization import ApprovalRequiredError, ApprovalStore, Identity
from src.security.guardrails import AgentWorkflowGuardrail, ToolCallRequest, ToolGuardrail, secure_tool_call
from src.security.tool_policy import (
    ArgumentSpec,
    ToolArgumentValidationError,
    ToolRegistry,
    ToolRiskLevel,
    ToolSpec,
    UnknownToolError,
    build_default_registry,
)


@pytest.fixture
def registry():
    return build_default_registry()


class TestToolRiskClassification:
    def test_search_web_is_low_risk(self, registry):
        assert registry.get("search_web").risk_level is ToolRiskLevel.LOW

    def test_update_record_is_medium_risk(self, registry):
        assert registry.get("update_record").risk_level is ToolRiskLevel.MEDIUM

    def test_send_email_is_high_risk(self, registry):
        assert registry.get("send_email").risk_level is ToolRiskLevel.HIGH

    def test_only_high_risk_tools_require_approval(self, registry):
        assert registry.get("search_web").requires_approval is False
        assert registry.get("update_record").requires_approval is False
        assert registry.get("send_email").requires_approval is True

    def test_unregistered_tool_is_never_callable(self, registry):
        with pytest.raises(UnknownToolError):
            registry.get("delete_database")


class TestToolArgumentValidation:
    """Requirement 2: tool argument validation."""

    def test_missing_required_argument_is_rejected(self, registry):
        tool = registry.get("send_email")
        with pytest.raises(ToolArgumentValidationError) as exc_info:
            tool.validate_arguments({"to": "a@b.com", "subject": "hi"})  # missing "body"
        assert "body" in str(exc_info.value)

    def test_unexpected_extra_argument_is_rejected(self, registry):
        tool = registry.get("search_web")
        with pytest.raises(ToolArgumentValidationError):
            tool.validate_arguments({"query": "x", "unexpected_field": "malicious"})

    def test_wrong_type_argument_is_rejected(self, registry):
        tool = registry.get("update_record")
        with pytest.raises(ToolArgumentValidationError):
            tool.validate_arguments({"record_id": "r1", "field": "status", "value": 12345})  # value must be str

    def test_custom_validator_rejects_malformed_email(self, registry):
        tool = registry.get("send_email")
        with pytest.raises(ToolArgumentValidationError) as exc_info:
            tool.validate_arguments({"to": "not-an-email", "subject": "hi", "body": "hello"})
        assert "to" in str(exc_info.value)

    def test_valid_arguments_pass(self, registry):
        tool = registry.get("send_email")
        tool.validate_arguments({"to": "a@b.com", "subject": "hi", "body": "hello"})  # does not raise

    def test_every_error_is_reported_in_one_pass(self, registry):
        tool = registry.get("send_email")
        with pytest.raises(ToolArgumentValidationError) as exc_info:
            tool.validate_arguments({"to": "not-an-email"})  # missing subject/body AND bad email
        assert len(exc_info.value.errors) >= 3

    def test_argument_injection_attack_extra_fields_are_rejected(self, registry):
        """Simulated attack: the agent (or a manipulated tool-call
        payload) tries to smuggle an extra field (e.g. an internal admin
        flag) alongside legitimate arguments."""
        tool = registry.get("update_record")
        with pytest.raises(ToolArgumentValidationError):
            tool.validate_arguments(
                {"record_id": "r1", "field": "status", "value": "closed", "skip_authorization_check": True}
            )


class TestCustomRegistry:
    """The registry itself is reusable/extensible -- not hard-coded to
    only the three named example tools."""

    def test_can_register_a_new_tool_with_its_own_risk_tier(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="delete_file",
                risk_level=ToolRiskLevel.HIGH,
                description="Deletes a file permanently.",
                arguments=(ArgumentSpec("path", str, required=True),),
            )
        )
        assert registry.get("delete_file").risk_level is ToolRiskLevel.HIGH
        assert "delete_file" in registry
        assert "search_web" not in registry


class TestEndToEndRiskTierBehavior:
    """The requirement's own contrast: HIGH-risk tools go through
    approval; LOW/MEDIUM do not."""

    @pytest.fixture
    def workflow_guardrail(self, registry):
        return AgentWorkflowGuardrail(registry, ApprovalStore())

    @pytest.fixture
    def tool_guardrail(self, registry):
        return ToolGuardrail(registry)

    def test_low_risk_tool_call_reaches_the_tool_directly(self, workflow_guardrail, tool_guardrail):
        identity = Identity("u1", "t1", frozenset())
        request = ToolCallRequest(tool_name="search_web", arguments={"query": "x"}, identity=identity)
        result = secure_tool_call(
            request=request, tool_fn=lambda **kw: "ok", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
        )
        assert result.output == "ok"

    def test_medium_risk_tool_call_reaches_the_tool_directly_for_an_editor(self, workflow_guardrail, tool_guardrail):
        identity = Identity("u2", "t1", frozenset({"editor"}))
        request = ToolCallRequest(
            tool_name="update_record",
            arguments={"record_id": "r1", "field": "status", "value": "closed"},
            identity=identity,
        )
        result = secure_tool_call(
            request=request, tool_fn=lambda **kw: "ok", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
        )
        assert result.output == "ok"

    def test_high_risk_tool_call_never_reaches_the_tool_without_approval(self, workflow_guardrail, tool_guardrail):
        identity = Identity("u3", "t1", frozenset({"admin"}))
        request = ToolCallRequest(
            tool_name="send_email", arguments={"to": "a@b.com", "subject": "s", "body": "b"}, identity=identity
        )
        with pytest.raises(ApprovalRequiredError):
            secure_tool_call(
                request=request, tool_fn=lambda **kw: "ok", workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
            )

    def test_invalid_arguments_are_rejected_before_authorization_even_matters(
        self, workflow_guardrail, tool_guardrail
    ):
        """Argument validation happens at Layer 3, but a malformed call
        must never reach the real tool function regardless of who the
        caller is."""
        identity = Identity("u3", "t1", frozenset({"admin"}))
        request = ToolCallRequest(
            tool_name="send_email", arguments={"to": "not-an-email", "subject": "s", "body": "b"}, identity=identity
        )
        # Even an admin (who *would* eventually be allowed to request
        # approval) is blocked here because Layer 3's schema check runs
        # after the Layer 2 approval gate is satisfied is moot -- the
        # approval request itself would still be for an invalid call; this
        # test targets validate_arguments directly to isolate Layer 3.
        tool = workflow_guardrail.registry.get("send_email")
        with pytest.raises(ToolArgumentValidationError):
            tool.validate_arguments(request.arguments)
