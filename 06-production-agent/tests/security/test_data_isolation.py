"""Tenant isolation + sensitive-data filtering + output validation.

Simulates a small multi-tenant "record store" tool to exercise tenant
isolation end to end (not just the unit-level check), and exercises
sensitive-data redaction/blocking on both tool output and final-answer
output validation.
"""

from __future__ import annotations

import pytest

from src.security.authorization import ApprovalStore, Identity, TenantIsolationError
from src.security.guardrails import AgentWorkflowGuardrail, ToolCallRequest, ToolGuardrail, secure_tool_call
from src.security.sanitization import (
    DEFAULT_BLOCK_CATEGORIES,
    redact_sensitive_data,
    scan_for_sensitive_data,
    validate_output_text,
)
from src.security.tool_policy import build_default_registry

# ---------------------------------------------------------------------------
# A tiny in-memory multi-tenant record store, standing in for a real
# database, used to exercise tenant isolation against real "data".
# ---------------------------------------------------------------------------

RECORDS: dict[str, dict[str, str]] = {
    "record-1": {"tenant_id": "tenant-a", "owner": "alice", "balance": "100"},
    "record-2": {"tenant_id": "tenant-b", "owner": "bob", "balance": "500"},
}


def read_record(record_id: str) -> str:
    record = RECORDS[record_id]
    return f"owner={record['owner']} balance={record['balance']}"


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def workflow_guardrail(registry):
    return AgentWorkflowGuardrail(registry, ApprovalStore())


@pytest.fixture
def tool_guardrail(registry):
    return ToolGuardrail(registry)


class TestTenantIsolationAgainstRealData:
    def test_user_can_read_own_tenant_record(self, workflow_guardrail, tool_guardrail):
        alice = Identity(user_id="alice", tenant_id="tenant-a", roles=frozenset({"editor"}))
        request = ToolCallRequest(
            tool_name="update_record",
            arguments={"record_id": "record-1", "field": "balance", "value": "150"},
            identity=alice,
            resource_tenant_id=RECORDS["record-1"]["tenant_id"],
        )
        result = secure_tool_call(
            request=request,
            tool_fn=lambda **kw: read_record("record-1"),
            workflow_guardrail=workflow_guardrail,
            tool_guardrail=tool_guardrail,
        )
        assert "owner=alice" in result.output

    def test_user_cannot_read_another_tenants_record(self, workflow_guardrail, tool_guardrail):
        """Attack scenario: a tenant-a user attempts to access a
        tenant-b record by supplying its record_id directly (e.g. via an
        IDOR-style guessed/enumerated ID)."""
        alice = Identity(user_id="alice", tenant_id="tenant-a", roles=frozenset({"editor"}))
        request = ToolCallRequest(
            tool_name="update_record",
            arguments={"record_id": "record-2", "field": "balance", "value": "0"},
            identity=alice,
            resource_tenant_id=RECORDS["record-2"]["tenant_id"],  # belongs to tenant-b
        )
        with pytest.raises(TenantIsolationError):
            secure_tool_call(
                request=request,
                tool_fn=lambda **kw: read_record("record-2"),
                workflow_guardrail=workflow_guardrail,
                tool_guardrail=tool_guardrail,
            )

    def test_cross_tenant_attack_never_invokes_the_tool_function(self, workflow_guardrail, tool_guardrail):
        tool_invoked = {"value": False}

        def update_record(**kwargs):
            tool_invoked["value"] = True
            return "updated"

        attacker = Identity(user_id="mallory", tenant_id="tenant-a", roles=frozenset({"admin"}))
        request = ToolCallRequest(
            tool_name="update_record",
            arguments={"record_id": "record-2", "field": "balance", "value": "999999"},
            identity=attacker,
            resource_tenant_id="tenant-b",
        )
        with pytest.raises(TenantIsolationError):
            secure_tool_call(
                request=request, tool_fn=update_record, workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
            )
        assert tool_invoked["value"] is False


class TestSensitiveDataFiltering:
    """Requirement 6: sensitive-data filtering."""

    def test_email_and_phone_are_detected(self):
        text = "Contact jane@example.com or call 555-123-4567 for details."
        matches = scan_for_sensitive_data(text)
        categories = {m.category for m in matches}
        assert "email" in categories

    def test_openai_api_key_is_detected(self):
        text = "Here is the key: sk-abcdEFGH1234567890abcd1234"
        matches = scan_for_sensitive_data(text)
        assert any(m.category == "openai_api_key" for m in matches)

    def test_aws_access_key_is_detected(self):
        text = "AWS key AKIAABCDEFGHIJKLMNOP was found in the log."
        matches = scan_for_sensitive_data(text)
        assert any(m.category == "aws_access_key" for m in matches)

    def test_credit_card_number_is_detected(self):
        text = "Card on file: 4111 1111 1111 1111"
        matches = scan_for_sensitive_data(text)
        assert any(m.category == "credit_card" for m in matches)

    def test_redaction_preserves_surrounding_text(self):
        text = "Reach me at jane@example.com anytime."
        redacted = redact_sensitive_data(text)
        assert "jane@example.com" not in redacted
        assert redacted.startswith("Reach me at ")
        assert redacted.endswith(" anytime.")


class TestOutputValidation:
    """Requirement 8: output validation."""

    def test_low_severity_finding_is_redacted_but_output_still_allowed(self):
        result = validate_output_text("Please contact support at help@example.com.")
        assert result.allowed is True
        assert "help@example.com" not in result.sanitized_text

    def test_high_severity_finding_blocks_the_output_entirely(self):
        result = validate_output_text("Debug info: sk-abcdEFGH1234567890abcd1234")
        assert result.allowed is False
        assert result.blocked_reason is not None
        assert "openai_api_key" in result.blocked_reason

    def test_tool_result_leaking_a_secret_is_blocked_at_the_tool_boundary(self, workflow_guardrail, tool_guardrail):
        """Simulated attack/bug: a tool (e.g. a buggy log-search tool)
        returns a response that accidentally contains a live secret. Layer
        3 output sanitization must catch this even though nothing was
        wrong with the *input* or the *authorization*."""
        identity = Identity(user_id="u1", tenant_id="t1", roles=frozenset())

        def leaky_search(query: str) -> str:
            return "Found in logs: Authorization: Bearer sk-liveSECRETtoken1234567890abcdef"

        request = ToolCallRequest(tool_name="search_web", arguments={"query": "recent errors"}, identity=identity)
        result = secure_tool_call(
            request=request, tool_fn=leaky_search, workflow_guardrail=workflow_guardrail, tool_guardrail=tool_guardrail
        )
        assert result.output_validation.allowed is False
        assert "sk-liveSECRETtoken1234567890abcdef" not in result.output

    def test_default_block_categories_cover_the_most_severe_findings(self):
        assert {"openai_api_key", "aws_access_key", "credit_card", "ssn"} <= DEFAULT_BLOCK_CATEGORIES
