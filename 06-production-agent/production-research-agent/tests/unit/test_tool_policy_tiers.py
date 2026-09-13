"""Unit tests: risk-tier registration in ``src.security.tool_policy``."""

from __future__ import annotations

import pytest

from src.security.tool_policy import (
    ArgumentSpec,
    ToolArgumentValidationError,
    ToolRiskLevel,
    ToolSpec,
    UnknownToolError,
    build_default_registry,
)


def test_default_registry_assigns_the_three_documented_risk_tiers():
    registry = build_default_registry()
    assert registry.get("search_web").risk_level is ToolRiskLevel.LOW
    assert registry.get("update_record").risk_level is ToolRiskLevel.MEDIUM
    assert registry.get("send_email").risk_level is ToolRiskLevel.HIGH


def test_only_high_risk_tools_require_approval():
    registry = build_default_registry()
    assert registry.get("search_web").requires_approval is False
    assert registry.get("update_record").requires_approval is False
    assert registry.get("send_email").requires_approval is True


def test_unregistered_tool_name_is_never_callable():
    registry = build_default_registry()
    with pytest.raises(UnknownToolError):
        registry.get("delete_everything")
    assert "delete_everything" not in registry
    assert "search_web" in registry


def test_send_email_argument_validation_rejects_a_malformed_address():
    registry = build_default_registry()
    tool = registry.get("send_email")
    with pytest.raises(ToolArgumentValidationError):
        tool.validate_arguments({"to": "not-an-email", "subject": "hi", "body": "hi"})


def test_send_email_argument_validation_accepts_a_well_formed_call():
    registry = build_default_registry()
    tool = registry.get("send_email")
    tool.validate_arguments({"to": "ops@example.com", "subject": "hi", "body": "hi"})


def test_missing_required_argument_is_rejected():
    registry = build_default_registry()
    tool = registry.get("update_record")
    with pytest.raises(ToolArgumentValidationError):
        tool.validate_arguments({"record_id": "r1", "field": "status"})  # missing "value"


def test_custom_tool_registration_round_trips():
    from src.security.tool_policy import ToolRegistry

    registry = ToolRegistry()
    spec = ToolSpec(
        name="delete_record",
        risk_level=ToolRiskLevel.HIGH,
        description="Irreversible delete.",
        arguments=(ArgumentSpec("record_id", str, required=True),),
    )
    registry.register(spec)
    assert registry.get("delete_record") is spec
    assert registry.get("delete_record").requires_approval is True
