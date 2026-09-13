"""Three-layer guardrail pipeline.

    Layer 1 -- Input Guardrail       : scan_prompt_injection / enforce_input_guardrail
    Layer 2 -- Agent/Workflow Guardrail: AgentWorkflowGuardrail (authorization,
                                         tenant isolation, identity propagation,
                                         HIGH-risk approval gate, output validation)
    Layer 3 -- Tool Guardrail        : ToolGuardrail (argument validation,
                                         output sanitization at the tool boundary)

:func:`secure_tool_call` composes all three end to end and is the one
function every tool invocation in this experiment should go through --
mirroring how ``src/reliability/retry.py::retry_call`` is the one pipeline
every tool call's retry/timeout/error-classification goes through.

    User input
       |
       v
    [Layer 1: Input Guardrail]  -- prompt injection / sensitive input scan
       |  (blocks here if flagged)
       v
    [Layer 2: Agent/Workflow Guardrail]
       |  - role authorization (check_role_authorization)
       |  - tenant isolation (check_tenant_isolation)
       |  - HIGH-risk tools: approval gate
       |      (Agent -> approval -> Tool, NOT Agent -> Tool)
       v
    [Layer 3: Tool Guardrail]
       |  - tool argument validation (ToolSpec.validate_arguments)
       |  - execute the real tool function
       |  - output validation / sensitive-data redaction
       v
    Tool result (sanitized)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from src.security.authorization import (
    ApprovalStore,
    Identity,
    check_role_authorization,
    check_tenant_isolation,
)
from src.security.sanitization import OutputValidationResult, validate_output_text
from src.security.tool_policy import ToolRegistry, ToolRiskLevel, ToolSpec

# ---------------------------------------------------------------------------
# Layer 1: Input Guardrail -- prompt injection detection.
# ---------------------------------------------------------------------------

DEFAULT_INPUT_BLOCK_THRESHOLD = 0.5


@dataclass(frozen=True)
class InjectionRule:
    name: str
    pattern: re.Pattern[str]
    weight: float
    # "critical" rules block regardless of the overall risk_score -- these
    # are patterns with essentially no legitimate use in a normal user
    # message (e.g. an embedded attempt to exfiltrate data to an external
    # URL, or an instruction to directly invoke a specific tool).
    critical: bool = False


# Weighted, pattern-based heuristics. Deliberately simple/regex-based
# (no LLM-based classifier) so detection is fast, free, deterministic, and
# testable without any network access -- a real production system should
# layer an LLM-based or ML classifier on top of this, not instead of it
# (defense in depth), see docs/04-SECURITY.md.
PROMPT_INJECTION_RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        "ignore_previous_instructions",
        re.compile(r"(?i)\bignore\s+(all|any|the)?\s*(previous|above|prior)\s+(instructions|prompts?|rules)\b"),
        weight=0.6,
    ),
    InjectionRule(
        "disregard_system_prompt",
        re.compile(r"(?i)\bdisregard\s+(all|the|your)?\s*(system|previous)\s+(prompt|instructions?)\b"),
        weight=0.6,
    ),
    InjectionRule(
        "jailbreak_persona",
        re.compile(r"(?i)\byou\s+are\s+(now|no\s+longer)\b.{0,40}\b(DAN|jailbroken|unrestricted|unfiltered)\b"),
        weight=0.7,
    ),
    InjectionRule(
        "reveal_system_prompt",
        re.compile(r"(?i)\b(reveal|show|print|repeat|output)\s+(your|the)\s+(hidden\s+)?(system\s+prompt|instructions|initial\s+prompt)\b"),
        weight=0.6,
    ),
    InjectionRule(
        "fake_role_delimiter",
        re.compile(r"(?im)^\s*(###\s*system\b|system\s*:|assistant\s*:)"),
        weight=0.5,
    ),
    InjectionRule(
        "exfiltration_attempt",
        re.compile(r"(?i)\bsend\s+(this|the|all)\s+(data|conversation|information|api\s*key|secret)s?\s+to\s+https?://"),
        weight=1.0,
        critical=True,
    ),
    InjectionRule(
        "direct_tool_invocation_injection",
        re.compile(r"(?i)\bcall\s+(the\s+)?(send_email|update_record|delete_\w+)\s*\("),
        weight=1.0,
        critical=True,
    ),
    InjectionRule(
        "unrestricted_persona_request",
        re.compile(r"(?i)\bact\s+as\s+(if\s+you\s+(were|are)\s+)?.{0,30}\b(no\s+restrictions|without\s+limitations|no\s+filter)\b"),
        weight=0.6,
    ),
    InjectionRule(
        "chained_instruction_override",
        re.compile(r"(?i)\btranslate\s+the\s+following.{0,60}\bthen\s+ignore\b"),
        weight=0.7,
    ),
    InjectionRule(
        "zero_width_obfuscation",
        re.compile("[\u200b\u200c\u200d\ufeff]"),
        weight=0.6,
    ),
    InjectionRule(
        "system_prompt_leak_marker",
        re.compile(r"(?i)begin\s+system\s+prompt"),
        weight=0.8,
    ),
    InjectionRule(
        "override_safety_guidelines",
        re.compile(r"(?i)\b(bypass|override|disable)\s+(the\s+)?(safety|content)\s+(guidelines|filters?|policy)\b"),
        weight=0.7,
    ),
    InjectionRule(
        "urgency_pressure_language",
        re.compile(r"(?i)\b(urgent|immediately|right away|act\s+now)\b.{0,40}\b(or\s+(you|your)|otherwise)\b"),
        weight=0.3,
    ),
)


@dataclass(frozen=True)
class InputGuardrailResult:
    allowed: bool
    risk_score: float
    matched_rules: tuple[str, ...]
    sanitized_text: str
    reason: str | None = None


class PromptInjectionDetected(RuntimeError):
    """Layer 1 blocked the input. Raised instead of silently passing a
    flagged prompt through to the LLM/agent -- the caller must handle this
    (e.g. reject the request, ask the user to rephrase) rather than treat
    the flagged text as trustworthy input."""

    def __init__(self, result: InputGuardrailResult):
        super().__init__(
            f"Prompt injection detected (risk_score={result.risk_score:.2f}, "
            f"matched_rules={list(result.matched_rules)!r}); blocking this input."
        )
        self.result = result


def _normalize_for_scanning(text: str) -> str:
    """NFKC-normalize before scanning so that trivial unicode-confusable
    obfuscation (full-width characters, alternate quote/space glyphs, etc.)
    does not defeat the plain-ASCII regexes above. Deliberately does NOT
    strip the zero-width characters themselves -- ``zero_width_obfuscation``
    needs to still see them."""
    return unicodedata.normalize("NFKC", text)


def scan_prompt_injection(
    text: str, *, block_threshold: float = DEFAULT_INPUT_BLOCK_THRESHOLD
) -> InputGuardrailResult:
    """Requirement 1: prompt injection detection. Scores ``text`` against
    every rule in :data:`PROMPT_INJECTION_RULES` and returns a structured
    result; does not raise. Use :func:`enforce_input_guardrail` to turn a
    flagged result into a hard block."""
    normalized = _normalize_for_scanning(text)
    matched: list[str] = []
    score = 0.0
    critical_hit = False
    for rule in PROMPT_INJECTION_RULES:
        if rule.pattern.search(normalized):
            matched.append(rule.name)
            score += rule.weight
            if rule.critical:
                critical_hit = True
    # Score can exceed 1.0 (deliberately -- more simultaneous signals
    # should read as strictly more suspicious); "allowed" is a threshold
    # comparison against the caller-configurable ``block_threshold``, plus
    # an unconditional block for any critical-rule hit regardless of score.
    allowed = not critical_hit and score < block_threshold

    from src.security.sanitization import redact_sensitive_data

    sanitized = redact_sensitive_data(text)
    reason = None
    if not allowed:
        reason = "critical prompt-injection pattern matched" if critical_hit else "risk_score exceeded block threshold"
    return InputGuardrailResult(
        allowed=allowed, risk_score=score, matched_rules=tuple(matched), sanitized_text=sanitized, reason=reason
    )


def enforce_input_guardrail(text: str, *, block_threshold: float = DEFAULT_INPUT_BLOCK_THRESHOLD) -> InputGuardrailResult:
    """Same scan as :func:`scan_prompt_injection`, but raises
    :class:`PromptInjectionDetected` when the input should be blocked
    (either a critical-rule hit, or ``risk_score >= block_threshold``)."""
    result = scan_prompt_injection(text, block_threshold=block_threshold)
    if not result.allowed:
        raise PromptInjectionDetected(result)
    return result


# ---------------------------------------------------------------------------
# Layer 2: Agent / Workflow Guardrail -- authorization, tenant isolation,
# identity propagation, HIGH-risk approval gate, output validation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRequest:
    """Everything Layer 2 needs to authorize one proposed tool call.
    ``identity`` (Requirement 4) is always explicit -- never looked up from
    ambient/thread-local state."""

    tool_name: str
    arguments: dict[str, Any]
    identity: Identity
    # The tenant that owns the resource this call would touch, if any
    # (e.g. the tenant_id embedded in a record_id). None means the tool has
    # no tenant-scoped resource to check (e.g. a pure read-only web search).
    resource_tenant_id: Optional[str] = None
    # Set on a *retry* of a HIGH-risk call, after
    # ``ApprovalStore.decide(..., approved=True)`` -- this is what proves
    # the call went through "Agent -> approval -> Tool".
    approval_request_id: Optional[str] = None


class AgentWorkflowGuardrail:
    """Layer 2. One instance is shared across a run (or a whole process),
    parameterized by the tool registry and the approval store it should
    check against."""

    def __init__(
        self,
        registry: ToolRegistry,
        approval_store: ApprovalStore,
        *,
        minimum_roles: Mapping[ToolRiskLevel, frozenset[str]] | None = None,
    ) -> None:
        self.registry = registry
        self.approvals = approval_store
        self.minimum_roles = minimum_roles

    def authorize_tool_call(self, request: ToolCallRequest) -> ToolSpec:
        """Requirements 3 + 5: role authorization and tenant isolation.
        Raises if either check fails; never partially applies one and
        skips the other."""
        tool = self.registry.get(request.tool_name)
        check_role_authorization(request.identity, tool, minimum_roles=self.minimum_roles)
        if request.resource_tenant_id is not None:
            check_tenant_isolation(request.identity, request.resource_tenant_id)
        return tool

    def enforce_high_risk_approval(self, request: ToolCallRequest, tool: ToolSpec) -> Optional[Any]:
        """Requirement 7: HIGH-risk tools go ``Agent -> approval -> Tool``.

        * Not HIGH risk: no-op, returns ``None`` immediately (LOW/MEDIUM
          tools go straight ``Agent -> Tool`` after Layer 2's authorization
          check, per the requirement's own contrast).
        * HIGH risk, no ``approval_request_id`` supplied yet: submits a new
          :class:`~src.security.authorization.ApprovalRequest` and raises
          :class:`~src.security.authorization.ApprovalRequiredError` --
          the tool function is never invoked on this call.
        * HIGH risk, ``approval_request_id`` supplied: looks it up and
          raises unless it has been APPROVED (raises
          :class:`ApprovalRequiredError` if still pending,
          :class:`ApprovalDeniedError` if denied) -- only a genuinely
          approved request lets execution continue.
        """
        if not tool.requires_approval:
            return None
        if request.approval_request_id is None:
            approval = self.approvals.submit(tool.name, request.arguments, request.identity)
            from src.security.authorization import ApprovalRequiredError

            raise ApprovalRequiredError(approval)
        return self.approvals.require_approved(request.approval_request_id)

    def validate_output(self, text: str) -> OutputValidationResult:
        """Requirement 8: output validation, applied at the agent/workflow
        boundary (e.g. right before returning a final answer to the user),
        independent of the tool-level sanitization in
        :class:`ToolGuardrail`."""
        return validate_output_text(text)


# ---------------------------------------------------------------------------
# Layer 3: Tool Guardrail -- argument validation + output sanitization.
# ---------------------------------------------------------------------------


class ToolGuardrail:
    """Layer 3. The last line of defense immediately around the real tool
    function: no argument reaches the tool unless it matches the
    registered schema, and no tool output leaves this layer unsanitized."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def validate_arguments(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolSpec:
        """Requirement 2: tool argument validation."""
        tool = self.registry.get(tool_name)
        tool.validate_arguments(arguments)
        return tool

    def sanitize_output(self, raw_output: str) -> OutputValidationResult:
        """Requirement 6 + 8 at the tool boundary: redact/flag sensitive
        data in whatever the tool returned, before it flows back into
        agent state."""
        return validate_output_text(raw_output)


# ---------------------------------------------------------------------------
# End-to-end composition of all three layers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecureToolCallResult:
    tool_name: str
    risk_level: ToolRiskLevel
    raw_output: Any
    output: str
    output_validation: OutputValidationResult
    input_guardrail: Optional[InputGuardrailResult] = None


def secure_tool_call(
    *,
    request: ToolCallRequest,
    tool_fn: Callable[..., Any],
    workflow_guardrail: AgentWorkflowGuardrail,
    tool_guardrail: ToolGuardrail,
    user_input: Optional[str] = None,
    input_block_threshold: float = DEFAULT_INPUT_BLOCK_THRESHOLD,
) -> SecureToolCallResult:
    """Runs one tool call through all three layers, in order:

    1. Layer 1 (only if ``user_input`` is given -- e.g. the user message
       that triggered this tool call): blocks on prompt injection.
    2. Layer 2: role authorization, tenant isolation, and -- for HIGH-risk
       tools -- the approval gate (raises instead of ever reaching the
       tool function if not yet approved).
    3. Layer 3: argument validation, then the real tool call, then output
       sanitization.

    Raises (never silently swallows) any guardrail failure from any layer;
    callers should catch the specific exception types they want to handle
    (``PromptInjectionDetected``, ``InsufficientRoleError``,
    ``TenantIsolationError``, ``ApprovalRequiredError``,
    ``ApprovalDeniedError``, ``ToolArgumentValidationError``).
    """
    input_result: Optional[InputGuardrailResult] = None
    if user_input is not None:
        input_result = enforce_input_guardrail(user_input, block_threshold=input_block_threshold)

    tool = workflow_guardrail.authorize_tool_call(request)
    workflow_guardrail.enforce_high_risk_approval(request, tool)

    tool_guardrail.validate_arguments(request.tool_name, request.arguments)
    raw_output = tool_fn(**request.arguments)
    output_validation = tool_guardrail.sanitize_output(str(raw_output))

    return SecureToolCallResult(
        tool_name=request.tool_name,
        risk_level=tool.risk_level,
        raw_output=raw_output,
        output=output_validation.sanitized_text,
        output_validation=output_validation,
        input_guardrail=input_result,
    )


__all__ = [
    "DEFAULT_INPUT_BLOCK_THRESHOLD",
    "InjectionRule",
    "PROMPT_INJECTION_RULES",
    "InputGuardrailResult",
    "PromptInjectionDetected",
    "scan_prompt_injection",
    "enforce_input_guardrail",
    "ToolCallRequest",
    "AgentWorkflowGuardrail",
    "ToolGuardrail",
    "SecureToolCallResult",
    "secure_tool_call",
]
