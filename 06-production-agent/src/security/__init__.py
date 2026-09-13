"""Security & Guardrails experiment.

Three layers, applied in order to every tool call an Agent makes:

    Layer 1 -- Input Guardrail        (``guardrails``: prompt injection detection)
    Layer 2 -- Agent/Workflow Guardrail (``guardrails``+``authorization``: role
                authorization, tenant isolation, identity propagation,
                HIGH-risk approval gate, output validation)
    Layer 3 -- Tool Guardrail          (``guardrails``+``tool_policy``: argument
                validation, output sanitization)

Modules
-------
``tool_policy``
    Tool risk classification (LOW/MEDIUM/HIGH), the tool registry, and
    argument-schema validation.
``authorization``
    ``Identity`` (explicit user-identity propagation), role-based
    authorization, tenant isolation, and the HIGH-risk approval workflow
    (``ApprovalStore``/``ApprovalRequiredError``/``ApprovalDeniedError``).
``sanitization``
    Sensitive-data detection/redaction and output validation, shared by
    both the input and tool-output guardrails.
``guardrails``
    The three-layer pipeline itself: ``scan_prompt_injection`` /
    ``enforce_input_guardrail`` (Layer 1), ``AgentWorkflowGuardrail``
    (Layer 2), ``ToolGuardrail`` (Layer 3), and ``secure_tool_call`` which
    composes all three end to end.

See ``docs/04-SECURITY.md`` for the full design write-up, the attack test
catalogue, and ``tests/security/`` for the automated test suite.
"""

from src.security.authorization import (
    ApprovalDeniedError,
    ApprovalRequest,
    ApprovalRequiredError,
    ApprovalStatus,
    ApprovalStore,
    AuthorizationError,
    DEFAULT_MINIMUM_ROLES,
    Identity,
    InsufficientRoleError,
    TenantIsolationError,
    check_role_authorization,
    check_tenant_isolation,
)
from src.security.guardrails import (
    AgentWorkflowGuardrail,
    DEFAULT_INPUT_BLOCK_THRESHOLD,
    InputGuardrailResult,
    PromptInjectionDetected,
    SecureToolCallResult,
    ToolCallRequest,
    ToolGuardrail,
    enforce_input_guardrail,
    scan_prompt_injection,
    secure_tool_call,
)
from src.security.sanitization import (
    DEFAULT_BLOCK_CATEGORIES,
    OutputValidationResult,
    SensitiveMatch,
    redact_sensitive_data,
    scan_for_sensitive_data,
    validate_output_text,
)
from src.security.tool_policy import (
    ArgumentSpec,
    ToolArgumentValidationError,
    ToolRegistry,
    ToolRiskLevel,
    ToolSpec,
    UnknownToolError,
    build_default_registry,
)

__all__ = [
    # authorization
    "ApprovalDeniedError",
    "ApprovalRequest",
    "ApprovalRequiredError",
    "ApprovalStatus",
    "ApprovalStore",
    "AuthorizationError",
    "DEFAULT_MINIMUM_ROLES",
    "Identity",
    "InsufficientRoleError",
    "TenantIsolationError",
    "check_role_authorization",
    "check_tenant_isolation",
    # guardrails
    "AgentWorkflowGuardrail",
    "DEFAULT_INPUT_BLOCK_THRESHOLD",
    "InputGuardrailResult",
    "PromptInjectionDetected",
    "SecureToolCallResult",
    "ToolCallRequest",
    "ToolGuardrail",
    "enforce_input_guardrail",
    "scan_prompt_injection",
    "secure_tool_call",
    # sanitization
    "DEFAULT_BLOCK_CATEGORIES",
    "OutputValidationResult",
    "SensitiveMatch",
    "redact_sensitive_data",
    "scan_for_sensitive_data",
    "validate_output_text",
    # tool_policy
    "ArgumentSpec",
    "ToolArgumentValidationError",
    "ToolRegistry",
    "ToolRiskLevel",
    "ToolSpec",
    "UnknownToolError",
    "build_default_registry",
]
