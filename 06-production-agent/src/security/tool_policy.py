"""Layer 3 support: Tool risk classification, registry, and argument
validation ("Tool Guardrail").

Requirement: every tool must be classified into one of three risk tiers:

    LOW    -- read-only (e.g. ``search_web``)
    MEDIUM -- write but reversible (e.g. ``update_record``)
    HIGH   -- external side effect, generally irreversible (e.g. ``send_email``)

The risk tier is looked up from a single :class:`ToolRegistry` -- nothing
in this project should ever decide "is this tool dangerous?" ad hoc at a
call site; it is always this registry's job, so
``guardrails.AgentWorkflowGuardrail`` (Layer 2) and
``guardrails.ToolGuardrail`` (Layer 3) both key every decision (role
requirement, approval requirement, argument schema) off the one
:class:`ToolSpec` for that tool name.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


class ToolRiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Ordering used for "at least this risky" comparisons (e.g. minimum-role
# tables keyed by risk level fall back to the *next lower* tier's
# requirement if a tier has none of its own -- not used by default here,
# but kept so future policy code doesn't need to invent its own ordering).
RISK_ORDER: dict[ToolRiskLevel, int] = {
    ToolRiskLevel.LOW: 0,
    ToolRiskLevel.MEDIUM: 1,
    ToolRiskLevel.HIGH: 2,
}


class UnknownToolError(KeyError):
    """Raised when a tool name has no registered :class:`ToolSpec`. A tool
    that has never been risk-classified must never be callable -- fail
    closed, not open."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Unknown tool {tool_name!r}: every tool must be registered with a "
            "risk level (LOW/MEDIUM/HIGH) via ToolRegistry.register() before "
            "it can be authorized or executed."
        )
        self.tool_name = tool_name


class ToolArgumentValidationError(ValueError):
    """Layer 3: raised when a tool call's arguments do not match its
    registered schema -- unexpected/missing/mistyped/invalid arguments are
    all rejected *before* the tool function is ever invoked."""

    def __init__(self, tool_name: str, errors: list[str]):
        super().__init__(f"Invalid arguments for tool {tool_name!r}: " + "; ".join(errors))
        self.tool_name = tool_name
        self.errors = errors


@dataclass(frozen=True)
class ArgumentSpec:
    """One argument's schema: type + required-ness + an optional extra
    validation rule (e.g. "looks like an email address")."""

    name: str
    type_: type
    required: bool = True
    validator: Callable[[Any], bool] | None = None
    description: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool's full policy: its risk tier and its argument
    schema. ``requires_approval`` is derived (never set independently of
    ``risk_level``) so a HIGH-risk tool can never accidentally be wired up
    to skip the approval gate."""

    name: str
    risk_level: ToolRiskLevel
    description: str
    arguments: tuple[ArgumentSpec, ...] = ()

    @property
    def requires_approval(self) -> bool:
        return self.risk_level is ToolRiskLevel.HIGH

    def validate_arguments(self, kwargs: Mapping[str, Any]) -> None:
        """Requirement 2: tool argument validation. Raises
        :class:`ToolArgumentValidationError` (never lets a malformed
        argument reach the real tool function) listing every problem found
        in one pass, not just the first."""
        errors: list[str] = []
        allowed_names = {spec.name for spec in self.arguments}
        for extra in kwargs:
            if extra not in allowed_names:
                errors.append(f"unexpected argument {extra!r}")

        for spec in self.arguments:
            if spec.name not in kwargs:
                if spec.required:
                    errors.append(f"missing required argument {spec.name!r}")
                continue
            value = kwargs[spec.name]
            if not isinstance(value, spec.type_):
                errors.append(
                    f"argument {spec.name!r} must be of type {spec.type_.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue
            if spec.validator is not None and not spec.validator(value):
                reason = spec.description or "custom validation rule"
                errors.append(f"argument {spec.name!r} failed validation ({reason})")

        if errors:
            raise ToolArgumentValidationError(self.name, errors)


class ToolRegistry:
    """The single source of truth for "which tools exist and how risky are
    they". Fails closed: an unregistered tool name is never callable."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        return spec

    def get(self, tool_name: str) -> ToolSpec:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise UnknownToolError(tool_name) from exc

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def all(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())


def _looks_like_email(value: str) -> bool:
    return "@" in value and "." in value.split("@")[-1]


def build_default_registry() -> ToolRegistry:
    """The three example tools named in the requirement, one per risk
    tier, each with a realistic argument schema."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_web",
            risk_level=ToolRiskLevel.LOW,
            description="Read-only web search; no state anywhere is changed.",
            arguments=(ArgumentSpec("query", str, required=True),),
        )
    )
    registry.register(
        ToolSpec(
            name="update_record",
            risk_level=ToolRiskLevel.MEDIUM,
            description="Writes one field of one record, but the write can be undone (reversible).",
            arguments=(
                ArgumentSpec("record_id", str, required=True),
                ArgumentSpec("field", str, required=True),
                ArgumentSpec("value", str, required=True),
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="send_email",
            risk_level=ToolRiskLevel.HIGH,
            description="External side effect: an email, once sent, cannot be recalled.",
            arguments=(
                ArgumentSpec(
                    "to", str, required=True, validator=_looks_like_email, description="must look like an email address"
                ),
                ArgumentSpec("subject", str, required=True),
                ArgumentSpec("body", str, required=True),
            ),
        )
    )
    return registry


__all__ = [
    "ToolRiskLevel",
    "RISK_ORDER",
    "UnknownToolError",
    "ToolArgumentValidationError",
    "ArgumentSpec",
    "ToolSpec",
    "ToolRegistry",
    "build_default_registry",
]
