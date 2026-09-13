"""Sensitive-data filtering and output validation.

Used from two places, deliberately sharing one implementation instead of
each guardrail layer inventing its own pattern list:

* Layer 1 (Input Guardrail, ``guardrails.scan_prompt_injection``) also
  flags sensitive data accidentally pasted into a prompt (e.g. a user
  pasting a real API key into their message) so it can be redacted before
  the text is ever sent to the LLM/logged/traced.
* Layer 3 (Tool Guardrail, ``guardrails.ToolGuardrail.sanitize_output``)
  scans and redacts/blocks whatever a tool call returns before it flows
  back into agent state or the final answer -- Requirement 6
  ("sensitive-data filtering") and Requirement 8 ("output validation").

Nothing here talks to an LLM; it is pure, dependency-free pattern
matching, so it is deterministic and safe to run on every single tool
call/response without any added latency or cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sensitive-data categories. Each pattern is intentionally conservative
# (favors some false positives over silently letting a real secret through
# -- "fail safe", matching this project's reliability philosophy in
# src/reliability/errors.py).
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_api_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{20,}"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+\b"),
    "phone": re.compile(r"\b\+?\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "secret_assignment": re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*\S+"),
}

# Categories severe enough that :func:`validate_output_text` blocks the
# output entirely (returns ``allowed=False``) rather than merely redacting
# it and letting a partially-redacted response through -- a leaked API key
# or full credit-card number is not safe to expose even in redacted form
# alongside the rest of the message, whereas an email/phone number is
# redacted but otherwise allowed through.
DEFAULT_BLOCK_CATEGORIES: frozenset[str] = frozenset(
    {"openai_api_key", "aws_access_key", "generic_bearer_token", "credit_card", "ssn", "secret_assignment"}
)


@dataclass(frozen=True)
class SensitiveMatch:
    category: str
    matched_text: str
    start: int
    end: int


def scan_for_sensitive_data(
    text: str, *, patterns: dict[str, re.Pattern[str]] | None = None
) -> tuple[SensitiveMatch, ...]:
    """Requirement 6: sensitive-data filtering, detection half. Returns
    every match found, across every category, in the order they appear."""
    patterns = patterns or SENSITIVE_PATTERNS
    matches: list[SensitiveMatch] = []
    for category, pattern in patterns.items():
        for m in pattern.finditer(text):
            matches.append(SensitiveMatch(category=category, matched_text=m.group(0), start=m.start(), end=m.end()))
    matches.sort(key=lambda m: m.start)
    return tuple(matches)


def redact_sensitive_data(text: str, *, patterns: dict[str, re.Pattern[str]] | None = None) -> str:
    """Requirement 6: sensitive-data filtering, redaction half. Replaces
    every match with ``[REDACTED:<category>]`` -- never silently drops the
    surrounding text, so a redacted response is still readable."""
    matches = scan_for_sensitive_data(text, patterns=patterns)
    if not matches:
        return text
    out = []
    cursor = 0
    for match in matches:
        if match.start < cursor:
            # Overlapping match (e.g. a secret_assignment span that
            # contains an email) -- skip; the outer/earlier match already
            # covers this text.
            continue
        out.append(text[cursor : match.start])
        out.append(f"[REDACTED:{match.category}]")
        cursor = match.end
    out.append(text[cursor:])
    return "".join(out)


@dataclass(frozen=True)
class OutputValidationResult:
    """Requirement 8: output validation. ``allowed=False`` means the
    output must not be returned to the caller at all (even redacted) --
    the caller should treat this as a tool/agent failure, not silently
    substitute the sanitized text."""

    allowed: bool
    sanitized_text: str
    violations: tuple[SensitiveMatch, ...]
    blocked_reason: str | None = None


def validate_output_text(
    text: str,
    *,
    block_on_categories: frozenset[str] = DEFAULT_BLOCK_CATEGORIES,
    patterns: dict[str, re.Pattern[str]] | None = None,
) -> OutputValidationResult:
    """Scans ``text`` (a tool result or an agent's final answer) for
    sensitive data. Low-severity categories (email/phone) are redacted but
    the output is still ``allowed``; high-severity categories
    (``block_on_categories``) cause ``allowed=False`` -- the caller must
    reject the output rather than pass along even a redacted version."""
    matches = scan_for_sensitive_data(text, patterns=patterns)
    sanitized = redact_sensitive_data(text, patterns=patterns)
    blocking = [m for m in matches if m.category in block_on_categories]
    if blocking:
        categories = sorted({m.category for m in blocking})
        return OutputValidationResult(
            allowed=False,
            sanitized_text=sanitized,
            violations=matches,
            blocked_reason=f"output contains high-severity sensitive data: {', '.join(categories)}",
        )
    return OutputValidationResult(allowed=True, sanitized_text=sanitized, violations=matches)


__all__ = [
    "SENSITIVE_PATTERNS",
    "DEFAULT_BLOCK_CATEGORIES",
    "SensitiveMatch",
    "scan_for_sensitive_data",
    "redact_sensitive_data",
    "OutputValidationResult",
    "validate_output_text",
]
