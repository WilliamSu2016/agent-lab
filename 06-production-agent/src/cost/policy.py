"""Three execution strategies (Fast / Balanced / Quality) and the
controlled-degradation ladder between them.

    Fast:      fewer agent calls, less context, a low-cost model.
    Balanced:  the default -- this project's existing
               ``DEFAULT_MAX_WORKERS``/``DEFAULT_MAX_ITERATIONS`` from
               ``src/multi_agent_research/state.py``.
    Quality:   more research/review iterations allowed.

Every mode's :class:`~src.cost.budget.Budget` is validated against the one
absolute :data:`~src.cost.budget.HARD_CEILING` at construction time (see
``budget.py``) -- Quality is "more headroom", never "no limit".

Controlled degradation (requirement: "Quality mode -> timeout -> 自动降级:
Quality -> Balanced -> Fast -> graceful failure"): :func:`run_with_degradation`
wraps a caller-supplied ``execute`` callable that takes one
:class:`ExecutionPolicy` and either returns a result or raises
:class:`~src.cost.budget.BudgetExceededError` (or ``TimeoutError``, treated
the same way). On such a failure, it re-tries with the next weaker policy;
if Fast also fails, it returns a structured :class:`GracefulFailure`
instead of raising -- a designed, observable stop, mirroring
``src/reliability/errors.py``'s ``ControlledFailure`` for the
Planner<->Reviewer loop budget, applied here to the mode-selection budget
instead.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

from src.cost.budget import Budget, BudgetExceededError, BudgetTracker


class ExecutionMode(str, enum.Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass(frozen=True)
class ExecutionPolicy:
    """Everything one mode controls: the hard :class:`Budget`, which model
    to call, and how much conversation/tool-result context to keep."""

    mode: ExecutionMode
    budget: Budget
    # None means "use whatever model config/Settings already specifies";
    # set only by Fast, to substitute a cheaper model. Never hard-coded as
    # a literal model name anywhere else in this project (see
    # ``config/settings.py``'s Configuration & Secrets requirement) -- this
    # is a *preference* a caller may honor by passing it into their own
    # ``config.Settings``/model-selection code, not a direct API call.
    model_name_override: Optional[str] = None
    # A proxy for "reduce context": the maximum number of characters of
    # accumulated research findings/trace passed into each subsequent
    # LLM call's prompt. Lower = cheaper & faster, at some quality cost.
    max_context_chars: int = 8000
    allow_parallel_workers: bool = True

    @property
    def max_agent_iterations(self) -> int:
        return self.budget.max_agent_iterations

    @property
    def max_workers(self) -> int:
        return self.budget.max_workers


# ---------------------------------------------------------------------------
# The three default policies.
# ---------------------------------------------------------------------------

FAST_POLICY = ExecutionPolicy(
    mode=ExecutionMode.FAST,
    budget=Budget(max_agent_iterations=1, max_workers=2, max_tokens=6_000, max_cost_usd=0.02, timeout_seconds=30.0),
    model_name_override="gpt-4o-mini",
    max_context_chars=2_000,
    allow_parallel_workers=True,
)

BALANCED_POLICY = ExecutionPolicy(
    mode=ExecutionMode.BALANCED,
    budget=Budget(max_agent_iterations=3, max_workers=6, max_tokens=40_000, max_cost_usd=0.20, timeout_seconds=120.0),
    model_name_override=None,
    max_context_chars=8_000,
    allow_parallel_workers=True,
)

QUALITY_POLICY = ExecutionPolicy(
    mode=ExecutionMode.QUALITY,
    budget=Budget(max_agent_iterations=6, max_workers=10, max_tokens=120_000, max_cost_usd=0.80, timeout_seconds=300.0),
    model_name_override=None,
    max_context_chars=24_000,
    allow_parallel_workers=True,
)

DEFAULT_POLICIES: dict[ExecutionMode, ExecutionPolicy] = {
    ExecutionMode.FAST: FAST_POLICY,
    ExecutionMode.BALANCED: BALANCED_POLICY,
    ExecutionMode.QUALITY: QUALITY_POLICY,
}

# Requirement: Balanced is the default mode.
DEFAULT_MODE = ExecutionMode.BALANCED

# The controlled-degradation ladder: Quality -> Balanced -> Fast -> (graceful failure).
DEGRADATION_ORDER: tuple[ExecutionMode, ...] = (ExecutionMode.QUALITY, ExecutionMode.BALANCED, ExecutionMode.FAST)


def next_degraded_mode(mode: ExecutionMode) -> Optional[ExecutionMode]:
    """The next weaker mode in the ladder, or ``None`` if ``mode`` is
    already Fast (meaning: the next step is graceful failure, not a
    weaker mode)."""
    try:
        index = DEGRADATION_ORDER.index(mode)
    except ValueError:
        raise ValueError(f"Unknown execution mode for degradation: {mode!r}") from None
    if index + 1 < len(DEGRADATION_ORDER):
        return DEGRADATION_ORDER[index + 1]
    return None


def get_policy(mode: ExecutionMode, policies: dict[ExecutionMode, ExecutionPolicy] = DEFAULT_POLICIES) -> ExecutionPolicy:
    return policies[mode]


# ---------------------------------------------------------------------------
# Controlled degradation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DegradationAttempt:
    """One attempt at running under one policy -- kept even for failed
    attempts, so the final :class:`DegradationOutcome` has a full audit
    trail of exactly which modes were tried and why each one failed."""

    mode: ExecutionMode
    succeeded: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class GracefulFailure:
    """Structured terminal state when even Fast mode could not complete
    within budget -- a designed, observable stop (mirrors
    ``src/reliability/errors.py::ControlledFailure``), never a crash and
    never a silent unbounded retry loop."""

    reason: str
    attempts: tuple[DegradationAttempt, ...]
    failure_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: float = field(default_factory=time.time)

    def to_trace_line(self) -> str:
        modes_tried = " -> ".join(a.mode.value for a in self.attempts)
        return f"GracefulFailure reason={self.reason!r} modes_tried={modes_tried!r} failure_id={self.failure_id}"


R = TypeVar("R")
Execute = Callable[[ExecutionPolicy], R]


@dataclass(frozen=True)
class DegradationOutcome:
    """Either ``result`` is set (some mode succeeded) or ``graceful_failure``
    is set (every mode down to Fast failed) -- never both, never neither."""

    result: Optional[object]
    final_mode: Optional[ExecutionMode]
    graceful_failure: Optional[GracefulFailure]
    attempts: tuple[DegradationAttempt, ...]

    @property
    def succeeded(self) -> bool:
        return self.graceful_failure is None


def run_with_degradation(
    execute: Execute,
    *,
    start_mode: ExecutionMode = ExecutionMode.QUALITY,
    policies: dict[ExecutionMode, ExecutionPolicy] = DEFAULT_POLICIES,
    on_degrade: Optional[Callable[[ExecutionMode, ExecutionMode, str], None]] = None,
) -> DegradationOutcome:
    """Runs ``execute(policy)`` starting at ``start_mode``; on
    :class:`~src.cost.budget.BudgetExceededError` or ``TimeoutError``,
    degrades to the next weaker mode (Quality -> Balanced -> Fast) and
    retries. If Fast also fails, returns a :class:`DegradationOutcome`
    whose ``graceful_failure`` is set -- callers must check
    ``outcome.succeeded`` rather than assuming a result is always present.

    ``on_degrade(from_mode, to_mode, reason)`` is an optional hook for
    logging/tracing each degradation step (e.g. into
    ``src/observability/logging.py``) without this module needing to
    import any specific logging backend itself.
    """
    mode: Optional[ExecutionMode] = start_mode
    attempts: list[DegradationAttempt] = []

    while mode is not None:
        policy = get_policy(mode, policies)
        try:
            result = execute(policy)
        except (BudgetExceededError, TimeoutError) as exc:
            reason = str(exc)
            attempts.append(DegradationAttempt(mode=mode, succeeded=False, error=reason))
            next_mode = next_degraded_mode(mode)
            if on_degrade is not None and next_mode is not None:
                on_degrade(mode, next_mode, reason)
            mode = next_mode
            continue
        attempts.append(DegradationAttempt(mode=mode, succeeded=True))
        return DegradationOutcome(result=result, final_mode=mode, graceful_failure=None, attempts=tuple(attempts))

    graceful_failure = GracefulFailure(
        reason="every execution mode (Quality -> Balanced -> Fast) exceeded its budget",
        attempts=tuple(attempts),
    )
    return DegradationOutcome(result=None, final_mode=None, graceful_failure=graceful_failure, attempts=tuple(attempts))


__all__ = [
    "ExecutionMode",
    "ExecutionPolicy",
    "FAST_POLICY",
    "BALANCED_POLICY",
    "QUALITY_POLICY",
    "DEFAULT_POLICIES",
    "DEFAULT_MODE",
    "DEGRADATION_ORDER",
    "next_degraded_mode",
    "get_policy",
    "DegradationAttempt",
    "GracefulFailure",
    "DegradationOutcome",
    "run_with_degradation",
]
