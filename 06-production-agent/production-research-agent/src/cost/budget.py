"""Hard budget caps + a mutable per-request tracker that enforces them.

Requirement: "建立 MAX_AGENT_ITERATIONS / MAX_WORKERS / MAX_TOKENS / MAX_COST
/ TIMEOUT。如果超过：进入 controlled degradation。" This module owns the
*definition* of a budget and the *detection* of a violation
(:class:`BudgetTracker` raises :class:`BudgetExceededError` the moment any
one dimension is crossed). It does not decide what to do about a
violation -- degrading Quality -> Balanced -> Fast -> graceful failure is
``policy.py``'s job, layered on top of this module.

Two kinds of limits exist here, and they are deliberately not the same
thing:

* :data:`HARD_CEILING` -- one absolute, mode-independent safety net. No
  :class:`Budget` (including a custom one built ad hoc by a caller) may
  ever exceed this, regardless of which of the three modes
  (Fast/Balanced/Quality) it belongs to -- this is what stops "Quality
  mode" from ever being able to accidentally authorize an unbounded run.
* the per-mode :class:`Budget` values in ``policy.py`` (``DEFAULT_POLICIES``)
  -- these are the actual caps enforced for a given run, always
  ``<= HARD_CEILING`` (validated in :meth:`Budget.__post_init__`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class HardCeiling:
    """The one absolute, mode-independent upper bound on every dimension.
    Every :class:`Budget`, no matter which mode it backs, must fit under
    this -- it is the safety net beneath Fast/Balanced/Quality, not one of
    the three modes itself."""

    max_agent_iterations: int = 10
    max_workers: int = 12
    max_tokens: int = 200_000
    max_cost_usd: float = 2.0
    timeout_seconds: float = 600.0


HARD_CEILING = HardCeiling()


@dataclass(frozen=True)
class Budget:
    """The five required limits for one execution policy."""

    max_agent_iterations: int
    max_workers: int
    max_tokens: int
    max_cost_usd: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        checks = (
            ("max_agent_iterations", self.max_agent_iterations, HARD_CEILING.max_agent_iterations),
            ("max_workers", self.max_workers, HARD_CEILING.max_workers),
            ("max_tokens", self.max_tokens, HARD_CEILING.max_tokens),
            ("max_cost_usd", self.max_cost_usd, HARD_CEILING.max_cost_usd),
            ("timeout_seconds", self.timeout_seconds, HARD_CEILING.timeout_seconds),
        )
        for name, value, ceiling in checks:
            if value <= 0:
                raise ValueError(f"Budget.{name} must be > 0, got {value!r}.")
            if value > ceiling:
                raise ValueError(
                    f"Budget.{name}={value!r} exceeds the absolute HARD_CEILING.{name}={ceiling!r}. "
                    "No mode may authorize a run above the hard ceiling."
                )


# ---------------------------------------------------------------------------
# Violation detection.
# ---------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised by :class:`BudgetTracker` the instant any one budget
    dimension is crossed. Carries enough structured detail
    (``dimension``/``used``/``limit``) for the caller (``policy.py``'s
    degradation controller) to log/trace it and decide the next mode --
    never just a bare ``RuntimeError("budget exceeded")`` string."""

    def __init__(self, dimension: str, used: float, limit: float):
        super().__init__(f"Budget exceeded: {dimension}={used!r} > limit={limit!r}")
        self.dimension = dimension
        self.used = used
        self.limit = limit


@dataclass
class BudgetUsage:
    """A live snapshot of what one run has consumed so far, independent of
    which :class:`Budget` it is being checked against -- the same usage
    counters are meaningful whether the run started in Fast, Balanced, or
    Quality mode."""

    agent_iterations: int = 0
    workers_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at


class BudgetTracker:
    """Wraps a :class:`Budget` + live :class:`BudgetUsage` and raises
    :class:`BudgetExceededError` the moment any dimension is crossed.
    Every call site that consumes budget (one more agent iteration, one
    more worker dispatched, N more tokens spent, $ more spent) should go
    through this tracker rather than comparing against the ``Budget``
    fields directly, so the violation is always detected -- and raised --
    at the same single choke point.
    """

    def __init__(self, budget: Budget, usage: Optional[BudgetUsage] = None):
        self.budget = budget
        self.usage = usage or BudgetUsage()

    def record_iteration(self) -> None:
        self.usage.agent_iterations += 1
        if self.usage.agent_iterations > self.budget.max_agent_iterations:
            raise BudgetExceededError("max_agent_iterations", self.usage.agent_iterations, self.budget.max_agent_iterations)

    def record_workers(self, count: int) -> None:
        self.usage.workers_used = max(self.usage.workers_used, count)
        if count > self.budget.max_workers:
            raise BudgetExceededError("max_workers", count, self.budget.max_workers)

    def record_tokens(self, tokens: int) -> None:
        self.usage.tokens_used += tokens
        if self.usage.tokens_used > self.budget.max_tokens:
            raise BudgetExceededError("max_tokens", self.usage.tokens_used, self.budget.max_tokens)

    def record_cost(self, cost_usd: float) -> None:
        self.usage.cost_usd += cost_usd
        if self.usage.cost_usd > self.budget.max_cost_usd:
            raise BudgetExceededError("max_cost_usd", self.usage.cost_usd, self.budget.max_cost_usd)

    def check_timeout(self) -> None:
        if self.usage.elapsed_seconds > self.budget.timeout_seconds:
            raise BudgetExceededError("timeout_seconds", self.usage.elapsed_seconds, self.budget.timeout_seconds)

    def check_all(self) -> None:
        """Re-validates every already-recorded dimension against the
        current budget -- used right after a degradation swaps in a
        smaller budget, to immediately detect "the usage so far already
        exceeds the *new*, smaller budget too" instead of waiting for the
        next ``record_*``/``check_timeout`` call."""
        if self.usage.agent_iterations > self.budget.max_agent_iterations:
            raise BudgetExceededError("max_agent_iterations", self.usage.agent_iterations, self.budget.max_agent_iterations)
        if self.usage.workers_used > self.budget.max_workers:
            raise BudgetExceededError("max_workers", self.usage.workers_used, self.budget.max_workers)
        if self.usage.tokens_used > self.budget.max_tokens:
            raise BudgetExceededError("max_tokens", self.usage.tokens_used, self.budget.max_tokens)
        if self.usage.cost_usd > self.budget.max_cost_usd:
            raise BudgetExceededError("max_cost_usd", self.usage.cost_usd, self.budget.max_cost_usd)
        self.check_timeout()


__all__ = [
    "HardCeiling",
    "HARD_CEILING",
    "Budget",
    "BudgetExceededError",
    "BudgetUsage",
    "BudgetTracker",
]
