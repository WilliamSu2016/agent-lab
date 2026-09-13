"""Cost & Latency Optimization for the Multi-Agent Research Agent.

    src/cost/
    ├── budget.py       # Budget / BudgetTracker / BudgetExceededError + the
    │                     one absolute HARD_CEILING every mode fits under
    ├── policy.py        # ExecutionMode (Fast/Balanced/Quality) + policies +
    │                     controlled degradation (run_with_degradation)
    └── estimator.py      # cost model: retrospective (from a trace) and
                           prospective (from a policy, before running)

See ``docs/07-COST-LATENCY.md`` for the full design write-up.
"""

from src.cost.budget import (
    HARD_CEILING,
    Budget,
    BudgetExceededError,
    BudgetTracker,
    BudgetUsage,
    HardCeiling,
)
from src.cost.estimator import (
    CostPrediction,
    RequestCostReport,
    estimate_request_cost,
    predict_cost,
)
from src.cost.policy import (
    BALANCED_POLICY,
    DEFAULT_MODE,
    DEFAULT_POLICIES,
    DEGRADATION_ORDER,
    FAST_POLICY,
    QUALITY_POLICY,
    DegradationAttempt,
    DegradationOutcome,
    ExecutionMode,
    ExecutionPolicy,
    GracefulFailure,
    get_policy,
    next_degraded_mode,
    run_with_degradation,
)

__all__ = [
    # budget
    "HardCeiling",
    "HARD_CEILING",
    "Budget",
    "BudgetExceededError",
    "BudgetUsage",
    "BudgetTracker",
    # policy
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
    # estimator
    "RequestCostReport",
    "estimate_request_cost",
    "CostPrediction",
    "predict_cost",
]
