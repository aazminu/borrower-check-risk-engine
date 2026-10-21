"""BorrowerCheck Risk Engine: Synthetic Identity, Income Verification, and Resilient Orchestration."""

from borrower_check.identity_graph import (
    ApplicantIdentity,
    BipartiteIdentityGraph,
    IdentityGraphAnalysis,
)
from borrower_check.income_verifier import (
    IncomeRiskVerifier,
    IncomeVerificationInput,
    IncomeVerificationResult,
    OccupationBenchmark,
    RiskTier,
)
from borrower_check.orchestrator import (
    AuditLogEntry,
    BorrowerCheckDecision,
    BorrowerCheckOrchestrator,
    BorrowerCheckRequest,
    CircuitBreaker,
    CircuitState,
    DecisionAction,
    IdempotencyConflictError,
)

__all__ = [
    "ApplicantIdentity",
    "AuditLogEntry",
    "BipartiteIdentityGraph",
    "BorrowerCheckDecision",
    "BorrowerCheckOrchestrator",
    "BorrowerCheckRequest",
    "CircuitBreaker",
    "CircuitState",
    "DecisionAction",
    "IdempotencyConflictError",
    "IdentityGraphAnalysis",
    "IncomeRiskVerifier",
    "IncomeVerificationInput",
    "IncomeVerificationResult",
    "OccupationBenchmark",
    "RiskTier",
]

__version__ = "0.2.0"

from .diagnostics import run_diagnostic_suite, validate_runtime_invariants
__all__ = list(__all__) + ["run_diagnostic_suite", "validate_runtime_invariants"]
