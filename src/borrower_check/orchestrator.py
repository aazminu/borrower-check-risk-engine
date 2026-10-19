"""Idempotent partner risk decision gateway with circuit-breaker recovery and tamper-evident audit logs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

from borrower_check.identity_graph import (
    ApplicantIdentity,
    BipartiteIdentityGraph,
    IdentityGraphAnalysis,
)
from borrower_check.income_verifier import (
    IncomeRiskVerifier,
    IncomeVerificationInput,
    IncomeVerificationResult,
    RiskTier,
)


class CircuitState(str, Enum):
    """State machine states for external credit/payroll bureau circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class DecisionAction(str, Enum):
    """Partner auto-lending underwriting routing actions."""

    AUTO_APPROVE = "AUTO_APPROVE"
    STIPULATION_REQUIRED = "STIPULATION_REQUIRED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DECLINE_FRAUD = "DECLINE_FRAUD"


class IdempotencyConflictError(RuntimeError):
    """Raised when an idempotency key is reused with a mutated request payload."""


class CircuitBreaker:
    """Protects real-time dealer checkout flows from upstream payroll/bureau latency spikes."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if (self._clock() - self._opened_at) >= self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(self) -> bool:
        current = self.state
        return current in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()


@dataclass(frozen=True)
class BorrowerCheckRequest:
    """Inbound partner risk verification payload."""

    idempotency_key: str
    correlation_id: str
    dealer_id: str
    identity: ApplicantIdentity
    income: IncomeVerificationInput

    def payload_fingerprint(self) -> str:
        """Compute canonical SHA-256 fingerprint of request parameters to enforce idempotency."""
        canonical = {
            "dealer_id": self.dealer_id,
            "application_id": self.identity.application_id,
            "ssn_hash": self.identity.ssn_hash,
            "phone": self.identity.phone,
            "email": self.identity.email,
            "address_hash": self.identity.address_hash,
            "device_fingerprint": self.identity.device_fingerprint,
            "stated_annual_income": self.income.stated_annual_income,
            "soc_code": self.income.soc_code,
            "region_code": self.income.region_code,
            "loan_amount": self.income.loan_amount,
        }
        encoded = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BorrowerCheckDecision:
    """Comprehensive fraud and income risk decision returned to the auto-lending platform."""

    idempotency_key: str
    correlation_id: str
    decision_action: DecisionAction
    risk_tier: RiskTier
    combined_probability: float
    identity_analysis: IdentityGraphAnalysis
    income_result: IncomeVerificationResult
    degraded_mode: bool
    cached_replay: bool
    audit_hash: str


@dataclass(frozen=True)
class AuditLogEntry:
    """Tamper-evident hash-chained telemetry record for regulatory and partner auditing."""

    sequence_id: int
    correlation_id: str
    idempotency_key: str
    dealer_id: str
    application_id: str
    decision_action: DecisionAction
    risk_tier: RiskTier
    degraded_mode: bool
    previous_hash: str
    audit_hash: str


class BorrowerCheckOrchestrator:
    """Coordinates bipartite graph analysis, external payroll enrichment, and idempotent decisioning."""

    def __init__(
        self,
        identity_graph: Optional[BipartiteIdentityGraph] = None,
        income_verifier: Optional[IncomeRiskVerifier] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        external_income_provider: Optional[Callable[[ str], Optional[float]]] = None,
    ) -> None:
        self.identity_graph = identity_graph or BipartiteIdentityGraph()
        self.income_verifier = income_verifier or IncomeRiskVerifier()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._external_income_provider = external_income_provider
        self._idempotency_store: Dict[str, Tuple[str, BorrowerCheckDecision]] = {}
        self._audit_log: List[AuditLogEntry] = []

    @property
    def audit_log(self) -> Tuple[AuditLogEntry, ...]:
        return tuple(self._audit_log)

    def _resolve_verified_income(self, request: BorrowerCheckRequest) -> Tuple[IncomeVerificationInput, bool]:
        """Fetch verified payroll income via circuit-breaker if not already supplied."""
        if request.income.verified_annual_income is not None or self._external_income_provider is None:
            return request.income, False

        if not self.circuit_breaker.allow_request():
            return request.income, True

        try:
            verified = self._external_income_provider(request.identity.ssn_hash)
            self.circuit_breaker.record_success()
            return replace(request.income, verified_annual_income=verified), False
        except Exception:
            self.circuit_breaker.record_failure()
            return request.income, True

    @staticmethod
    def _select_action(
        risk_tier: RiskTier,
        income_result: IncomeVerificationResult,
        degraded_mode: bool,
    ) -> DecisionAction:
        if risk_tier == RiskTier.CRITICAL_SYNTHETIC:
            return DecisionAction.DECLINE_FRAUD
        if risk_tier == RiskTier.HIGH:
            return DecisionAction.MANUAL_REVIEW
        if risk_tier == RiskTier.MEDIUM or income_result.income_inflation_flag:
            return DecisionAction.STIPULATION_REQUIRED
        if degraded_mode:
            # Never silently auto-approve without paystub stipulation when external bureau is degraded
            return DecisionAction.STIPULATION_REQUIRED
        return DecisionAction.AUTO_APPROVE

    def _append_audit_entry(
        self,
        request: BorrowerCheckRequest,
        decision_action: DecisionAction,
        risk_tier: RiskTier,
        degraded_mode: bool,
    ) -> str:
        prev_hash = self._audit_log[-1].audit_hash if self._audit_log else ("0" * 64)
        seq_id = len(self._audit_log) + 1
        record_str = (
            f"{seq_id}|{request.correlation_id}|{request.idempotency_key}|"
            f"{request.dealer_id}|{request.identity.application_id}|"
            f"{decision_action.value}|{risk_tier.value}|{degraded_mode}|{prev_hash}"
        )
        digest = hashlib.sha256(record_str.encode("utf-8")).hexdigest()
        entry = AuditLogEntry(
            sequence_id=seq_id,
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            dealer_id=request.dealer_id,
            application_id=request.identity.application_id,
            decision_action=decision_action,
            risk_tier=risk_tier,
            degraded_mode=degraded_mode,
            previous_hash=prev_hash,
            audit_hash=digest,
        )
        self._audit_log.append(entry)
        return digest

    def evaluate_application(self, request: BorrowerCheckRequest) -> BorrowerCheckDecision:
        """Process a borrower check request idempotently and record a hash-chained audit entry."""
        fingerprint = request.payload_fingerprint()
        cached = self._idempotency_store.get(request.idempotency_key)
        if cached is not None:
            cached_fp, cached_decision = cached
            if cached_fp != fingerprint:
                raise IdempotencyConflictError(
                    f"Idempotency key '{request.idempotency_key}' reused with different payload."
                )
            return replace(cached_decision, cached_replay=True)

        identity_analysis = self.identity_graph.ingest_and_evaluate(request.identity)
        enriched_income_input, degraded_mode = self._resolve_verified_income(request)
        income_result = self.income_verifier.evaluate(enriched_income_input, identity_analysis)

        action = self._select_action(income_result.risk_tier, income_result, degraded_mode)
        audit_hash = self._append_audit_entry(request, action, income_result.risk_tier, degraded_mode)

        decision = BorrowerCheckDecision(
            idempotency_key=request.idempotency_key,
            correlation_id=request.correlation_id,
            decision_action=action,
            risk_tier=income_result.risk_tier,
            combined_probability=income_result.calibrated_probability,
            identity_analysis=identity_analysis,
            income_result=income_result,
            degraded_mode=degraded_mode,
            cached_replay=False,
            audit_hash=audit_hash,
        )
        self._idempotency_store[request.idempotency_key] = (fingerprint, decision)
        return decision

    def verify_audit_chain(self) -> bool:
        """Verify cryptographic integrity of the entire in-memory audit log chain."""
        expected_prev = "0" * 64
        for entry in self._audit_log:
            if entry.previous_hash != expected_prev:
                return False
            record_str = (
                f"{entry.sequence_id}|{entry.correlation_id}|{entry.idempotency_key}|"
                f"{entry.dealer_id}|{entry.application_id}|"
                f"{entry.decision_action.value}|{entry.risk_tier.value}|"
                f"{entry.degraded_mode}|{entry.previous_hash}"
            )
            recomputed = hashlib.sha256(record_str.encode("utf-8")).hexdigest()
            if recomputed != entry.audit_hash:
                return False
            expected_prev = entry.audit_hash
        return True
