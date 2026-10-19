"""Comprehensive unit tests for BorrowerCheck Risk Engine."""

from __future__ import annotations

import pytest

from borrower_check import (
    ApplicantIdentity,
    BipartiteIdentityGraph,
    BorrowerCheckOrchestrator,
    BorrowerCheckRequest,
    CircuitBreaker,
    CircuitState,
    DecisionAction,
    IdempotencyConflictError,
    IncomeRiskVerifier,
    IncomeVerificationInput,
    RiskTier,
)


def _clean_identity(
    app_id: str = "APP-1001",
    ssn_hash: str = "ssn_clean_001",
    phone: str = "+14045550101",
    email: str = "alex.mercer@example.com",
    address_hash: str = "addr_dunwoody_01",
    device_fp: str = "dev_ios_991",
    timestamp: float = 1_700_000_000.0,
    file_age: int = 72,
    au_lines: int = 0,
    primary_lines: int = 5,
) -> ApplicantIdentity:
    return ApplicantIdentity(
        application_id=app_id,
        ssn_hash=ssn_hash,
        phone=phone,
        email=email,
        address_hash=address_hash,
        device_fingerprint=device_fp,
        timestamp=timestamp,
        credit_file_age_months=file_age,
        authorized_user_tradelines=au_lines,
        primary_tradelines=primary_lines,
    )


def test_clean_applicant_zero_bipartite_entropy() -> None:
    graph = BipartiteIdentityGraph()
    analysis = graph.ingest_and_evaluate(_clean_identity())
    assert analysis.shared_ssn_identities == 1
    assert analysis.shared_phone_ssns == 1
    assert analysis.bipartite_entropy_score == 0.0
    assert analysis.thin_file_au_abuse_flag is False
    assert analysis.connected_component_size == 1
    assert analysis.synthetic_cluster_score == 0.0
    assert analysis.risk_reasons == ()


def test_ssn_fragmentation_across_multiple_profiles() -> None:
    graph = BipartiteIdentityGraph()
    graph.ingest_and_evaluate(_clean_identity(app_id="A1", phone="+14045550101", email="one@example.com"))
    analysis = graph.ingest_and_evaluate(
        _clean_identity(app_id="A2", phone="+14045550999", email="two@example.com", address_hash="addr_02")
    )
    assert analysis.shared_ssn_identities == 2
    assert analysis.bipartite_entropy_score > 0.0
    assert any(r.startswith("SSN_FRAGMENTATION") for r in analysis.risk_reasons)


def test_synthetic_identity_ring_shared_phone_and_device() -> None:
    graph = BipartiteIdentityGraph()
    for idx in range(1, 4):
        analysis = graph.ingest_and_evaluate(
            _clean_identity(
                app_id=f"RING-{idx}",
                ssn_hash=f"ssn_synth_{idx}",
                phone="+14045559999",
                device_fp="dev_emulator_01",
                email=f"ring{idx}@example.com",
                address_hash=f"addr_{idx}",
                timestamp=1_700_000_000.0 + idx * 30,
            )
        )
    assert analysis.shared_phone_ssns == 3
    assert analysis.shared_device_ssns == 3
    assert analysis.connected_component_size == 3
    assert analysis.synthetic_cluster_score >= 0.60
    assert "SHARED_PHONE_ACROSS_3_SSNS" in analysis.risk_reasons
    assert "SYNTHETIC_RING_CLUSTER_SIZE_3" in analysis.risk_reasons


def test_thin_file_authorized_user_piggybacking_detection() -> None:
    graph = BipartiteIdentityGraph()
    analysis = graph.ingest_and_evaluate(
        _clean_identity(file_age=5, au_lines=4, primary_lines=0)
    )
    assert analysis.thin_file_au_abuse_flag is True
    assert "THIN_FILE_AUTHORIZED_USER_PIGGYBACKING" in analysis.risk_reasons
    assert analysis.synthetic_cluster_score >= 0.28


def test_sliding_window_velocity_1h_burst_and_24h_pruning() -> None:
    graph = BipartiteIdentityGraph()
    base_ts = 1_700_000_000.0
    # Submit 3 applications within 10 minutes sharing a device fingerprint
    for i in range(3):
        res = graph.ingest_and_evaluate(
            _clean_identity(
                app_id=f"VEL-{i}",
                ssn_hash=f"ssn_v_{i}",
                phone=f"+1404555011{i}",
                device_fp="dev_shared_kiosk",
                timestamp=base_ts + i * 120.0,
            )
        )
    assert res.velocity_1h_count == 3
    assert any(r.startswith("VELOCITY_BURST_1H") for r in res.risk_reasons)

    # An application 25 hours later should prune the earlier 24h window
    later_res = graph.ingest_and_evaluate(
        _clean_identity(
            app_id="VEL-LATER",
            ssn_hash="ssn_v_99",
            phone="+14045550199",
            device_fp="dev_shared_kiosk",
            timestamp=base_ts + 90_000.0,
        )
    )
    assert later_res.velocity_1h_count == 1
    assert later_res.velocity_24h_count == 1


def test_bls_income_z_score_normal_software_developer() -> None:
    verifier = IncomeRiskVerifier()
    result = verifier.evaluate(
        IncomeVerificationInput(
            stated_annual_income=135_000.0,
            soc_code="15-1252",
            region_code="ATL-GA",
            existing_monthly_debt=1_200.0,
            proposed_auto_payment=650.0,
            vehicle_msrp=42_000.0,
            loan_amount=36_000.0,
        )
    )
    assert -0.5 < result.occupation_z_score < 0.5
    assert result.income_inflation_flag is False
    assert result.risk_tier == RiskTier.LOW
    assert result.adverse_action_codes == ()


def test_bls_income_inflation_outlier_z_score_flags_adverse_code() -> None:
    verifier = IncomeRiskVerifier()
    # Stated $115,000 for Fast Food Counter Worker (median ~$29.5k, std ~$6.1k)
    result = verifier.evaluate(
        IncomeVerificationInput(
            stated_annual_income=115_000.0,
            soc_code="35-3023",
            region_code="NATIONAL",
            existing_monthly_debt=800.0,
            proposed_auto_payment=950.0,
            vehicle_msrp=55_000.0,
            loan_amount=52_000.0,
        )
    )
    assert result.occupation_z_score > 10.0
    assert result.income_inflation_flag is True
    assert "INC_01_OCCUPATION_ZSCORE_OUTLIER" in result.adverse_action_codes
    assert result.risk_tier in (RiskTier.MEDIUM, RiskTier.HIGH)


def test_verified_payroll_mismatch_recomputes_true_dti() -> None:
    verifier = IncomeRiskVerifier()
    result = verifier.evaluate(
        IncomeVerificationInput(
            stated_annual_income=120_000.0,
            verified_annual_income=48_000.0,
            soc_code="43-4051",
            region_code="ATL-GA",
            existing_monthly_debt=1_400.0,
            proposed_auto_payment=800.0,
            vehicle_msrp=38_000.0,
            loan_amount=36_000.0,
        )
    )
    # Discrepancy = (120k - 48k) / 48k = 1.50 (150% inflation)
    assert result.income_discrepancy_ratio == pytest.approx(1.50)
    # Effective monthly income uses verified $48k / 12 = $4,000 -> post-loan DTI = 2200 / 4000 = 0.55
    assert result.post_loan_dti == pytest.approx(0.55)
    assert "INC_02_VERIFIED_PAYROLL_MISMATCH" in result.adverse_action_codes
    assert "DTI_01_EXCESSIVE_POST_LOAN_OBLIGATION" in result.adverse_action_codes
    assert result.risk_tier == RiskTier.HIGH


def test_invalid_income_or_msrp_raises_value_error() -> None:
    verifier = IncomeRiskVerifier()
    with pytest.raises(ValueError, match="stated_annual_income"):
        verifier.evaluate(IncomeVerificationInput(stated_annual_income=0.0, soc_code="15-1252"))
    with pytest.raises(ValueError, match="vehicle_msrp"):
        verifier.evaluate(
            IncomeVerificationInput(stated_annual_income=80_000.0, soc_code="15-1252", vehicle_msrp=-10.0)
        )


def test_circuit_breaker_transitions_closed_open_half_open_closed() -> None:
    now = [100.0]
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=15.0, clock=lambda: now[0])
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

    # Advance clock past recovery timeout
    now[0] = 116.0
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request() is True

    # Success in HALF_OPEN resets circuit to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_orchestrator_auto_approves_clean_verified_borrower() -> None:
    orchestrator = BorrowerCheckOrchestrator(
        external_income_provider=lambda ssn: 130_000.0
    )
    req = BorrowerCheckRequest(
        idempotency_key="idem-clean-01",
        correlation_id="corr-001",
        dealer_id="DEALER-COX-ATL-01",
        identity=_clean_identity(),
        income=IncomeVerificationInput(
            stated_annual_income=130_000.0,
            soc_code="15-1252",
            region_code="ATL-GA",
            existing_monthly_debt=900.0,
            proposed_auto_payment=550.0,
            vehicle_msrp=40_000.0,
            loan_amount=32_000.0,
        ),
    )
    decision = orchestrator.evaluate_application(req)
    assert decision.decision_action == DecisionAction.AUTO_APPROVE
    assert decision.risk_tier == RiskTier.LOW
    assert decision.degraded_mode is False
    assert decision.cached_replay is False
    assert orchestrator.verify_audit_chain() is True


def test_orchestrator_idempotent_replay_returns_cached_decision() -> None:
    orchestrator = BorrowerCheckOrchestrator()
    req = BorrowerCheckRequest(
        idempotency_key="idem-replay-99",
        correlation_id="corr-099",
        dealer_id="DEALER-COX-ATL-01",
        identity=_clean_identity(),
        income=IncomeVerificationInput(
            stated_annual_income=125_000.0,
            verified_annual_income=125_000.0,
            soc_code="15-1252",
            existing_monthly_debt=800.0,
            proposed_auto_payment=500.0,
        ),
    )
    first = orchestrator.evaluate_application(req)
    second = orchestrator.evaluate_application(req)
    assert first.cached_replay is False
    assert second.cached_replay is True
    assert first.audit_hash == second.audit_hash
    assert len(orchestrator.audit_log) == 1


def test_orchestrator_idempotency_conflict_raises_error() -> None:
    orchestrator = BorrowerCheckOrchestrator()
    req1 = BorrowerCheckRequest(
        idempotency_key="idem-conflict-1",
        correlation_id="corr-1",
        dealer_id="DEALER-01",
        identity=_clean_identity(),
        income=IncomeVerificationInput(stated_annual_income=90_000.0, soc_code="29-1141"),
    )
    req2 = BorrowerCheckRequest(
        idempotency_key="idem-conflict-1",
        correlation_id="corr-2",
        dealer_id="DEALER-01",
        identity=_clean_identity(),
        income=IncomeVerificationInput(stated_annual_income=190_000.0, soc_code="29-1141"),
    )
    orchestrator.evaluate_application(req1)
    with pytest.raises(IdempotencyConflictError):
        orchestrator.evaluate_application(req2)


def test_orchestrator_degraded_mode_when_external_bureau_throws() -> None:
    def failing_provider(ssn: str) -> float:
        raise TimeoutError("Upstream payroll bureau timeout")

    cb = CircuitBreaker(failure_threshold=2)
    orchestrator = BorrowerCheckOrchestrator(
        circuit_breaker=cb,
        external_income_provider=failing_provider,
    )
    req = BorrowerCheckRequest(
        idempotency_key="idem-deg-1",
        correlation_id="corr-deg-1",
        dealer_id="DEALER-01",
        identity=_clean_identity(),
        income=IncomeVerificationInput(
            stated_annual_income=125_000.0,
            soc_code="15-1252",
            existing_monthly_debt=600.0,
            proposed_auto_payment=450.0,
        ),
    )
    decision = orchestrator.evaluate_application(req)
    assert decision.degraded_mode is True
    assert decision.decision_action == DecisionAction.STIPULATION_REQUIRED
    assert cb.failure_count == 1


def test_orchestrator_declines_critical_synthetic_identity_ring() -> None:
    orchestrator = BorrowerCheckOrchestrator()
    for idx in range(1, 4):
        req = BorrowerCheckRequest(
            idempotency_key=f"idem-ring-{idx}",
            correlation_id=f"corr-ring-{idx}",
            dealer_id="DEALER-01",
            identity=_clean_identity(
                app_id=f"APP-RING-{idx}",
                ssn_hash=f"ssn_ring_{idx}",
                phone="+14045558888",
                device_fp="dev_ring_shared",
                email=f"synth{idx}@example.com",
                address_hash="addr_drop_house",
                timestamp=1_700_000_000.0 + idx * 45.0,
                file_age=6,
                au_lines=4,
                primary_lines=0,
            ),
            income=IncomeVerificationInput(
                stated_annual_income=95_000.0,
                soc_code="11-1021",
                existing_monthly_debt=500.0,
                proposed_auto_payment=700.0,
            ),
        )
        decision = orchestrator.evaluate_application(req)

    assert decision.risk_tier == RiskTier.CRITICAL_SYNTHETIC
    assert decision.decision_action == DecisionAction.DECLINE_FRAUD
    assert "ID_01_THIN_FILE_AU_ABUSE" in decision.income_result.adverse_action_codes
    assert "ID_02_BIPARTITE_PII_COLLISION" in decision.income_result.adverse_action_codes


def test_audit_chain_verification_detects_multi_record_integrity() -> None:
    orchestrator = BorrowerCheckOrchestrator()
    for idx in range(1, 5):
        req = BorrowerCheckRequest(
            idempotency_key=f"idem-chain-{idx}",
            correlation_id=f"corr-chain-{idx}",
            dealer_id="DEALER-COX-ATL",
            identity=_clean_identity(
                app_id=f"APP-CHAIN-{idx}",
                ssn_hash=f"ssn_chain_{idx}",
                phone=f"+1404555020{idx}",
                email=f"user{idx}@example.com",
                device_fp=f"dev_{idx}",
            ),
            income=IncomeVerificationInput(
                stated_annual_income=90_000.0,
                verified_annual_income=90_000.0,
                soc_code="29-1141",
            ),
        )
        orchestrator.evaluate_application(req)

    assert len(orchestrator.audit_log) == 4
    assert orchestrator.verify_audit_chain() is True
    assert orchestrator.audit_log[1].previous_hash == orchestrator.audit_log[0].audit_hash
