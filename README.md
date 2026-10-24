# BorrowerCheck Risk Engine

Real-time auto-lending synthetic identity graph, statistical income verification, and fault-tolerant partner decision orchestration service modeled on fintech fraud-prevention integrations (`BorrowerCheck`) in retail automotive finance.

## Overview

Automotive F&I (Finance & Insurance) checkout pipelines face two primary classes of application fraud that traditional univariate credit bureau pulls frequently miss:
1. **Synthetic Identity Fraud Rings**: Fabricated or "Frankenstein" credit profiles that pair a newly minted or stolen SSN/TIN with recycled burner phones, drop-house physical addresses, shared device fingerprints, and thin-file Authorized User (AU) tradeline piggybacking.
2. **Stated Income Misrepresentation**: Inflated stated salaries designed to suppress Debt-to-Income (DTI) ratios on high-LTV vehicle originations when real-time payroll bureau APIs experience latency spikes or partial coverage.

`borrower-check-risk-engine` provides a zero-dependency, deterministic Python reference implementation of a partner risk orchestration service combining a **bipartite PII linkage graph**, **BLS occupation/region wage Z-score anomaly detection**, **idempotent decision caching**, **circuit-breaker degraded fallback**, and **hash-chained audit telemetry**.

---

## Architecture

```
                         +-----------------------------------------+
                         |     Partner Dealer Checkout Gateway     |
                         |  (Idempotency Key + Correlation ID)     |
                         +--------------------+--------------------+
                                              |
                                              v
                         +-----------------------------------------+
                         |       BorrowerCheckOrchestrator         |
                         |  - SHA-256 Payload Fingerprint Check    |
                         |  - Hash-Chained Audit Log Emitter       |
                         +--------+-----------------------+--------+
                                  |                       |
              +-------------------+                       +-------------------+
              v                                                               v
+-----------------------------------+                           +-----------------------------------+
|      BipartiteIdentityGraph       |                           |        IncomeRiskVerifier         |
|  - SSN <-> (Phone, Email, Addr,   |                           |  - BLS SOC x Metro Wage Quantiles |
|    Device) Bipartite Graph        |                           |  - Circuit-Breaker Payroll Lookup |
|  - Neighborhood Shannon Entropy   |                           |  - Pre/Post-Loan DTI & LTV Shock  |
|  - 1h / 24h Sliding Window Bursts |                           |  - Calibrated Logistic Log-Odds   |
|  - Thin-File AU Piggybacking Flag |                           |  - FCRA Adverse Action Reason Map |
+-----------------------------------+                           +-----------------------------------+
```

### 1. Synthetic Identity Bipartite Graph (`borrower_check.identity_graph`)
* **Bipartite PII Topology**: Tracks edges between SSN hashes ($U$) and contact/device attributes ($V = \{\text{phone}, \text{email}, \text{address}, \text{device\_fp}\}$).
* **Shannon Degree Entropy**: Measures structural fragmentation across an applicant's local bipartite neighborhood:
  $$H = -\sum_{i} p_i \log_2(p_i), \quad p_i = \frac{\deg(v_i)}{\sum_j \deg(v_j)}$$
  Legitimate borrowers maintain a 1-to-1 PII mapping ($H = 0$), whereas synthetic identity rings sharing phones or devices across multiple SSNs produce elevated entropy and multi-hop connected components.
* **Authorized User (AU) Tradeline Piggybacking**: Flags thin credit files ($\text{age} < 12\text{ months}$) where $\ge 3$ authorized user tradelines mask the absence of primary revolving or installment history.
* **Sliding-Window Velocity**: Evaluates 1-hour ($3600\text{ s}$) and 24-hour ($86400\text{ s}$) application bursts per SSN, phone, address, and device fingerprint.

### 2. BLS Occupation Wage Benchmarking (`borrower_check.income_verifier`)
* **Regionally Adjusted Z-Score**: Evaluates stated annual income $Y_{\text{stated}}$ against Bureau of Labor Statistics (BLS) Standard Occupational Classification (SOC) medians $\mu_{\text{SOC}}$ and standard deviations $\sigma_{\text{SOC}}$ scaled by metro cost-of-living multipliers $m_{\text{region}}$:
  $$Z = \frac{Y_{\text{stated}} - m_{\text{region}} \mu_{\text{SOC}}}{m_{\text{region}} \sigma_{\text{SOC}}}$$
* **Calibrated Logistic Risk Classifier**: Combines bipartite synthetic cluster scores, occupation $Z$-scores, payroll verification discrepancies, post-loan DTI, and vehicle LTV into a calibrated probability $P(\text{fraud}) = (1 + e^{-\eta})^{-1}$ mapped to four operational risk tiers:
  * `LOW` $\rightarrow$ `AUTO_APPROVE`
  * `MEDIUM` $\rightarrow$ `STIPULATION_REQUIRED` (automated paystub/W-2 upload request)
  * `HIGH` $\rightarrow$ `MANUAL_REVIEW`
  * `CRITICAL_SYNTHETIC` $\rightarrow$ `DECLINE_FRAUD`

### 3. Resilient Partner Orchestration (`borrower_check.orchestrator`)
* **Circuit-Breaker Degradation**: Wraps external payroll/tax-transcript enrichment calls in a `CLOSED -> OPEN -> HALF_OPEN` state machine. When an upstream bureau times out, the orchestrator degrades gracefully to local statistical + graph heuristics and routes otherwise clean applications to `STIPULATION_REQUIRED` rather than failing dealer checkout requests.
* **Strict Idempotency**: Replaying a request with an existing `idempotency_key` returns the exact cached decision (`cached_replay=True`), while mutating payload attributes under an existing key raises `IdempotencyConflictError`.
* **Cryptographic Audit Chain**: Every decision appends an `AuditLogEntry` chained via SHA-256 (`previous_hash || sequence_id || correlation_id || decision`), verifiable via `verify_audit_chain()`.

---

## Quickstart

```bash
pip install -e ".[dev]"
pytest -v
```

### Example Usage

```python
from borrower_check import (
    ApplicantIdentity,
    BorrowerCheckOrchestrator,
    BorrowerCheckRequest,
    IncomeVerificationInput,
)

orchestrator = BorrowerCheckOrchestrator(
    external_income_provider=lambda ssn_hash: 132_000.0
)

request = BorrowerCheckRequest(
    idempotency_key="req-2026-0001",
    correlation_id="trace-atl-98412",
    dealer_id="DEALER-ATL-014",
    identity=ApplicantIdentity(
        application_id="APP-9001",
        ssn_hash="e3b0c44298fc1c149afbf4c8996fb924",
        phone="+14045550142",
        email="borrower@example.com",
        address_hash="addr_dunwoody_30346",
        device_fingerprint="dev_safari_macos_18",
        timestamp=1_710_000_000.0,
    ),
    income=IncomeVerificationInput(
        stated_annual_income=132_000.0,
        soc_code="15-1252",  # Software Developer
        region_code="ATL-GA",
        existing_monthly_debt=950.0,
        proposed_auto_payment=610.0,
        vehicle_msrp=39_500.0,
        loan_amount=33_000.0,
    ),
)

decision = orchestrator.evaluate_application(request)
print(decision.decision_action, decision.risk_tier, decision.audit_hash)
```

## Author

**Abdullah Azmi** — [abdullahazmi.com](https://abdullahazmi.com) · [@aazminu](https://github.com/aazminu)


## CLI Diagnostics & Invariant Verification

Run the built-in deterministic diagnostic probe and invariant self-check directly via the module CLI:

```bash
PYTHONPATH=src python -m borrower_check --self-test --json
```

Run the complete regression test suite:

```bash
PYTHONPATH=src pytest -q
```
