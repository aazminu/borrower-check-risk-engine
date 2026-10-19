"""Statistical BLS occupation wage benchmarking, DTI payment shock, and calibrated risk scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple

from borrower_check.identity_graph import IdentityGraphAnalysis


class RiskTier(str, Enum):
    """Calibrated borrower fraud and credit misrepresentation risk tiers."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL_SYNTHETIC = "CRITICAL_SYNTHETIC"


@dataclass(frozen=True)
class OccupationBenchmark:
    """Bureau of Labor Statistics (BLS) occupational wage distribution parameters."""

    soc_code: str
    title: str
    median_annual_income: float
    std_dev_annual_income: float
    p90_annual_income: float


DEFAULT_BLS_BENCHMARKS: Dict[str, OccupationBenchmark] = {
    "15-1252": OccupationBenchmark("15-1252", "Software Developer", 127260.0, 34000.0, 178500.0),
    "29-1141": OccupationBenchmark("29-1141", "Registered Nurse", 86070.0, 19500.0, 116400.0),
    "11-1021": OccupationBenchmark("11-1021", "General and Operations Manager", 101280.0, 38000.0, 162000.0),
    "41-2031": OccupationBenchmark("41-2031", "Retail Salesperson", 34700.0, 9200.0, 48500.0),
    "53-3032": OccupationBenchmark("53-3032", "Heavy and Tractor-Trailer Truck Driver", 54320.0, 13400.0, 74900.0),
    "35-3023": OccupationBenchmark("35-3023", "Fast Food and Counter Worker", 29540.0, 6100.0, 38200.0),
    "43-4051": OccupationBenchmark("43-4051", "Customer Service Representative", 41800.0, 9800.0, 56900.0),
}

REGIONAL_WAGE_MULTIPLIERS: Dict[str, float] = {
    "NATIONAL": 1.00,
    "ATL-GA": 1.03,
    "DFW-TX": 1.02,
    "CHI-IL": 1.06,
    "NYC-NY": 1.28,
    "SF-CA": 1.35,
    "MIDWEST-NONMETRO": 0.88,
}


@dataclass(frozen=True)
class IncomeVerificationInput:
    """Stated and verified financial attributes for an auto loan application."""

    stated_annual_income: float
    soc_code: str
    region_code: str = "NATIONAL"
    verified_annual_income: Optional[float] = None
    existing_monthly_debt: float = 0.0
    proposed_auto_payment: float = 0.0
    vehicle_msrp: float = 35000.0
    loan_amount: float = 30000.0


@dataclass(frozen=True)
class IncomeVerificationResult:
    """Output of BLS occupation Z-score verification, DTI analysis, and logistic calibration."""

    occupation_z_score: float
    income_discrepancy_ratio: float
    pre_loan_dti: float
    post_loan_dti: float
    payment_shock_ratio: float
    ltv_ratio: float
    income_inflation_flag: bool
    log_odds_score: float
    calibrated_probability: float
    risk_tier: RiskTier
    adverse_action_codes: Tuple[str, ...] = field(default_factory=tuple)


class IncomeRiskVerifier:
    """Evaluates stated income plausibility against BLS distributions and combines with identity signals."""

    def __init__(
        self,
        benchmarks: Optional[Dict[str, OccupationBenchmark]] = None,
        region_multipliers: Optional[Dict[str, float]] = None,
    ) -> None:
        self._benchmarks = dict(DEFAULT_BLS_BENCHMARKS)
        if benchmarks:
            self._benchmarks.update(benchmarks)
        self._region_multipliers = dict(REGIONAL_WAGE_MULTIPLIERS)
        if region_multipliers:
            self._region_multipliers.update(region_multipliers)

    def evaluate(
        self,
        income_input: IncomeVerificationInput,
        identity_analysis: Optional[IdentityGraphAnalysis] = None,
    ) -> IncomeVerificationResult:
        """Compute statistical income anomaly metrics and calibrated joint fraud probability."""
        if income_input.stated_annual_income <= 0:
            raise ValueError("stated_annual_income must be positive")
        if income_input.vehicle_msrp <= 0:
            raise ValueError("vehicle_msrp must be positive")

        benchmark = self._benchmarks.get(
            income_input.soc_code,
            OccupationBenchmark(income_input.soc_code, "Unclassified Occupation", 58000.0, 18000.0, 86000.0),
        )
        multiplier = self._region_multipliers.get(income_input.region_code, 1.00)

        regional_median = benchmark.median_annual_income * multiplier
        regional_std = benchmark.std_dev_annual_income * multiplier

        z_score = (income_input.stated_annual_income - regional_median) / regional_std

        # Compare stated income against verified payroll/tax transcript income if present
        if income_input.verified_annual_income is not None and income_input.verified_annual_income > 0:
            discrepancy_ratio = (
                income_input.stated_annual_income - income_input.verified_annual_income
            ) / income_input.verified_annual_income
            effective_annual_income = min(
                income_input.stated_annual_income, income_input.verified_annual_income
            )
        else:
            discrepancy_ratio = 0.0
            effective_annual_income = income_input.stated_annual_income

        monthly_income = effective_annual_income / 12.0
        pre_loan_dti = income_input.existing_monthly_debt / monthly_income
        post_loan_dti = (
            income_input.existing_monthly_debt + income_input.proposed_auto_payment
        ) / monthly_income

        baseline_debt = max(income_input.existing_monthly_debt, 250.0)
        payment_shock_ratio = (
            income_input.existing_monthly_debt + income_input.proposed_auto_payment
        ) / baseline_debt
        ltv_ratio = income_input.loan_amount / income_input.vehicle_msrp

        income_inflation_flag = (z_score >= 2.5) or (discrepancy_ratio >= 0.25)

        adverse_codes: List[str] = []
        if z_score >= 2.5:
            adverse_codes.append("INC_01_OCCUPATION_ZSCORE_OUTLIER")
        if discrepancy_ratio >= 0.25:
            adverse_codes.append("INC_02_VERIFIED_PAYROLL_MISMATCH")
        if post_loan_dti > 0.50:
            adverse_codes.append("DTI_01_EXCESSIVE_POST_LOAN_OBLIGATION")
        if ltv_ratio > 1.25:
            adverse_codes.append("COL_01_EXCESSIVE_VEHICLE_LTV")

        # Calibrated logistic regression log-odds combining income & bipartite identity graph
        log_odds = -2.65
        if z_score > 1.5:
            log_odds += 0.65 * (z_score - 1.5)
        if discrepancy_ratio > 0.10:
            log_odds += 3.20 * discrepancy_ratio
        if post_loan_dti > 0.40:
            log_odds += 2.80 * (post_loan_dti - 0.40)
        if ltv_ratio > 1.10:
            log_odds += 1.60 * (ltv_ratio - 1.10)

        synthetic_score = 0.0
        if identity_analysis is not None:
            synthetic_score = identity_analysis.synthetic_cluster_score
            log_odds += 4.25 * synthetic_score
            if identity_analysis.thin_file_au_abuse_flag:
                log_odds += 0.85
                adverse_codes.append("ID_01_THIN_FILE_AU_ABUSE")
            if identity_analysis.shared_ssn_identities > 1 or identity_analysis.shared_phone_ssns > 1:
                adverse_codes.append("ID_02_BIPARTITE_PII_COLLISION")
            if identity_analysis.velocity_1h_count >= 3:
                log_odds += 0.75
                adverse_codes.append("VEL_01_RAPID_APPLICATION_BURST")

        calibrated_prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, log_odds))))

        if synthetic_score >= 0.60 or (calibrated_prob >= 0.80 and synthetic_score >= 0.35):
            tier = RiskTier.CRITICAL_SYNTHETIC
        elif calibrated_prob >= 0.65 or income_inflation_flag and post_loan_dti > 0.48:
            tier = RiskTier.HIGH
        elif calibrated_prob >= 0.30 or income_inflation_flag or post_loan_dti > 0.43:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.LOW

        return IncomeVerificationResult(
            occupation_z_score=round(z_score, 4),
            income_discrepancy_ratio=round(discrepancy_ratio, 4),
            pre_loan_dti=round(pre_loan_dti, 4),
            post_loan_dti=round(post_loan_dti, 4),
            payment_shock_ratio=round(payment_shock_ratio, 4),
            ltv_ratio=round(ltv_ratio, 4),
            income_inflation_flag=income_inflation_flag,
            log_odds_score=round(log_odds, 4),
            calibrated_probability=round(calibrated_prob, 4),
            risk_tier=tier,
            adverse_action_codes=tuple(adverse_codes),
        )
