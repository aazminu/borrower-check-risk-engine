"""Deterministic runtime diagnostics and invariant verification for `borrower_check`."""

from __future__ import annotations

import time
from typing import Any

from .identity_graph import ApplicantIdentity, BipartiteIdentityGraph

def _domain_probe(seed: int) -> dict:
    graph = BipartiteIdentityGraph()
    ident = ApplicantIdentity(
        application_id="APP-1001", ssn_hash="ssn_clean_001", phone="+14045550101",
        email="alex.mercer@example.com", address_hash="addr_dunwoody_01",
        device_fingerprint="dev_ios_991", timestamp=1700000000.0,
        credit_file_age_months=72, authorized_user_tradelines=0, primary_tradelines=5
    )
    analysis = graph.ingest_and_evaluate(ident)
    return {
        "shared_ssn_identities": analysis.shared_ssn_identities,
        "bipartite_entropy_score": round(analysis.bipartite_entropy_score, 4),
        "synthetic_cluster_score": round(analysis.synthetic_cluster_score, 4),
        "invariants_ok": (analysis.shared_ssn_identities == 1 and analysis.bipartite_entropy_score == 0.0 and not analysis.thin_file_au_abuse_flag),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `borrower-check-risk-engine` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "borrower-check-risk-engine",
        "package": "borrower_check",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `borrower_check`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "borrower-check-risk-engine",
        "package": "borrower_check",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
