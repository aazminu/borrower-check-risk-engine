# Changelog — `borrower-check-risk-engine`

All notable engineering changes to `borrower-check-risk-engine` (`borrower_check`) are documented in this file.

## [0.3.0] — 2026-10-24
### Added
- Documented CLI diagnostic usage (`python -m borrower_check --self-test --json`) and deterministic invariant verification workflows in `README.md`.
- Added release history and semantic versioning audit trail in `CHANGELOG.md`.

## [0.2.1] — 2026-10-23
### Added
- Added automated CLI and runtime invariant regression suite in `tests/test_diagnostics_cli.py` covering seed determinism, negative-seed rejection, JSON output schema, and `--self-test` exit codes.

## [0.2.0] — 2026-10-21
### Added
- Added `src/borrower_check/diagnostics.py` (`run_diagnostic_suite`, `validate_runtime_invariants`) and module CLI runner `src/borrower_check/__main__.py`.
- Exported diagnostic helpers in `borrower_check.__all__`.

## [0.1.0] — 2026-10-19
### Added
- Initial architecture and core test suite: Real-time synthetic-identity bipartite graph scoring, BLS income discrepancy verification, and sliding-window velocity fraud engine (BorrowerCheck integration architecture).
