# AMAAM Project Instructions

This project implements the Adaptive Multi-Asset Allocation Model (AMAAM),
a quantitative asset allocation system. The full specification is in
AMAAM_SPECIFICATION.md. Treat it as the authoritative reference for all
implementation decisions, module designs, and parameter values.

## Non-negotiable rules:
- Never hard-code parameter values in logic code. All numerical parameters
  live in config/default_config.py and are imported from there.
- All public functions require NumPy-style docstrings and complete type hints.
- Comments explain WHY a decision was made, not WHAT the code does.
- No module should exceed ~400 lines. Split if needed.
- snake_case for functions and variables. PascalCase for classes.
  UPPER_SNAKE_CASE for constants.
- Line length maximum: 100 characters.
- Imports ordered: stdlib, then third-party, then local, with blank lines between groups.

## Testing:
- Every factor module gets a corresponding test file in tests/.
- Run tests after each phase before proceeding to the next.

## Data:
- data/raw/ and data/processed/ are gitignored. Do not commit data files.
- The spec's Section 4.3 defines all required data validation checks.