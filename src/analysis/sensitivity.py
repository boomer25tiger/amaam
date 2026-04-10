"""
Sensitivity analysis for AMAAM.

Runs systematic parameter sweeps to characterize the robustness of the model:
(1) factor weight sensitivity (wM from 0.20 to 0.60 in 0.05 steps, with wV and
wC absorbing the remainder equally), (2) selection count sensitivity (top 4–7
from the main sleeve), (3) weighting scheme comparison (equal vs inverse-vol),
and (4) rebalancing frequency comparison (monthly vs bi-weekly). All analysis
is performed on the development period only (through December 2017). See
Section 6 of the specification.
"""
