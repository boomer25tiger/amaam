"""
ATR Trend/Breakout System factor (T) for AMAAM.

Implements a daily ATR-based breakout signal that assigns T = +2 (uptrend) when
today's high exceeds the upper band, T = -2 (downtrend) when today's low falls
below the lower band, and retains the prior value otherwise. The signal captures
directional bias and enters TRank as a raw value (not ranked). See Section 3.5
of the specification for band definitions and sign-convention resolution notes.
"""
