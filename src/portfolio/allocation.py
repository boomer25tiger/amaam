"""
Portfolio allocation logic for AMAAM.

Implements the two-sleeve allocation pipeline: applies the momentum filter to
main sleeve selections, redirects weight from negative-momentum main-sleeve
assets to the hedging sleeve, applies the hedging sleeve's own momentum filter
(replacing failures with SHY), and assembles the final portfolio weights.
Handles all edge cases: full main-sleeve momentum failure (100% to hedging),
full hedging failure (100% to SHY), and tie-splitting. See Sections 3.7 and
9.11 of the specification.
"""
