# Identification audit

## Estimand

The observed object is the equally weighted mean of the rank-10 minus rank-11
outcome difference. It compares two drivers at adjacent realized ranks and is a
finite-rank assignment contrast.

## Assignment mechanism

In included standard-format weekends, ten drivers advance from Q2 to Q3.
Treatment is deterministic in Q2 rank. Both rank and the underlying Q2
lap-time distance are observed.

## Weakest indispensable causal assumptions

An own-treatment interpretation requires adjacent-rank comparability for
untreated potential outcomes and a no-interference representation even though
advancing one driver changes the composition of the fixed-capacity Q3 field.
The institutional rule does not supply either condition.

## Diagnostics

- Original-score gap distribution and ECDF.
- Q1 and prior-season-history contrasts.
- Missing-history balance and complete-pair sensitivity.
- Continuous log2-gap associations.
- Q3 boundary versus adjacent rank gradients 8-13.
- Four sampling-dependence calculations, season sign enumeration, extreme-gap
  deletion, and leave-one-season-out checks.

## Remaining limitations

- Only ten season clusters are available.
- Sign symmetry is not institutionally guaranteed.
- The score gap is not randomly assigned.
- Race outcomes may depend on all Q3 participants.
- More races improve precision for the finite-rank contrast but do not force
  the adjacent-rank baseline gap to zero.
