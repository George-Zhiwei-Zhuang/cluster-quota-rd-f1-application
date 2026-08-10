# Formula One Q3 Fixed-Quota Diagnostic Illustration

This package rebuilds the Formula One application in Section 11 of
`Cluster_Based_RDD_Rewritten(20260808-123526).docx` from frozen public API
responses. It implements explicit race-audit rules, missing-aware prior-season
history variables, four sampling-based uncertainty calculations, adjacent-rank
diagnostics, and all main and supplementary figures.

## Research question and estimand

During the standard knockout format, ten drivers advance from Q2 to Q3. Each
race weekend is a bounded cluster. The focal observed estimand is the equally
weighted cross-race mean of the rank-10 minus rank-11 difference:

```text
Delta_Y = mean_g [Y_(10):g - Y_(11):g].
```

This is a finite-rank assignment contrast, not a conventional continuous-score
RD estimand and not, without additional comparability and no-interference
assumptions, an own-treatment causal effect.

## Data and frozen snapshot

- Source: [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the maintained
  Ergast-compatible public API. The qualifying endpoint and response fields
  follow its [qualifying documentation](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/qualifying.md).
- Retrieval date: recorded for every page in `data/raw/jolpica/manifest.json`.
- Coverage: qualifying results for 2010-2019 and race results for 2009-2019.
- Raw pages: 100 JSON responses retained with SHA-256 checksums.
- Audited weekends: 198.
- Rule-based analysis sample: 193 weekends and 3,010 classified Q2
  driver-race observations.

The raw archive is included, so the default run is offline and does not depend
on a future API revision. Set `REFRESH_RAW=1` only for an intentional refresh.

## Race inclusion rules

An included weekend must have:

1. between 12 and 17 recorded Q2 times;
2. distinct positive Q2 times at ranks 10 and 11;
3. race-result records for all classified Q2 drivers;
4. a realized ten-place Q3 advancement rule under the standard format.

Five weekends are excluded. Their identities and reasons are generated as
`output/tables/table_a1_excluded_races.*`. Manual institutional exclusions are
declared in `config/manual_exclusions.csv`; all other exclusions are generated
from the raw records.

## Prior-season history and rookie treatment

The rebuild does not assign a performance rate of zero to a driver with no
prior-season starts.

- `has_prior_history = 1` if the driver had at least one counted start in the
  preceding season.
- Prior-season total points are filled with zero only in an explicitly named
  exposure-inclusive sensitivity.
- Prior-season points and points per start use a complete focal pair in the
  main predetermined-performance diagnostics.
- A separate no-prior-history rank-10 minus rank-11 contrast is reported.
- Neighboring-rank history diagnostics use a common complete-history race
  sample across every rank entering the contrast.

There are 130 complete-history focal pairs; 63 of 193 pairs have at least one
driver without a prior-season start.

## Uncertainty calculations

The package reports:

1. race-i.i.d. intervals;
2. season-clustered intervals using a t(9) reference distribution;
3. 20,000 complete-season block-bootstrap replications;
4. race-by-driver two-way cluster-robust intervals after absorbing race fixed
   effects.

It also enumerates all 2^10 season sign patterns for the race-point boundary
excess. These are sampling-based sensitivities, not assignment-randomization
procedures, because Q3 assignment is deterministic in Q2 rank.

## Reproduction

Python 3.10 or newer is recommended. Install
`environment/requirements.txt`, then run:

```bash
bash run_all.sh
```

To choose a specific interpreter:

```bash
PYTHON_BIN=/path/to/python bash run_all.sh
```

To refresh the public API pages intentionally:

```bash
REFRESH_RAW=1 bash run_all.sh
```

With frozen raw inputs, expected runtime is under one minute. A live refresh is
slower because requests respect the public API rate limit.

## Main rebuilt results

- Race-points contrast: 0.679.
- Top-ten-finish contrast: 0.067.
- Starting-grid contrast: -2.461 positions.
- Complete-pair prior-season-points contrast: 1.073.
- Complete-pair prior-season points-per-start contrast: 0.071.
- Score gaps range from 0.002 to 1.564 seconds.

The score-gap and neighboring-rank lessons survive. The stronger draft claim
about predetermined performance imbalance is sensitive to zero coding and
should be revised. See `docs/deviations.md` and
`output/tables/table_a8_draft_vs_rebuild.*`. Exact paragraph-level numerical
replacements for Section 11 are collected in
`docs/manuscript-revision-notes.md`.

## Output map

Main tables:

- `output/tables/table_2_primary_contrasts.*`
- `output/tables/table_3_continuous_gap_diagnostics.*`
- `output/tables/table_4_boundary_vs_neighboring.*`

Main figures:

- `output/figures/figure_5_score_gap_distribution.*`
- `output/figures/figure_6_gap_quartile_diagnostics.*`
- `output/figures/figure_7_quota_vs_neighboring_gradients.*`
- `output/figures/figure_8_dependence_and_extreme_gap_sensitivity.*`

Rookie/history and other appendices:

- `output/figures/figure_a1_rookie_history_sensitivity.*`
- `output/figures/figure_a2_leave_one_season_out.*`
- `output/tables/table_a1_...` through `table_a8_...`

Validation logs, software versions, and output checksums are in `output/logs/`.
The complete package-level checksum list is `MANIFEST.sha256`.

## Dependency map

```text
code/00_download  -> frozen raw JSON
code/01_clean     -> flat qualifying, race-result, and history files
code/02_construct -> audited race sample and rank panels
code/03_analyze   -> estimates and diagnostics
code/04_tables    -> publication and appendix tables
code/05_figures   -> PNG and PDF figures
code/06_diagnostics -> integrity and reproducibility checks
```

## Interpretation limits

The application is diagnostic. More race weekends improve precision for the
finite-rank contrast but do not eliminate the adjacent-rank baseline gap.
Advancing one driver changes the composition of a fixed-capacity Q3 field, so
no-interference is demanding. The package does not convert the Formula One
comparison into a causal RD design.

## License

The replication code is released under the MIT License. The Formula One data
remain subject to the terms and provenance of their original provider.
