# Variable dictionary

| Variable | Definition | Source and construction | Missingness rule | Unit |
| --- | --- | --- | --- | --- |
| race_id | Grand Prix key | Season plus zero-padded round | Never missing in analysis sample | String |
| q2_sec | Best recorded Q2 lap time | Parse Jolpica mm:ss.sss to seconds | Missing means no recorded Q2 time | Seconds |
| q2_rank | Rank within recorded Q2 times | Sort q2_sec; official position and driver ID break residual ordering ambiguity | Missing outside Q2 | Integer |
| treated_q3 | Focal Q3 advancement indicator | 1(q2_rank <= 10) in included standard-format weekends | Undefined in excluded cancelled/nonstandard sessions | Binary |
| score_gap_sec | Rank-11 minus rank-10 Q2 lap time | q2_sec(11) - q2_sec(10) | Weekend excluded if nonpositive or rank 11 unavailable | Seconds |
| race_points | Grand Prix championship points | Jolpica result points | Required for all classified Q2 drivers | Points |
| top_ten | Official classified position 1-10 | 1(finish_position <= 10) | Missing without result record | Binary |
| grid_position | Official starting-grid position | Jolpica grid field | Missing without result record | Position |
| prior_season_starts | Counted starts in season t-1 | Excludes did-not-start, did-not-qualify, did-not-prequalify, and withdrawn statuses | Zero if no counted start | Count |
| has_prior_history | At least one start in season t-1 | 1(prior_season_starts > 0) | Never missing | Binary |
| no_prior_history | No start in season t-1 | 1 - has_prior_history | Never missing | Binary |
| prior_season_points | Total points in season t-1 | Sum Jolpica race points by driver-season | Missing in main analysis if has_prior_history=0 | Points |
| prior_points_zero | Exposure-inclusive sensitivity | Fill prior_season_points with zero only for labeled sensitivity | Never missing | Points |
| prior_points_per_start | Prior points divided by counted starts | prior_season_points / prior_season_starts | Missing, not zero, when starts=0 | Points per start |
| complete_prior_pair | Both focal drivers have prior-season starts | Product of rank-10 and rank-11 history indicators | Never missing | Binary |
| q1_advantage_sec | Same-weekend Q1 advantage of rank 10 | q1_sec(11) - q1_sec(10) | Missing if either Q1 time unavailable | Seconds |
