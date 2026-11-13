# Stage A2.4 — Clustered uncertainty and calibration report

## Scope and execution

- 3 synthetic seeds × 3 landmarks × 2 test partitions × 3 horizons × 2 causes.
- 108 seed-level metric cells.
- 200 bootstrap replicates per cell for account-level and company-level resampling.
- Calibration intercept/slope estimated with IPCW-weighted logistic calibration regression.
- Industry-stratified calibration and reliability bins produced for Enterprise Software, Logistics and SaaS.

## Aggregate future-test calibration observations

Across seed-level cells and both causes:

| Horizon | Mean calibration intercept | Mean calibration slope | Mean account-bootstrap CI width | Mean company-bootstrap CI width |
|---:|---:|---:|---:|---:|
| 90 days | 0.446 | 1.494 | 0.0195 | 0.0327 |
| 180 days | -0.037 | 0.950 | 0.0145 | 0.0168 |
| 365 days | 0.129 | 0.904 | 0.0142 | 0.0153 |

Ideal calibration intercept is 0 and slope is 1. These averages indicate the 90-day predictions need particular calibration attention. A slope above 1 indicates predictions are too compressed relative to observed outcomes under the fitted calibration regression; it does not by itself establish a simple uniform under- or overprediction direction. At 180 and 365 days, average slopes are closer to 1, though subgroup variation remains.

## Industry heterogeneity

Future-test industry averages show different calibration patterns. Examples:
- At 90 days, Won calibration slopes averaged approximately 1.60 for Enterprise Software, 1.30 for Logistics and 1.12 for SaaS.
- At 90 days, Lost slopes averaged approximately 1.10 for Enterprise Software, 0.78 for Logistics and 1.26 for SaaS.
- At 180/365 days, slopes move closer to 1 in several groups, but do not do so uniformly.

These are descriptive averages across seed-level fits, not evidence of statistically significant industry differences. Industry subgroup metrics should be accompanied by the per-seed counts and uncertainty intervals.

## Bootstrap interpretation

- Account bootstrap resamples account clusters with replacement within each seed/test cell.
- Company bootstrap resamples company clusters with replacement within each seed/test cell.
- Confidence intervals are percentile intervals from 200 resamples.
- The reported cross-seed summary averages seed-level interval endpoints. It is **not** a pooled three-seed confidence interval.
- Unseen-company test sets contain only 18 held-out companies per seed, so company-bootstrap intervals are coarse and should be treated as a sensitivity analysis.
- Bootstrap is performed on fixed predictions; it does not refit the model inside each resample. These intervals quantify test-sample cluster uncertainty, not full training-procedure uncertainty.

## Files

- `cluster_bootstrap_calibration.csv`: 108 seed-level cells with account/company bootstrap intervals and calibration coefficients.
- `cluster_bootstrap_calibration_summary.csv`: mean/std and averaged seed-level interval endpoints.
- `industry_calibration_metrics.csv`: 324 industry-stratified metric/calibration cells.
- `industry_reliability_bins.csv`: 3,240 IPCW-weighted reliability-bin records.
- `analysis_metadata.json`: replicate count and methodological caveats.

## Decision

Do not select a production model. The current evidence supports further calibration work, especially at 90 days and within industry subgroups. Next research actions:
1. Plot reliability curves from the saved bins.
2. Add bootstrap uncertainty for calibration intercept/slope.
3. Increase bootstrap replicates to at least 1,000 for final manuscript tables.
4. Increase the number of independent simulator seeds and test new company populations.
5. Consider a training-only calibration layer, validated without touching test data.
