# Stage A2.5 — Training-only calibration experiment

## Design

For each seed and landmark, the original within-company train/dev partition was split by account group into:
- model-fit subset (approximately 80%);
- calibration subset (approximately 20%).

No account group appears in both subsets. The model was refit on the fit subset, and the calibration mappings were learned only from predictions on the held-out calibration subset. Future-test and unseen-company predictions were generated from that same fit-only model. Test outcomes were used only for evaluation.

The censoring Kaplan–Meier estimator was also fitted on the model-fit opportunities only. Calibration and test scoring use the same IPCW cause-specific target construction and 0.05 censor-survival floor used in the reference pipeline.

Compared:
- no calibration;
- sigmoid calibration (weighted logistic regression on the logit of predicted CIF);
- isotonic calibration (weighted isotonic regression).

Separate mappings were fitted for Won/Lost, landmark and horizon.

## Results

Across 54 seed-level cells per test partition (3 seeds × 3 landmarks × 3 horizons × 2 causes):

| Test partition | Calibration | Mean change in IPCW Brier vs uncalibrated | Cells improved |
|---|---|---:|---:|
| Future test | Sigmoid | -0.00068 | 37/54 |
| Future test | Isotonic | +0.00049 | 18/54 |
| Unseen-company test | Sigmoid | -0.00054 | 37/54 |
| Unseen-company test | Isotonic | +0.00076 | 18/54 |

Negative change is an improvement. Sigmoid calibration produces a small, fairly consistent average improvement on both test partitions. Isotonic calibration does not generalize reliably in this experiment.

## Critical joint-probability caveat

Won and Lost CIFs were calibrated independently. Their sum can therefore exceed 1, violating the competing-risk probability simplex:
- Uncalibrated: 0% of evaluated rows exceeded 1.
- Sigmoid: approximately 1.2% across both test partitions.
- Isotonic: approximately 5.3–5.5%.

Consequently, the independently calibrated CIFs are useful as a marginal calibration experiment, but **must not be treated as a valid final competing-risk probability output**. A future calibration method must preserve non-negativity, monotonicity over horizons, and `CIF_Won + CIF_Lost <= 1` jointly. A simple renormalization may repair the sum but can undo marginal calibration; it requires separate validation.

## Reliability diagrams

Twelve reliability diagrams are generated (two test partitions × three horizons × two causes). Curves pool the three seeds for visualization only; the numerical tables preserve seed-level results. IPCW weights are used for observed incidence in each bin.

## Files

- `calibration_test_metrics.csv`: per-seed/test/landmark/horizon/cause/method metrics.
- `calibration_test_summary.csv`: descriptive mean and standard deviation across three seeds.
- `calibration_test_predictions.csv`: row-level predictions, targets and IPCW weights for reliability plots and audit.
- `calibration_group_splits.csv`: fit/calibration opportunity and account-group counts; overlap must be zero.
- `calibrator_fit_diagnostics.csv`: calibrator fit status and calibration-sample diagnostics.
- `reliability_plots/`: 12 PNG reliability diagrams.
- `STAGE_A2_5_CALIBRATION_REPORT.md`: this report.

## Limitations

- Only three synthetic seeds; results are preliminary.
- Calibration subset is account-group held out but not a separate company-level holdout.
- The calibration subset is sampled from the same train/dev population; it does not test distribution shift.
- Sigmoid improvement is modest and no production champion is selected.
- Independent cause-specific calibration breaks joint CIF coherence in a minority of rows.
- No bootstrap confidence intervals for calibration deltas yet.
- This is a reference reimplementation, not exact reproduction of historical experiments.

## Decision

Retain the uncalibrated reference model as the research baseline. Sigmoid calibration is a promising marginal calibration candidate, not a deployable output layer. Next, investigate a joint competing-risk calibration method that preserves the CIF simplex and test industry-stratified calibration under the same group-held-out design.
