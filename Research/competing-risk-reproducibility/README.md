# Personalized Competing-Risk Reproducibility Pack v1

## Status

This is a **reference reimplementation** of the documented personalized discrete-time competing-risk method. It is not a byte-for-byte reproduction of the historical v1/v2 runs.

The original experiment-specific training scripts were not present in the archived result ZIPs. The original repository contains the earlier product/baseline `train_model.py` and `evaluate_models.py`, but these scripts do not implement the multi-landmark multinomial competing-risk experiment. Therefore, this pack makes the methodology executable and explicit while preserving a clear distinction between reconstructed and historical results.

## Scope

- Synthetic Sales Simulator v0.4 only.
- Three landmarks: day 0, day 30, day 60.
- Outcomes: no event, Won, Lost.
- Right-censoring: only fully observed person-period intervals are included for censored observations.
- Intervals: 0–30, 30–60, ..., 330–360, 360–365 days.
- Forecast horizons: 90, 180, 365 days after the landmark.
- Model: scikit-learn multinomial logistic regression with one-hot interval indicators, numeric standardization and median/mode imputation.
- Test partitions: future opportunities and unseen companies.
- No restricted simulator truth is read by the model.

## Evaluation implemented in v1.1

The runner now estimates a censoring Kaplan–Meier curve on the training/dev partition and calculates cause-specific IPCW Brier scores at 90, 180 and 365 days. For each cause, observed Won/Lost events by the horizon receive inverse censoring-survival weights at the event time; observations event-free through the horizon receive weights at the horizon; observations censored before the horizon contribute zero.

The competing cause is a known non-case for the cause-specific target. Censoring weights are floored at 0.05 to avoid unbounded weights. The output includes evaluable counts, maximum/mean positive weights and estimated censoring survival at the horizon.

This is a transparent IPCW implementation and is not yet guaranteed to match the exact historical v2 estimator. Independent censoring remains an assumption; the weight floor and censoring model require sensitivity analysis.

## Run

First extract `ai_sales_synthetic_sales_simulator_v04.zip` to a directory, e.g.:

```text
data/
  seed_20261121/
  seed_20261122/
  seed_20261123/
```

Install the dependencies:

```bash
python -m pip install numpy pandas scikit-learn
```

Run all seeds and landmarks:

```bash
python scripts/run_reference_pipeline.py --data-dir ./data --output-dir ./outputs
```

Run one seed and one landmark:

```bash
python scripts/run_reference_pipeline.py --data-dir ./data --output-dir ./outputs --seed 20261121 --landmarks day_0
```

Outputs:
- `reference_metrics.csv`
- `run_metadata.json`
- `opportunity_predictions.csv` — held-out per-opportunity CIF predictions, outcomes and split metadata

## Execution results

- IPCW unit tests: 3/3 passed.
- Dataset structural/split validation: 30/30 checks passed across three seeds.
- Full reference run: all three seeds × all three landmarks × both test partitions × both causes × three horizons completed.
- Cumulative-incidence bounds are asserted at prediction time.

The full run writes raw per-seed metrics and a mean/standard-deviation summary under `ipcw_all_seeds_all_landmarks/`. It also saves held-out per-opportunity predictions in `opportunity_predictions.csv`; the latest full-run prediction audit is under `prediction_level_full_run/`.

The resulting scores are **not** the historical reported results and must not be mixed with the original v1/v2 tables. The model hyperparameters and some preprocessing choices were reconstructed because the original experiment scripts were not preserved.

## Known differences from historical experiments

1. Historical model hyperparameters and preprocessing details were not preserved in the result archives.
2. The exact historical censoring-KM implementation and IPCW weight handling are not recovered here.
3. The historical v2 model explicitly included industry; this reference runner includes industry as a categorical predictor.
4. The historical v3 ablation sampled person-period rows after expansion. This pack does not reproduce that ablation.
5. The reference implementation has not yet been validated against historical per-opportunity predictions because those predictions were not packaged.

## Reproducibility requirements before manuscript use

- Recover the original experiment scripts or label this reference implementation as a new, fully specified rerun.
- Freeze dependency versions.
- Increase bootstrap replicates and validate calibration layers on a separate training/dev calibration split.
- Preserve train/test group IDs and sample counts.
- Add bootstrap resampling at account and company levels.
- Verify cumulative incidence invariants:
  - Won CIF is non-decreasing over time.
  - Lost CIF is non-decreasing over time.
  - Won CIF + Lost CIF <= 1.
  - Event-free probability = 1 - Won CIF - Lost CIF.
- Re-run all seeds and landmarks after the evaluation implementation is validated.

## Safety and data governance

The source v0.4 archive contains `simulator_truth_RESTRICTED.csv`. The reference runner does not load it. Do not include restricted truth files in a public dataset release. Keep the original repository untouched; this pack is isolated under `/mnt/data`.


## Stage A2.4 — clustered uncertainty and calibration

The `cluster_calibration_200/` directory contains account- and company-cluster bootstrap intervals (200 resamples per seed-level cell), IPCW-weighted calibration intercept/slope, and industry-stratified reliability bins. See `STAGE_A2_4_CALIBRATION_REPORT.md` for interpretation and limitations. The bootstrap keeps predictions fixed; it does not capture model refitting uncertainty. Cross-seed averaged interval endpoints are not pooled confidence intervals.


## Stage A2.5 — training-only calibration

See `training_only_calibration_v1/STAGE_A2_5_CALIBRATION_REPORT.md`. The experiment uses account-group-held-out calibration data and evaluates sigmoid/isotonic mappings on future and unseen-company tests. Independent cause-specific calibrators can violate CIF joint probability coherence; do not deploy these outputs as-is.
