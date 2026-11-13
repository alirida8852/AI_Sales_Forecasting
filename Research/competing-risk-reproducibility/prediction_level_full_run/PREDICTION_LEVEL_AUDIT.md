# Prediction-level audit

## Output

- File: `opportunity_predictions.csv`
- Rows: 89,655
- Unique prediction keys: seed + opportunity_id + landmark + horizon_days
- Unique keys verified: True
- Test partitions: {'unseen_company_test': 45555, 'within_company_future_test': 44100}
- Landmarks: {'day_0': 32025, 'day_30': 31356, 'day_60': 26274}
- Horizons: {90: 29885, 180: 29885, 365: 29885}
- Outcome labels retained: {'lost': 41436, 'won': 36465, 'open': 11754}

Each record retains opportunity, company, account, split, landmark timestamp, observed outcome/close/censoring metadata, prediction partition, horizon, Won CIF, Lost CIF and event-free probability.

## Integrity checks

- Prediction keys are unique.
- CIFs are non-negative and each cause-specific CIF is bounded by 1.
- Sum of Won and Lost CIF never exceeds 1 (maximum observed sum: 0.999999999293).
- Won CIF is non-decreasing across 90, 180 and 365 days for every opportunity-landmark.
- Lost CIF is non-decreasing across 90, 180 and 365 days for every opportunity-landmark.
- No duplicate prediction keys.

## Interpretation

The prediction file contains held-out future and unseen-company predictions only; training predictions are not included. This is intentional to keep evaluation records distinct from model fitting data. The file is suitable for downstream calibration plots, error analysis and clustered bootstrap evaluation.

These predictions come from the reference reimplementation, not the historical v2 experiment.
