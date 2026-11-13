# AI Sales Forecasting — Research Branch Package

This archive is intended to be added to a new branch of the existing
`alirida8852/AI_Sales_Forecasting` repository.

## Suggested branch

`aatif-work`

## Contents

The package is placed under:

`aatif-work/`

It contains the reference competing-risk pipeline, validation scripts, prediction-level outputs,
IPCW evaluation, cluster-bootstrap and calibration analyses, reliability diagrams, reports,
configuration, manifest and pinned Python dependencies.

## Add to a branch

1. Clone the existing repository and create a new branch.
2. Extract this archive at the repository root.
3. Review the added `research/` directory.
4. Commit and push only after reviewing the diff and confirming that no unrelated application files changed.

Example:

```bash
git checkout -b aatif-work
# Extract this archive into the repository root.
git status
git add aatif-work
git commit -m "Add reproducible competing-risk research pipeline"
git push -u origin aatif-work
```

## Important scope notes

- This is an additive research package, **not a complete replacement repository**.
- It does not include the original frontend/backend application source.
- It does not include the raw synthetic simulator dataset.
- It does not include restricted simulator truth files.
- It does include derived held-out predictions and experimental CSV outputs.
- The reference implementation is not an exact reproduction of missing historical experiment scripts.
- Current calibration experiments are exploratory; independently calibrated Won/Lost CIFs can violate joint probability coherence.
- No production champion is selected, and no real CRM performance claim is made.

See the package README and `REPRODUCIBILITY_MANIFEST.json` before using results in a manuscript.
