# Frequency-residual metastasis classifier

This repository is the reproducibility package for the final 10.96M
specificity-constrained frequency-residual classifier.

## Final model
- Code: `src/model/train_final_model.py`
- Checkpoint: `checkpoints/final_model/best_validation_constraint.pt`
- Config: `configs/final_model/run_config.json`
- History: `configs/final_model/training_history.json`
- Summary: `configs/final_model/summary.json`

## Reported analyses
- PCam bootstrap: `analysis/statistics/bootstrap_confidence_intervals.py`
- Paired comparator statistics: `analysis/statistics/paired_comparator_statistics.py`
- Comparator benchmark: `baselines/run_baseline_benchmarks.py`
- WILDS full test: `analysis/external_validation/`
- Center-1 OOD validation: `analysis/external_validation/analyze_center1_ood_validation.py`
- Attribution: `analysis/attribution/`

## Provenance
`manifests/HISTORICAL_SOURCE_PROVENANCE.csv` records sanitized historical source provenance for the public package.
`manifests/SHA256_MANIFEST.csv` and `.json` provide file-level SHA-256 hashes.

The original experiments were archived before the project was maintained as a Git-tracked
snapshot. The new revision commit should therefore be cited together with the SHA-256 manifest;
it should not be described as the historical training commit.

## Adding data later
Use:
- `data/source_data/` for compact manuscript source-data tables
- `data/derived/` for new derived results
- `data/additional/` for later reviewer-response data

Large raw benchmark archives are intentionally excluded.


