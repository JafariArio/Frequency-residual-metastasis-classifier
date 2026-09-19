# Reproduction guide

This repository contains the code, checkpoint, configurations, prediction
files, statistical analyses, external-validation scripts, attribution scripts,
and provenance records used for the reported frequency-residual metastasis
classification study.

## 1. Authoritative final-model package

The authoritative final model is represented by:

- `src/model/train_final_model.py`
- `configs/final_model/run_config.json`
- `configs/final_model/training_history.json`
- `configs/final_model/summary.json`
- `checkpoints/final_model/best_validation_constraint.pt`

The authoritative prediction files are stored under `predictions/`.

The original experiments were completed before the project was maintained as a
Git-tracked public snapshot. Therefore, the Git commit for this repository
identifies the public reproducibility package and must not be described as the
historical training commit.

Historical source provenance is recorded in:

- `manifests/HISTORICAL_SOURCE_PROVENANCE.csv`

Current public-file hashes are recorded in:

- `manifests/SHA256_MANIFEST.csv`
- `manifests/SHA256_MANIFEST.json`

## 2. Python environment

The repository-packaging environment used Python 3.13.15.

Create an isolated environment from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Several package versions in `requirements.txt` are exactly pinned because they
were directly verified in the packaging environment. Dependencies whose exact
version was not captured are listed without invented version numbers.

After installation, a fully resolved environment can optionally be recorded:

```powershell
python -m pip freeze > environment_freeze.txt
```

## 3. Verify the public package

Check that the principal artifacts exist:

```powershell
Test-Path "src\model\train_final_model.py"
Test-Path "checkpoints\final_model\best_validation_constraint.pt"
Test-Path "configs\final_model\run_config.json"
Test-Path "predictions\pcam\test_predictions.csv"
Test-Path "predictions\wilds_full_test\test_predictions.csv"
```

Compile all Python scripts:

```powershell
Get-ChildItem . -Recurse -File -Filter "*.py" |
ForEach-Object {
    python -m py_compile $_.FullName
}
```

## 4. Reproduce PCam bootstrap confidence intervals

The canonical PCam test prediction file is:

`predictions/pcam/test_predictions.csv`

Run:

```powershell
python analysis\statistics\bootstrap_confidence_intervals.py `
    --pred-csv "predictions\pcam\test_predictions.csv" `
    --outdir "results\pcam\bootstrap" `
    --n-bootstrap 2000 `
    --seed 20260517 `
    --ci-level 0.95 `
    --bootstrap-mode stratified `
    --save-draws `
    --write-xlsx `
    --verify-final-model
```

## 5. Reproduce paired comparator statistics

Canonical predictions are stored as:

- `predictions/pcam/test_predictions.csv`
- `predictions/baselines/resnet18/test_predictions.csv`
- `predictions/baselines/efficientnet_b0/test_predictions.csv`
- `predictions/baselines/convnext_tiny/test_predictions.csv`
- `predictions/baselines/deit_tiny/test_predictions.csv`

Run:

```powershell
python analysis\statistics\paired_comparator_statistics.py `
    --patheom-csv "predictions\pcam\test_predictions.csv" `
    --baseline-predictions-root "predictions\baselines" `
    --outdir "results\statistics\comparator_statistics" `
    --baselines resnet18 efficientnet_b0 convnext_tiny deit_tiny `
    --save-draws
```

## 6. Final-model training

The archived configuration is provided in:

`configs/final_model/run_config.json`

The selected final-model variant is:

`p7_r2_sens_reg_11m`

A new training run may be launched with:

```powershell
python src\model\train_final_model.py `
    --root "<PATH_TO_PCAM_DATA>" `
    --variant p7_r2_sens_reg_11m `
    --outdir "runs\final_model_retrain"
```

Use the values in `configs/final_model/run_config.json` for the archived
hyperparameter configuration.

The supplied checkpoint is the archived checkpoint used for the reported
final-model analyses. A newly trained model should not be assumed to reproduce
the archived checkpoint bit-for-bit because hardware, library versions, and
numerical execution can affect optimization.

## 7. Baseline benchmarking

The standardized comparator implementation is:

`baselines/run_baseline_benchmarks.py`

After installing all requirements:

```powershell
python baselines\run_baseline_benchmarks.py --help
```

Reported baseline prediction files are already provided under:

`predictions/baselines/`

Rerunning baseline training is therefore not required to reproduce the paired
statistical comparisons.

## 8. CAMELYON17-WILDS external evaluation

External-validation code is under:

`analysis/external_validation/`

The workflow includes:

1. `prepare_wilds_full_test.py`
2. `evaluate_wilds_full_test.py`
3. `prepare_wilds_ood_validation.py`
4. `evaluate_wilds_ood_validation.py`
5. `analyze_center1_ood_validation.py`

The reported full-test prediction file is:

`predictions/wilds_full_test/test_predictions.csv`

Center-1 OOD-validation predictions are stored under:

`predictions/center1_ood_validation/`

The external-evaluation scripts load the public final-model implementation and
the archived final checkpoint rather than a separate external-domain model.

## 9. Attribution analysis

Attribution code is stored under:

`analysis/attribution/`

PCam analyses:

- `generate_attribution_maps.py`
- `summarize_attribution_metrics.py`
- `build_attribution_galleries.py`
- `summarize_attribution_supplement.py`

CAMELYON17-WILDS analyses:

- `generate_wilds_attribution_maps.py`
- `build_wilds_attribution_galleries.py`

The principal displayed attribution maps use SmoothGrad with 16 noisy samples
and noise standard deviation 0.03. The full-dataset attribution summary uses a
separate batched gradient procedure implemented in the summary script and
should not be described as identical to the 16-sample SmoothGrad procedure.

## 10. Data

Large raw benchmark archives are intentionally excluded from the repository.

Repository data directories are organized as:

- `data/source_data/`
- `data/derived/`
- `data/additional/`
- `data/raw/`

PCam and CAMELYON17-WILDS data should be obtained from their original public
sources according to the corresponding dataset terms and prepared using the
supplied scripts.

## 11. Integrity verification

The SHA-256 manifest can be checked with:

```powershell
Import-Csv "manifests\SHA256_MANIFEST.csv" |
ForEach-Object {
    $path = Join-Path (Get-Location) ($_.path -replace "/", "\")
    if (Test-Path $path) {
        $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $_.sha256.ToLower()) {
            Write-Host "MISMATCH: $($_.path)"
        }
    }
}
```

No mismatch output indicates that the checked files agree with the stored
SHA-256 values.

## 12. Interpretation of the PCam test split

The official PCam test split was available during architecture development.
Consequently, results on this split represent in-benchmark test evaluation
rather than evaluation on a strictly untouched final holdout.

The frozen CAMELYON17-WILDS evaluation provides the independent cross-domain
assessment performed after final model and PCam operating-point selection.
