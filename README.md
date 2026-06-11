# Frequency-residual metastasis classifier

This repository contains code for patch-level metastasis classification in histopathology using a specificity-aware frequency-residual architecture.

## Contents

- `src/freqres_pathology/models/`: model definitions
- `src/freqres_pathology/data/`: PCam loading and image transforms
- `src/freqres_pathology/eval/`: metrics, bootstrap intervals, and paired comparisons
- `scripts/`: command-line entry points for training and evaluation
- `tests/`: lightweight import checks

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Example commands

Evaluate a prediction CSV:

```bash
python scripts/evaluate_predictions.py --input predictions.csv --threshold 0.220 --out-dir outputs/evaluation
```

Compute bootstrap confidence intervals:

```bash
python scripts/bootstrap_ci.py --input predictions.csv --threshold 0.220 --out outputs/bootstrap_ci.csv
```

Train on PCam HDF5 files:

```bash
python scripts/train_pcam.py --data-root /path/to/pcam --out-dir outputs/run01 --epochs 20
```

Prediction CSV files should contain a binary label column and a positive-class score column. Standard names such as `y_true`, `label`, `y_score`, `score`, or `probability` are detected automatically.
